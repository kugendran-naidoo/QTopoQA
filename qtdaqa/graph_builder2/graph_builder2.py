#!/usr/bin/env python3
"""
qtdaqa.graph_builder

Generates PyG graph files (.pt) ONLY, with configurable feature construction.

Inputs
- --dataset-dir: folder containing per-target subfolders with .pdb decoys
- --work-dir:    folder for intermediates (interface, topo, node feature CSVs)
- --out-graphs:  destination for graph .pt files (one per decoy)
- --log-dir:     folder for logs (a per-run timestamped file is created)

Parallelism (CLI):
- --parallel N: number of decoys to process in parallel across all targets
  (cross-target concurrency). If omitted, falls back to jobs in other.json.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio.PDB import PDBParser

GRAPH_BUILDER2_DIR = Path(__file__).resolve().parent
LIB_DIR = GRAPH_BUILDER2_DIR / "lib"
# Ensure the local helpers (e.g., log_dirs.py) are importable even when this module
# is executed as a script rather than installed as a package.
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from directory_permissions import ensure_tree_readable, ensure_tree_readwrite
from log_dirs import LogDirectoryInfo, prepare_log_directory
from parallel_executor import normalise_worker_count
from new_calculate_interface import process_pdb_file
from new_topological_features import (
    ResidueDescriptor,
    TopologicalConfig,
    compute_features_for_residues,
)

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Error: pandas is required for node feature extraction.") from exc


def _locate_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        candidate = parent / "topoqa" / "src"
        if candidate.exists():
            return parent
    raise RuntimeError("Unable to locate repo root containing 'topoqa/src'.")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
TOPOQA_SRC = REPO_ROOT / "topoqa" / "src"
if str(TOPOQA_SRC) not in sys.path:
    sys.path.insert(0, str(TOPOQA_SRC))

from node_fea_df import node_fea  # type: ignore  # noqa: E402


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
INTERFACE_CUTOFF = 10.0
TOPOLOGY_NEIGHBOR_DISTANCE = 8.0
TOPOLOGY_FILTRATION_CUTOFF = 8.0
TOPOLOGY_MIN_PERSISTENCE = 0.01
# Mirror the legacy extractor's element groups so we emit 7 × (5 + 15) = 140 values.
TOPOLOGY_ELEMENT_FILTERS: Sequence[Sequence[str]] = (
    ("C",),
    ("N",),
    ("O",),
    ("C", "N"),
    ("C", "O"),
    ("N", "O"),
    ("C", "N", "O"),
)

NODE_FEATURE_DROP_NA = False
INTERFACE_COORD_DECIMALS = 3


def _format_element_filters(filters: Sequence[Sequence[str]]) -> str:
    parts: List[str] = []
    for group in filters:
        parts.append("{" + ",".join(group) + "}")
    return ", ".join(parts)


def _round_interface_file(path: Path, decimals: int) -> None:
    """Normalise coordinate precision so downstream scaling matches legacy output."""
    if decimals < 0:
        return
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    format_str = f"{{:.{decimals}f}}"
    updated_lines: List[str] = []
    changed = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            updated_lines.append("")
            continue
        parts = stripped.split()
        if len(parts) < 4:
            updated_lines.append(stripped)
            continue
        descriptor, coord_tokens = parts[0], parts[1:]
        try:
            rounded_coords = [format_str.format(float(value)) for value in coord_tokens]
        except ValueError:
            updated_lines.append(stripped)
            continue
        normalised_line = " ".join([descriptor, *rounded_coords])
        if normalised_line != stripped:
            changed = True
        updated_lines.append(normalised_line)

    if changed:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(updated_lines))
            handle.write("\n")


def _trim_suffix(stem: str, suffixes: tuple[str, ...]) -> str:
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem = stem.rstrip("_- .")
            lower = stem.lower()
    return stem


def _normalise_interface_name(name: str) -> str:
    return _trim_suffix(Path(name).stem, (".interface", "interface", "iface"))


def _normalise_topology_name(name: str) -> str:
    return _trim_suffix(Path(name).stem, (".topology", "topology", "node_topo"))


def _gather_interface_files(root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for pattern in ("*.interface.txt", "*.txt"):
        for path in root.rglob(pattern):
            if path.is_file():
                mapping.setdefault(_normalise_interface_name(path.name), []).append(path)
        if mapping:
            break
    return mapping


def _gather_topology_files(root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for pattern in ("*.topology.csv", "*.csv"):
        for path in root.rglob(pattern):
            if path.is_file():
                mapping.setdefault(_normalise_topology_name(path.name), []).append(path)
        if mapping:
            break
    return mapping


def _select_single_path(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    return sorted(paths, key=lambda p: (len(p.parts), str(p)))[0]


def _relative_node_output_path(base_dir: Path, interface_file: Path, model: str) -> Path:
    try:
        relative = interface_file.relative_to(base_dir)
        return relative.parent / f"{model}.csv"
    except ValueError:
        return Path(f"{model}.csv")


def _detect_node_fea_signature() -> int:
    try:
        return len(inspect.signature(node_fea).parameters)
    except (TypeError, ValueError):
        return -1


@dataclass
class NodeFeatureTask:
    model: str
    structure_path: Path
    interface_path: Path
    topology_path: Path
    output_path: Path


def _configure_main_logger(run_log_dir: Path) -> logging.Logger:
    """Initialise the main logger that writes to both file and stdout."""
    logger = logging.getLogger("graph_builder2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(run_log_dir / "graph_builder2.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def _create_per_pdb_logger(log_path: Path) -> logging.Logger:
    """Create a dedicated logger for a single PDB processing run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger(name=str(log_path))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _is_text_file(path: Path, *, chunk_size: int = 4096) -> bool:
    """Return ``True`` if the file appears to contain text data."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(chunk_size)
    except OSError:
        return False

    if b"\x00" in chunk:
        return False

    if not chunk:
        return True

    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        try:
            chunk.decode("latin-1")
        except UnicodeDecodeError:
            return False

    return True


def _check_dataset_readable(dataset_dir: Path) -> None:
    """Ensure the dataset directory hierarchy can be read."""
    try:
        ensure_tree_readable(dataset_dir)
    except PermissionError as exc:
        raise PermissionError(
            f"dataset directory '{dataset_dir}' is not readable"
        ) from exc


def _collect_structure_files(dataset_dir: Path) -> tuple[List[Path], List[Path]]:
    """Find .pdb and .cif files in the dataset directory and validate them."""
    try:
        pdb_files = [path for path in dataset_dir.rglob("*.pdb") if path.is_file()]
        cif_files = [path for path in dataset_dir.rglob("*.cif") if path.is_file()]
    except OSError as exc:
        raise RuntimeError(
            f"failed to scan dataset directory '{dataset_dir}'"
        ) from exc

    if not pdb_files and not cif_files:
        raise RuntimeError(
            f"dataset directory '{dataset_dir}' does not contain any '.pdb' or '.cif' files."
        )

    for label, files in ((".pdb", pdb_files), (".cif", cif_files)):
        for file_path in files:
            if not _is_text_file(file_path):
                raise RuntimeError(
                    f"{label} file '{file_path}' is not recognised as a text file."
                )

    return pdb_files, cif_files


def _check_rw_directories(paths: list[tuple[Path, str]]) -> None:
    """Ensure each directory in the list is readable and writeable."""
    for path, label in paths:
        try:
            ensure_tree_readwrite(path)
        except PermissionError as exc:
            raise PermissionError(
                f"{label} '{path}' is not read/write accessible"
            ) from exc


def _ensure_empty_directories(paths: list[tuple[Path, str]]) -> None:
    """Ensure each directory in the list is empty."""
    for path, label in paths:
        try:
            first_entry = next(path.iterdir(), None)
        except OSError as exc:
            raise RuntimeError(f"failed to inspect {label} '{path}'") from exc

        if first_entry is not None:
            raise RuntimeError(
                f"{label} '{path}' must be empty before running (found '{first_entry.name}')"
            )


def _collect_residue_descriptors(pdb_path: Path) -> List[ResidueDescriptor]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("topology", str(pdb_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDB file '{pdb_path}': {exc}") from exc

    descriptors: List[ResidueDescriptor] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, seq_number, insertion_code = residue.id
                if hetero_flag != " ":
                    continue
                insertion = insertion_code if insertion_code.strip() else " "
                resname = residue.get_resname()
                descriptor_str = f"c<{chain.id}>r<{seq_number}>"
                if insertion.strip():
                    descriptor_str += f"i<{insertion}>"
                descriptor_str += f"R<{resname}>"
                descriptors.append(ResidueDescriptor.from_string(descriptor_str))
    return descriptors


def _load_interface_descriptors(
    interface_path: Path,
    interface_cutoff: float,
) -> Tuple[List[ResidueDescriptor], Optional[str]]:
    descriptors: List[ResidueDescriptor] = []
    try:
        with interface_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                residue_token = stripped.split(None, 1)[0]
                try:
                    descriptors.append(ResidueDescriptor.from_string(residue_token))
                except ValueError:
                    return [], (
                        f"Interface file '{interface_path}' contains an unrecognised descriptor: "
                        f"'{residue_token}'"
                    )
    except FileNotFoundError:
        return [], f"Interface file not found: {interface_path}"
    except OSError as exc:
        return [], f"Failed to read interface file '{interface_path}': {exc}"

    if not descriptors:
        cutoff_str = f"{interface_cutoff:g}A"
        return [], f"No interface residues with cutoff of {cutoff_str} recorded in '{interface_path}'"

    return descriptors, None


def _write_topology_log(log_path: Path, lines: List[str]) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _generate_topology_features(
    pdb_path: Path,
    output_path: Path,
    config: TopologicalConfig,
    interface_path: Optional[Path] = None,
    interface_cutoff: float = INTERFACE_CUTOFF,
) -> tuple[int, Optional[str]]:
    if interface_path is not None:
        descriptors, error = _load_interface_descriptors(interface_path, interface_cutoff)
        if error:
            return 0, error
    else:
        descriptors = _collect_residue_descriptors(pdb_path)
        if not descriptors:
            return 0, "No standard residues found"

    frame = compute_features_for_residues(
        pdb_path,
        descriptors,
        config,
    )
    frame.to_csv(output_path, index=False)
    return len(descriptors), None


def _stage_node_feature_inputs(model: str, interface_path: Path, topology_path: Path) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix=f"node_features_{model}_")
    root = Path(temp_dir.name)
    iface_dir = root / "interface"
    topo_dir = root / "topology"
    iface_dir.mkdir(parents=True, exist_ok=True)
    topo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(interface_path, iface_dir / f"{model}.txt")
    shutil.copyfile(topology_path, topo_dir / f"{model}.csv")
    return temp_dir, iface_dir, topo_dir


def _initialise_node_fea(model: str, structure_path: Path, interface_dir: Path, topo_dir: Path, signature_size: int):
    iface = str(interface_dir)
    topo = str(topo_dir)
    if signature_size == 3:
        return node_fea(str(structure_path), iface, topo)
    if signature_size == 4:
        return node_fea(model, str(structure_path.parent), iface, topo)
    try:
        return node_fea(str(structure_path), iface, topo)
    except TypeError:
        return node_fea(model, str(structure_path.parent), iface, topo)


def _process_node_feature_task(
    task: NodeFeatureTask,
    signature_size: int,
    drop_na: bool,
) -> tuple[str, Optional[str]]:
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        temp_dir, iface_dir, topo_dir = _stage_node_feature_inputs(task.model, task.interface_path, task.topology_path)
        extractor = _initialise_node_fea(task.model, task.structure_path, iface_dir, topo_dir, signature_size)
        result = extractor.calculate_fea()
        if isinstance(result, tuple):
            fea_df = result[0]
        else:
            fea_df = result
        if drop_na:
            pd.set_option("future.no_silent_downcasting", True)
            fea_df.replace("NA", pd.NA, inplace=True)
            fea_df = fea_df.dropna()
        fea_df.to_csv(task.output_path, index=False)
        return task.model, None
    except Exception as exc:  # pragma: no cover
        return task.model, str(exc)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _build_node_feature_tasks(
    structure_map: Dict[str, Path],
    interface_dir: Path,
    topology_dir: Path,
    output_dir: Path,
) -> List[NodeFeatureTask]:
    interface_map = _gather_interface_files(interface_dir)
    topology_map = _gather_topology_files(topology_dir)
    shared_models = sorted(set(interface_map) & set(topology_map))

    tasks: List[NodeFeatureTask] = []
    for model in shared_models:
        interface_path = _select_single_path(interface_map.get(model, []))
        topology_path = _select_single_path(topology_map.get(model, []))
        structure_path = structure_map.get(model)
        if not interface_path or not topology_path or not structure_path:
            continue
        output_rel = _relative_node_output_path(interface_dir, interface_path, model)
        output_path = (output_dir / output_rel).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append(
            NodeFeatureTask(
                model=model,
                structure_path=structure_path,
                interface_path=interface_path,
                topology_path=topology_path,
                output_path=output_path,
            )
        )
    return tasks



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Configurable graph-only builder for QTopoQA")

    p.add_argument("-d", "--dataset-dir",
                   metavar="/datasets/Dockground_MAF2",
                   type=str, 
                   required=True, 
                   default=None,
                   help="Folder containing targets with .pdb decoys")

    p.add_argument("-w", "--work-dir", 
                   metavar="./work",
                   type=str, 
                   required=True, 
                   default=None,
                   help="Folder for intermediate files")

    p.add_argument("-o", "--out-graphs", 
                   metavar="./graph_data",
                   type=str, 
                   required=True, 
                   default=None,
                   help="Folder for graph .pt files (one per decoy)")

    p.add_argument("-l", "--log-dir", 
                   metavar="./logs",
                   type=str, 
                   required=True, 
                   default=None,
                   help="Folder for logs")

    p.add_argument("-p", "--parallel",
                   metavar="4",
                   type=int,
                   default=None,
                   help="Optional number of worker processes for parallel jobs",
                  )

    # If no arguments at all → print usage and exit
    if len(sys.argv) < 4:
        p.print_usage()
        sys.exit(1)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:

    # parse CLI arguments
    args = parse_args(argv)

    # CLI parameters - generate fully resolved directories
    dataset_dir = Path(args.dataset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_graphs = Path(args.out_graphs).resolve()
    log_root = Path(args.log_dir).resolve()

    try:
        log_info: LogDirectoryInfo = prepare_log_directory(log_root, run_prefix="graph_builder2")
    except Exception as exc:
        print(f"Error: unable to prepare log directory '{log_root}': {exc}", file=sys.stderr)
        return 2

    run_log_dir = log_info.run_dir
    logger = _configure_main_logger(run_log_dir)
    logger.info("=== Starting graph_builder2 run ===")
    logger.info(
        "Defaults: interface_cutoff=%.1f Å, topology_neighbor_distance=%.1f Å, "
        "topology_filtration_cutoff=%.1f Å, topology_min_persistence=%.4f, "
        "topology_element_filters=%s",
        INTERFACE_CUTOFF,
        TOPOLOGY_NEIGHBOR_DISTANCE,
        TOPOLOGY_FILTRATION_CUTOFF,
        TOPOLOGY_MIN_PERSISTENCE,
        _format_element_filters(TOPOLOGY_ELEMENT_FILTERS),
    )

    raw_args = list(argv) if argv is not None else sys.argv[1:]
    logger.info("CLI raw arguments: %s", raw_args)
    logger.info(
        "CLI parameters: dataset_dir=%s, work_dir=%s, out_graphs=%s, log_dir=%s, parallel=%s",
        args.dataset_dir,
        args.work_dir,
        args.out_graphs,
        args.log_dir,
        args.parallel,
    )

    parallel_cfg = normalise_worker_count(args.parallel, default_workers=8)
    worker_count = parallel_cfg.workers if parallel_cfg.workers is not None else 1
    logger.info("Parallel worker configuration: requested=%s, effective=%d", args.parallel, worker_count)

    try:
        _check_dataset_readable(dataset_dir)
        pdb_files, cif_files = _collect_structure_files(dataset_dir)
        structure_map: Dict[str, Path] = {}
        for path in list(pdb_files) + list(cif_files):
            structure_map.setdefault(path.stem, path)
        _check_rw_directories(
            [
                (work_dir, "work directory"),
                (out_graphs, "output graph directory"),
                (log_root, "log directory"),
            ]
        )
        _ensure_empty_directories(
            [
                (work_dir, "work directory"),
                (out_graphs, "output graph directory"),
            ]
        )
    except (PermissionError, RuntimeError) as exc:
        logger.error("Fatal validation error: %s", exc)
        return 2

    pdb_count = len(pdb_files)
    cif_count = len(cif_files)

    interface_dir = work_dir / "interface"
    logger.info(
        "Dataset verification complete: %d PDB files, %d CIF files. Preparing interface outputs.",
        pdb_count,
        cif_count,
    )
    try:
        interface_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Unable to prepare interface directory '%s': %s", interface_dir, exc)
        return 2

    interface_log_dir = run_log_dir / "interface_logs"
    interface_log_dir.mkdir(parents=True, exist_ok=True)
    interface_failure_entries: List[tuple[Path, Path, str]] = []

    tasks: List[tuple[Path, Path, Path]] = []
    for pdb_path in sorted(pdb_files):
        try:
            relative = pdb_path.relative_to(dataset_dir)
            relative_path = Path(relative)
        except ValueError:
            relative_path = Path(pdb_path.name)
        output_parent = interface_dir / relative_path.parent
        log_parent = interface_log_dir / relative_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        log_parent.mkdir(parents=True, exist_ok=True)
        output_path = output_parent / f"{pdb_path.stem}.interface.txt"
        log_path = log_parent / f"{pdb_path.stem}.log"
        tasks.append((pdb_path, output_path, log_path))

    generated_interface_files = 0
    interface_elapsed = 0.0

    if tasks:
        logger.info(
            "Beginning interface residue extraction for %d PDB files using %d worker(s)",
            len(tasks),
            worker_count,
        )
        interface_start = time.perf_counter()
        try:
            if worker_count <= 1:
                for pdb_path, output_path, log_path in tasks:
                    try:
                        residue_count = process_pdb_file(pdb_path, output_path, cutoff=INTERFACE_CUTOFF)
                        _round_interface_file(output_path, INTERFACE_COORD_DECIMALS)
                        generated_interface_files += 1
                        with log_path.open("w", encoding="utf-8") as handle:
                            handle.write(f"Interface residues: {residue_count}\n")
                    except Exception as exc:
                        interface_failure_entries.append((pdb_path, log_path, str(exc)))
                        with log_path.open("w", encoding="utf-8") as handle:
                            handle.write(f"Failure: {exc}\n")
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_to_task = {
                        executor.submit(process_pdb_file, pdb_path, output_path, INTERFACE_CUTOFF): (pdb_path, output_path, log_path)
                        for pdb_path, output_path, log_path in tasks
                    }
                    for future in as_completed(future_to_task):
                        pdb_path, output_path, log_path = future_to_task[future]
                        try:
                            residue_count = future.result()
                            _round_interface_file(output_path, INTERFACE_COORD_DECIMALS)
                            generated_interface_files += 1
                            with log_path.open("w", encoding="utf-8") as handle:
                                handle.write(f"Interface residues: {residue_count}\n")
                        except Exception as exc:
                            interface_failure_entries.append((pdb_path, log_path, str(exc)))
                            with log_path.open("w", encoding="utf-8") as handle:
                                handle.write(f"Failure: {exc}\n")
        except Exception as exc:
            logger.exception("Unexpected error during interface extraction: %s", exc)
            return 2
        interface_elapsed = time.perf_counter() - interface_start
    else:
        logger.warning("No PDB files found for interface extraction.")
    logger.info("Interface extraction elapsed time: %.2f s", interface_elapsed)

    topology_dir = work_dir / "topology"
    try:
        topology_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Unable to prepare topology directory '%s': %s", topology_dir, exc)
        return 2

    node_feature_dir = work_dir / "node_features"
    try:
        node_feature_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Unable to prepare node feature directory '%s': %s", node_feature_dir, exc)
        return 2

    topology_log_dir = run_log_dir / "topology_logs"
    topology_log_dir.mkdir(parents=True, exist_ok=True)

    topology_failure_entries: List[tuple[Path, Path, str]] = []
    topology_config = TopologicalConfig(
        neighbor_distance=TOPOLOGY_NEIGHBOR_DISTANCE,
        filtration_cutoff=TOPOLOGY_FILTRATION_CUTOFF,
        min_persistence=TOPOLOGY_MIN_PERSISTENCE,
        element_filters=TOPOLOGY_ELEMENT_FILTERS,
        workers=None,
        log_progress=False,
    )

    topology_tasks: List[tuple[Path, Path, Path, Path]] = []
    for pdb_path in sorted(pdb_files):
        try:
            relative = pdb_path.relative_to(dataset_dir)
            relative_path = Path(relative)
        except ValueError:
            relative_path = Path(pdb_path.name)
        topology_output_parent = topology_dir / relative_path.parent
        topology_log_parent = topology_log_dir / relative_path.parent
        topology_output_parent.mkdir(parents=True, exist_ok=True)
        topology_log_parent.mkdir(parents=True, exist_ok=True)
        interface_path = interface_dir / relative_path.parent / f"{pdb_path.stem}.interface.txt"
        topology_output_path = topology_output_parent / f"{pdb_path.stem}.topology.csv"
        topology_log_path = topology_log_parent / f"{pdb_path.stem}.log"
        topology_tasks.append((pdb_path, interface_path, topology_output_path, topology_log_path))

    generated_topology_files = 0
    topology_elapsed = 0.0

    if topology_tasks:
        logger.info(
            "Beginning topological feature extraction for %d PDB files using %d worker(s)",
            len(topology_tasks),
            worker_count,
        )
        topology_start = time.perf_counter()
        try:
            if worker_count <= 1:
                for pdb_path, interface_path, output_path, log_path in topology_tasks:
                    try:
                        residue_count, error_msg = _generate_topology_features(
                            pdb_path,
                            output_path,
                            topology_config,
                            interface_path=interface_path,
                            interface_cutoff=INTERFACE_CUTOFF,
                        )
                    except Exception as exc:
                        residue_count, error_msg = 0, str(exc)
                    if error_msg:
                        topology_failure_entries.append((pdb_path, log_path, error_msg))
                        _write_topology_log(
                            log_path,
                            [
                                f"PDB: {pdb_path}",
                                f"Status: FAILURE",
                                f"Error: {error_msg}",
                            ],
                        )
                    else:
                        generated_topology_files += 1
                        _write_topology_log(
                            log_path,
                            [
                                f"PDB: {pdb_path}",
                                "Status: SUCCESS",
                                f"Residues processed: {residue_count}",
                                f"Output file: {output_path}",
                            ],
                        )
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_to_task = {
                        executor.submit(
                            _generate_topology_features,
                            pdb_path,
                            output_path,
                            topology_config,
                            interface_path,
                            INTERFACE_CUTOFF,
                        ): (pdb_path, output_path, log_path)
                        for pdb_path, interface_path, output_path, log_path in topology_tasks
                    }
                    for future in as_completed(future_to_task):
                        pdb_path, output_path, log_path = future_to_task[future]
                        try:
                            residue_count, error_msg = future.result()
                        except Exception as exc:
                            residue_count, error_msg = 0, str(exc)
                        if error_msg:
                            topology_failure_entries.append((pdb_path, log_path, error_msg))
                            _write_topology_log(
                                log_path,
                                [
                                    f"PDB: {pdb_path}",
                                    f"Status: FAILURE",
                                    f"Error: {error_msg}",
                                ],
                            )
                        else:
                            generated_topology_files += 1
                            _write_topology_log(
                                log_path,
                                [
                                    f"PDB: {pdb_path}",
                                    "Status: SUCCESS",
                                    f"Residues processed: {residue_count}",
                                    f"Output file: {output_path}",
                                ],
                            )
        except Exception as exc:
            logger.exception("Unexpected error during topology feature extraction: %s", exc)
            return 2
        topology_elapsed = time.perf_counter() - topology_start
    else:
        logger.warning("No PDB files found for topological feature extraction.")
    logger.info("Topological feature extraction elapsed time: %.2f s", topology_elapsed)

    logger.info("Dataset directory: %s", dataset_dir)
    logger.info("Work directory: %s", work_dir)
    logger.info("Output graph directory: %s", out_graphs)
    logger.info("Run log directory: %s", run_log_dir)
    logger.info("PDB file count: %d", pdb_count)
    logger.info("CIF file count: %d", cif_count)
    logger.info("Interface output directory: %s", interface_dir)
    logger.info("Per-PDB interface logs directory: %s", interface_log_dir)
    logger.info("Interface success count: %d", generated_interface_files)
    node_feature_tasks = _build_node_feature_tasks(structure_map, interface_dir, topology_dir, node_feature_dir)
    node_feature_success = 0
    node_feature_failures: List[Tuple[str, str]] = []
    node_elapsed = 0.0

    if node_feature_tasks:
        logger.info(
            "Beginning node feature extraction for %d model(s) using %d worker(s)",
            len(node_feature_tasks),
            worker_count,
        )
        node_start = time.perf_counter()
        signature_size = _detect_node_fea_signature()
        if worker_count <= 1:
            for task in node_feature_tasks:
                model, error = _process_node_feature_task(task, signature_size, NODE_FEATURE_DROP_NA)
                if error:
                    node_feature_failures.append((task.model, error))
                    logger.warning("Node feature extraction failed for %s: %s", task.model, error)
                else:
                    node_feature_success += 1
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _process_node_feature_task,
                        task,
                        signature_size,
                        NODE_FEATURE_DROP_NA,
                    ): task
                    for task in node_feature_tasks
                }
                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        model, error = future.result()
                    except Exception as exc:  # pragma: no cover
                        node_feature_failures.append((task.model, str(exc)))
                        logger.warning("Node feature extraction failed for %s: %s", task.model, exc)
                    else:
                        if error:
                            node_feature_failures.append((model, error))
                            logger.warning("Node feature extraction failed for %s: %s", model, error)
                        else:
                            node_feature_success += 1
        node_elapsed = time.perf_counter() - node_start
    else:
        logger.info("No node feature tasks to process (missing interface/topology pairs)")

    logger.info("Topology output directory: %s", topology_dir)
    logger.info("Per-PDB topology logs directory: %s", topology_log_dir)
    logger.info("Topology success count: %d", generated_topology_files)
    logger.info("Node feature output directory: %s", node_feature_dir)
    logger.info("Node feature success count: %d", node_feature_success)
    logger.info("Node feature extraction elapsed time: %.2f s", node_elapsed)

    if interface_failure_entries:
        logger.warning("Interface failure count: %d", len(interface_failure_entries))
        for failed_path, log_path, error in interface_failure_entries:
            try:
                label = failed_path.relative_to(dataset_dir)
            except ValueError:
                label = failed_path
            logger.warning(" - %s failed with %s (see log: %s)", label, error, log_path)
    else:
        logger.info("All interface extractions completed successfully.")

    if topology_failure_entries:
        logger.warning("Topology failure count: %d", len(topology_failure_entries))
        for failed_path, log_path, error in topology_failure_entries:
            try:
                label = failed_path.relative_to(dataset_dir)
            except ValueError:
                label = failed_path
            logger.warning(" - %s failed with %s (see log: %s)", label, error, log_path)
    else:
        logger.info("All topological feature extractions completed successfully.")

    if node_feature_failures:
        logger.warning("Node feature failure count: %d", len(node_feature_failures))
        for model, error in node_feature_failures:
            logger.warning(" - %s failed with %s", model, error)
    else:
        logger.info("All node feature extractions completed successfully.")

    logger.info("=== graph_builder2 run completed ===")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
