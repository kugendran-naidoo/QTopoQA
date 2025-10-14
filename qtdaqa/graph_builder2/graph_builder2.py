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
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

GRAPH_BUILDER2_DIR = Path(__file__).resolve().parent
LIB_DIR = GRAPH_BUILDER2_DIR / "lib"
# Ensure the local helpers (e.g., log_dirs.py) are importable even when this module
# is executed as a script rather than installed as a package.
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from directory_permissions import ensure_tree_readable, ensure_tree_readwrite
from log_dirs import LogDirectoryInfo, prepare_log_directory
from parallel_executor import ParallelConfig, normalise_worker_count
from new_calculate_interface import process_pdb_file


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _configure_main_logger(run_log_dir: Path) -> logging.Logger:
    """Initialise the main logger that writes to both file and stdout."""
    logger = logging.getLogger("graph_builder2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

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

    p.add_argument("--parallel",
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

    try:
        _check_dataset_readable(dataset_dir)
        pdb_files, cif_files = _collect_structure_files(dataset_dir)
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
        relative = pdb_path.relative_to(dataset_dir)
        relative_path = Path(relative)
        output_parent = interface_dir / relative_path.parent
        log_parent = interface_log_dir / relative_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        log_parent.mkdir(parents=True, exist_ok=True)
        output_path = output_parent / f"{pdb_path.stem}.interface.txt"
        log_path = log_parent / f"{pdb_path.stem}.log"
        tasks.append((pdb_path, output_path, log_path))

    generated_interface_files = 0

    if tasks:
        logger.info(
            "Beginning interface residue extraction for %d PDB files using 8 workers",
            len(tasks),
        )
        try:
            with ProcessPoolExecutor(max_workers=8) as executor:
                future_to_task = {
                    executor.submit(process_pdb_file, pdb_path, output_path): (pdb_path, output_path, log_path)
                    for pdb_path, output_path, log_path in tasks
                }
                for future in as_completed(future_to_task):
                    pdb_path, output_path, log_path = future_to_task[future]
                    try:
                        residue_count = future.result()
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
    else:
        logger.warning("No PDB files found for interface extraction.")

    # Placeholder for upcoming parallel work; normalise the CLI input so later
    # code can simply check ``parallel_cfg.workers``.
    parallel_cfg = normalise_worker_count(args.parallel, default_workers=None)
    logger.info("Parallel worker configuration: %s", parallel_cfg.workers)

    logger.info("Dataset directory: %s", dataset_dir)
    logger.info("Work directory: %s", work_dir)
    logger.info("Output graph directory: %s", out_graphs)
    logger.info("Run log directory: %s", run_log_dir)
    logger.info("PDB file count: %d", pdb_count)
    logger.info("CIF file count: %d", cif_count)
    logger.info("Interface output directory: %s", interface_dir)
    logger.info("Per-PDB interface logs directory: %s", interface_log_dir)
    logger.info("Interface success count: %d", generated_interface_files)

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

    logger.info("=== graph_builder2 run completed ===")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
