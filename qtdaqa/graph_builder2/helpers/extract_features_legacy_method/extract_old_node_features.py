#!/usr/bin/env python3
"""Extract legacy node features (basic + topology) using the original topoqa stack."""
from __future__ import annotations

import argparse
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Error: pandas is required to run this script.") from exc

# Make topoqa/src importable
REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOQA_SRC = REPO_ROOT / "topoqa" / "src"
if str(TOPOQA_SRC) not in sys.path:
    sys.path.insert(0, str(TOPOQA_SRC))

from node_fea_df import node_fea  # type: ignore  # noqa: E402


def _trim_suffix(stem: str, suffixes: tuple[str, ...]) -> str:
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem = stem.rstrip("_- .")
            break
    return stem


def _normalise_interface_name(name: str) -> str:
    stem = Path(name).stem
    stem = _trim_suffix(stem, (".interface", "interface", "iface"))
    return stem


def _normalise_topology_name(name: str) -> str:
    stem = Path(name).stem
    stem = _trim_suffix(stem, (".topology", "topology", "node_topo"))
    return stem


def _gather_interface_files(root: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for pattern in ("*.interface.txt", "*.txt"):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            key = _normalise_interface_name(path.name)
            mapping.setdefault(key, []).append(path)
        if mapping:
            break
    return mapping


def _gather_topology_files(root: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for pattern in ("*.topology.csv", "*.csv"):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            key = _normalise_topology_name(path.name)
            mapping.setdefault(key, []).append(path)
        if mapping:
            break
    return mapping


def _index_structure_files(dataset_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for ext in (".pdb", ".cif", ".mmcif"):
        for path in dataset_dir.rglob(f"*{ext}"):
            if path.is_file():
                mapping.setdefault(path.stem, path)
    return mapping


def _select_single(path_list: list[Path]) -> Optional[Path]:
    if not path_list:
        return None
    if len(path_list) == 1:
        return path_list[0]
    # Prefer shortest relative path (most specific) to break ties deterministically
    return sorted(path_list, key=lambda p: (len(p.parts), str(p)))[0]


def _detect_node_fea_signature() -> int:
    try:
        signature = inspect.signature(node_fea)
        return len(signature.parameters)
    except (TypeError, ValueError):
        return -1


@dataclass
class ExtractionTask:
    model: str
    pdb_path: Path
    interface_path: Path
    topology_path: Path
    output_path: Path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory containing structures and precomputed subdirectories 'interface_ca' and 'node_topo'. (required)",
    )
    parser.add_argument(
        "--interface-features-dir",
        type=Path,
        required=True,
        help="Directory containing interface feature files (one .interface.txt per model). (required)",
    )
    parser.add_argument(
        "--topo-features-dir",
        type=Path,
        required=True,
        help="Directory containing topology feature CSV files (one per model). (required)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write node feature CSV files. (required)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker threads (1 = no parallelism).",
    )
    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows containing NA/None before writing output.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional limit on the number of models to process (0 = all).",
    )
    return parser.parse_args(list(argv))


def _relative_output_path(root: Path, interface_file: Path, model: str) -> Path:
    try:
        relative = interface_file.relative_to(root)
        return relative.with_suffix(".csv").with_name(f"{model}.csv")
    except ValueError:
        return Path(f"{model}.csv")


def _create_task(
    structure_map: dict[str, Path],
    interface_root: Path,
    topology_root: Path,
    model: str,
    interface_path: Path,
    topology_path: Path,
    output_root: Path,
) -> Optional[ExtractionTask]:
    pdb_path = structure_map.get(model)
    if pdb_path is None:
        print(f"[warn] Skipping {model}: no structure file found.")
        return None
    output_rel = _relative_output_path(interface_root, interface_path, model)
    output_path = (output_root / output_rel).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return ExtractionTask(
        model=model,
        pdb_path=pdb_path,
        interface_path=interface_path,
        topology_path=topology_path,
        output_path=output_path,
    )


def _build_tasks(
    structure_map: dict[str, Path],
    interface_dir: Path,
    topology_dir: Path,
    output_dir: Path,
    max_models: int,
) -> list[ExtractionTask]:
    interface_map = _gather_interface_files(interface_dir)
    topology_map = _gather_topology_files(topology_dir)

    shared_models = sorted(set(interface_map) & set(topology_map))

    tasks: list[ExtractionTask] = []
    for model in shared_models:
        iface_path = _select_single(interface_map.get(model, []))
        topo_path = _select_single(topology_map.get(model, []))
        if iface_path is None or topo_path is None:
            print(f"[warn] Skipping {model}: duplicate or missing interface/topology files.")
            continue
        task = _create_task(
            structure_map,
            interface_dir,
            topology_dir,
            model,
            iface_path,
            topo_path,
            output_dir,
        )
        if task:
            tasks.append(task)
        if max_models and len(tasks) >= max_models:
            break
    return tasks


def _initialise_node_fea(
    model: str,
    pdb_path: Path,
    interface_dir: Path,
    topo_dir: Path,
    signature_size: int,
):
    iface = str(interface_dir)
    topo = str(topo_dir)
    if signature_size == 3:
        return node_fea(str(pdb_path), iface, topo)
    if signature_size == 4:
        return node_fea(model, str(pdb_path.parent), iface, topo)
    try:
        return node_fea(str(pdb_path), iface, topo)
    except TypeError:
        return node_fea(model, str(pdb_path.parent), iface, topo)


def _prepare_feature_directories(task: ExtractionTask) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    tmp_root = tempfile.TemporaryDirectory(prefix=f"node_fea_{task.model}_")
    tmp_base = Path(tmp_root.name)
    interface_dir = tmp_base / "interface"
    topo_dir = tmp_base / "topology"
    interface_dir.mkdir(parents=True, exist_ok=True)
    topo_dir.mkdir(parents=True, exist_ok=True)

    expected_interface = interface_dir / f"{task.model}.txt"
    expected_topology = topo_dir / f"{task.model}.csv"

    shutil.copyfile(task.interface_path, expected_interface)
    shutil.copyfile(task.topology_path, expected_topology)

    return tmp_root, interface_dir, topo_dir


def _process_task(task: ExtractionTask, drop_na: bool, signature_size: int) -> tuple[str, Optional[str]]:
    tmp_handle: Optional[tempfile.TemporaryDirectory] = None
    try:
        tmp_handle, iface_dir, topo_dir = _prepare_feature_directories(task)
        nf = _initialise_node_fea(task.model, task.pdb_path, iface_dir, topo_dir, signature_size)
        result = nf.calculate_fea()
        if isinstance(result, tuple):
            fea_df = result[0]
        else:
            fea_df = result
        if drop_na:
            pd.set_option("future.no_silent_downcasting", True)
            fea_df.replace("NA", pd.NA, inplace=True)
            fea_df = fea_df.dropna()
        fea_df.to_csv(task.output_path, index=False)
        if tmp_handle is not None:
            tmp_handle.cleanup()
        return task.model, None
    except Exception as exc:  # pragma: no cover - operational logging
        if tmp_handle is not None:
            tmp_handle.cleanup()
        return task.model, str(exc)


def run_extraction(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    interface_dir = args.interface_features_dir.resolve()
    topology_dir = args.topo_features_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("Configuration:")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  interface_dir: {interface_dir}")
    print(f"  topology_dir: {topology_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  jobs: {args.jobs}  # number of worker threads (1 = sequential)")
    print(f"  drop_na: {args.drop_na}  # drop rows containing NA before saving")
    print(f"  max_models: {args.max_models}  # optional limit on models to process (0 = all)")

    if not dataset_dir.is_dir():
        print(f"[error] dataset directory not found: {dataset_dir}")
        return 2
    if not interface_dir.is_dir():
        print(f"[error] interface directory not found: {interface_dir}")
        return 2
    if not topology_dir.is_dir():
        print(f"[error] topology directory not found: {topology_dir}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(
        _index_structure_files(dataset_dir),
        interface_dir,
        topology_dir,
        output_dir,
        args.max_models,
    )

    if not tasks:
        print("[warn] No matching models found; nothing to do.")
        return 0

    signature_size = _detect_node_fea_signature()

    success = 0
    failures: list[tuple[str, str]] = []

    if args.jobs and args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_map = {
                executor.submit(_process_task, task, args.drop_na, signature_size): task.model for task in tasks
            }
            for future in as_completed(future_map):
                model = future_map[future]
                model_name, error = future.result()
                if error:
                    failures.append((model_name, error))
                    print(f"[warn] {model_name}: {error}")
                else:
                    success += 1
    else:
        for task in tasks:
            model_name, error = _process_task(task, args.drop_na, signature_size)
            if error:
                failures.append((model_name, error))
                print(f"[warn] {model_name}: {error}")
            else:
                success += 1

    print("Summary:")
    print(f"  processed: {len(tasks)}")
    print(f"  succeeded: {success}")
    print(f"  failed:    {len(failures)}")
    if failures:
        print("Failures:")
        for model, error in failures:
            print(f"  {model}: {error}")
        return 1
    return 0


def describe_source() -> None:
    print("This script reuses code from topoqa/src/node_fea_df.py (node_fea class).")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    return run_extraction(args)


if __name__ == "__main__":
    describe_source()
    raise SystemExit(main(sys.argv[1:]))
