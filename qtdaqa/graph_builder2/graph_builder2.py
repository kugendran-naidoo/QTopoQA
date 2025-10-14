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
import sys
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
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    pdb_count = len(pdb_files)
    cif_count = len(cif_files)

    # add log directory and logs with <run_prefix>.<timestamp>
    log_info: LogDirectoryInfo = prepare_log_directory(log_root, run_prefix="graph_builder2")
    log_dir = log_info.root_dir
    run_log_dir = log_info.run_dir

    # Placeholder for upcoming parallel work; normalise the CLI input so later
    # code can simply check ``parallel_cfg.workers``.
    parallel_cfg = normalise_worker_count(args.parallel, default_workers=None)
    _ = parallel_cfg  # placeholder until processing is integrated

    # CLI parameter
    print(f"dataset_dir: ", dataset_dir)
    print(f"work_dir: ", work_dir)
    print(f"out_graphs: ", out_graphs)
    print(f"run_log_dir: ", run_log_dir)
    print(f"pdb_file_count: ", pdb_count)
    print(f"cif_file_count: ", cif_count)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
