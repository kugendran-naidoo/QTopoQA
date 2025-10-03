#!/usr/bin/env python3
"""
qtdaqa.graph_builder

Generates PyG graph files (.pt) ONLY, with configurable feature construction.

Inputs
- --train-validate-list: list containing targets for training/validation
                         - match list to dataset-dir with .pdb decoys
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
from typing import Optional, Sequence

GRAPH_BUILDER2_DIR = Path(__file__).resolve().parent
LIB_DIR = GRAPH_BUILDER2_DIR / "lib"
# Ensure the local helpers (e.g., log_dirs.py) are importable even when this module
# is executed as a script rather than installed as a package.
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from log_dirs import LogDirectoryInfo, prepare_log_directory
from parallel_executor import ParallelConfig, normalise_worker_count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Configurable graph-only builder for QTopoQA")

    p.add_argument("-t", "--train-validate-target-list",
                   metavar="train_validate.csv",
                   type=str, 
                   required=True, 
                   default=None,
                   help="Target list containing for training/validation")

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
    if len(sys.argv) == 1:
        p.print_usage()
        sys.exit(1)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:

    # parse CLI arguments
    args = parse_args(argv)

    # CLI parameter
    train_val_file = args.train_validate_target_list
    print(f"file: ",train_val_file)

    # CLI parameters - generate fully resolved directories
    dataset_dir = Path(args.dataset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_graphs = Path(args.out_graphs).resolve()

    # add log directory and logs
    log_info: LogDirectoryInfo = prepare_log_directory(Path(args.log_dir), run_prefix="graph_builder2")
    log_dir = log_info.root_dir
    run_log_dir = log_info.run_dir

    # Placeholder for upcoming parallel work; normalise the CLI input so later
    # code can simply check ``parallel_cfg.workers``.
    parallel_cfg = normalise_worker_count(args.parallel, default_workers=None)
    _ = parallel_cfg  # placeholder until processing is integrated

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
