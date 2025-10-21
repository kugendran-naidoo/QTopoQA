#!/usr/bin/env python3
"""CLI wrapper that delegates .pt graph generation to the library helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

from lib.pt_writer import DEFAULT_ARR_CUTOFF, generate_pt_files


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interface-dir", type=Path, required=True,
                        help="Directory containing interface files (.interface.txt). (required)")
    parser.add_argument("--topology-dir", type=Path, required=True,
                        help="Directory containing topology CSV files (.topology.csv). (required)")
    parser.add_argument("--node-dir", type=Path, required=True,
                        help="Directory containing node feature CSV files (.csv). (required)")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Directory containing source structures (.pdb). (required)")
    parser.add_argument("--output-pt-dir", type=Path, required=True,
                        help="Directory where .pt graphs will be written. (required)")
    parser.add_argument("--arr-cutoff", nargs="*", default=list(DEFAULT_ARR_CUTOFF),
                        help="Edge distance cutoffs passed to create_graph (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of worker threads (1 = sequential).")
    parser.add_argument("--log-dir", type=Path, default=Path("pt_logs"),
                        help="Directory to store pt generation logs.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    arr_cutoff: Sequence[str] = args.arr_cutoff or list(DEFAULT_ARR_CUTOFF)

    result = generate_pt_files(
        interface_dir=args.interface_dir,
        topology_dir=args.topology_dir,
        node_dir=args.node_dir,
        dataset_dir=args.dataset_dir,
        output_pt_dir=args.output_pt_dir,
        jobs=args.jobs,
        arr_cutoff=arr_cutoff,
        log_dir=args.log_dir,
    )

    print("=== write_pt_file summary ===")
    print(f"Processed : {result.processed}")
    print(f"Succeeded : {result.success_count}")
    print(f"Failed    : {len(result.failures)}")
    print(f"Logs      : {result.log_dir}")
    if result.failures:
        print("Failures:")
        for model, error, log_path in result.failures:
            print(f"  - {model}: {error} (log: {log_path})")

    return 0 if not result.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
