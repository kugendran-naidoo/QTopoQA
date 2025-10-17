#!/usr/bin/env python3
"""Extract interface features using the legacy cal_interface implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import sys
sys.path.insert(0, "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa/src")

from get_interface import cal_interface


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/datasets/training/Dockground_MAF2"),
        help="Directory containing the Dockground_MAF2 PDBs (default: %(default)s)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff (Å) for interface detection (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_old_method/work/interface"),
        help="Directory to write interface text files (default: %(default)s)",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    print("Configuration:")
    print(f"  dataset_dir: {args.dataset_dir}")
    print(f"  cutoff: {args.cutoff}")
    print(f"  output_dir: {args.output_dir}")

    pdb_files = sorted(args.dataset_dir.rglob("*.pdb"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdb_path in pdb_files:
        cal = cal_interface(str(pdb_path), cut=args.cutoff)
        output_path = output_dir / f"{pdb_path.stem}.interface.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cal.find_and_write(str(output_path))
        print(f"Wrote {output_path}")

    return 0


def describe_source() -> None:
    print("This script reuses code from:")
    print("  topoqa/src/get_interface.py (cal_interface class)")


if __name__ == "__main__":
    import sys

    describe_source()
    raise SystemExit(main(sys.argv[1:]))
