#!/usr/bin/env python3
"""Compare interface outputs between two work directories."""

from __future__ import annotations

import argparse
import filecmp
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare interface output files between two directories."
    )
    parser.add_argument(
        "interface_1",
        type=Path,
        help="Directory containing the first set of interface outputs",
    )
    parser.add_argument(
        "interface_2",
        type=Path,
        help="Directory containing the second set of interface outputs",
    )
    return parser.parse_args()


def gather_files(base_dir: Path) -> dict[Path, Path]:
    return {
        path.relative_to(base_dir): path
        for path in base_dir.rglob("*")
        if path.is_file()
    }


def main() -> int:
    args = parse_args()
    baseline_dir = args.interface_1.resolve()
    candidate_dir = args.interface_2.resolve()

    if not baseline_dir.is_dir():
        print(f"Error: baseline directory does not exist: {baseline_dir}")
        return 2
    if not candidate_dir.is_dir():
        print(f"Error: candidate directory does not exist: {candidate_dir}")
        return 2

    baseline_files = gather_files(baseline_dir)
    candidate_files = gather_files(candidate_dir)

    missing_in_candidate = sorted(set(baseline_files) - set(candidate_files))
    missing_in_baseline = sorted(set(candidate_files) - set(baseline_files))

    identical = 0
    different = 0

    for relative_path in sorted(set(baseline_files) & set(candidate_files)):
        baseline_path = baseline_files[relative_path]
        candidate_path = candidate_files[relative_path]
        if filecmp.cmp(baseline_path, candidate_path, shallow=False):
            identical += 1
        else:
            different += 1

    total_shared = identical + different
    print(f"Baseline directory: {baseline_dir}")
    print(f"Candidate directory: {candidate_dir}")
    print(f"Shared files compared: {total_shared}")
    print(f"  Identical files:     {identical}")
    print(f"  Different files:     {different}")
    print(f"Missing in candidate:  {len(missing_in_candidate)}")
    print(f"Missing in baseline:   {len(missing_in_baseline)}")

    if missing_in_candidate:
        print("  Files missing in candidate:")
        for path in missing_in_candidate:
            print(f"    {path}")
    if missing_in_baseline:
        print("  Files missing in baseline:")
        for path in missing_in_baseline:
            print(f"    {path}")

    if different or missing_in_candidate or missing_in_baseline:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
