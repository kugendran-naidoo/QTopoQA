#!/usr/bin/env python3
"""Compare topological feature CSVs produced by two runs.

The script builds on ideas from ``compare_topo_features2.py``:

* align rows by a stable identifier (``ID``) before comparison so ordering
  mismatches do not trigger false positives;
* compute absolute numeric differences and report the maximum, treating values
  within a configurable tolerance as equivalent.

Strengths of these ideas: they are simple, easy to inspect, and they tolerate
floating‑point noise.  Shortcomings: they still rely on matching column names
exactly and only examine CSVs that exist in both trees.

Suggested improvements (not implemented here):

* hash canonicalised numeric arrays as an additional guard;
* emit per-column statistics (mean/std of deltas) to spot systemic drift;
* optionally visualise distributions of differences for manual review.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ComparisonResult:
    relative_path: Path
    max_abs_diff: float
    exceeded: List[Tuple[str, str, float]]  # (row_id, column, diff)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        type=Path,
        help="Directory containing the baseline topology CSVs (e.g. output/work/topology)",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="Directory containing the new topology CSVs to compare",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Maximum absolute difference tolerated (default: %(default)s)",
    )
    return parser.parse_args(list(argv))


def collect_csvs(root: Path) -> Dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in root.rglob("*.csv")
        if path.is_file()
    }


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ID" in df.columns:
        df = df.set_index("ID", drop=False)
    return df


def compare_dataframes(
    df_ref: pd.DataFrame,
    df_new: pd.DataFrame,
    tolerance: float,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    if not df_ref.columns.equals(df_new.columns):
        raise ValueError("Column names/order differ between dataframes")
    if len(df_ref) != len(df_new):
        raise ValueError("Row counts differ between dataframes")

    numeric_cols = df_ref.select_dtypes(include=[np.number]).columns
    abs_diff = (df_ref[numeric_cols] - df_new[numeric_cols]).abs()
    max_abs_diff = float(abs_diff.to_numpy().max(initial=0.0))

    exceeded: List[Tuple[str, str, float]] = []
    mask = abs_diff > tolerance
    if mask.any().any():
        row_indices, col_indices = np.where(mask.to_numpy())
        id_index = df_ref.index
        for r_idx, c_idx in zip(row_indices, col_indices):
            row_id = str(id_index[r_idx])
            column = numeric_cols[c_idx]
            diff = float(abs_diff.iloc[r_idx, c_idx])
            exceeded.append((row_id, column, diff))

    return max_abs_diff, exceeded


def compare_directories(
    reference_dir: Path,
    candidate_dir: Path,
    tolerance: float,
) -> Tuple[List[ComparisonResult], List[Path], List[Path]]:
    reference_csvs = collect_csvs(reference_dir)
    candidate_csvs = collect_csvs(candidate_dir)

    shared_paths = sorted(set(reference_csvs) & set(candidate_csvs))
    missing_in_candidate = sorted(set(reference_csvs) - set(candidate_csvs))
    missing_in_reference = sorted(set(candidate_csvs) - set(reference_csvs))

    results: List[ComparisonResult] = []
    for rel_path in shared_paths:
        df_ref = load_dataframe(reference_csvs[rel_path])
        df_new = load_dataframe(candidate_csvs[rel_path])
        max_diff, exceeded = compare_dataframes(df_ref, df_new, tolerance)
        results.append(
            ComparisonResult(
                relative_path=rel_path,
                max_abs_diff=max_diff,
                exceeded=exceeded,
            )
        )

    return results, missing_in_candidate, missing_in_reference


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    reference_dir = args.reference.resolve()
    candidate_dir = args.candidate.resolve()

    if not reference_dir.is_dir():
        print(f"Reference directory not found: {reference_dir}", file=sys.stderr)
        return 2
    if not candidate_dir.is_dir():
        print(f"Candidate directory not found: {candidate_dir}", file=sys.stderr)
        return 2

    results, missing_candidate, missing_reference = compare_directories(
        reference_dir,
        candidate_dir,
        tolerance=args.tolerance,
    )

    print(f"Reference root: {reference_dir}")
    print(f"Candidate root: {candidate_dir}")
    print(f"Shared CSV files: {len(results)}")
    print(f"Missing in candidate: {len(missing_candidate)}")
    print(f"Missing in reference: {len(missing_reference)}")

    if missing_candidate:
        print("  -> Missing in candidate:")
        for rel_path in missing_candidate:
            print(f"     {rel_path}")
    if missing_reference:
        print("  -> Missing in reference:")
        for rel_path in missing_reference:
            print(f"     {rel_path}")

    mismatches = 0
    for result in results:
        if result.exceeded:
            mismatches += 1
            print(
                f"{result.relative_path} : max_abs_diff={result.max_abs_diff:.3e} "
                f"(differences > {args.tolerance:.1e})"
            )
            for row_id, column, diff in result.exceeded:
                print(
                    f"    row={row_id}, column='{column}', diff={diff:.3e}"
                )
        else:
            print(
                f"{result.relative_path} : max_abs_diff={result.max_abs_diff:.3e} "
                f"(within tolerance)"
            )

    print(
        f"Summary: {len(results) - mismatches} files within tolerance, "
        f"{mismatches} files exceeding tolerance."
    )

    return 0 if mismatches == 0 and not missing_candidate and not missing_reference else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
