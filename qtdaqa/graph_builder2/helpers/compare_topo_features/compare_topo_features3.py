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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help (with defaults) on error."""

    def error(self, message: str) -> None:  # pragma: no cover
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")





@dataclass
class ComparisonResult:
    relative_path: Path
    reference_path: Path
    candidate_path: Path
    max_abs_diff: float
    exceeded: List[Tuple[str, str, float]]  # (row_id, column, diff)
    error: Optional[str] = None


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        default=0.0,
        help="Maximum absolute difference tolerated (default: %(default)s)",
    )
    parser.add_argument(
        "--diff-report",
        type=Path,
        default=Path("topo_diff_report.txt"),
        help="Write detailed differences to this file (default: %(default)s)",
    )
    parser.add_argument(
        "--same-report",
        type=Path,
        default=Path("topo_same_report.txt"),
        help="Write identical file pairs to this file (default: %(default)s)",
    )
    return parser.parse_args(list(argv))


def canonical_csv_key(path: Path) -> str:
    """Normalise CSV filenames so minor naming variants align."""
    name = path.name
    if name.endswith(".csv"):
        stem = name[:-4]
        if stem.endswith(".topology"):
            return stem[: -len(".topology")] + ".csv"
    return name


def collect_csvs(root: Path) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    for path in root.rglob('*.csv'):
        if path.is_file():
            key = canonical_csv_key(path)
            files.setdefault(key, []).append(path)
    if not files:
        for path in root.glob('*.csv'):
            if path.is_file():
                key = canonical_csv_key(path)
                files.setdefault(key, []).append(path)
    return files


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
        ref_cols = list(df_ref.columns)
        new_cols = list(df_new.columns)
        ref_only = [col for col in ref_cols if col not in new_cols]
        new_only = [col for col in new_cols if col not in ref_cols]
        if ref_only or new_only:
            raise ValueError(
                "Column names differ between dataframes "
                f"(reference-only: {ref_only}, candidate-only: {new_only})"
            )
        df_new = df_new[ref_cols]
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

    ref_keys = set(reference_csvs.keys())
    cand_keys = set(candidate_csvs.keys())

    shared_keys = sorted(ref_keys & cand_keys)
    missing_in_candidate_keys = sorted(ref_keys - cand_keys)
    missing_in_reference_keys = sorted(cand_keys - ref_keys)

    results: List[ComparisonResult] = []
    for key in shared_keys:
        ref_paths = reference_csvs[key]
        cand_paths = candidate_csvs[key]
        if len(ref_paths) != 1 or len(cand_paths) != 1:
            raise ValueError(
                f"Encountered multiple files named '{key}' in one of the directories; please disambiguate."
            )
        reference_path = ref_paths[0]
        candidate_path = cand_paths[0]
        df_ref = load_dataframe(reference_path)
        df_new = load_dataframe(candidate_path)
        try:
            max_diff, exceeded = compare_dataframes(df_ref, df_new, tolerance)
            error: Optional[str] = None
        except ValueError as exc:
            max_diff = float("nan")
            exceeded = []
            error = str(exc)
        try:
            rel_reference = reference_path.relative_to(reference_dir)
        except ValueError:
            rel_reference = Path(reference_path.name)
        results.append(
            ComparisonResult(
                relative_path=rel_reference,
                reference_path=reference_path,
                candidate_path=candidate_path,
                max_abs_diff=max_diff,
                exceeded=exceeded,
                error=error,
            )
        )

    missing_in_candidate: List[Path] = []
    for key in missing_in_candidate_keys:
        ref_paths = reference_csvs[key]
        if len(ref_paths) != 1:
            raise ValueError(
                f"Encountered multiple files matching canonical name '{key}' in reference directory; please disambiguate."
            )
        path = ref_paths[0]
        try:
            missing_in_candidate.append(path.relative_to(reference_dir))
        except ValueError:
            missing_in_candidate.append(Path(path.name))

    missing_in_reference: List[Path] = []
    for key in missing_in_reference_keys:
        cand_paths = candidate_csvs[key]
        if len(cand_paths) != 1:
            raise ValueError(
                f"Encountered multiple files matching canonical name '{key}' in candidate directory; please disambiguate."
            )
        path = cand_paths[0]
        try:
            missing_in_reference.append(path.relative_to(candidate_dir))
        except ValueError:
            missing_in_reference.append(Path(path.name))

    return results, missing_in_candidate, missing_in_reference


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    reference_dir = args.reference.resolve()
    candidate_dir = args.candidate.resolve()

    log_path = Path("topo_run.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        def log_print(*objects, **kwargs):
            print(*objects, **kwargs)
            print(*objects, **kwargs, file=log_file)

        if not reference_dir.is_dir():
            log_print(f"Reference directory not found: {reference_dir}", file=sys.stderr)
            return 2
        if not candidate_dir.is_dir():
            log_print(f"Candidate directory not found: {candidate_dir}", file=sys.stderr)
            return 2

        results, missing_candidate, missing_reference = compare_directories(
            reference_dir,
            candidate_dir,
            tolerance=args.tolerance,
        )

        log_print(f"Reference root: {reference_dir}")
        log_print(f"Candidate root: {candidate_dir}")
        log_print(f"Tolerance: {args.tolerance}")
        log_print(f"Shared CSV files: {len(results)}")
        log_print(f"Missing in candidate: {len(missing_candidate)}")
        log_print(f"Missing in reference: {len(missing_reference)}")

        mismatches = 0
        diff_lines: List[str] = []
        same_lines: List[str] = []
        for rel_path in missing_candidate:
            diff_lines.append(f"MISSING in candidate: {reference_dir / rel_path}")
        for rel_path in missing_reference:
            diff_lines.append(f"MISSING in reference: {candidate_dir / rel_path}")
        for result in results:
            if result.error:
                mismatches += 1
                diff_lines.append(
                    f"ERROR: {result.relative_path}\n"
                    f"  reference: {result.reference_path}\n"
                    f"  candidate: {result.candidate_path}\n"
                    f"  detail: {result.error}"
                )
            elif result.exceeded:
                mismatches += 1
                header = (
                    f"DIFFERENT: {result.relative_path}\n"
                    f"  reference: {result.reference_path}\n"
                    f"  candidate: {result.candidate_path}"
                )
                diff_lines.append(header)
                for row_id, column, diff in result.exceeded:
                    diff_lines.append(
                        f"    row={row_id}, column='{column}', diff={diff:.3e}"
                    )
            else:
                same_lines.append(
                    f"SAME: {result.relative_path}\n"
                    f"  reference: {result.reference_path}\n"
                    f"  candidate: {result.candidate_path}"
                )

        log_print(
            f"Summary: {len(results) - mismatches} files within tolerance, "
            f"{mismatches} files exceeding tolerance."
        )

        if diff_lines:
            diff_path = args.diff_report.resolve()
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
            log_print(f"Detailed difference report written to {diff_path}")
        else:
            log_print("No differences detected; no diff report written.")

        if same_lines:
            same_path = args.same_report.resolve()
            same_path.parent.mkdir(parents=True, exist_ok=True)
            same_path.write_text("\n".join(same_lines) + "\n", encoding="utf-8")
            log_print(f"Identical file pairs written to {same_path}")
        else:
            log_print("No identical file pairs recorded; no same report written.")

        return 0 if mismatches == 0 and not missing_candidate and not missing_reference else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
