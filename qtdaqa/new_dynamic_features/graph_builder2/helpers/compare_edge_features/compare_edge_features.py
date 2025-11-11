#!/usr/bin/env python3
"""Compare edge feature CSV dumps between two work directories."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help (with defaults) on error."""

    def error(self, message: str) -> None:  # pragma: no cover - CLI convenience
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


@dataclass
class ComparisonResult:
    relative_path: Path
    reference_path: Path
    candidate_path: Path
    max_abs_diff: float
    exceeded: List[Tuple[str, str, float]]
    error: Optional[str] = None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Compare edge CSV dumps between baseline and candidate directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline_dir", type=Path, help="Directory containing baseline edge CSV dumps.")
    parser.add_argument("candidate_dir", type=Path, help="Directory containing candidate edge CSV dumps.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Maximum absolute difference tolerated for numeric fields.",
    )
    parser.add_argument(
        "--diff-report",
        type=Path,
        default=Path("edge_diff_report.txt"),
        help="Write detailed differences to this file.",
    )
    parser.add_argument(
        "--same-report",
        type=Path,
        default=Path("edge_same_report.txt"),
        help="Write identical file pairs to this file.",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=Path("edge_missing_report.txt"),
        help="Write missing file listings to this file.",
    )
    parser.add_argument(
        "--run-report",
        type=Path,
        default=Path("edge_run.log"),
        help="Write console output to this log file.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def canonical_edge_key(path: Path) -> str:
    name = path.name
    if name.endswith(".csv"):
        return name
    return name


def collect_csvs(root: Path) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    for path in root.rglob("*.csv"):
        if path.is_file():
            key = canonical_edge_key(path)
            files.setdefault(key, []).append(path)
    if not files:
        for path in root.glob("*.csv"):
            if path.is_file():
                key = canonical_edge_key(path)
                files.setdefault(key, []).append(path)
    return files


Row = Dict[str, str]


def _row_sort_key(row: Row, *, cols: Sequence[str]) -> Tuple:
    key_components: List[Tuple[str, str]] = []
    if "src_idx" in row and "dst_idx" in row:
        key_components.append(("src_idx", row["src_idx"]))
        key_components.append(("dst_idx", row["dst_idx"]))
    for col in cols:
        if col not in ("src_idx", "dst_idx"):
            key_components.append((col, row.get(col, "")))
    return tuple(key_components)


def _parse_csv(path: Path) -> Tuple[List[Row], List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [dict(row) for row in reader]
    return rows, reader.fieldnames


def _both_numeric(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        float(a)
        float(b)
        return True
    except (TypeError, ValueError):
        return False


def compare_rows(
    ref_rows: List[Row],
    cand_rows: List[Row],
    columns: Sequence[str],
    tolerance: float,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    if len(ref_rows) != len(cand_rows):
        raise ValueError("Row counts differ between CSV files.")

    sort_key = lambda row: _row_sort_key(row, cols=columns)
    ref_sorted = sorted(ref_rows, key=sort_key)
    cand_sorted = sorted(cand_rows, key=sort_key)

    exceeded: List[Tuple[str, str, float]] = []
    max_abs_diff = 0.0

    for idx, (ref_row, cand_row) in enumerate(zip(ref_sorted, cand_sorted)):
        if "src_idx" in ref_row and "dst_idx" in ref_row:
            row_label = f"{ref_row.get('src_idx')}->{ref_row.get('dst_idx')}"
        else:
            row_label = f"row_{idx}"
        for col in columns:
            ref_val = ref_row.get(col)
            cand_val = cand_row.get(col)
            if ref_val is None or cand_val is None:
                raise ValueError(f"Column '{col}' missing in one of the rows.")
            if _both_numeric(ref_val, cand_val):
                diff = abs(float(ref_val) - float(cand_val))
                max_abs_diff = max(max_abs_diff, diff)
                if diff > tolerance:
                    exceeded.append((row_label, col, diff))
            else:
                if str(ref_val) != str(cand_val):
                    raise ValueError(
                        f"Non-numeric column '{col}' differs: {ref_val!r} vs {cand_val!r}"
                    )
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
    missing_in_candidate = sorted(ref_keys - cand_keys)
    missing_in_reference = sorted(cand_keys - ref_keys)

    results: List[ComparisonResult] = []

    for key in shared_keys:
        ref_paths = reference_csvs[key]
        cand_paths = candidate_csvs[key]
        if len(ref_paths) != 1 or len(cand_paths) != 1:
            raise ValueError(
                f"Encountered multiple files named '{key}' in one of the directories; please disambiguate."
            )
        ref_path = ref_paths[0]
        cand_path = cand_paths[0]
        ref_rows, ref_columns = _parse_csv(ref_path)
        cand_rows, cand_columns = _parse_csv(cand_path)
        if set(ref_columns) != set(cand_columns):
            ref_only = sorted(set(ref_columns) - set(cand_columns))
            cand_only = sorted(set(cand_columns) - set(ref_columns))
            raise ValueError(
                f"Column names differ between files (baseline-only: {ref_only}, candidate-only: {cand_only})"
            )
        columns = sorted(ref_columns)
        relative = ref_path.relative_to(reference_dir)
        try:
            max_diff, exceeded = compare_rows(ref_rows, cand_rows, columns, tolerance)
            error: Optional[str] = None
        except ValueError as exc:
            max_diff = float("nan")
            exceeded = []
            error = str(exc)
        results.append(
            ComparisonResult(
                relative_path=relative,
                reference_path=ref_path,
                candidate_path=cand_path,
                max_abs_diff=max_diff,
                exceeded=exceeded,
                error=error,
            )
        )

    missing_ref_paths = [reference_dir / key for key in missing_in_candidate]
    missing_cand_paths = [candidate_dir / key for key in missing_in_reference]
    return results, missing_ref_paths, missing_cand_paths


class _TeeStdout:
    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def __getattr__(self, attr):
        return getattr(self._primary, attr)


def _write_reports(
    results: List[ComparisonResult],
    diff_report: Path,
    same_report: Path,
    missing_report: Path,
    missing_in_candidate: List[Path],
    missing_in_reference: List[Path],
) -> Tuple[int, int]:
    identical = 0
    different = 0

    with diff_report.open("w", encoding="utf-8") as diff_handle, same_report.open(
        "w", encoding="utf-8"
    ) as same_handle:
        for result in results:
            if result.error:
                different += 1
                diff_handle.write(f"{result.relative_path} ERROR: {result.error}\n")
            elif result.exceeded:
                different += 1
                diff_handle.write(
                    f"{result.relative_path} max_abs_diff={result.max_abs_diff:.6g}, "
                    f"num_exceeded={len(result.exceeded)}\n"
                )
                for row_label, column, diff in result.exceeded[:20]:
                    diff_handle.write(f"  {row_label} :: {column} diff={diff:.6g}\n")
                if len(result.exceeded) > 20:
                    diff_handle.write(f"  ... {len(result.exceeded) - 20} more differences\n")
            else:
                identical += 1
                same_handle.write(f"{result.relative_path} identical (max_abs_diff=0)\n")

    with missing_report.open("w", encoding="utf-8") as miss_handle:
        if missing_in_candidate:
            miss_handle.write("Missing in candidate:\n")
            for path in missing_in_candidate:
                miss_handle.write(f"  {path}\n")
        if missing_in_reference:
            if missing_in_candidate:
                miss_handle.write("\n")
            miss_handle.write("Missing in baseline:\n")
            for path in missing_in_reference:
                miss_handle.write(f"  {path}\n")

    return identical, different


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()

    if not baseline_dir.exists():
        raise SystemExit(f"Baseline directory does not exist: {baseline_dir}")
    if not candidate_dir.exists():
        raise SystemExit(f"Candidate directory does not exist: {candidate_dir}")

    args.diff_report.parent.mkdir(parents=True, exist_ok=True)
    args.same_report.parent.mkdir(parents=True, exist_ok=True)
    args.missing_report.parent.mkdir(parents=True, exist_ok=True)
    args.run_report.parent.mkdir(parents=True, exist_ok=True)

    with args.run_report.open("w", encoding="utf-8") as log_handle:
        tee = _TeeStdout(sys.stdout, log_handle)
        original_stdout = sys.stdout
        sys.stdout = tee
        try:
            print("=== compare_edge_features run ===")
            print("CLI parameters:")
            print(f"  baseline_dir: {args.baseline_dir}")
            print(f"  candidate_dir: {args.candidate_dir}")
            print(f"  tolerance: {args.tolerance}")
            print(f"  diff_report: {args.diff_report}")
            print(f"  same_report: {args.same_report}")
            print(f"  missing_report: {args.missing_report}")
            print(f"  run_report: {args.run_report}")
            print(f"  Reference root: {baseline_dir}")
            print(f"  Candidate root: {candidate_dir}")

            results, missing_ref_paths, missing_cand_paths = compare_directories(
                baseline_dir,
                candidate_dir,
                args.tolerance,
            )

            identical, different = _write_reports(
                results,
                args.diff_report,
                args.same_report,
                args.missing_report,
                missing_ref_paths,
                missing_cand_paths,
            )

            print(f"Compared files: {len(results)}")
            print(f"Identical files: {identical}")
            print(f"Different files: {different}")
            print(f"Missing in candidate: {len(missing_ref_paths)}")
            print(f"Missing in baseline: {len(missing_cand_paths)}")
        finally:
            sys.stdout = original_stdout
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
