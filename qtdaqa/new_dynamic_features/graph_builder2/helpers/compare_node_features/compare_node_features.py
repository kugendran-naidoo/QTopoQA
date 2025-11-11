#!/usr/bin/env python3
"""Compare node feature CSV outputs between two directories."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List


class _TeeStdout:
    """Mirror writes to both stdout and a log file."""

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


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text before exiting on errors."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Compare node feature CSV files between two directories."
    )
    parser.add_argument(
        "node_dir_1",
        type=Path,
        help="Directory containing the first set of node feature CSV files",
    )
    parser.add_argument(
        "node_dir_2",
        type=Path,
        help="Directory containing the second set of node feature CSV files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("node_feature_diff_report.txt"),
        help="Path to write a report of differences (default: %(default)s)",
    )
    parser.add_argument(
        "--same-report",
        type=Path,
        default=Path("node_feature_same_report.txt"),
        help="Path to write a report of identical files (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Floating point absolute tolerance for numeric comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=0.0,
        help="Floating point relative tolerance for numeric comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--flatten-baseline",
        action="store_true",
        dest="flatten_baseline",
        default=True,
        help="Recursively search baseline directory for node feature files (default: flatten).",
    )
    parser.add_argument(
        "--no-flatten-baseline",
        action="store_false",
        dest="flatten_baseline",
        help="Disable recursive search for baseline directory.",
    )
    parser.add_argument(
        "--flatten-candidate",
        action="store_true",
        dest="flatten_candidate",
        default=True,
        help="Recursively search candidate directory for node feature files (default: flatten).",
    )
    parser.add_argument(
        "--no-flatten-candidate",
        action="store_false",
        dest="flatten_candidate",
        help="Disable recursive search for candidate directory.",
    )
    return parser.parse_args()


def _format_path_for_display(path: Path) -> str:
    """Return a shorter, human-friendly representation of a path."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        # Fall back to filename for off-tree locations to keep output concise.
        return path.name or str(path)


def _normalise_name(name: str) -> str:
    stem = Path(name).stem
    lower = stem.lower()
    for suffix in (".node_fea", "node_fea", "node", ".csv"):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem = stem.rstrip("_-. ")
            lower = stem.lower()
    return stem


def gather_files(directory: Path, *, flatten: bool) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    patterns = ["*.csv"]
    if flatten:
        iterator = directory.rglob
    else:
        iterator = directory.glob
    for pattern in patterns:
        for path in iterator(pattern):
            if path.is_file():
                key = _normalise_name(path.name)
                files.setdefault(key, []).append(path)
        if files:
            break
    return files


def parse_csv(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return data
        if not header or header[0] != "ID":
            raise ValueError(f"{path}: first column must be 'ID'")
        for row in reader:
            if not row:
                continue
            if row[0] in data:
                raise ValueError(f"{path}: duplicate ID '{row[0]}'")
            values: List[float] = []
            for token in row[1:]:
                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise ValueError(f"{path}: invalid numeric value '{token}'") from exc
            data[row[0]] = values
    return data


def compare_files(
    baseline_path: Path,
    candidate_path: Path,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, List[str]]:
    baseline_data = parse_csv(baseline_path)
    candidate_data = parse_csv(candidate_path)

    diffs: List[str] = []
    is_same = True
    all_keys = set(baseline_data) | set(candidate_data)

    for key in sorted(all_keys):
        base_values = baseline_data.get(key)
        cand_values = candidate_data.get(key)
        if base_values is None:
            diffs.append(f"  Missing in baseline: {key}")
            is_same = False
            continue
        if cand_values is None:
            diffs.append(f"  Missing in candidate: {key}")
            is_same = False
            continue
        if len(base_values) != len(cand_values):
            diffs.append(
                f"  Length mismatch for {key}: baseline={len(base_values)} candidate={len(cand_values)}"
            )
            is_same = False
            continue
        for idx, (b_val, c_val) in enumerate(zip(base_values, cand_values)):
            if not math.isclose(b_val, c_val, rel_tol=rel_tol, abs_tol=abs_tol):
                diffs.append(
                    f"  {key} value[{idx}]: baseline={b_val:.12g} candidate={c_val:.12g} diff={abs(b_val - c_val):.3e}"
                )
                is_same = False
    return is_same, diffs


def _run() -> int:
    args = parse_args()
    baseline_dir = args.node_dir_1.resolve()
    candidate_dir = args.node_dir_2.resolve()

    print("=== compare_node_features run ===")
    print("CLI parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    if not baseline_dir.is_dir():
        print(f"Error: baseline directory does not exist: {baseline_dir}")
        return 2
    if not candidate_dir.is_dir():
        print(f"Error: candidate directory does not exist: {candidate_dir}")
        return 2

    baseline_files = gather_files(baseline_dir, flatten=args.flatten_baseline)
    candidate_files = gather_files(candidate_dir, flatten=args.flatten_candidate)

    missing_in_candidate = sorted(set(baseline_files) - set(candidate_files))
    missing_in_baseline = sorted(set(candidate_files) - set(baseline_files))

    identical = 0
    different = 0
    differences_report: List[str] = []
    same_report: List[str] = []

    shared_keys = sorted(set(baseline_files) & set(candidate_files))
    for key in shared_keys:
        b_paths = baseline_files[key]
        c_paths = candidate_files[key]
        if len(b_paths) != 1 or len(c_paths) != 1:
            warning = f"Warning: multiple files named {key} detected; skipping comparison."
            differences_report.append(warning)
            different += 1
            continue
        baseline_path = b_paths[0]
        candidate_path = c_paths[0]

        same, diffs = compare_files(
            baseline_path,
            candidate_path,
            args.tolerance,
            args.rel_tolerance,
        )
        if same:
            identical += 1
            same_report.append(
                f"SAME: {key}\n  baseline: {str(baseline_path.resolve())}\n  candidate: {str(candidate_path.resolve())}"
            )
        else:
            different += 1
            header = (
                f"DIFFERENT: {key}\n  baseline: {str(baseline_path.resolve())}\n  candidate: {str(candidate_path.resolve())}"
            )
            differences_report.append("\n".join([header] + diffs))

    total_shared = identical + different

    print(f"Baseline directory: {baseline_dir}")
    print(f"Candidate directory: {candidate_dir}")
    print(f"Shared files compared: {total_shared}")
    print(f"  Identical files:     {identical}")
    print(f"  Different files:     {different}")
    print(f"Missing in candidate:  {len(missing_in_candidate)}")
    print(f"Missing in baseline:   {len(missing_in_baseline)}")

    for filename in missing_in_candidate:
        paths = [str(p.resolve()) for p in baseline_files.get(filename, [])]
        if paths:
            block = [f"MISSING in candidate: {filename}"] + [f"  baseline: {p}" for p in paths]
            differences_report.append("\n".join(block))
        else:
            differences_report.append(f"MISSING in candidate: {filename}")
    for filename in missing_in_baseline:
        paths = [str(p.resolve()) for p in candidate_files.get(filename, [])]
        if paths:
            block = [f"MISSING in baseline: {filename}"] + [f"  candidate: {p}" for p in paths]
            differences_report.append("\n".join(block))
        else:
            differences_report.append(f"MISSING in baseline: {filename}")

    if differences_report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(differences_report) + "\n", encoding="utf-8")
        print(f"Detailed differences written to {_format_path_for_display(report_path)}")
    else:
        print("No differences detected; no report written.")

    if same_report:
        same_path = args.same_report.resolve()
        same_path.parent.mkdir(parents=True, exist_ok=True)
        same_path.write_text("\n".join(same_report) + "\n", encoding="utf-8")
        print(f"Identical file pairs written to {_format_path_for_display(same_path)}")
    else:
        print("No identical file pairs recorded; no same-report written.")

    if different or missing_in_candidate or missing_in_baseline:
        return 1
    return 0


def main() -> int:
    log_path = Path("node_feature_compare_run.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_handle:
        sys.stdout = _TeeStdout(original_stdout, log_handle)
        try:
            return _run()
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
