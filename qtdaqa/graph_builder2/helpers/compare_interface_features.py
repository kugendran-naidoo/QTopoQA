#!/usr/bin/env python3
"""Compare interface outputs between two work directories."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List


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
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("interface_diff_report.txt"),
        help="Path to write a report of differences (default: %(default)s)",
    )
    parser.add_argument(
        "--same-report",
        type=Path,
        default=Path("interface_same_report.txt"),
        help="Path to write a report of identical files (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Floating point tolerance for coordinate comparison (default: %(default)s)",
    )
    return parser.parse_args()


def gather_files(base_dir: Path) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    for path in base_dir.rglob("*.interface.txt"):
        if path.is_file():
            files.setdefault(path.name, []).append(path)
    if not files:
        for path in base_dir.glob("*.interface.txt"):
            if path.is_file():
                files.setdefault(path.name, []).append(path)
    return files


def parse_line(line: str) -> tuple[str, List[float]]:
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Line does not contain at least 4 fields: {line!r}")
    key = parts[0]
    coords = []
    for token in parts[3:]:
        try:
            coords.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Could not parse coordinate '{token}' in line {line!r}") from exc
    return key, coords


def load_interface_file(path: Path) -> Dict[str, List[float]]:
    entries: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            key, coords = parse_line(line)
            entries[key] = coords
    return entries


def compare_files(
    baseline_path: Path,
    candidate_path: Path,
    tolerance: float,
) -> tuple[bool, List[str]]:
    baseline_entries = load_interface_file(baseline_path)
    candidate_entries = load_interface_file(candidate_path)

    diffs: List[str] = []
    is_same = True
    all_keys = set(baseline_entries) | set(candidate_entries)

    for key in sorted(all_keys):
        base_coords = baseline_entries.get(key)
        cand_coords = candidate_entries.get(key)
        if base_coords is None:
            diffs.append(f"  Missing in baseline: {key}")
            is_same = False
            continue
        if cand_coords is None:
            diffs.append(f"  Missing in candidate: {key}")
            is_same = False
            continue
        if len(base_coords) != len(cand_coords):
            diffs.append(
                f"  Coordinate length mismatch for {key}: baseline={len(base_coords)} candidate={len(cand_coords)}"
            )
            is_same = False
            continue
        for idx, (b_val, c_val) in enumerate(zip(base_coords, cand_coords)):
            if not math.isclose(b_val, c_val, rel_tol=0.0, abs_tol=tolerance):
                diffs.append(
                    f"  {key} coord[{idx}]: baseline={b_val:.12g} candidate={c_val:.12g} diff={abs(b_val - c_val):.3e}"
                )
                is_same = False
    return is_same, diffs


def _run() -> int:
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
    differences_report: List[str] = []
    same_report: List[str] = []

    for filename in sorted(set(baseline_files.keys()) & set(candidate_files.keys())):
        b_paths = baseline_files[filename]
        c_paths = candidate_files[filename]
        if len(b_paths) != 1 or len(c_paths) != 1:
            warning = f"Warning: multiple files named {filename} detected; skipping comparison."
            print(warning)
            differences_report.append(warning)
            different += 1
            continue
        baseline_path = b_paths[0]
        candidate_path = c_paths[0]

        same, diffs = compare_files(baseline_path, candidate_path, args.tolerance)
        if same:
            identical += 1
            same_report.append(
                f"SAME: {filename}\n  baseline: {baseline_path}\n  candidate: {candidate_path}"
            )
        else:
            different += 1
            header = (
                f"DIFFERENT: {filename}\n  baseline: {baseline_path}\n  candidate: {candidate_path}"
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

    if missing_in_candidate:
        print("  Files missing in candidate:")
        for path in missing_in_candidate:
            print(f"    {path}")
            differences_report.append(f"MISSING in candidate: {path}")
    if missing_in_baseline:
        print("  Files missing in baseline:")
        for path in missing_in_baseline:
            print(f"    {path}")
            differences_report.append(f"MISSING in baseline: {path}")

    if differences_report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(differences_report) + "\n", encoding="utf-8")
        print(f"Detailed differences written to {report_path}")
    else:
        print("No differences detected; no report written.")

    if same_report:
        same_path = args.same_report.resolve()
        same_path.parent.mkdir(parents=True, exist_ok=True)
        same_path.write_text("\n".join(same_report) + "\n", encoding="utf-8")
        print(f"Identical file pairs written to {same_path}")
    else:
        print("No identical file pairs recorded; no same-report written.")

    if different or missing_in_candidate or missing_in_baseline:
        return 1
    return 0


def main() -> int:
    log_path = Path("interface_run.log").resolve()
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
