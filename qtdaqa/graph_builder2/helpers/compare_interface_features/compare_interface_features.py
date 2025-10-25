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


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help (including defaults) on error."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Compare interface output files between two directories."
    )
    parser.add_argument(
        "baseline_dir",
        type=Path,
        help="Directory containing the baseline interface outputs",
    )
    parser.add_argument(
        "candidate_dir",
        type=Path,
        help="Directory containing the candidate interface outputs",
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
        default=0.0,
        help=(
            "Absolute tolerance for coordinate comparison; default 0.0 requires exact match "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=0.0,
        help=(
            "Relative tolerance for coordinate comparison; default 0.0 requires exact match "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--flatten-baseline",
        action="store_true",
        default=False,
        help="Recursively search baseline directory for interface files.",
    )
    parser.add_argument(
        "--flatten-candidate",
        action="store_true",
        default=False,
        help="Recursively search candidate directory for interface files.",
    )
    return parser.parse_args()


def _normalise_filename(name: str) -> str:
    if name.endswith(".interface.txt"):
        return name[: -len(".interface.txt")] + ".txt"
    return name


def gather_files(base_dir: Path, *, flatten_subdirs: bool) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    if flatten_subdirs:
        for path in base_dir.rglob("*.interface.txt"):
            if path.is_file():
                key = _normalise_filename(path.name)
                files.setdefault(key, []).append(path)
        if not files:
            for path in base_dir.rglob("*.txt"):
                if path.is_file():
                    key = _normalise_filename(path.name)
                    files.setdefault(key, []).append(path)
    else:
        for path in base_dir.glob("*.interface.txt"):
            if path.is_file():
                key = _normalise_filename(path.name)
                files.setdefault(key, []).append(path)
        if not files:
            for path in base_dir.rglob("*.interface.txt"):
                if path.is_file():
                    key = _normalise_filename(path.name)
                    files.setdefault(key, []).append(path)
        if not files:
            for path in base_dir.glob("*.txt"):
                if path.is_file():
                    key = _normalise_filename(path.name)
                    files.setdefault(key, []).append(path)
            if not files:
                for path in base_dir.rglob("*.txt"):
                    if path.is_file():
                        key = _normalise_filename(path.name)
                        files.setdefault(key, []).append(path)
    return files


def parse_line(line: str) -> tuple[str, List[float]]:
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Line does not contain at least 4 fields: {line!r}")
    key = parts[0]
    coords = []
    for token in parts[1:]:
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
    rel_tolerance: float,
) -> tuple[bool, List[str]]:
    baseline_entries: Dict[str, List[float]] | None = None
    candidate_entries: Dict[str, List[float]] | None = None
    baseline_error: ValueError | None = None
    candidate_error: ValueError | None = None

    try:
        baseline_entries = load_interface_file(baseline_path)
    except ValueError as exc:
        baseline_error = exc

    try:
        candidate_entries = load_interface_file(candidate_path)
    except ValueError as exc:
        candidate_error = exc

    diffs: List[str] = []

    if baseline_error or candidate_error:
        if baseline_error:
            diffs.append(f"  Malformed baseline file ({baseline_path}): {baseline_error}")
        if candidate_error:
            diffs.append(f"  Malformed candidate file ({candidate_path}): {candidate_error}")
        return False, diffs

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
            if not math.isclose(b_val, c_val, rel_tol=rel_tolerance, abs_tol=tolerance):
                diffs.append(
                    f"  {key} coord[{idx}]: baseline={b_val:.12g} candidate={c_val:.12g} diff={abs(b_val - c_val):.3e}"
                )
                is_same = False
    return is_same, diffs


def _run() -> int:
    args = parse_args()
    print("=== compare_interface_features run ===")
    print("CLI parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()

    if not baseline_dir.is_dir():
        print(f"Error: baseline directory does not exist: {baseline_dir}")
        return 2
    if not candidate_dir.is_dir():
        print(f"Error: candidate directory does not exist: {candidate_dir}")
        return 2

    baseline_files = gather_files(baseline_dir, flatten_subdirs=args.flatten_baseline)
    candidate_files = gather_files(candidate_dir, flatten_subdirs=args.flatten_candidate)

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
                f"SAME: {filename}\n  baseline: {str(baseline_path.resolve())}\n  candidate: {str(candidate_path.resolve())}"
            )
        else:
            different += 1
            header = (
                f"DIFFERENT: {filename}\n  baseline: {str(baseline_path.resolve())}\n  candidate: {str(candidate_path.resolve())}"
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
        for filename in missing_in_candidate:
            entries = baseline_files.get(filename, [])
            if entries:
                block = [f"MISSING in candidate: {filename}"] + [
                    f"  baseline: {str(p.resolve())}" for p in entries
                ]
                differences_report.append("\n".join(block))
            else:
                differences_report.append(f"MISSING in candidate: {filename}")
    if missing_in_baseline:
        for filename in missing_in_baseline:
            entries = candidate_files.get(filename, [])
            if entries:
                block = [f"MISSING in baseline: {filename}"] + [
                    f"  candidate: {str(p.resolve())}" for p in entries
                ]
                differences_report.append("\n".join(block))
            else:
                differences_report.append(f"MISSING in baseline: {filename}")

    if differences_report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n\n".join(differences_report) + "\n", encoding="utf-8")
        print(f"Detailed differences written to {report_path.name}")
    else:
        print("No differences detected; no report written.")

    if same_report:
        same_path = args.same_report.resolve()
        same_path.parent.mkdir(parents=True, exist_ok=True)
        same_path.write_text("\n".join(same_report) + "\n", encoding="utf-8")
        print(f"Identical file pairs written to {same_path.name}")
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
    parser.add_argument(
        "--flatten-baseline",
        action="store_true",
        help="Recursively search baseline directory for interface files (default: only direct children).",
    )
    parser.add_argument(
        "--flatten-candidate",
        action="store_true",
        help="Recursively search candidate directory for interface files (default: only direct children).",
    )
