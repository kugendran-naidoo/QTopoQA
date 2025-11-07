#!/usr/bin/env python3
"""Aggregate per-target summary metrics into consolidated CSV files."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple


LOG = logging.getLogger("results_summary")


def _last_row(csv_path: Path) -> List[str]:
    """Return the columns from the last non-empty, non-comment line."""
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return [part.strip() for part in line.split(",")]
    raise ValueError(f"{csv_path} contains no data rows.")


def collect_target_values(
    results_dir: Path,
    column_indexes: Sequence[int],
    formatter: Callable[[List[str]], str],
    *,
    allow_missing: bool = False,
) -> List[Tuple[str, str]]:
    """Collect (target, formatted_value) pairs from per-target summary files."""
    pairs: List[Tuple[str, str]] = []
    missing: List[str] = []
    targets_checked = 0
    for target_dir in sorted(results_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        summary_path = target_dir / f"{target}.summary_metrics.csv"
        targets_checked += 1
        if not summary_path.exists():
            missing.append(target)
            continue
        try:
            row = _last_row(summary_path)
            for index in column_indexes:
                if index >= len(row):
                    raise ValueError(
                        f"Line in {summary_path} has fewer than {index + 1} columns."
                    )
            value = formatter(row)
        except Exception as exc:
            LOG.warning("Skipping %s due to error: %s", summary_path, exc)
            continue
        pairs.append((target, value))

    if missing:
        message = "Targets missing summary files: %s" % ", ".join(sorted(missing))
        if allow_missing:
            LOG.warning(message)
        else:
            raise RuntimeError(message)
    if targets_checked == 0:
        raise RuntimeError(f"No target directories found under {results_dir}")
    return pairs


def write_summary(
    results_dir: Path,
    rows: Sequence[Tuple[str, str]],
    filename: str,
    header: Sequence[str],
) -> Path:
    """Write aggregated results to results_dir/filename."""
    output_path = results_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for target, value in rows:
            writer.writerow([target, value])
    return output_path


def _format_dockq(row: List[str], column_index: int = 4) -> str:
    raw_value = row[column_index]
    try:
        number = float(raw_value)
    except ValueError:
        return raw_value
    return f"{number:.3f}"


def generate_dockq_summary(results_dir: Path, *, allow_missing: bool = False) -> Path:
    """Create final_results_dockq.csv using column 5 (DockQ) rounded to 3 decimals."""
    pairs = collect_target_values(
        results_dir,
        column_indexes=[4],
        formatter=lambda row: _format_dockq(row, 4),
        allow_missing=allow_missing,
    )
    if not pairs and not allow_missing:
        raise RuntimeError(f"No summary metrics found under {results_dir}")
    return write_summary(results_dir, sorted(pairs), "final_results_dockq.csv", ["target", "dockq"])


def _format_hit_rate(row: List[str], offsets: Sequence[int]) -> str:
    values = [row[index] for index in offsets]
    return "/".join(values)


def generate_hit_rate_summary(results_dir: Path, *, allow_missing: bool = False) -> Path:
    """Create final_results_hit_rate.csv using columns 6/7/8 joined by '/'."""
    offsets = [5, 6, 7]
    pairs = collect_target_values(
        results_dir,
        column_indexes=offsets,
        formatter=lambda row: _format_hit_rate(row, offsets),
        allow_missing=allow_missing,
    )
    if not pairs and not allow_missing:
        raise RuntimeError(f"No summary metrics found under {results_dir}")
    return write_summary(
        results_dir,
        sorted(pairs),
        "final_results_hit_rate.csv",
        ["target", "hit_rate"],
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-target summary metrics into a single CSV.")
    parser.add_argument("--results-dir", type=Path, default=Path("output/BM55-AF2/results"), help="Directory containing per-target folders.")
    parser.add_argument(
        "--mode",
        choices=["dockq", "hit-rate", "all"],
        default="all",
        help="Which summary to generate.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail when some targets are missing summary files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")
    outputs: List[Path] = []
    if args.mode in ("dockq", "all"):
        outputs.append(generate_dockq_summary(results_dir, allow_missing=args.allow_missing))
    if args.mode in ("hit-rate", "all"):
        outputs.append(generate_hit_rate_summary(results_dir, allow_missing=args.allow_missing))
    for path in outputs:
        LOG.info("Summary written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
