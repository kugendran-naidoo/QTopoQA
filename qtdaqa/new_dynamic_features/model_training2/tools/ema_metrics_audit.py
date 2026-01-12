#!/usr/bin/env python3
"""Audit training runs for missing EMA metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from qtdaqa.new_dynamic_features.model_training2 import train_cli


def _load_run_metadata(run_dir: Path) -> Dict[str, object] | None:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _get_metric_value(data: Dict[str, object], metric: str) -> object:
    if metric in data:
        return data.get(metric)
    if metric.startswith("ema_"):
        ema_block = data.get("ema_metrics")
        if isinstance(ema_block, dict):
            return ema_block.get(metric.replace("ema_", "", 1))
    return data.get(metric)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit training runs for missing EMA metrics.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=train_cli.RUN_ROOT,
        help="Path to training_runs2 directory.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ema_val_loss", "ema_tuning_hit_rate_023"],
        help="Metrics to check for missing values.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path for the audit report.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print each run with metric values (useful for debugging).",
    )
    args = parser.parse_args()

    training_root = args.training_root.expanduser().resolve()
    metrics = list(args.metrics)
    missing_by_metric: Dict[str, List[str]] = {m: [] for m in metrics}
    missing_any: List[str] = []
    present_runs: List[str] = []
    total_runs = 0

    for run_dir in sorted(training_root.iterdir()):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        data = _load_run_metadata(run_dir)
        if data is None:
            continue
        total_runs += 1
        run_name = run_dir.name
        present_runs.append(run_name)
        missing = False
        for metric in metrics:
            value = _get_metric_value(data, metric)
            if value is None:
                missing_by_metric[metric].append(run_name)
                missing = True
        if missing:
            missing_any.append(run_name)
        if args.print_all:
            metric_values = ", ".join(f"{m}={_get_metric_value(data, m)}" for m in metrics)
            print(f"{run_name}: {metric_values}")

    print(f"training_root={training_root}")
    print(f"total_runs={total_runs}")
    print(f"metrics={metrics}")
    print(f"runs_missing_any={len(missing_any)}")
    for metric in metrics:
        runs = missing_by_metric[metric]
        print(f"missing_{metric}={len(runs)}")
        for run_name in runs:
            print(f"  {run_name}")

    if args.json:
        report = {
            "training_root": str(training_root),
            "total_runs": total_runs,
            "metrics": metrics,
            "runs_missing_any": missing_any,
            "missing_by_metric": missing_by_metric,
        }
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json_report={args.json.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
