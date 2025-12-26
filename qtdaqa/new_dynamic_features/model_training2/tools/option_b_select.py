#!/usr/bin/env python3
"""
Select a checkpoint using Option B:
  1) pick top-K runs by best val_loss
  2) choose the run with the best tuning metric among those
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from qtdaqa.new_dynamic_features.model_training2 import train_cli


def _metric_direction(metric_name: str) -> str:
    return "max" if metric_name in train_cli.MAXIMIZE_METRICS else "min"


def _resolve_val_loss_checkpoint(summary: Dict[str, object], run_dir: Path) -> str | None:
    alternates = summary.get("alternate_checkpoints") if isinstance(summary, dict) else None
    if isinstance(alternates, dict):
        candidate = alternates.get("val_loss")
        if candidate:
            path = Path(str(candidate))
            if path.exists():
                return str(path)
    best_link = run_dir / "model_checkpoints" / "val_loss_best.ckpt"
    if best_link.exists():
        return str(best_link.resolve())
    best_checkpoint = summary.get("best_checkpoint") if isinstance(summary, dict) else None
    if best_checkpoint:
        path = Path(str(best_checkpoint))
        if path.exists():
            return str(path)
    return None


def _extract_metric(summary: Dict[str, object], metric: str) -> float | None:
    value = summary.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_runs(training_root: Path) -> List[Tuple[Path, Dict[str, object]]]:
    runs: List[Tuple[Path, Dict[str, object]]] = []
    for run_dir in sorted(training_root.iterdir()):
        if run_dir.is_symlink():
            continue
        if not run_dir.is_dir() or not (run_dir / "run_metadata.json").exists():
            continue
        summary = train_cli._summarise_run(run_dir)
        runs.append((run_dir, summary))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Option B checkpoint selector.")
    parser.add_argument("--training-root", type=Path, default=train_cli.RUN_ROOT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--shortlist-metric",
        default="best_val_loss",
        choices=(
            "best_val_loss",
            "ema_val_loss",
        ),
        help="Metric used to build the top-K shortlist before applying the tuning metric.",
    )
    parser.add_argument(
        "--tuning-metric",
        default="best_val_tuning_rank_spearman",
        choices=(
            "best_val_tuning_rank_spearman",
            "best_val_tuning_rank_regret",
            "best_tuning_rank_spearman",
            "best_tuning_rank_regret",
            "ema_tuning_rank_spearman",
            "ema_tuning_rank_regret",
        ),
    )
    parser.add_argument("--emit-checkpoint-only", action="store_true")
    args = parser.parse_args()

    training_root = args.training_root.resolve()
    runs = _collect_runs(training_root)
    if not runs:
        raise SystemExit(f"No runs found under {training_root}")

    ranked: List[Tuple[float, Path, Dict[str, object]]] = []
    for run_dir, summary in runs:
        shortlist_value = _extract_metric(summary, args.shortlist_metric)
        if shortlist_value is None:
            continue
        ranked.append((shortlist_value, run_dir, summary))
    if not ranked:
        raise SystemExit(f"No runs with {args.shortlist_metric} found.")

    ranked.sort(key=lambda item: item[0], reverse=(_metric_direction(args.shortlist_metric) == "max"))
    top_k = max(1, int(args.top_k))
    shortlist = ranked[:top_k]

    tuning_metric = args.tuning_metric
    direction = _metric_direction(tuning_metric)
    best = None
    for shortlist_value, run_dir, summary in shortlist:
        tuning_value = _extract_metric(summary, tuning_metric)
        if tuning_value is None:
            continue
        score = tuning_value if direction == "max" else -tuning_value
        candidate = (score, shortlist_value, run_dir, summary)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        # fall back to best val_loss if tuning metric missing
        shortlist_value, run_dir, summary = shortlist[0]
    else:
        _, shortlist_value, run_dir, summary = best

    checkpoint = _resolve_val_loss_checkpoint(summary, run_dir)
    if not checkpoint:
        raise SystemExit(f"No checkpoint found for run {run_dir.name}")

    if args.emit_checkpoint_only:
        print(checkpoint)
        return 0

    best_val_loss = _extract_metric(summary, "best_val_loss")
    tuning_value = _extract_metric(summary, tuning_metric)
    print(f"run={run_dir.name}")
    if best_val_loss is not None:
        print(f"best_val_loss={best_val_loss}")
    print(f"shortlist_metric={args.shortlist_metric}")
    print(f"shortlist_metric_value={shortlist_value}")
    print(f"{tuning_metric}={tuning_value}")
    print(f"checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
