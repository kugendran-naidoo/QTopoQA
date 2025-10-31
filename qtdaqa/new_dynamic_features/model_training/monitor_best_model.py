#!/usr/bin/env python3
"""Monitor the current best TopoQA checkpoint by validation MSE.

Usage examples
--------------
  python monitor_best_model.py --run-id training_run_2025-10-28_18-16-16
  python monitor_best_model.py --run-dir qtdaqa/new_dynamic_features/model_training/training_runs/latest
  python monitor_best_model.py --run-id training_run_2025-10-28_18-16-16 --follow --interval 30

Outputs a JSON payload describing the best checkpoint found so far, including
its filesystem path and the primary learning parameters that produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_training import train_cli


def _extract_learning_params(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "learning_rate",
        "batch_size",
        "num_epochs",
        "seed",
        "accumulate_grad_batches",
        "attention_head",
        "pooling_type",
    ]
    return {key: config.get(key) for key in keys if key in config}


def _summarise_best(run_dir: Path) -> Dict[str, Any]:
    payload = train_cli._summarise_run(run_dir)  # type: ignore[attr-defined]
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    feature_metadata = payload.get("feature_metadata", {}) if isinstance(payload, dict) else {}
    edge_schema = feature_metadata.get("edge_schema", {}) if isinstance(feature_metadata, dict) else {}
    top_val_losses = payload.get("top_val_losses", []) if isinstance(payload, dict) else []
    top_selection = payload.get("top_selection_metrics", []) if isinstance(payload, dict) else []
    selection_enabled = bool(payload.get("selection_metric_enabled")) if isinstance(payload, dict) else False
    runtime_estimate = payload.get("runtime_estimate") if isinstance(payload, dict) else None

    best_checkpoint = payload.get("best_checkpoint") if isinstance(payload, dict) else None
    checkpoint_name = Path(best_checkpoint).name if best_checkpoint else None
    selection_best = top_selection[0] if top_selection else None
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "best_val_loss": payload.get("best_val_loss") if isinstance(payload, dict) else None,
        "best_epoch": payload.get("best_epoch") if isinstance(payload, dict) else None,
        "best_checkpoint_path": best_checkpoint,
        "best_checkpoint_name": checkpoint_name,
        "learning_parameters": _extract_learning_params(config),
        "edge_feature_dim": edge_schema.get("dim"),
        "notes": payload.get("run_metadata", {}).get("notes") if isinstance(payload.get("run_metadata"), dict) else None,
        "top_val_losses": top_val_losses,
        "selection_metric_enabled": selection_enabled,
        "best_selection": selection_best,
        "selection_alternates": top_selection[1:] if top_selection else [],
        "runtime_estimate": runtime_estimate,
    }


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    try:
        return train_cli._resolve_run_dir_argument(args.run_id, args.run_dir)  # type: ignore[attr-defined]
    except train_cli.CLIError as exc:  # type: ignore[attr-defined]
        raise SystemExit(f"[monitor_best_model] {exc}")


def _render_once(run_dir: Path) -> None:
    summary = _summarise_best(run_dir)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report the current best checkpoint for a training run.")
    parser.add_argument("--run-id", type=str, help="Run identifier under training_runs/ (e.g., training_run_2025-01-01_12-00-00).")
    parser.add_argument("--run-dir", type=Path, help="Explicit path to a training run directory.")
    parser.add_argument("--follow", action="store_true", help="Poll indefinitely, emitting summaries every interval seconds.")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds when --follow is enabled.")
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(args)
    if args.follow:
        try:
            while True:
                _render_once(run_dir)
                time.sleep(max(1, args.interval))
        except KeyboardInterrupt:
            return 0
    else:
        _render_once(run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
