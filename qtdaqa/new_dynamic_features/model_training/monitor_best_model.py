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
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_training import train_cli


def _to_repo_relative(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return path_value
    try:
        candidate = Path(path_value)
    except (TypeError, ValueError):
        return path_value
    bases = [
        getattr(train_cli, "RUN_ROOT", None),
        REPO_ROOT,
    ]
    for resolver in (candidate.resolve, lambda: candidate):
        try:
            resolved = resolver()
        except Exception:
            continue
        for base in bases:
            if base is None:
                continue
            try:
                return str(
                    (Path("training_runs") / resolved.relative_to(base))
                    if base == getattr(train_cli, "RUN_ROOT", None)
                    else resolved.relative_to(base)
                )
            except Exception:
                continue
    for base in bases:
        if base is None:
            continue
        try:
            return str(
                (Path("training_runs") / candidate.relative_to(base))
                if base == getattr(train_cli, "RUN_ROOT", None)
                else candidate.relative_to(base)
            )
        except Exception:
            continue
    return str(candidate)


def _extract_learning_params(
    config: Dict[str, Any],
    training_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    keys = [
        "learning_rate",
        "batch_size",
        "num_epochs",
        "seed",
        "accumulate_grad_batches",
        "attention_head",
        "pooling_type",
    ]
    params: Dict[str, Any] = {key: config.get(key) for key in keys if key in config}
    if isinstance(training_parameters, dict):
        trainer_cfg = training_parameters.get("trainer")
        dataloader_cfg = training_parameters.get("dataloader")
        if isinstance(trainer_cfg, dict):
            for key in ("num_epochs", "accumulate_grad_batches", "precision", "accelerator", "devices"):
                if key in trainer_cfg:
                    params.setdefault(key, trainer_cfg[key])
        if isinstance(dataloader_cfg, dict):
            for key in ("batch_size", "num_workers", "seed"):
                if key in dataloader_cfg:
                    params.setdefault(key, dataloader_cfg[key])
        if training_parameters.get("fast_dev_run") is not None:
            params.setdefault("fast_dev_run", training_parameters.get("fast_dev_run"))
    return {key: value for key, value in params.items() if value is not None}


def _collect_warnings(
    best_checkpoint: Optional[str],
    selection_enabled: bool,
    run_metadata: Any,
    progress: Optional[Dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if not best_checkpoint:
        warnings.append("No checkpoint has been saved yet.")
    if not selection_enabled:
        warnings.append("Selection metric is disabled; ranking relies on validation loss only.")

    completed = None
    if isinstance(run_metadata, dict):
        completed = run_metadata.get("completed")

    if completed is None:
        remaining_epochs = None
        if isinstance(progress, dict):
            remaining_epochs = progress.get("remaining_epochs")
        if isinstance(remaining_epochs, (int, float)) and remaining_epochs > 0:
            warnings.append("Run appears in progress; best checkpoint may change as training continues.")
        else:
            warnings.append("Run metadata does not yet include a completion timestamp.")

    return warnings


def _format_metric_block(summary: Dict[str, Any], indent: str = "   ", include_alt: bool = False) -> list[str]:
    primary_metric = summary.get("selection_primary_metric") or "val_loss"
    primary_value = summary.get("best_val_loss")
    if primary_metric == "selection_metric":
        primary_value = summary.get("best_selection_metric") or primary_value
    lines = [f"{indent}primary_metric: {primary_metric} = {primary_value}"]
    secondary_enabled = bool(summary.get("selection_metric_enabled"))
    if secondary_enabled:
        secondary_value = summary.get("best_selection_val_spearman")
        lines.append(f"{indent}secondary_metric: val_spearman_corr = {secondary_value}")
    else:
        lines.append(f"{indent}secondary_metric: None")
    val_loss = summary.get("best_val_loss")
    if val_loss is not None:
        lines.append(f"{indent}val_loss: {val_loss}")
    sel_metric = summary.get("best_selection_metric")
    if sel_metric is not None:
        lines.append(f"{indent}selection_metric: {sel_metric}")
    checkpoint = summary.get("best_checkpoint_path") or summary.get("best_checkpoint")
    if checkpoint:
        lines.append(f"{indent}checkpoint: {checkpoint}")
    if include_alt:
        alternates = summary.get("selection_alternates") or []
        if alternates:
            alt_entry = alternates[0]
            alt_val = alt_entry.get("selection_metric")
            alt_epoch = alt_entry.get("epoch")
            label = "   alt_selection_rank: "
            if alt_epoch is not None:
                label += f"(epoch={alt_epoch}, selection_metric = {alt_val})"
            else:
                label += f"(selection_metric = {alt_val})"
            lines.append(label)
        else:
            lines.append(f"{indent}alt_selection_rank: (selection metric ranks current run)")
    return lines


def _summarise_best(run_dir: Path, *, metrics_limit: Optional[int] = None) -> Dict[str, Any]:
    payload = train_cli._summarise_run(run_dir)  # type: ignore[attr-defined]
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    training_parameters = payload.get("training_parameters") if isinstance(payload, dict) else None
    feature_metadata = payload.get("feature_metadata", {}) if isinstance(payload, dict) else {}
    edge_schema = feature_metadata.get("edge_schema", {}) if isinstance(feature_metadata, dict) else {}
    top_val_losses = payload.get("top_val_losses", []) if isinstance(payload, dict) else []
    top_selection = payload.get("top_selection_metrics", []) if isinstance(payload, dict) else []
    selection_enabled = bool(payload.get("selection_metric_enabled")) if isinstance(payload, dict) else False
    runtime_estimate = payload.get("runtime_estimate") if isinstance(payload, dict) else None
    progress = payload.get("progress") if isinstance(payload, dict) else None
    recent_metrics: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        candidate = payload.get("recent_metrics")
        if isinstance(candidate, list):
            recent_metrics = [item for item in candidate if isinstance(item, dict)]
    latest_metric = payload.get("latest_metric") if isinstance(payload, dict) else None

    best_checkpoint = payload.get("best_checkpoint") if isinstance(payload, dict) else None
    checkpoint_name = Path(best_checkpoint).name if best_checkpoint else None
    selection_best = top_selection[0] if top_selection else None

    metric_label = "val_loss"
    metric_value: Optional[Any] = payload.get("best_val_loss") if isinstance(payload, dict) else None
    metric_epoch = payload.get("best_epoch") if isinstance(payload, dict) else None
    if selection_enabled and selection_best and selection_best.get("selection_metric") is not None:
        metric_label = "selection_metric"
        metric_value = selection_best.get("selection_metric")
        if selection_best.get("epoch") is not None:
            metric_epoch = selection_best.get("epoch")

    def _format_number(value: Any) -> Optional[str]:
        if isinstance(value, float):
            return f"{value:.5f}"
        if isinstance(value, int):
            return str(value)
        return str(value) if value is not None else None

    summary_parts: list[str] = []
    if checkpoint_name:
        summary_parts.append(checkpoint_name)
    if metric_epoch is not None:
        summary_parts.append(f"epoch={metric_epoch}")
    metric_repr = _format_number(metric_value)
    if metric_repr is not None:
        summary_parts.append(f"{metric_label}={metric_repr}")
    if isinstance(progress, dict):
        progress_percent = progress.get("progress_percent")
        if isinstance(progress_percent, (int, float)):
            summary_parts.append(f"{progress_percent:.1f}% complete")
        eta = progress.get("eta")
        if eta:
            summary_parts.append(f"ETA {eta}")
    best_summary_line = " | ".join(summary_parts) if summary_parts else None

    if metrics_limit is not None:
        if metrics_limit > 0:
            recent_metrics = recent_metrics[-metrics_limit:]
        else:
            recent_metrics = []

    summary: Dict[str, Any] = {
        "run_dir": _to_repo_relative(str(run_dir)),
        "run_name": run_dir.name,
        "best_val_loss": payload.get("best_val_loss") if isinstance(payload, dict) else None,
        "best_epoch": payload.get("best_epoch") if isinstance(payload, dict) else None,
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_path": _to_repo_relative(best_checkpoint),
        "best_checkpoint_name": checkpoint_name,
        "learning_parameters": _extract_learning_params(config, training_parameters),
        "edge_feature_dim": edge_schema.get("dim"),
        "notes": payload.get("run_metadata", {}).get("notes") if isinstance(payload.get("run_metadata"), dict) else None,
        "top_val_losses": top_val_losses,
        "selection_metric_enabled": selection_enabled,
        "best_selection": selection_best,
        "selection_alternates": top_selection[1:] if top_selection else [],
        "selection_primary_metric": payload.get("selection_primary_metric") if isinstance(payload, dict) else None,
        "runtime_estimate": runtime_estimate,
        "progress": progress,
        "best_summary_line": best_summary_line,
        "warnings": _collect_warnings(best_checkpoint, selection_enabled, payload.get("run_metadata", {}), progress),
        "selection_primary_metric": payload.get("selection_primary_metric") if isinstance(payload, dict) else None,
        "best_selection_metric": payload.get("best_selection_metric") if isinstance(payload, dict) else None,
        "best_selection_val_spearman": payload.get("best_selection_val_spearman") if isinstance(payload, dict) else None,
    }

    def _normalise_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(metric)
        source = payload.get("source")
        if source:
            payload["source"] = _to_repo_relative(source)
        return payload

    if isinstance(latest_metric, dict):
        summary["latest_metric"] = _normalise_metric(latest_metric)
    if metrics_limit and recent_metrics:
        summary["recent_metrics"] = [_normalise_metric(item) for item in recent_metrics]
    checkpoint_symlinks = payload.get("checkpoint_symlinks")
    if isinstance(checkpoint_symlinks, list):
        normalised_symlinks: List[Dict[str, Any]] = []
        for entry in checkpoint_symlinks:
            if not isinstance(entry, dict):
                continue
            normalised = dict(entry)
            path_value = normalised.get("path")
            if path_value:
                normalised["path"] = _to_repo_relative(path_value)
            normalised_symlinks.append(normalised)
        if normalised_symlinks:
            summary["checkpoint_symlinks"] = normalised_symlinks

    return summary


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    try:
        return train_cli._resolve_run_dir_argument(args.run_id, args.run_dir)  # type: ignore[attr-defined]
    except train_cli.CLIError as exc:  # type: ignore[attr-defined]
        raise SystemExit(f"[monitor_best_model] {exc}")


def _render_table(summary: Dict[str, Any], metrics_limit: Optional[int]) -> None:
    def _fmt_metric(entry: Dict[str, Any]) -> str:
        parts = []
        epoch = entry.get("epoch")
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        val_loss = entry.get("val_loss")
        if isinstance(val_loss, float):
            parts.append(f"val_loss={val_loss:.5f}")
        elif val_loss is not None:
            parts.append(f"val_loss={val_loss}")
        spearman = entry.get("val_spearman_corr")
        if isinstance(spearman, float):
            parts.append(f"spearman={spearman:.3f}")
        selection_metric = entry.get("selection_metric")
        if isinstance(selection_metric, float):
            parts.append(f"selection={selection_metric:.5f}")
        return ", ".join(parts) if parts else "(no metrics)"

    print(f"Run: {summary.get('run_name')} ({summary.get('run_dir')})")
    if summary.get("best_summary_line"):
        print(f"Best: {summary['best_summary_line']}")
    else:
        checkpoint = summary.get("best_checkpoint_name") or "N/A"
        print(f"Best checkpoint: {checkpoint}")
    alt_flag = summary.get("selection_metric_enabled", False)
    for line in _format_metric_block(summary, include_alt=alt_flag):
        print(line)

    learning = summary.get("learning_parameters") or {}
    if learning:
        ordered = ", ".join(f"{key}={value}" for key, value in sorted(learning.items()))
        print(f"Learning params: {ordered}")

    warnings = summary.get("warnings") or []
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")

    latest_metric = summary.get("latest_metric")
    if isinstance(latest_metric, dict):
        print(f"Latest metric: {_fmt_metric(latest_metric)}")

    recent_metrics = summary.get("recent_metrics")
    if metrics_limit and recent_metrics:
        print("Recent epochs:")
        for entry in recent_metrics:
            print(f"  - {_fmt_metric(entry)}")
    symlinks = summary.get("checkpoint_symlinks") or []
    if symlinks:
        print("Checkpoint ranking:")
        for entry in symlinks:
            target = entry.get("path")
            parts = [entry.get("name", "")]
            if target:
                parts.append(f"-> {target}")
            sel = entry.get("selection_metric")
            if isinstance(sel, (int, float)):
                parts.append(f"selection_metric={sel:.6f}")
            val_loss = entry.get("val_loss")
            if isinstance(val_loss, (int, float)):
                parts.append(f"val_loss={val_loss:.6f}")
            epoch = entry.get("epoch")
            if isinstance(epoch, int):
                parts.append(f"epoch={epoch}")
            print(f"  - {' '.join(part for part in parts if part)}")


def _render_once(run_dir: Path, args: argparse.Namespace) -> None:
    metrics_limit = args.metrics if args.metrics is not None else None
    summary = _summarise_best(run_dir, metrics_limit=metrics_limit)
    if args.output_format == "json":
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        _render_table(summary, metrics_limit)


def _render_top_runs(root: Path, count: int) -> None:
    limit = max(1, count)
    ranked = train_cli.rank_runs(root)  # type: ignore[attr-defined]
    if not ranked:
        print(f"[monitor_best_model] No runs with selection/val metrics found under {root}")
        return

    visible = min(limit, len(ranked))
    print(f"Top {visible} runs (primary metric ascending):")
    for idx, (metric_name, metric_value, summary) in enumerate(ranked[:visible], start=1):
        run_name = summary.get("run_name") or summary.get("run_dir") or "(unknown run)"
        print(f"{idx}. {run_name}")
        summary_copy = dict(summary)
        checkpoint = summary_copy.get("best_checkpoint")
        if checkpoint and "best_checkpoint_path" not in summary_copy:
            summary_copy["best_checkpoint_path"] = _to_repo_relative(str(checkpoint))
        summary_copy.setdefault("selection_primary_metric", summary.get("selection_primary_metric"))
        summary_copy.setdefault("best_selection_val_spearman", summary.get("best_selection_val_spearman"))
        for line in _format_metric_block(summary_copy, include_alt=True):
            print(line)
        print("")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report the current best checkpoint for a training run.")
    parser.add_argument("--run-id", type=str, help="Run identifier under training_runs/ (e.g., training_run_2025-01-01_12-00-00).")
    parser.add_argument("--run-dir", type=Path, help="Explicit path to a training run directory.")
    parser.add_argument("--follow", action="store_true", help="Poll indefinitely, emitting summaries every interval seconds.")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds when --follow is enabled.")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "table"),
        default="json",
        help="Select output format (json or table).",
    )
    parser.add_argument(
        "--metrics",
        type=int,
        default=0,
        help="Include the last N epochs of metrics in the output (0 disables the snippet).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=train_cli.RUN_ROOT,
        help="Training run root (used when --top is supplied).",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Show the top N runs (by primary metric) instead of a single run summary.",
    )
    args = parser.parse_args(argv)

    if args.top is not None:
        if args.top <= 0:
            parser.error("--top must be >= 1")
        if args.follow:
            parser.error("--top cannot be combined with --follow")
        if args.run_id or args.run_dir:
            parser.error("--top cannot be combined with --run-id/--run-dir")
        root = (args.root or train_cli.RUN_ROOT).resolve()
        if not root.exists():
            parser.error(f"Training root does not exist: {root}")
        _render_top_runs(root, args.top)
        return 0

    run_dir = _resolve_run_dir(args)
    if args.follow:
        try:
            while True:
                _render_once(run_dir, args)
                time.sleep(max(1, args.interval))
        except KeyboardInterrupt:
            return 0
    else:
        _render_once(run_dir, args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
