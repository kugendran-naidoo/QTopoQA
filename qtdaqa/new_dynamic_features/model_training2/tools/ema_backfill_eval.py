#!/usr/bin/env python3
"""
Backfill EMA/SWA validation + tuning metrics for existing runs.

Runs a validation pass using the averaged checkpoint and stores the results
in run_metadata.json under "ema_metrics" so Option B can remain eval-blind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from pytorch_lightning import Trainer, seed_everything

from qtdaqa.new_dynamic_features.model_training2.model_train_topoqa_cpu import (
    CpuTopoQAModule,
    TuningMetricsCallback,
    _safe_scalar,
    _setup_logging,
    build_dataloaders,
    load_config,
)
from qtdaqa.new_dynamic_features.model_training2.run_metadata import update_run_metadata


def _resolve_config(run_dir: Path) -> Path:
    candidates = [
        run_dir / "config" / "config.yaml",
        run_dir / "config" / "original_config.yaml",
        run_dir / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No config.yaml found under {run_dir}")


def _resolve_averaged_checkpoint(run_dir: Path, method: str) -> Optional[Path]:
    method = method.lower()
    candidates = []
    if method in {"ema", "swa"}:
        candidates.append(run_dir / "model_checkpoints" / f"averaged_{method}.chkpt")
    candidates.extend(
        [
            run_dir / "model_checkpoints" / "averaged_ema.chkpt",
            run_dir / "model_checkpoints" / "averaged_swa.chkpt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_edge_schema(feature_metadata, cfg_edge_schema: Dict[str, object], logger) -> Dict[str, object]:
    resolved = dict(feature_metadata.edge_schema)
    configured = dict(cfg_edge_schema or {})
    metadata_edge_dim = resolved.get("dim")
    config_edge_dim = configured.get("dim")
    if config_edge_dim is not None and metadata_edge_dim is not None:
        if int(config_edge_dim) != int(metadata_edge_dim):
            logger.warning(
                "Config edge_schema.dim=%s differs from metadata dim=%s; using metadata value.",
                config_edge_dim,
                metadata_edge_dim,
            )
    if metadata_edge_dim is not None:
        resolved["dim"] = int(metadata_edge_dim)
    elif config_edge_dim is not None:
        resolved["dim"] = int(config_edge_dim)
    for key, value in configured.items():
        if key == "dim":
            continue
        if value is not None:
            resolved[key] = value
    if "dim" not in resolved or resolved["dim"] is None:
        raise RuntimeError("Edge feature dimension could not be resolved from metadata or config.")
    return resolved


def _build_model(cfg, feature_metadata, logger):
    resolved_edge_schema = _resolve_edge_schema(feature_metadata, cfg.edge_schema, logger)
    feature_metadata.edge_schema = dict(resolved_edge_schema)

    feature_metadata_dict = feature_metadata.to_dict()
    feature_metadata_dict["topology_schema"] = feature_metadata.topology_schema or cfg.topology_schema
    if feature_metadata.topology_schema_spec:
        feature_metadata_dict["topology_schema_spec"] = feature_metadata.topology_schema_spec
    if feature_metadata.feature_config:
        feature_metadata_dict["feature_config"] = feature_metadata.feature_config

    edge_dim_value = int(resolved_edge_schema["dim"])
    node_dim_value = feature_metadata.node_schema.get("dim")
    node_dim = int(node_dim_value) if node_dim_value is not None else None
    if node_dim is None:
        logger.warning("Node feature dimension unavailable; falling back to architecture default.")

    return CpuTopoQAModule(
        lr=cfg.learning_rate,
        pooling_type=cfg.pooling_type,
        lr_scheduler_factor=cfg.lr_scheduler_factor,
        lr_scheduler_patience=cfg.lr_scheduler_patience,
        num_net=1,
        edge_dim=edge_dim_value,
        node_dim=node_dim,
        heads=cfg.attention_head,
        self_loops=cfg.model_self_loops,
        self_loop_fill=cfg.model_self_loop_fill,
        residual=cfg.model_residual,
        edge_schema=cfg.edge_schema,
        feature_metadata=feature_metadata_dict,
        rank_loss_weight=cfg.rank_loss_weight,
        rank_loss_margin=cfg.rank_loss_margin,
        rank_loss_mode=cfg.rank_loss_mode,
        rank_loss_grouped=cfg.rank_loss_grouped,
    )


def _record_ema_metrics(
    run_dir: Path,
    method: str,
    checkpoint_path: Path,
    metrics: Dict[str, object],
) -> None:
    def _mutate(metadata: Dict[str, object]) -> None:
        ema_block = metadata.setdefault("ema_metrics", {})
        if not isinstance(ema_block, dict):
            ema_block = {}
            metadata["ema_metrics"] = ema_block
        ema_block["checkpoint"] = str(checkpoint_path)
        ema_block["method"] = method
        for key in (
            "val_loss",
            "val_pearson_corr",
            "val_spearman_corr",
            "val_rank_spearman",
            "val_rank_regret",
            "tuning_rank_spearman",
            "tuning_rank_regret",
        ):
            value = _safe_scalar(metrics.get(key))
            if value is not None:
                ema_block[key] = value

    update_run_metadata(run_dir, _mutate)


def _iter_runs(root: Path) -> Iterable[Path]:
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_symlink():
            continue
        if not run_dir.is_dir():
            continue
        if not (run_dir / "run_metadata.json").exists():
            continue
        yield run_dir


def _should_skip(run_dir: Path, force: bool) -> bool:
    if force:
        return False
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(metadata.get("ema_metrics"), dict)


def evaluate_run(run_dir: Path, *, method: str, force: bool) -> bool:
    if _should_skip(run_dir, force):
        print(f"[ema_backfill] skip (already has ema_metrics): {run_dir}")
        return False

    config_path = _resolve_config(run_dir)
    cfg = load_config(config_path)
    cfg.save_dir = run_dir
    cfg.fast_dev_run = False

    logger = _setup_logging(run_dir)
    logger.info("EMA backfill starting for %s", run_dir)

    seed_everything(cfg.seed, workers=True)
    _train_loader, val_loader, tuning_loader, _, feature_metadata, _ = build_dataloaders(cfg, logger)
    model = _build_model(cfg, feature_metadata, logger)

    avg_checkpoint = _resolve_averaged_checkpoint(run_dir, method)
    if avg_checkpoint is None:
        logger.warning("No averaged checkpoint found for %s (method=%s)", run_dir, method)
        return False
    method_used = "swa" if "swa" in avg_checkpoint.name else "ema"

    callbacks = []
    if tuning_loader is not None:
        callbacks.append(TuningMetricsCallback(logger, tuning_loader, eval_every=1))

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    metrics_list = trainer.validate(
        model,
        dataloaders=val_loader,
        ckpt_path=str(avg_checkpoint),
    )
    if not metrics_list:
        logger.warning("No validation metrics returned for %s", run_dir)
        return False

    metrics = metrics_list[0]
    _record_ema_metrics(run_dir, method_used, avg_checkpoint, metrics)
    logger.info("EMA backfill completed for %s", run_dir)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill EMA/SWA validation metrics for existing runs.")
    parser.add_argument("--run-dir", type=Path, help="Single run directory to process.")
    default_root = Path(__file__).resolve().parents[1] / "training_runs2"
    parser.add_argument(
        "--training-root",
        type=Path,
        default=default_root,
        help="Root folder containing run directories.",
    )
    parser.add_argument(
        "--method",
        choices=("ema", "swa", "auto"),
        default="auto",
        help="Averaging method to use when selecting the averaged checkpoint.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing ema_metrics if present.")
    args = parser.parse_args()

    if args.run_dir and args.training_root:
        runs = [args.run_dir.resolve()]
    elif args.run_dir:
        runs = [args.run_dir.resolve()]
    else:
        runs = list(_iter_runs(args.training_root.resolve()))

    if not runs:
        print("[ema_backfill] No runs found.")
        return 1

    total = 0
    updated = 0
    for run in runs:
        total += 1
        method = args.method
        if method == "auto":
            try:
                cfg_method = load_config(_resolve_config(run)).variance_reduction_method
            except Exception:
                cfg_method = None
            method = cfg_method if cfg_method in {"ema", "swa"} else "ema"
        if evaluate_run(run, method=method, force=args.force):
            updated += 1

    print(f"[ema_backfill] completed {updated}/{total} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
