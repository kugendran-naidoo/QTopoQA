#!/usr/bin/env python3
"""
CPU-only training entry point for the TopoQA regression model.

This script replaces the legacy GPU-specific launchers with a parameterised,
well-validated pipeline that:
  * loads graph data from .pt files while attaching regression targets;
  * shuffles training batches and keeps validation deterministic;
  * applies early stopping and learning-rate scheduling to combat plateaus;
  * runs entirely on CPU hardware (accelerator='cpu').

Configuration is provided through a YAML file (default: config.yaml) whose
paths are resolved relative to the repository root.  Overrides are available
via CLI flags for quick experiments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import re
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import platform

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

try:  # Prefer absolute import so script execution still works
    from qtdaqa.new_dynamic_features.model_training2.run_metadata import (
        record_checkpoint_paths,
        record_selection_metadata,
        resolve_checkpoint_path,
        update_run_metadata,
    )
except ImportError:  # pragma: no cover - fallback for ad-hoc launches
    from run_metadata import (
        record_checkpoint_paths,
        record_selection_metadata,
        resolve_checkpoint_path,
        update_run_metadata,
    )

try:  # Optional MLflow dependency
    from pytorch_lightning.loggers import MLFlowLogger  # type: ignore
    _MLFLOW_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - MLflow optional
    MLFlowLogger = None  # type: ignore
    _MLFLOW_IMPORT_ERROR = exc
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model"
COMMON_DIR = SCRIPT_DIR.parent / "common"
for path in (MODEL_DIR, COMMON_DIR):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_metadata import GraphFeatureMetadata, load_graph_feature_metadata  # type: ignore  # noqa: E402
try:
    from gat_5_edge1 import GNN_edge1_edgepooling  # type: ignore  # noqa: E402
except ImportError as exc:  # pragma: no cover - helpful hint for misconfigured launches
    raise ImportError(
        "Unable to import the GNN model. If you are running this script directly, ensure "
        "you launch it from the repository root (so PYTHONPATH includes qtdaqa/new_dynamic_features/model_training2) "
        "or run it as a module: 'python -m qtdaqa.new_dynamic_features.model_training2.model_train_topoqa_cpu ...'."
    ) from exc
try:
    from common.graph_cache import GraphTensorCache  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    GraphTensorCache = None  # type: ignore
from common.metadata_artifacts import write_feature_metadata_artifacts  # type: ignore  # noqa: E402
try:
    from builder_metadata import load_builder_info_from_metadata, log_builder_provenance  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    from qtdaqa.new_dynamic_features.model_training2.builder_metadata import (  # type: ignore  # noqa: E402
        load_builder_info_from_metadata,
        log_builder_provenance,
    )
try:
    from common.ranking_metrics import compute_grouped_ranking_metrics  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - optional helper
    compute_grouped_ranking_metrics = None
from common.resume_guard import validate_resume_checkpoint  # type: ignore  # noqa: E402

@dataclasses.dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: str = "./mlruns"
    experiment: str = "dynamic_topoqa"
    run_name: Optional[str] = None
    log_artifacts: bool = True
    tags: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingConfig:
    graph_dir: Path
    train_label_file: Path
    val_label_file: Path
    save_dir: Path
    tuning_label_file: Optional[Path] = None
    metadata_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    accelerator: str = "cpu"
    devices: int = 1
    attention_head: int = 8
    pooling_type: str = "mean"
    model_self_loops: bool = False
    model_self_loop_fill: object = "mean"
    model_residual: bool = False
    batch_size: int = 16
    learning_rate: float = 5e-3
    num_epochs: int = 200
    accumulate_grad_batches: int = 32
    seed: int = 222
    num_workers: int = 0
    precision: int = 32
    lr_scheduler_type: str = "reduce_on_plateau"
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    fast_dev_run: bool = False
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None
    log_lr: bool = False
    progress_bar_refresh_rate: int = 0
    log_every_n_steps: int = 100
    use_val_spearman_as_secondary: bool = True
    spearman_secondary_min_delta: float = 0.0
    spearman_secondary_weight: float = 1.0
    selection_primary_metric: str = "val_loss"
    tuning_max_samples: Optional[int] = None
    tuning_fraction: Optional[float] = None
    tuning_eval_every: int = 1
    tuning_min_groups_ge2: int = 200
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 0.0
    rank_loss_mode: str = "hinge"
    rank_loss_grouped: bool = True
    variance_reduction_enabled: bool = False
    variance_reduction_method: str = "topk_avg"
    variance_reduction_top_k: int = 3
    variance_reduction_ema_decay: float = 0.999
    variance_reduction_swa_start: float = 0.75
    variance_reduction_save_every_epochs: int = 0
    variance_reduction_resume: bool = True
    coverage_minimum: float = 0.0
    coverage_fail_on_missing: bool = True
    graph_load_profiling: bool = False
    graph_load_top_k: int = 5
    enable_graph_cache: bool = True
    graph_cache_size: int = 256
    canonicalize_on_load: bool = False
    mlflow: MlflowConfig = dataclasses.field(default_factory=MlflowConfig)
    edge_schema: dict = dataclasses.field(default_factory=dict)
    topology_schema: dict = dataclasses.field(default_factory=dict)


def _normalise_ratio(value: float | int | str, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric < 0:
        return 0.0
    if numeric > 1.0:
        numeric = numeric / 100.0
    return min(max(numeric, 0.0), 1.0)


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_swa_start_epoch(value: float, total_epochs: int) -> int:
    if total_epochs <= 0:
        return 0
    if value < 1.0:
        return max(0, min(total_epochs - 1, int(round(total_epochs * value))))
    return max(0, min(total_epochs - 1, int(round(value))))


PRIMARY_METRIC_CHOICES = {
    "val_loss",
    "selection_metric",
    "val_rank_spearman",
    "val_rank_regret",
    "tuning_rank_spearman",
    "tuning_rank_regret",
}

MAXIMIZE_METRICS = {
    "val_spearman_corr",
    "val_rank_spearman",
    "tuning_rank_spearman",
}


def _metric_mode(metric_name: str) -> str:
    return "max" if metric_name in MAXIMIZE_METRICS else "min"


def _merge_defaults_into_edge_schema(feature_metadata: "GraphFeatureMetadata") -> None:
    """Merge recorded defaults into edge_schema.module_params to make schemas explicit."""
    edge_schema = feature_metadata.edge_schema or {}
    module_id = edge_schema.get("module")
    if not module_id:
        return

    registry = feature_metadata.module_registry if isinstance(feature_metadata.module_registry, dict) else {}
    defaults: Dict[str, object] = {}
    candidates: List[Dict[str, object]] = []
    edge_entry = registry.get("edge")
    if isinstance(edge_entry, dict):
        candidates.append(edge_entry)
    candidates.extend([entry for entry in registry.values() if isinstance(entry, dict)])
    for entry in candidates:
        module_value = entry.get("id") or entry.get("module")
        if module_value != module_id:
            continue
        if isinstance(entry.get("defaults"), dict):
            defaults.update(entry["defaults"])
        if isinstance(entry.get("params"), dict):
            defaults.update(entry["params"])
        break

    if not defaults:
        return

    current_params = edge_schema.get("module_params") if isinstance(edge_schema, dict) else None
    merged = {**defaults, **(current_params or {})}
    feature_metadata.edge_schema = dict(edge_schema)
    feature_metadata.edge_schema["module_params"] = merged


def _resolve_path(
    raw: str | Path,
    base: Path,
    *,
    allow_missing: bool = False,
    fallbacks: Optional[Sequence[Path]] = None,
) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    primary = (base / path).resolve()
    if primary.exists() or allow_missing:
        return primary
    fallbacks = fallbacks or ()
    for fallback in fallbacks:
        candidate = (Path(fallback) / path).resolve()
        if candidate.exists() or allow_missing:
            return candidate
    return primary


def load_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the root")

    base_dir = path.parent.resolve()

    if "paths" not in data:
        def _get(name: str, default=None):
            return data.get(name, default)

        graph_dir = _resolve_path(_get("graph_dir"), base_dir, fallbacks=(SCRIPT_DIR,))
        train_label_file = _resolve_path(_get("train_label_file"), base_dir, fallbacks=(SCRIPT_DIR,))
        val_label_file = _resolve_path(_get("val_label_file"), base_dir, fallbacks=(SCRIPT_DIR,))
        save_dir = _resolve_path(_get("save_dir", "./training_runs2"), base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)

        metadata_path_raw = _get("metadata_path")
        summary_path_raw = _get("summary_path")
        metadata_path = (
            _resolve_path(metadata_path_raw, base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)
            if metadata_path_raw
            else None
        )
        summary_path = (
            _resolve_path(summary_path_raw, base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)
            if summary_path_raw
            else None
        )

        coverage_cfg = _get("coverage", {})
        if not isinstance(coverage_cfg, dict):
            coverage_cfg = {}
        profiling_cfg = _get("profiling", {})
        if not isinstance(profiling_cfg, dict):
            profiling_cfg = {}
        coverage_minimum = _normalise_ratio(
            coverage_cfg.get("minimum_percent", coverage_cfg.get("minimum", 0.0)), default=0.0
        )
        coverage_fail_raw = coverage_cfg.get("fail_on_missing")
        coverage_fail_on_missing = True if coverage_fail_raw is None else bool(coverage_fail_raw)
        try:
            graph_load_top_k = int(profiling_cfg.get("graph_load_top_k", 5))
        except (TypeError, ValueError):
            graph_load_top_k = 5
        graph_load_top_k = max(1, graph_load_top_k)
        graph_load_profiling = bool(profiling_cfg.get("graph_load", False))
        canonicalize_on_load = bool(_get("canonicalize_on_load", False))
        tuning_cfg = _get("tuning", {})
        if not isinstance(tuning_cfg, dict):
            tuning_cfg = {}
        tuning_label_raw = (
            tuning_cfg.get("labels")
            or _get("tuning_label_file")
            or _get("tuning_labels")
        )
        tuning_label_file = (
            _resolve_path(tuning_label_raw, base_dir, fallbacks=(SCRIPT_DIR,))
            if tuning_label_raw
            else None
        )
        tuning_max_samples = tuning_cfg.get("max_samples")
        try:
            tuning_max_samples = int(tuning_max_samples) if tuning_max_samples is not None else None
        except (TypeError, ValueError):
            tuning_max_samples = None
        tuning_fraction_raw = tuning_cfg.get("fraction")
        tuning_fraction = _normalise_ratio(tuning_fraction_raw, default=0.0) if tuning_fraction_raw is not None else None
        tuning_eval_every = tuning_cfg.get("eval_every", 1)
        try:
            tuning_eval_every = max(1, int(tuning_eval_every))
        except (TypeError, ValueError):
            tuning_eval_every = 1
        tuning_min_groups_raw = tuning_cfg.get("min_groups_ge2", tuning_cfg.get("min_groups", 200))
        try:
            tuning_min_groups_ge2 = int(tuning_min_groups_raw) if tuning_min_groups_raw is not None else 0
        except (TypeError, ValueError):
            tuning_min_groups_ge2 = 0
        tuning_min_groups_ge2 = max(0, tuning_min_groups_ge2)

        ranking_cfg = _get("ranking_loss", {})
        if not isinstance(ranking_cfg, dict):
            ranking_cfg = {}
        rank_loss_weight = float(ranking_cfg.get("weight", 0.0) or 0.0)
        rank_loss_margin = float(ranking_cfg.get("margin", 0.0) or 0.0)
        rank_loss_mode = str(ranking_cfg.get("mode", "hinge") or "hinge")
        rank_loss_grouped = bool(ranking_cfg.get("grouped", True))
        variance_cfg = _get("variance_reduction", {})
        if not isinstance(variance_cfg, dict):
            variance_cfg = {}
        variance_reduction_enabled = bool(variance_cfg.get("enabled", False))
        variance_reduction_method = str(variance_cfg.get("method", "topk_avg") or "topk_avg")
        try:
            variance_reduction_top_k = int(variance_cfg.get("top_k", 3))
        except (TypeError, ValueError):
            variance_reduction_top_k = 3
        variance_reduction_ema_decay = _coerce_float(variance_cfg.get("ema_decay"), 0.999)
        if not 0.0 < variance_reduction_ema_decay <= 1.0:
            variance_reduction_ema_decay = 0.999
        variance_reduction_swa_start = _coerce_float(variance_cfg.get("swa_start"), 0.75)
        if variance_reduction_swa_start < 0.0:
            variance_reduction_swa_start = 0.75
        variance_reduction_save_every_epochs = int(variance_cfg.get("save_every_epochs", 0) or 0)
        if variance_reduction_save_every_epochs < 0:
            variance_reduction_save_every_epochs = 0
        variance_reduction_resume = bool(variance_cfg.get("resume", True))

        cfg = TrainingConfig(
            graph_dir=graph_dir,
            train_label_file=train_label_file,
            val_label_file=val_label_file,
            tuning_label_file=tuning_label_file,
            save_dir=save_dir,
            metadata_path=metadata_path,
            summary_path=summary_path,
            accelerator=str(_get("accelerator", "cpu")),
            devices=int(_get("devices", 1)),
            attention_head=int(_get("attention_head", 8)),
            pooling_type=str(_get("pooling_type", "mean")),
            model_self_loops=bool(_get("self_loops", False)),
            model_self_loop_fill=_get("self_loop_fill", "mean"),
            model_residual=bool(_get("residual", False)),
            batch_size=int(_get("batch_size", 16)),
            learning_rate=float(_get("learning_rate", 5e-3)),
            num_epochs=int(_get("num_epochs", 200)),
            accumulate_grad_batches=int(_get("accumulate_grad_batches", 32)),
            seed=int(_get("seed", 222)),
            num_workers=int(_get("num_workers", 0)),
            precision=int(_get("precision", 32)),
            lr_scheduler_type=str(_get("lr_scheduler_type", "reduce_on_plateau")),
            early_stopping_patience=int(_get("early_stopping_patience", 20)),
            lr_scheduler_patience=int(_get("lr_scheduler_patience", 10)),
            lr_scheduler_factor=float(_get("lr_scheduler_factor", 0.5)),
            progress_bar_refresh_rate=int(_get("progress_bar_refresh_rate", 0)),
            log_every_n_steps=int(_get("log_every_n_steps", 100)),
            use_val_spearman_as_secondary=bool(_get("use_val_spearman_as_secondary", True)),
            spearman_secondary_min_delta=float(_get("spearman_secondary_min_delta", 0.0)),
            spearman_secondary_weight=float(_get("spearman_secondary_weight", 1.0)),
            coverage_minimum=coverage_minimum,
            coverage_fail_on_missing=coverage_fail_on_missing,
            graph_load_profiling=graph_load_profiling,
            graph_load_top_k=graph_load_top_k,
            mlflow=MlflowConfig(),
            edge_schema=dict(_get("edge_schema", {})),
            topology_schema=dict(_get("topology_schema", {})),
            canonicalize_on_load=canonicalize_on_load,
            tuning_max_samples=tuning_max_samples,
            tuning_fraction=tuning_fraction,
            tuning_eval_every=tuning_eval_every,
            tuning_min_groups_ge2=tuning_min_groups_ge2,
            rank_loss_weight=rank_loss_weight,
            rank_loss_margin=rank_loss_margin,
            rank_loss_mode=rank_loss_mode,
            rank_loss_grouped=rank_loss_grouped,
            variance_reduction_enabled=variance_reduction_enabled,
            variance_reduction_method=variance_reduction_method,
            variance_reduction_top_k=variance_reduction_top_k,
            variance_reduction_ema_decay=variance_reduction_ema_decay,
            variance_reduction_swa_start=variance_reduction_swa_start,
            variance_reduction_save_every_epochs=variance_reduction_save_every_epochs,
            variance_reduction_resume=variance_reduction_resume,
        )
        return cfg

    def _section(name: str) -> Dict[str, object]:
        value = data.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"Section '{name}' must be a mapping")
        return value

    paths_cfg = _section("paths")
    if not paths_cfg:
        raise ValueError("Configuration section 'paths' is required")

    def _require(cfg: Dict[str, object], key: str) -> object:
        if key not in cfg:
            raise KeyError(f"Missing required configuration key: paths.{key}")
        return cfg[key]

    graph_dir = _resolve_path(_require(paths_cfg, "graph"), base_dir, fallbacks=(SCRIPT_DIR,))
    train_label_file = _resolve_path(_require(paths_cfg, "train_labels"), base_dir, fallbacks=(SCRIPT_DIR,))
    val_label_file = _resolve_path(_require(paths_cfg, "val_labels"), base_dir, fallbacks=(SCRIPT_DIR,))
    save_dir = _resolve_path(paths_cfg.get("save_dir", "./training_runs2"), base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)

    metadata_path = paths_cfg.get("metadata")
    if metadata_path is not None:
        metadata_path = _resolve_path(metadata_path, base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)

    summary_path = paths_cfg.get("summary")
    if summary_path is not None:
        summary_path = _resolve_path(summary_path, base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)

    model_cfg = _section("model")
    pooling_type = str(model_cfg.get("pooling_type", "mean"))
    attention_head = int(model_cfg.get("attention_head", 8))
    model_self_loops = bool(model_cfg.get("self_loops", False))
    model_self_loop_fill = model_cfg.get("self_loop_fill", "mean")
    model_residual = bool(model_cfg.get("residual", False))

    dataloader_cfg = _section("dataloader")
    batch_size = int(dataloader_cfg.get("batch_size", 16))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    seed = int(dataloader_cfg.get("seed", 222))
    canonicalize_on_load = bool(dataloader_cfg.get("canonicalize_on_load", False))

    trainer_cfg = _section("trainer")
    accelerator = str(trainer_cfg.get("accelerator", "cpu"))
    devices = int(trainer_cfg.get("devices", 1))
    precision = int(trainer_cfg.get("precision", 32))
    num_epochs = int(trainer_cfg.get("num_epochs", 200))
    accumulate_grad_batches = int(trainer_cfg.get("accumulate_grad_batches", 32))

    optimizer_cfg = _section("optimizer")
    learning_rate = float(optimizer_cfg.get("learning_rate", 5e-3))

    scheduler_cfg = _section("scheduler")
    lr_scheduler_type = str(scheduler_cfg.get("type", "reduce_on_plateau"))
    lr_scheduler_factor = float(scheduler_cfg.get("factor", 0.5))
    lr_scheduler_patience = int(scheduler_cfg.get("patience", 10))

    early_cfg = _section("early_stopping")
    early_stopping_patience = int(early_cfg.get("patience", 20))

    selection_cfg = _section("selection")
    use_val_spearman = bool(selection_cfg.get("use_val_spearman", True))
    spearman_min_delta = float(selection_cfg.get("spearman_min_delta", 0.0))
    spearman_weight = float(selection_cfg.get("spearman_weight", 1.0))
    primary_metric_raw = selection_cfg.get("primary_metric", "val_loss")
    if isinstance(primary_metric_raw, str):
        selection_primary_metric = primary_metric_raw.strip().lower() or "val_loss"
    else:
        selection_primary_metric = "val_loss"
    if selection_primary_metric not in PRIMARY_METRIC_CHOICES:
        raise ValueError(
            "selection.primary_metric must be one of: "
            + ", ".join(sorted(PRIMARY_METRIC_CHOICES))
        )

    logging_cfg = _section("logging")
    progress_bar_refresh_rate = int(logging_cfg.get("progress_bar_refresh_rate", 0))
    log_every_n_steps = int(logging_cfg.get("log_every_n_steps", 100))

    mlflow_cfg = _section("mlflow")
    mlflow_config = MlflowConfig(
        enabled=bool(mlflow_cfg.get("enabled", False)),
        tracking_uri=str(mlflow_cfg.get("tracking_uri", "./mlruns")),
        experiment=str(mlflow_cfg.get("experiment", "dynamic_topoqa")),
        run_name=mlflow_cfg.get("run_name"),
        log_artifacts=bool(mlflow_cfg.get("log_artifacts", True)),
        tags=dict(mlflow_cfg.get("tags", {}) or {}),
    )

    coverage_cfg = _section("coverage")
    coverage_minimum = _normalise_ratio(
        coverage_cfg.get("minimum_percent", coverage_cfg.get("minimum", 0.0)), default=0.0
    )
    coverage_fail_raw = coverage_cfg.get("fail_on_missing")
    coverage_fail_on_missing = True if coverage_fail_raw is None else bool(coverage_fail_raw)

    profiling_cfg = _section("profiling")
    graph_load_profiling = bool(profiling_cfg.get("graph_load", False))
    try:
        graph_load_top_k = int(profiling_cfg.get("graph_load_top_k", 5))
    except (TypeError, ValueError):
        graph_load_top_k = 5
    graph_load_top_k = max(1, graph_load_top_k)

    cache_cfg = _section("cache")
    enable_graph_cache = bool(cache_cfg.get("enable_graph_cache", True))
    try:
        graph_cache_size = int(cache_cfg.get("graph_cache_size", 256))
    except (TypeError, ValueError):
        graph_cache_size = 256
    graph_cache_size = max(1, graph_cache_size)

    tuning_cfg = _section("tuning")
    tuning_label_raw = tuning_cfg.get("labels") or paths_cfg.get("tuning_labels")
    tuning_label_file = (
        _resolve_path(tuning_label_raw, base_dir, fallbacks=(SCRIPT_DIR,))
        if tuning_label_raw
        else None
    )
    tuning_max_samples = tuning_cfg.get("max_samples")
    try:
        tuning_max_samples = int(tuning_max_samples) if tuning_max_samples is not None else None
    except (TypeError, ValueError):
        tuning_max_samples = None
    tuning_fraction_raw = tuning_cfg.get("fraction")
    tuning_fraction = _normalise_ratio(tuning_fraction_raw, default=0.0) if tuning_fraction_raw is not None else None
    tuning_eval_every = tuning_cfg.get("eval_every", 1)
    try:
        tuning_eval_every = max(1, int(tuning_eval_every))
    except (TypeError, ValueError):
        tuning_eval_every = 1
    tuning_min_groups_raw = tuning_cfg.get("min_groups_ge2", tuning_cfg.get("min_groups", 200))
    try:
        tuning_min_groups_ge2 = int(tuning_min_groups_raw) if tuning_min_groups_raw is not None else 0
    except (TypeError, ValueError):
        tuning_min_groups_ge2 = 0
    tuning_min_groups_ge2 = max(0, tuning_min_groups_ge2)

    ranking_cfg = _section("ranking_loss")
    rank_loss_weight = float(ranking_cfg.get("weight", 0.0) or 0.0)
    rank_loss_margin = float(ranking_cfg.get("margin", 0.0) or 0.0)
    rank_loss_mode = str(ranking_cfg.get("mode", "hinge") or "hinge")
    rank_loss_grouped = bool(ranking_cfg.get("grouped", True))
    variance_cfg = _section("variance_reduction")
    variance_reduction_enabled = bool(variance_cfg.get("enabled", False))
    variance_reduction_method = str(variance_cfg.get("method", "topk_avg") or "topk_avg")
    try:
        variance_reduction_top_k = int(variance_cfg.get("top_k", 3))
    except (TypeError, ValueError):
        variance_reduction_top_k = 3
    variance_reduction_ema_decay = _coerce_float(variance_cfg.get("ema_decay"), 0.999)
    if not 0.0 < variance_reduction_ema_decay <= 1.0:
        variance_reduction_ema_decay = 0.999
    variance_reduction_swa_start = _coerce_float(variance_cfg.get("swa_start"), 0.75)
    if variance_reduction_swa_start < 0.0:
        variance_reduction_swa_start = 0.75
    variance_reduction_save_every_epochs = int(variance_cfg.get("save_every_epochs", 0) or 0)
    if variance_reduction_save_every_epochs < 0:
        variance_reduction_save_every_epochs = 0
    variance_reduction_resume = bool(variance_cfg.get("resume", True))

    cfg = TrainingConfig(
        graph_dir=graph_dir,
        train_label_file=train_label_file,
        val_label_file=val_label_file,
        tuning_label_file=tuning_label_file,
        save_dir=save_dir,
        metadata_path=metadata_path,
        summary_path=summary_path,
        accelerator=accelerator,
        devices=devices,
        attention_head=attention_head,
        pooling_type=pooling_type,
        model_self_loops=model_self_loops,
        model_self_loop_fill=model_self_loop_fill,
        model_residual=model_residual,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        seed=seed,
        num_workers=num_workers,
        precision=precision,
        lr_scheduler_type=lr_scheduler_type,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        log_every_n_steps=log_every_n_steps,
        use_val_spearman_as_secondary=use_val_spearman,
        spearman_secondary_min_delta=spearman_min_delta,
        spearman_secondary_weight=spearman_weight,
        selection_primary_metric=selection_primary_metric,
        tuning_max_samples=tuning_max_samples,
        tuning_fraction=tuning_fraction,
        tuning_eval_every=tuning_eval_every,
        tuning_min_groups_ge2=tuning_min_groups_ge2,
        rank_loss_weight=rank_loss_weight,
        rank_loss_margin=rank_loss_margin,
        rank_loss_mode=rank_loss_mode,
        rank_loss_grouped=rank_loss_grouped,
        variance_reduction_enabled=variance_reduction_enabled,
        variance_reduction_method=variance_reduction_method,
        variance_reduction_top_k=variance_reduction_top_k,
        variance_reduction_ema_decay=variance_reduction_ema_decay,
        variance_reduction_swa_start=variance_reduction_swa_start,
        variance_reduction_save_every_epochs=variance_reduction_save_every_epochs,
        variance_reduction_resume=variance_reduction_resume,
        coverage_minimum=coverage_minimum,
        coverage_fail_on_missing=coverage_fail_on_missing,
        graph_load_profiling=graph_load_profiling,
        graph_load_top_k=graph_load_top_k,
        enable_graph_cache=enable_graph_cache,
        graph_cache_size=graph_cache_size,
        mlflow=mlflow_config,
        edge_schema=dict(data.get("edge_schema", {})),
        topology_schema=dict(data.get("topology_schema", {})),
        canonicalize_on_load=canonicalize_on_load,
    )
    return cfg


def load_label_map(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
    import pandas as pd  # defer heavy imports until needed
    import numpy as np

    df = pd.read_csv(csv_path)
    if "MODEL" not in df or "dockq" not in df:
        raise ValueError(f"Label CSV {csv_path} missing MODEL/dockq columns")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dockq"])
    mapping = {str(row.MODEL): float(row.dockq) for row in df.itertuples()}
    if not mapping:
        raise ValueError(f"No valid labels found in {csv_path}")
    return mapping


def _build_graph_lookup(graph_dir: Path) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = {}
    for path in sorted(graph_dir.rglob("*.pt")):
        stem_key = path.stem
        lookup.setdefault(stem_key, []).append(path)
        rel_key = path.relative_to(graph_dir).as_posix()
        rel_key = rel_key[:-3] if rel_key.endswith(".pt") else rel_key
        lookup.setdefault(rel_key, []).append(path)
    return lookup


def _gather_samples(
    graph_dir: Path,
    label_map: Dict[str, float],
    graph_lookup: Dict[str, List[Path]],
) -> Tuple[List[Tuple[str, Path, torch.Tensor]], List[str]]:
    samples: List[Tuple[str, Path, torch.Tensor]] = []
    missing: List[str] = []
    for model, label in label_map.items():
        candidate = graph_dir / f"{model}.pt"
        if candidate.exists():
            tensor = torch.tensor([label], dtype=torch.float32)
            samples.append((model, candidate, tensor))
            continue

        matches = graph_lookup.get(model) or graph_lookup.get(model.replace("__", "/"))
        if matches:
            resolved = matches[0]
            tensor = torch.tensor([label], dtype=torch.float32)
            samples.append((model, resolved, tensor))
            continue

        missing.append(model)
    samples.sort(key=lambda item: item[0])
    return samples, missing


class GraphLoadProfiler:
    def __init__(self, enabled: bool, top_k: int, num_workers: int, logger: Optional[logging.Logger]):
        top_k = max(1, int(top_k)) if top_k else 5
        if enabled and num_workers > 0:
            if logger is not None:
                logger.warning(
                    "Graph load profiling is only supported with dataloader.num_workers=0; disabling profiler."
                )
            enabled = False
        self.enabled = bool(enabled)
        self.top_k = top_k
        self._records: List[Tuple[float, str]] = []

    def record(self, path: Path, duration: float) -> None:
        if not self.enabled:
            return
        entry = (float(duration), str(path))
        self._records.append(entry)

    def finalise(self, save_dir: Optional[Path], logger: logging.Logger) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        records = list(self._records)
        if not records:
            logger.info("Graph load profiling enabled but no samples were collected.")
            return None

        durations = sorted(duration for duration, _ in records)
        count = len(durations)
        total = sum(durations)
        mean = total / count
        median = statistics.median(durations)
        p95 = durations[int(min(count - 1, max(0, round(0.95 * (count - 1)))))]
        maximum_duration, maximum_path = max(records, key=lambda entry: entry[0])
        top_records = sorted(records, key=lambda entry: entry[0], reverse=True)[: self.top_k]
        summary = {
            "count": count,
            "mean_ms": mean * 1000.0,
            "median_ms": median * 1000.0,
            "p95_ms": p95 * 1000.0,
            "max_ms": maximum_duration * 1000.0,
            "max_path": maximum_path,
            "top_samples": [
                {"path": path, "ms": duration * 1000.0} for duration, path in top_records
            ],
        }

        logger.info(
            "Graph load profiling: count=%d mean=%.2fms median=%.2fms p95=%.2fms max=%.2fms (%s)",
            count,
            summary["mean_ms"],
            summary["median_ms"],
            summary["p95_ms"],
            summary["max_ms"],
            maximum_path,
        )
        top_preview = ", ".join(f"{item['ms']:.2f}ms:{item['path']}" for item in summary["top_samples"])
        logger.info("Graph load profiling slowest paths: %s", top_preview)

        if save_dir:
            output_path = Path(save_dir) / "graph_load_profile.json"
            try:
                with output_path.open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
                summary["output_path"] = str(output_path)
                logger.info("Graph load profiling summary written to %s", output_path)
            except OSError as exc:
                logger.warning("Unable to write graph load profiling summary: %s", exc)
        return summary


class GraphRegressionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, Path, torch.Tensor]],
        profiler: Optional[GraphLoadProfiler] = None,
        cache: Optional[GraphTensorCache] = None,
        canonicalize_on_load: bool = False,
    ):
        self.samples = list(samples)
        self.profiler = profiler
        self.cache = cache
        self.canonicalize_on_load = canonicalize_on_load

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        model, path, label = self.samples[idx]
        def _load_tensor(p: Path):
            return torch.load(p)

        start = None
        if self.profiler and self.profiler.enabled:
            start = time.perf_counter()

        if self.cache:
            data = self.cache.get(path, _load_tensor)
        else:
            data = _load_tensor(path)

        if self.canonicalize_on_load:
            data = canonicalize_graph(data)

        if self.profiler and self.profiler.enabled and start is not None:
            duration = time.perf_counter() - start
            self.profiler.record(path, duration)

        if hasattr(data, "batch"):
            batch_attr = getattr(data, "batch")
            if batch_attr is None or not torch.is_tensor(batch_attr):
                delattr(data, "batch")
        data.y = label.clone()
        data.name = model
        return data


def collate_graphs(batch: List[Data]) -> List[Batch]:
    if not batch:
        raise ValueError("Received empty batch")
    merged = Batch.from_data_list(batch)
    graph_names = [str(getattr(item, "name", "")) for item in batch]
    merged.graph_names = graph_names
    merged._graph_names = graph_names
    return [merged]


def _node_permutation(x: torch.Tensor) -> torch.Tensor:
    x_cpu = x.detach().cpu()
    n = x_cpu.size(0)
    if n == 0:
        return torch.arange(0, device=x.device, dtype=torch.long)

    keys = [np.arange(n)]
    x_np = x_cpu.numpy()
    if x_np.ndim == 1:
        keys.append(x_np)
    else:
        for col in x_np.T[::-1]:
            keys.append(col)
    order = np.lexsort(tuple(keys))
    return torch.from_numpy(order).to(x.device, dtype=torch.long)


def _edge_permutation(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.arange(0, device=edge_index.device, dtype=torch.long)

    edge_cpu = edge_index.detach().cpu()
    src = edge_cpu[0].numpy()
    dst = edge_cpu[1].numpy()
    e = src.shape[0]

    keys = [np.arange(e)]
    if edge_attr is not None and torch.is_tensor(edge_attr) and edge_attr.numel() > 0:
        attr_np = edge_attr.detach().cpu().numpy()
        if attr_np.ndim == 1:
            keys.append(attr_np)
        else:
            for col in attr_np.T[::-1]:
                keys.append(col)
    keys.append(dst)
    keys.append(src)
    order = np.lexsort(tuple(keys))
    return torch.from_numpy(order).to(edge_index.device, dtype=torch.long)


def canonicalize_graph(data: Data) -> Data:
    """Return a canonicalized copy of the graph (sorted nodes/edges) for deterministic loading."""
    if not isinstance(data, Data):
        return data

    # Work on a shallow clone to avoid mutating cached objects.
    data = data.clone()

    x = getattr(data, "x", None)
    if isinstance(x, torch.Tensor) and x.numel() > 0:
        perm = _node_permutation(x)
        data.x = x[perm]

        batch_attr = getattr(data, "batch", None)
        if isinstance(batch_attr, torch.Tensor) and batch_attr.numel() == perm.numel():
            data.batch = batch_attr[perm]

        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)

        edge_index = getattr(data, "edge_index", None)
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            remapped = perm_inv[edge_index]
            edge_attr = getattr(data, "edge_attr", None)
            if edge_attr is not None and not torch.is_tensor(edge_attr):
                edge_attr = None
            order = _edge_permutation(remapped, edge_attr)
            data.edge_index = remapped[:, order]

            for edge_field in ("edge_attr", "edge_weight"):
                value = getattr(data, edge_field, None)
                if isinstance(value, torch.Tensor) and value.size(0) == order.numel():
                    setattr(data, edge_field, value[order])
    else:
        # No node features; fall back to sorting edges only if present.
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            order = _edge_permutation(edge_index, edge_attr if torch.is_tensor(edge_attr) else None)
            data.edge_index = edge_index[:, order]
            if isinstance(edge_attr, torch.Tensor) and edge_attr.size(0) == order.numel():
                data.edge_attr = edge_attr[order]

    return data


class CpuTopoQAModule(GNN_edge1_edgepooling):
    def __init__(
        self,
        *,
        lr: float,
        pooling_type: str,
        lr_scheduler_factor: float,
        lr_scheduler_patience: int,
        edge_schema: Optional[Dict[str, object]] = None,
        feature_metadata: Optional[Dict[str, object]] = None,
        rank_loss_weight: float = 0.0,
        rank_loss_margin: float = 0.0,
        rank_loss_mode: str = "hinge",
        rank_loss_grouped: bool = True,
        **kwargs,
    ):
        super().__init__(
            init_lr=lr,
            pooling_type=pooling_type,
            mode="zuhe",
            edge_schema=edge_schema,
            rank_loss_weight=rank_loss_weight,
            rank_loss_margin=rank_loss_margin,
            rank_loss_mode=rank_loss_mode,
            rank_loss_grouped=rank_loss_grouped,
            **kwargs,
        )
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.feature_metadata = feature_metadata or {}
        self.model_hparams: Dict[str, object] = {
            "lr": lr,
            "pooling_type": pooling_type,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "rank_loss_weight": rank_loss_weight,
            "rank_loss_margin": rank_loss_margin,
            "rank_loss_mode": rank_loss_mode,
            "rank_loss_grouped": rank_loss_grouped,
        }
        for key in ("heads", "edge_dim", "node_dim", "num_net", "self_loops", "self_loop_fill", "residual"):
            if key in kwargs:
                self.model_hparams[key] = kwargs[key]
        if edge_schema is not None:
            self.model_hparams["edge_schema_source"] = edge_schema.get("source") if isinstance(edge_schema, dict) else None

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        checkpoint["feature_metadata"] = self.feature_metadata
        checkpoint["model_hyperparameters"] = self.model_hparams

    def on_load_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        self.feature_metadata = checkpoint.get("feature_metadata", {})
        self.model_hparams = checkpoint.get("model_hyperparameters", getattr(self, "model_hparams", {}))



def _setup_logging(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_cpu_topoqa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def _summarise_coverage(
    logger: logging.Logger,
    split: str,
    total_labels: int,
    usable_graphs: int,
    missing: Sequence[str],
) -> None:
    coverage = usable_graphs / total_labels if total_labels else 0.0
    logger.info(
        "[%s] label rows: %d | usable graphs: %d | coverage: %.2f%%",
        split,
        total_labels,
        usable_graphs,
        coverage * 100.0,
    )
    if missing:
        preview = ", ".join(missing[:5])
        logger.error(
            "[%s] %d label entries missing matching .pt graphs (first examples: %s). "
            "Regenerate graph tensors or adjust coverage settings to proceed safely.",
            split,
            len(missing),
            preview,
        )


def _write_coverage_report(save_dir: Path, coverage: Dict[str, Dict[str, object]]) -> Path:
    report = {}
    for split, stats in coverage.items():
        total = int(stats.get("total", 0))
        usable = int(stats.get("usable", 0))
        missing = list(stats.get("missing", []))
        coverage_ratio = (usable / total) if total else 0.0
        stats["coverage"] = coverage_ratio
        report[split] = {
            "total": total,
            "usable": usable,
            "coverage": coverage_ratio,
            "missing": missing,
        }
    path = save_dir / "dataset_coverage.json"
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    except OSError as exc:
        raise RuntimeError(f"Unable to write dataset coverage report: {exc}")
    return path



def _evaluate_coverage(
    cfg: TrainingConfig,
    coverage: Dict[str, Dict[str, object]],
    logger: logging.Logger,
    report_path: Path,
) -> None:
    min_ratio = max(0.0, min(1.0, cfg.coverage_minimum))
    missing_issues: List[str] = []
    threshold_issues: List[str] = []

    for split, stats in coverage.items():
        total = int(stats.get("total", 0))
        usable = int(stats.get("usable", 0))
        missing = list(stats.get("missing", []))
        ratio = stats.get("coverage", (usable / total) if total else 0.0)

        if missing:
            message = (
                f"{split} split: {len(missing)}/{total} labels missing graph tensors "
                f"(coverage {ratio * 100.0:.2f}%)."
            )
            logger.error(
                "[coverage] %s See %s for details.", message, report_path
            )
            if cfg.coverage_fail_on_missing:
                missing_issues.append(message)

        if ratio < min_ratio:
            threshold_message = (
                f"{split} split coverage {ratio * 100.0:.2f}% is below the required "
                f"{min_ratio * 100.0:.2f}%."
            )
            logger.error(
                "[coverage] %s Regenerate graphs or lower coverage.minimum_percent to continue.", threshold_message
            )
            threshold_issues.append(threshold_message)

    error_messages: List[str] = []
    if threshold_issues:
        error_messages.extend(threshold_issues)
    if missing_issues:
        error_messages.extend(missing_issues)

    if error_messages:
        joined = " ".join(error_messages)
        raise RuntimeError(
            f"Coverage requirements not met. {joined} "
            "Review dataset_coverage.json or regenerate missing graph tensors."
        )


def _log_feature_summary(logger: logging.Logger, metadata: GraphFeatureMetadata) -> None:
    node_dim = metadata.node_schema.get("dim")
    edge_dim = metadata.edge_schema.get("dim")
    sample_nodes = metadata.sample_node_count
    sample_edges = metadata.sample_edge_count
    if metadata.builder:
        log_builder_provenance(logger, metadata.builder)

    if metadata.metadata_path and metadata.module_registry:
        logger.info("Feature metadata source: %s", metadata.metadata_path)
        if metadata.summary_path:
            logger.info("Graph builder summary: %s", metadata.summary_path)

        interface_entry = metadata.module_registry.get("interface", {})
        interface_id = interface_entry.get("id") or "interface stage"
        cutoff = (interface_entry.get("defaults") or {}).get("cutoff")
        interface_line = f"Interface: {interface_id}"
        if cutoff is not None:
            interface_line += f" (cutoff {cutoff:g} Å)"
        if sample_nodes is not None:
            interface_line += f", yielding {sample_nodes} interface residues"
        interface_line += "."
        logger.info(interface_line)

        topology_entry = metadata.module_registry.get("topology", {})
        topology_id = topology_entry.get("id") or "topology stage"
        topo_defaults = topology_entry.get("defaults") or {}
        topo_details = []
        filtration = topo_defaults.get("filtration_cutoff")
        if filtration is not None:
            topo_details.append(f"filtration_cutoff {filtration:g} Å")
        min_persistence = topo_defaults.get("min_persistence")
        if min_persistence is not None:
            topo_details.append(f"min_persistence {min_persistence:g}")
        topology_line = f"Topology: {topology_id}"
        if topo_details:
            topology_line += " (" + ", ".join(topo_details) + ")"
        if node_dim is not None:
            topology_line += f" supporting the {node_dim}-dimensional node features"
        topology_line += "."
        logger.info(topology_line)

        node_entry = metadata.module_registry.get("node", {})
        node_id = node_entry.get("id") or "node stage"
        node_line = f"Node: {node_id}"
        if node_dim is not None:
            node_line += f"; metadata lists {node_dim} feature columns matching the .pt tensor"
        node_line += "."
        logger.info(node_line)

        edge_entry = metadata.module_registry.get("edge", {})
        edge_id = metadata.edge_schema.get("module") or edge_entry.get("id") or "edge stage"
        variant = metadata.edge_schema.get("variant")
        bands = metadata.edge_schema.get("bands") or []
        module_params = metadata.edge_schema.get("module_params") or {}
        contact_threshold = module_params.get("contact_threshold")
        histogram_bins = module_params.get("histogram_bins") or []

        edge_details = []
        if variant:
            edge_details.append(f"variant {variant}")
        if bands:
            edge_details.append("bands " + "/".join(str(b) for b in bands))
        if histogram_bins:
            if len(histogram_bins) >= 2:
                edge_details.append(f"histogram bins {histogram_bins[0]:g}–{histogram_bins[-1]:g} Å")
            else:
                edge_details.append("histogram bins available")
        if contact_threshold is not None:
            edge_details.append(f"contact_threshold {contact_threshold:g} Å")

        edge_line = f"Edge: {edge_id}"
        if edge_details:
            edge_line += " (" + ", ".join(edge_details) + ")"
        if sample_edges is not None and edge_dim is not None:
            edge_line += f"; .pt confirms {sample_edges} directed edges with {edge_dim} feature values each"
        elif edge_dim is not None:
            edge_line += f"; edge vectors hold {edge_dim} feature values"
        edge_line += "."
        logger.info(edge_line)

        logger.info(
            "\u2192 Total feature dimensionality observed and confirmed: node_dim = %s, edge_dim = %s.",
            node_dim if node_dim is not None else "unknown",
            edge_dim if edge_dim is not None else "unknown",
        )
    else:
        warning_suffix = f" ({metadata.sample_graph})" if metadata.sample_graph else ""
        logger.warning(
            "graph_metadata.json not found; summarising feature schema from sample tensors%s.",
            warning_suffix,
        )
        if sample_nodes is not None:
            logger.info(
                "Interface: %s interface residues identified (from node tensor).",
                sample_nodes,
            )
        else:
            logger.info("Interface: interface residues inferred from node tensor.")
        if node_dim is not None:
            logger.info(
                "Topology: topology-derived features present; node vectors are %s-dimensional.",
                node_dim,
            )
            logger.info(
                "Node: per-residue feature vectors have %s values each.",
                node_dim,
            )
        else:
            logger.info(
                "Topology: topology-derived features present; node vector dimensionality unavailable."
            )
            logger.info("Node: per-residue feature dimensionality could not be inferred.")
        if sample_edges is not None and edge_dim is not None:
            logger.info(
                "Edge: %s directed edges recorded; each edge vector holds %s values.",
                sample_edges,
                edge_dim,
            )
        elif edge_dim is not None:
            logger.info("Edge: edge vectors hold %s values (edge count unavailable).", edge_dim)
        else:
            logger.info("Edge: edge feature dimensionality could not be inferred.")
        logger.info(
            "\u2192 Total feature dimensionality observed in tensors: node_dim = %s, edge_dim = %s.",
            node_dim if node_dim is not None else "unknown",
            edge_dim if edge_dim is not None else "unknown",
        )


_CHECKPOINT_NEW_RE = re.compile(
    r"checkpoint\.(?:sel-(?P<sel>[0-9eE+\-.]+)_)?val-(?P<val>[0-9eE+\-.]+)_epoch(?P<epoch>\d+)"
)
_CHECKPOINT_LEGACY_RE = re.compile(r"model\.([0-9eE+\-.]+)")


def _parse_checkpoint_score(path: Path) -> Optional[float]:
    name = path.name
    match = _CHECKPOINT_NEW_RE.search(name)
    if match:
        sel_value = match.group("sel")
        val_value = match.group("val")
        token = sel_value or val_value
        if token is not None:
            try:
                return float(token)
            except ValueError:
                pass
    legacy_match = _CHECKPOINT_LEGACY_RE.search(name)
    if legacy_match:
        token = legacy_match.group(1).split("-", 1)[0].rstrip(".")
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _format_checkpoint_stem(path: Path) -> str:
    stem = path.stem
    match = _CHECKPOINT_NEW_RE.search(stem)
    if match:
        parts = []
        sel_value = match.group("sel")
        val_value = match.group("val")
        epoch_value = match.group("epoch")
        if sel_value is not None:
            try:
                sel_float = float(sel_value)
                parts.append(f"sel-{sel_float:.5f}")
            except ValueError:
                parts.append(f"sel{sel_value}")
        if val_value is not None:
            try:
                val_float = float(val_value)
                parts.append(f"val-{val_float:.5f}")
            except ValueError:
                parts.append(f"val{val_value}")
        if epoch_value is not None:
            try:
                parts.append(f"epoch{int(epoch_value):03d}")
            except ValueError:
                parts.append(f"epoch{epoch_value}")
        if parts:
            return "checkpoint." + "_".join(parts)
    legacy_match = _CHECKPOINT_LEGACY_RE.search(stem)
    if legacy_match:
        token = legacy_match.group(1)
        try:
            value = float(token)
            return f"model.{value:.5f}"
        except ValueError:
            return f"model.{token}"
    return stem


def _rank_checkpoints(
    checkpoint_dir: Path,
    limit: int = 3,
    scores: Optional[Dict[str, object]] = None,
    mode: str = "min",
) -> List[Path]:
    if scores:
        ranked_score_list: List[Tuple[float, Path]] = []
        for path_str, score in scores.items():
            try:
                metric = float(score)
            except (TypeError, ValueError):
                continue
            path = Path(path_str)
            if not path.exists():
                candidate = checkpoint_dir / path.name
                if candidate.exists():
                    path = candidate
                else:
                    continue
            ranked_score_list.append((metric, path.resolve()))
        ranked_score_list.sort(key=lambda item: item[0], reverse=(mode == "max"))
        return [path for _, path in ranked_score_list[:limit]]

    ranked: List[Tuple[float, Path]] = []
    for candidate in checkpoint_dir.glob("*.chkpt"):
        score = _parse_checkpoint_score(candidate)
        if score is None:
            continue
        ranked.append((score, candidate.resolve()))
    ranked.sort(key=lambda item: item[0], reverse=(mode == "max"))
    return [path for _, path in ranked[:limit]]


def _average_checkpoints(
    checkpoint_paths: Sequence[Path],
    output_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    if not checkpoint_paths:
        return None
    state_dicts = []
    base_checkpoint = None
    for path in checkpoint_paths:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            logger.warning("Skipping checkpoint %s during averaging: %s", path, exc)
            continue
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            logger.warning("Checkpoint %s lacks state_dict; skipping.", path)
            continue
        if base_checkpoint is None:
            base_checkpoint = ckpt
        state_dicts.append(state)
    if not state_dicts or base_checkpoint is None:
        return None

    keys = list(state_dicts[0].keys())
    averaged: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [sd[key] for sd in state_dicts if key in sd]
        if not tensors:
            continue
        stacked = torch.stack([t.detach().to(torch.float32) for t in tensors], dim=0)
        averaged[key] = stacked.mean(dim=0)

    base_checkpoint["state_dict"] = averaged
    base_checkpoint["averaged_from"] = [str(p) for p in checkpoint_paths]
    out_path = output_dir / f"checkpoint.avg_top{len(state_dicts)}.chkpt"
    try:
        torch.save(base_checkpoint, out_path)
        logger.info("Saved averaged checkpoint to %s", out_path)
        return out_path
    except Exception as exc:
        logger.warning("Failed to save averaged checkpoint: %s", exc)
        return None


def _create_checkpoint_symlinks(
    checkpoint_dir: Path, ranked_paths: Sequence[Path], logger: logging.Logger
) -> None:
    names = ["best.ckpt", "second_best.ckpt", "third_best.ckpt"]
    for name, target_path in zip(names, ranked_paths):
        symlink_path = checkpoint_dir / name
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(target_path)
            logger.info("Symlinked %s to %s", name, target_path.name)
        except OSError as exc:
            logger.warning("Unable to create %s symlink: %s", name, exc)
    for name in names[len(ranked_paths) :]:
        orphan = checkpoint_dir / name
        if orphan.exists() or orphan.is_symlink():
            try:
                orphan.unlink()
            except OSError:
                pass


def _normalise_checkpoint_file(path: Path, logger: logging.Logger) -> Path:
    if not path.exists():
        return path
    if path.suffix == ".chkpt":
        return path
    stem = _format_checkpoint_stem(path)
    candidate = path.with_name(f"{stem}.chkpt")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{stem}_dup{counter}.chkpt")
        counter += 1
    path.rename(candidate)
    logger.info("Renamed checkpoint file: %s -> %s", path.name, candidate.name)
    return candidate


def _create_named_symlink(symlink_path: Path, target_path: Path, logger: logging.Logger) -> None:
    try:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(target_path)
        logger.info("Symlinked %s to %s", symlink_path.name, target_path.name)
    except OSError as exc:
        logger.warning("Unable to create %s symlink: %s", symlink_path.name, exc)


def _safe_scalar(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        try:
            return float(value.detach().cpu().item())
        except (ValueError, RuntimeError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_metric(metrics: Dict[str, object], candidates: Sequence[str]) -> Tuple[Optional[str], Optional[float]]:
    for name in candidates:
        if name in metrics:
            value = _safe_scalar(metrics[name])
            if value is not None:
                return name, value
    return None, None


class SelectionMetricLogger(Callback):
    def __init__(
        self,
        logger: logging.Logger,
        enabled: bool,
        weight: float,
        min_delta: float,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.enabled = enabled
        self.weight = weight
        self.min_delta = min_delta

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss = _safe_scalar(metrics.get("val_loss"))
        if val_loss is None:
            return

        if not self.enabled:
            metrics["selection_metric"] = torch.tensor(val_loss, device=pl_module.device if hasattr(pl_module, "device") else None)
            return

        val_spearman = _safe_scalar(metrics.get("val_spearman_corr"))
        if val_spearman is None:
            selection_metric = val_loss
        else:
            improvement = max(0.0, val_spearman - self.min_delta)
            selection_metric = val_loss - self.weight * improvement

        metrics["selection_metric"] = selection_metric
        metrics["selection_metric"] = torch.tensor(selection_metric, device=pl_module.device if hasattr(pl_module, "device") else None)
        self.logger.info(
            "Selection metric (val_loss - %.3f*max(0,val_spearman_corr-%.3f)) = %.6f",
            self.weight,
            self.min_delta,
            selection_metric,
        )


class TuningMetricsCallback(Callback):
    def __init__(
        self,
        logger: logging.Logger,
        dataloader: Optional[DataLoader],
        eval_every: int = 1,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.dataloader = dataloader
        self.eval_every = max(1, int(eval_every))

    def _extract_names(self, batch) -> List[str]:
        expected = getattr(batch, "num_graphs", None)
        for attr in ("_graph_names", "graph_names", "name", "model_name"):
            names = getattr(batch, attr, None)
            if names is None:
                continue
            if torch.is_tensor(names):
                names = names.detach().cpu().tolist()
            elif hasattr(names, "tolist") and not isinstance(names, (list, tuple, str)):
                try:
                    names = names.tolist()
                except Exception:
                    names = [str(names)]
            if isinstance(names, (list, tuple)):
                flattened = []
                for item in names:
                    if isinstance(item, (list, tuple)):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                names_list = [str(x) for x in flattened if str(x)]
            else:
                names_list = [str(names)] if str(names) else []
            if not names_list:
                continue
            if expected is None or expected <= 1 or len(names_list) == expected:
                return names_list
            # Mismatch: likely batch-level name; fall through to per-graph extraction.
            break
        if hasattr(batch, "to_data_list"):
            try:
                data_list = batch.to_data_list()
                names_list = [str(getattr(item, "name", "")) for item in data_list]
                names_list = [name for name in names_list if name]
                if names_list:
                    return names_list
            except Exception:
                pass
        return []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        if self.dataloader is None or compute_grouped_ranking_metrics is None:
            return
        if (trainer.current_epoch + 1) % self.eval_every != 0:
            return

        was_training = pl_module.training
        pl_module.eval()

        preds: List[float] = []
        trues: List[float] = []
        names: List[str] = []
        with torch.no_grad():
            for batch_list in self.dataloader:
                raw_batch = batch_list[0] if isinstance(batch_list, (list, tuple)) else batch_list
                names.extend(self._extract_names(raw_batch))
                batch = raw_batch.to(pl_module.device)
                scores = pl_module.forward([batch])
                preds.extend(scores.detach().cpu().view(-1).tolist())
                trues.extend(batch.y.detach().cpu().view(-1).tolist())

        if not preds or not trues or not names:
            self.logger.warning(
                "Tuning metrics skipped due to missing data (preds=%d, trues=%d, names=%d).",
                len(preds),
                len(trues),
                len(names),
            )
            if was_training:
                pl_module.train()
            return

        aligned = min(len(preds), len(trues), len(names))
        if aligned < len(preds) or aligned < len(trues) or aligned < len(names):
            self.logger.warning(
                "Tuning metrics length mismatch (preds=%d, trues=%d, names=%d); truncating to %d.",
                len(preds),
                len(trues),
                len(names),
                aligned,
            )
        preds = preds[:aligned]
        trues = trues[:aligned]
        names = names[:aligned]

        summary = compute_grouped_ranking_metrics(names, preds, trues)
        pl_module.log("tuning_rank_regret", summary.mean_regret, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("tuning_rank_spearman", summary.mean_spearman, on_epoch=True, prog_bar=False, logger=True)
        self.logger.info(
            "Tuning ranking metrics: regret=%.6f spearman=%.6f (groups=%d, skipped=%d)",
            summary.mean_regret,
            summary.mean_spearman,
            summary.group_count,
            summary.skipped_groups,
        )

        if was_training:
            pl_module.train()


class WeightAveragingCallback(Callback):
    def __init__(
        self,
        logger: logging.Logger,
        method: str,
        output_path: Path,
        ema_decay: float = 0.999,
        swa_start_epoch: int = 0,
        save_every_epochs: int = 0,
        resume: bool = True,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.method = method
        self.output_path = output_path
        self.ema_decay = ema_decay
        self.swa_start_epoch = max(0, int(swa_start_epoch))
        self.save_every_epochs = max(0, int(save_every_epochs))
        self.resume = resume
        self._avg_state: Optional[Dict[str, torch.Tensor]] = None
        self._avg_count = 0
        self.averaged_path: Optional[Path] = None

    def _snapshot_state(self, state_dict: Dict[str, object]) -> Dict[str, torch.Tensor]:
        snapshot: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                snapshot[key] = value.detach().cpu().clone()
        return snapshot

    def _update_average(self, state_dict: Dict[str, object]) -> None:
        current = self._snapshot_state(state_dict)
        if not current:
            return
        if self._avg_state is None:
            self._avg_state = current
            self._avg_count = 1
            return
        self._avg_count += 1
        if self.method == "ema":
            decay = self.ema_decay
            for key, tensor in current.items():
                if key in self._avg_state:
                    self._avg_state[key].mul_(decay).add_(tensor, alpha=1.0 - decay)
        else:  # swa
            count = float(self._avg_count)
            for key, tensor in current.items():
                if key in self._avg_state:
                    self._avg_state[key].add_(tensor - self._avg_state[key], alpha=1.0 / count)

    def _load_average_checkpoint(self) -> None:
        if not self.output_path.exists():
            return
        try:
            ckpt = torch.load(self.output_path, map_location="cpu")
        except Exception as exc:
            self.logger.warning("Failed to load averaged checkpoint %s: %s", self.output_path, exc)
            return
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            self.logger.warning("Averaged checkpoint %s missing state_dict; skipping resume.", self.output_path)
            return
        self._avg_state = self._snapshot_state(state)
        self._avg_count = 1
        meta = ckpt.get("averaging_state") if isinstance(ckpt, dict) else None
        if isinstance(meta, dict):
            count = meta.get("avg_count")
            method = meta.get("method")
            if isinstance(method, str) and method != self.method:
                self.logger.warning(
                    "Averaged checkpoint method %s does not match current method %s; skipping resume.",
                    method,
                    self.method,
                )
                self._avg_state = None
                self._avg_count = 0
                return
            if isinstance(count, (int, float)) and count > 0:
                self._avg_count = int(count)
        self.logger.info("Resumed %s averaging state from %s (avg_count=%d).", self.method, self.output_path, self._avg_count)

    def _save_average_checkpoint(self, trainer, pl_module) -> None:
        if self._avg_state is None:
            self.logger.warning("Weight averaging requested but no averaged state was collected.")
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        original_state = self._snapshot_state(pl_module.state_dict())
        device = next(pl_module.parameters()).device
        averaged_state = {key: value.to(device) for key, value in self._avg_state.items()}

        try:
            pl_module.load_state_dict(averaged_state, strict=False)
            trainer.save_checkpoint(str(self.output_path))
            try:
                ckpt = torch.load(self.output_path, map_location="cpu")
                if isinstance(ckpt, dict):
                    ckpt["averaging_state"] = {
                        "method": self.method,
                        "avg_count": self._avg_count,
                        "ema_decay": self.ema_decay,
                    }
                    torch.save(ckpt, self.output_path)
            except Exception as exc:
                self.logger.warning("Failed to embed averaging metadata into %s: %s", self.output_path, exc)
            self.averaged_path = self.output_path
            self.logger.info("Saved %s averaged checkpoint to %s", self.method, self.output_path)
        finally:
            if original_state:
                restored_state = {key: value.to(device) for key, value in original_state.items()}
                pl_module.load_state_dict(restored_state, strict=False)

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        if self.resume:
            self._load_average_checkpoint()

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        if self.method == "swa" and trainer.current_epoch < self.swa_start_epoch:
            return
        self._update_average(pl_module.state_dict())
        if self.save_every_epochs and (trainer.current_epoch + 1) % self.save_every_epochs == 0:
            self._save_average_checkpoint(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        self._save_average_checkpoint(trainer, pl_module)


class EpochSummaryLogger(Callback):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        train_name, train_value = _select_metric(metrics, ("train_loss_epoch", "train_loss", "train_mse"))
        val_name, val_value = _select_metric(metrics, ("val_loss", "val_mse", "val_mean_dockq_loss"))

        best_val = None
        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_cb is not None:
            best_val = _safe_scalar(getattr(checkpoint_cb, "best_model_score", None))

        parts = [f"Epoch {epoch:03d}"]
        if train_name and train_value is not None:
            parts.append(f"{train_name}={train_value:.6f}")
        if val_name and val_value is not None:
            parts.append(f"{val_name}={val_value:.6f}")
        if best_val is not None:
            parts.append(f"best_val_loss={best_val:.6f}")

        if len(parts) > 1:
            self.logger.info(" | ".join(parts))


class EpochTimingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._epoch_start: Optional[float] = None
        self.epoch_durations: List[float] = []

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking or self._epoch_start is None:
            return
        duration = max(0.0, time.perf_counter() - self._epoch_start)
        self.epoch_durations.append(duration)
        self._epoch_start = None


class TrainingProgressLogger(Callback):
    def __init__(self, logger: logging.Logger, checkpoints: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.logger = logger
        self.default_checkpoints = checkpoints or (0.25, 0.5, 0.75)
        self._remaining: List[float] = []
        self._total_batches: int = 0

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        total = trainer.num_training_batches or 0
        self._total_batches = max(int(total), 1)
        self._remaining = list(self.default_checkpoints)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:  # type: ignore[override]
        if trainer.sanity_checking or not self._remaining:
            return
        progress = (batch_idx + 1) / self._total_batches
        triggered = [cp for cp in self._remaining if progress >= cp]
        if not triggered:
            return
        for cp in triggered:
            self._remaining.remove(cp)
        epoch = trainer.current_epoch
        percent = int(progress * 100)
        self.logger.info(
            "Epoch %03d | %2d%% complete (batch %d/%d)",
            epoch,
            percent,
            batch_idx + 1,
            self._total_batches,
        )

def build_dataloaders(cfg: TrainingConfig, logger: logging.Logger):
    train_labels = load_label_map(cfg.train_label_file)
    val_labels = load_label_map(cfg.val_label_file)
    tuning_labels = load_label_map(cfg.tuning_label_file) if cfg.tuning_label_file else None

    graph_lookup = _build_graph_lookup(cfg.graph_dir)

    train_samples, train_missing = _gather_samples(cfg.graph_dir, train_labels, graph_lookup)
    val_samples, val_missing = _gather_samples(cfg.graph_dir, val_labels, graph_lookup)
    tuning_samples: List[Tuple[str, Path, torch.Tensor]] = []
    tuning_missing: List[str] = []
    if tuning_labels:
        tuning_samples, tuning_missing = _gather_samples(cfg.graph_dir, tuning_labels, graph_lookup)

    if not train_samples:
        raise RuntimeError("No training graphs found; aborting.")
    if not val_samples:
        raise RuntimeError("No validation graphs found; aborting.")
    if tuning_labels and not tuning_samples:
        logger.warning("No tuning graphs found; tuning metrics will be skipped.")

    _summarise_coverage(
        logger,
        "train",
        total_labels=len(train_labels),
        usable_graphs=len(train_samples),
        missing=train_missing,
    )
    _summarise_coverage(
        logger,
        "val",
        total_labels=len(val_labels),
        usable_graphs=len(val_samples),
        missing=val_missing,
    )
    if tuning_labels:
        _summarise_coverage(
            logger,
            "tuning",
            total_labels=len(tuning_labels),
            usable_graphs=len(tuning_samples),
            missing=tuning_missing,
        )

    ordered_models: List[str] = []
    seen_models = set()
    for _, path, _ in [*train_samples, *val_samples]:
        relative_key = path.relative_to(cfg.graph_dir).as_posix()
        if relative_key.endswith(".pt"):
            relative_key = relative_key[:-3]
        if relative_key not in seen_models:
            ordered_models.append(relative_key)
            seen_models.add(relative_key)

    feature_metadata = load_graph_feature_metadata(
        cfg.graph_dir,
        sample_models=ordered_models,
        metadata_path=cfg.metadata_path,
        summary_path=cfg.summary_path,
    )
    _merge_defaults_into_edge_schema(feature_metadata)

    def _load_feature_config_payload() -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        fc_snapshot = None
        fc_payload = None
        builder = feature_metadata.builder if isinstance(feature_metadata.builder, dict) else {}
        if builder and isinstance(builder.get("feature_config"), dict):
            fc_snapshot = builder["feature_config"]
        if feature_metadata.feature_config:
            fc_snapshot = feature_metadata.feature_config
        if isinstance(fc_snapshot, dict):
            text = fc_snapshot.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    fc_payload = yaml.safe_load(text) or {}
                except Exception:
                    fc_payload = None
            if fc_payload is None:
                path_val = fc_snapshot.get("path")
                if path_val:
                    try:
                        fc_payload = yaml.safe_load(Path(str(path_val)).read_text()) or {}
                    except Exception:
                        fc_payload = None
        return fc_snapshot, fc_payload

    fc_snapshot, fc_payload = _load_feature_config_payload()

    topology_dim: Optional[int] = None
    if feature_metadata.metadata_path:
        try:
            meta_raw = json.loads(Path(feature_metadata.metadata_path).read_text())
            topo_val = meta_raw.get("topology_feature_dim")
            if topo_val is not None:
                topology_dim = int(topo_val)
        except Exception:
            topology_dim = None

    registry: Dict[str, object] = feature_metadata.module_registry if isinstance(feature_metadata.module_registry, dict) else {}

    def _update_stage(stage: str, schema: Optional[Dict[str, object]]) -> None:
        nonlocal registry
        entry = registry.get(stage, {}) if isinstance(registry, dict) else {}
        if not isinstance(entry, dict):
            entry = {}
        fc_stage = fc_payload.get(stage) if isinstance(fc_payload, dict) else {}
        fc_params = fc_stage.get("params") if isinstance(fc_stage, dict) else {}
        module_id = None
        if isinstance(fc_stage, dict):
            module_id = fc_stage.get("module")
        if not module_id:
            module_id = entry.get("module") or entry.get("id")
        params: Dict[str, object] = {}
        defaults = entry.get("defaults")
        if isinstance(defaults, dict):
            params.update(defaults)
        if isinstance(entry.get("params"), dict):
            params.update(entry["params"])
        if isinstance(fc_params, dict):
            params.update(fc_params)
        entry["module"] = module_id
        if params:
            entry["params"] = params
        registry[stage] = entry
        if schema is not None:
            if module_id:
                schema["module"] = module_id
            if params:
                schema["module_params"] = params
            schema.setdefault("source", feature_metadata.metadata_path)

    topology_schema: Dict[str, object] = {}
    _update_stage("edge", feature_metadata.edge_schema)
    _update_stage("node", feature_metadata.node_schema)
    _update_stage("topology", topology_schema)
    if topology_dim is not None:
        topology_schema["dim"] = topology_dim
    feature_metadata.topology_schema = topology_schema
    feature_metadata.module_registry = registry
    if fc_snapshot:
        feature_metadata.feature_config = fc_snapshot

    def _fallback_builder_from_graph_metadata() -> Optional[Dict[str, object]]:
        """Best-effort builder recovery when feature_metadata.builder is missing."""
        from builder_metadata import load_builder_info_from_metadata  # type: ignore

        builder_info = None
        # First try the recorded metadata_path
        builder_info = load_builder_info_from_metadata(feature_metadata.metadata_path)
        if builder_info:
            logger.warning(
                "Recovered builder info from graph metadata path because checkpoint metadata lacked builder block."
            )
            return builder_info
        # Fallback: if metadata_path is unset, attempt graph_dir/graph_metadata.json
        candidate = cfg.graph_dir / "graph_metadata.json"
        if candidate.exists():
            builder_info = load_builder_info_from_metadata(str(candidate))
            if builder_info:
                logger.warning(
                    "Recovered builder info from %s because checkpoint metadata lacked builder block.", candidate
                )
        return builder_info

    load_profiler = GraphLoadProfiler(cfg.graph_load_profiling, cfg.graph_load_top_k, cfg.num_workers, logger)

    cache_instance = None
    if cfg.enable_graph_cache and GraphTensorCache is not None:
        cache_instance = GraphTensorCache(max_items=cfg.graph_cache_size)
        logger.info(
            "Graph cache enabled (max_items=%d). Disable via selection cache.enable_graph_cache=false in config.",
            cfg.graph_cache_size,
        )

    def _sample_subset(
        samples: List[Tuple[str, Path, torch.Tensor]],
        *,
        fraction: Optional[float],
        max_samples: Optional[int],
        seed: int,
    ) -> List[Tuple[str, Path, torch.Tensor]]:
        if not samples:
            return samples
        target = len(samples)
        if fraction is not None:
            target = max(1, int(round(len(samples) * fraction)))
        if max_samples is not None:
            target = min(target, int(max_samples))
        if target >= len(samples):
            return samples
        rng = random.Random(seed)
        subset = rng.sample(samples, target)
        subset.sort(key=lambda item: item[0])
        return subset

    def _group_stats(samples: List[Tuple[str, Path, torch.Tensor]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for model, _, _ in samples:
            key = model.split("_", 1)[0] if model else ""
            counts[key] = counts.get(key, 0) + 1
        groups_ge2 = sum(1 for size in counts.values() if size >= 2)
        singletons = sum(1 for size in counts.values() if size == 1)
        max_group = max(counts.values()) if counts else 0
        return {
            "groups": len(counts),
            "groups_ge2": groups_ge2,
            "singletons": singletons,
            "max_group_size": max_group,
        }

    tuning_samples = _sample_subset(
        tuning_samples,
        fraction=cfg.tuning_fraction,
        max_samples=cfg.tuning_max_samples,
        seed=cfg.seed,
    )
    tuning_group_stats = _group_stats(tuning_samples) if tuning_samples else {}
    if tuning_group_stats:
        logger.info(
            "Tuning group stats: groups=%d groups>=2=%d singletons=%d max_group=%d",
            tuning_group_stats["groups"],
            tuning_group_stats["groups_ge2"],
            tuning_group_stats["singletons"],
            tuning_group_stats["max_group_size"],
        )

    train_dataset = GraphRegressionDataset(
        train_samples,
        profiler=load_profiler if load_profiler.enabled else None,
        cache=cache_instance,
        canonicalize_on_load=cfg.canonicalize_on_load or bool(os.environ.get("CANONICALIZE_GRAPHS_ON_LOAD")),
    )
    val_dataset = GraphRegressionDataset(
        val_samples,
        profiler=load_profiler if load_profiler.enabled else None,
        cache=cache_instance,
        canonicalize_on_load=cfg.canonicalize_on_load or bool(os.environ.get("CANONICALIZE_GRAPHS_ON_LOAD")),
    )
    tuning_dataset = (
        GraphRegressionDataset(
            tuning_samples,
            profiler=load_profiler if load_profiler.enabled else None,
            cache=cache_instance,
            canonicalize_on_load=cfg.canonicalize_on_load or bool(os.environ.get("CANONICALIZE_GRAPHS_ON_LOAD")),
        )
        if tuning_samples
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        persistent_workers=False,
    )
    tuning_loader = None
    if tuning_dataset is not None:
        tuning_loader = DataLoader(
            tuning_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_graphs,
            persistent_workers=False,
        )
    coverage = {
        "train": {
            "total": len(train_labels),
            "usable": len(train_samples),
            "missing": list(train_missing),
            "coverage": (len(train_samples) / len(train_labels)) if train_labels else 0.0,
        },
        "val": {
            "total": len(val_labels),
            "usable": len(val_samples),
            "missing": list(val_missing),
            "coverage": (len(val_samples) / len(val_labels)) if val_labels else 0.0,
        },
    }
    if tuning_labels:
        coverage["tuning"] = {
            "total": len(tuning_labels),
            "usable": len(tuning_samples),
            "missing": list(tuning_missing),
            "coverage": (len(tuning_samples) / len(tuning_labels)) if tuning_labels else 0.0,
        }
        if tuning_group_stats:
            coverage["tuning"].update(tuning_group_stats)
    return train_loader, val_loader, tuning_loader, coverage, feature_metadata, load_profiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only TopoQA training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration (default: config.yaml in this directory)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Enable Lightning fast_dev_run (1 train/val batch smoke test).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Optional fraction/absolute number of train batches per epoch.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Optional fraction/absolute number of validation batches per epoch.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override output directory for this training run (created by wrapper).",
    )
    parser.add_argument(
        "--log-lr",
        action="store_true",
        help="Log learning rate each epoch via callbacks.",
    )
    parser.add_argument(
        "--trial-label",
        type=str,
        default=None,
        help="Optional identifier provided by run_training.sh for logging.",
    )
    parser.add_argument(
        "--git-commit",
        type=str,
        default=None,
        help="Git commit hash recorded by the launcher.",
    )
    parser.add_argument(
        "--git-dirty",
        action="store_true",
        help="Flag indicating the repository had uncommitted changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = _resolve_path(args.config, Path(__file__).resolve().parent)
    cfg = load_config(config_path)

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None

    if args.fast_dev_run:
        cfg.fast_dev_run = True
    if args.limit_train_batches is not None:
        cfg.limit_train_batches = args.limit_train_batches
    if args.limit_val_batches is not None:
        cfg.limit_val_batches = args.limit_val_batches
    if run_dir is not None:
        cfg.save_dir = run_dir
    cfg.log_lr = bool(args.log_lr)

    logger = _setup_logging(cfg.save_dir)
    record_selection_metadata(
        cfg.save_dir,
        primary_metric=cfg.selection_primary_metric,
        use_val_spearman=cfg.use_val_spearman_as_secondary,
        spearman_weight=cfg.spearman_secondary_weight,
        spearman_min_delta=cfg.spearman_secondary_min_delta,
        primary_mode=_metric_mode(cfg.selection_primary_metric or "val_loss"),
        tuning_enabled=bool(cfg.tuning_label_file),
        tuning_eval_every=cfg.tuning_eval_every,
    )

    if args.trial_label:
        logger.info("trial = %s", args.trial_label)
    if args.git_commit or args.git_dirty:
        commit = args.git_commit or "unknown"
        status = "dirty" if args.git_dirty else "clean"
        logger.info("git_commit = %s (%s)", commit, status)

    logger.info("Training package root: %s", SCRIPT_DIR)
    logger.info("Configuration loaded from %s", config_path)

    for field in dataclasses.fields(cfg):
        logger.info("  %s = %s", field.name, getattr(cfg, field.name))

    required_paths = [
        cfg.graph_dir,
        cfg.train_label_file,
        cfg.val_label_file,
    ]
    if cfg.tuning_label_file is not None:
        required_paths.append(cfg.tuning_label_file)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    seed_everything(cfg.seed, workers=True)

    warnings.filterwarnings(
        "ignore",
        message="GPU available but not used.*",
        module="pytorch_lightning.trainer.setup",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'train_dataloader' does not have many workers.*",
        module="pytorch_lightning.trainer.connectors.data_connector",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'val_dataloader' does not have many workers.*",
        module="pytorch_lightning.trainer.connectors.data_connector",
    )

    train_loader, val_loader, tuning_loader, coverage, feature_metadata, load_profiler = build_dataloaders(cfg, logger)
    load_profile_summary: Optional[Dict[str, object]] = None

    resolved_edge_schema = dict(feature_metadata.edge_schema)
    configured_edge_schema = dict(cfg.edge_schema or {})

    metadata_edge_dim = resolved_edge_schema.get("dim")
    config_edge_dim = configured_edge_schema.get("dim")
    if config_edge_dim is not None and metadata_edge_dim is not None:
        if int(config_edge_dim) != int(metadata_edge_dim):
            logger.warning(
                "Config edge_schema.dim=%s differs from metadata dim=%s; using metadata value.",
                config_edge_dim,
                metadata_edge_dim,
            )
    if metadata_edge_dim is not None:
        resolved_edge_schema["dim"] = int(metadata_edge_dim)
    elif config_edge_dim is not None:
        resolved_edge_schema["dim"] = int(config_edge_dim)

    for key, value in configured_edge_schema.items():
        if key == "dim":
            continue
        if value is not None:
            resolved_edge_schema[key] = value

    if "dim" not in resolved_edge_schema or resolved_edge_schema["dim"] is None:
        raise RuntimeError("Edge feature dimension could not be resolved from metadata or config.")

    coverage_path = _write_coverage_report(cfg.save_dir, coverage)
    logger.info("Dataset coverage report written to %s", coverage_path)
    _evaluate_coverage(cfg, coverage, logger, coverage_path)

    if load_profiler.enabled:
        logger.info(
            "Graph load profiling enabled (top_k=%d); summary will be captured after training.",
            cfg.graph_load_top_k,
        )

    cfg.edge_schema = resolved_edge_schema
    feature_metadata.edge_schema = dict(resolved_edge_schema)
    builder_info = feature_metadata.builder or load_builder_info_from_metadata(feature_metadata.metadata_path)
    if builder_info is None:
        builder_info = _fallback_builder_from_graph_metadata()
    if builder_info:
        feature_metadata.builder = builder_info

    feature_metadata_dict = feature_metadata.to_dict()
    # Prefer enriched topology schema with resolved params/dim
    feature_metadata_dict["topology_schema"] = feature_metadata.topology_schema or cfg.topology_schema
    if feature_metadata.topology_schema_spec:
        feature_metadata_dict["topology_schema_spec"] = feature_metadata.topology_schema_spec
    if feature_metadata.feature_config:
        feature_metadata_dict["feature_config"] = feature_metadata.feature_config
        sha = feature_metadata.feature_config.get("sha256")  # type: ignore[arg-type]
        if sha:
            feature_metadata_dict["feature_config_sha256"] = sha

    metadata_path, schema_summary_path = write_feature_metadata_artifacts(
        cfg.save_dir, feature_metadata, feature_metadata_dict, logger
    )

    _log_feature_summary(logger, feature_metadata)

    if cfg.use_val_spearman_as_secondary:
        logger.info(
            "Secondary selection metric enabled: val_spearman_corr (weight=%.3f, min_delta=%.3f)",
            cfg.spearman_secondary_weight,
            cfg.spearman_secondary_min_delta,
        )
    else:
        logger.info("Secondary selection metric disabled; using val_loss only.")

    if feature_metadata.builder is None:
        logger.warning(
            "Graph metadata did not contain a builder block; downstream runs will lack builder options (e.g., rounding)."
        )

    node_schema_for_log: Dict[str, object] = dict(feature_metadata.node_schema)
    node_columns = node_schema_for_log.get("columns")
    if isinstance(node_columns, list):
        node_schema_for_log["column_count"] = len(node_columns)
        node_schema_for_log["columns_preview"] = node_columns[:5]
        node_schema_for_log.pop("columns")

    module_registry_log: Dict[str, Dict[str, object]] = {}
    for kind, info in feature_metadata.module_registry.items():
        if isinstance(info, dict):
            module_registry_log[kind] = {
                "id": info.get("id"),
                "alias": info.get("alias"),
                "jobs": info.get("jobs"),
                "summary": info.get("summary"),
            }

    metadata_for_log = {
        "edge_schema": feature_metadata.edge_schema,
        "node_schema": node_schema_for_log,
        "metadata_path": feature_metadata.metadata_path,
        "summary_path": feature_metadata.summary_path,
        "module_registry": module_registry_log,
        "notes": feature_metadata.notes,
    }
    logger.info("Resolved feature metadata: %s", json.dumps(metadata_for_log, indent=2))

    edge_dim_value = int(resolved_edge_schema["dim"])
    node_dim_value = feature_metadata.node_schema.get("dim")
    node_dim = int(node_dim_value) if node_dim_value is not None else None
    if node_dim is None:
        logger.warning("Node feature dimension unavailable; falling back to architecture default.")

    if cfg.num_workers == 0:
        if platform.system().lower() == "darwin":
            logger.info(
                "DataLoader num_workers=0 (recommended for macOS to avoid spawn issues; configurable via config.yaml -> num_workers)."
            )
        else:
            logger.info(
                "DataLoader num_workers=0; consider increasing num_workers in config.yaml "
                "to >1 for better throughput on non-macOS systems."
            )
    elif platform.system().lower() == "darwin":
        logger.info(
            "Detected macOS with num_workers=%d. If you observe instability, revert to 0 in config.yaml.",
            cfg.num_workers,
        )

    model = CpuTopoQAModule(
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

    checkpoint_dir = cfg.save_dir / "model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt_path = validate_resume_checkpoint(
        cfg.save_dir,
        checkpoint_dir,
        args.resume_from,
        cfg.num_epochs,
        logger,
    )

    primary_metric = (cfg.selection_primary_metric or "val_loss").lower()
    if primary_metric == "selection_metric" and not cfg.use_val_spearman_as_secondary:
        logger.warning(
            "selection.primary_metric=selection_metric requires use_val_spearman_as_secondary=true; enabling it for this run."
        )
        cfg.use_val_spearman_as_secondary = True
    tuning_groups_ge2 = 0
    if isinstance(coverage.get("tuning"), dict):
        try:
            tuning_groups_ge2 = int(coverage["tuning"].get("groups_ge2", 0))
        except (TypeError, ValueError):
            tuning_groups_ge2 = 0
    tuning_min_groups_ge2 = max(0, int(cfg.tuning_min_groups_ge2))
    if primary_metric.startswith("tuning_") and tuning_groups_ge2 < tuning_min_groups_ge2:
        logger.warning(
            "selection.primary_metric=%s requested but tuning groups>=2 is %d (<%d); falling back to val_loss.",
            primary_metric,
            tuning_groups_ge2,
            tuning_min_groups_ge2,
        )
        primary_metric = "val_loss"
    if primary_metric.startswith("tuning_") and (tuning_loader is None or compute_grouped_ranking_metrics is None):
        logger.warning(
            "selection.primary_metric=%s requested but tuning metrics are unavailable; falling back to val_loss.",
            primary_metric,
        )
        primary_metric = "val_loss"
    if primary_metric not in PRIMARY_METRIC_CHOICES:
        primary_metric = "val_loss"

    monitor_metric_name = primary_metric
    monitor_mode = _metric_mode(monitor_metric_name)
    alternate_metric_name: Optional[str] = "val_loss" if primary_metric != "val_loss" else "selection_metric"
    alternate_mode = _metric_mode(alternate_metric_name) if alternate_metric_name else "min"

    if monitor_metric_name == "selection_metric":
        checkpoint_filename = "checkpoint.sel-{selection_metric:.5f}_val-{val_loss:.5f}_epoch{epoch:03d}"
    elif monitor_metric_name == "val_loss":
        checkpoint_filename = "checkpoint.val-{val_loss:.5f}_epoch{epoch:03d}"
    else:
        checkpoint_filename = f"checkpoint.{monitor_metric_name}-{{{monitor_metric_name}:.5f}}_val-{{val_loss:.5f}}_epoch{{epoch:03d}}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=checkpoint_filename,
        auto_insert_metric_name=False,
        monitor=monitor_metric_name,
        mode=monitor_mode,
        save_top_k=3,
        save_last=False,
    )
    alternate_checkpoint_cb: Optional[ModelCheckpoint] = None
    alternate_symlink_name: Optional[str] = None
    if alternate_metric_name is not None:
        alt_dir = checkpoint_dir / f"{alternate_metric_name}_checkpoints"
        alt_dir.mkdir(parents=True, exist_ok=True)
        if alternate_metric_name == "selection_metric":
            alt_filename = "alt_selection.sel-{selection_metric:.5f}_val-{val_loss:.5f}_epoch{epoch:03d}"
        elif alternate_metric_name == "val_loss":
            alt_filename = "alt_val.val-{val_loss:.5f}_epoch{epoch:03d}"
        else:
            alt_filename = f"alt_{alternate_metric_name}.metric-{{{alternate_metric_name}:.5f}}_epoch{{epoch:03d}}"
        alternate_checkpoint_cb = ModelCheckpoint(
            dirpath=str(alt_dir),
            filename=alt_filename,
            auto_insert_metric_name=False,
            monitor=alternate_metric_name,
            mode=alternate_mode,
            save_top_k=1,
            save_last=False,
        )
        alternate_symlink_name = f"{alternate_metric_name}_best.ckpt"
    early_stop_cb = EarlyStopping(
        monitor=monitor_metric_name,
        mode=monitor_mode,
        patience=cfg.early_stopping_patience,
        verbose=True,
    )
    lr_callbacks: List[Callback] = []
    if args.log_lr:
        lr_callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    selection_callback = SelectionMetricLogger(
        logger,
        enabled=cfg.use_val_spearman_as_secondary,
        weight=cfg.spearman_secondary_weight,
        min_delta=cfg.spearman_secondary_min_delta,
    )
    tuning_callback = None
    if tuning_loader is not None and compute_grouped_ranking_metrics is not None:
        tuning_callback = TuningMetricsCallback(
            logger,
            tuning_loader,
            eval_every=cfg.tuning_eval_every,
        )
    weight_avg_callback = None
    if cfg.variance_reduction_enabled and cfg.variance_reduction_method in {"ema", "swa"}:
        swa_start_epoch = _resolve_swa_start_epoch(cfg.variance_reduction_swa_start, cfg.num_epochs)
        output_path = (cfg.save_dir / "model_checkpoints" / f"averaged_{cfg.variance_reduction_method}.ckpt")
        weight_avg_callback = WeightAveragingCallback(
            logger,
            cfg.variance_reduction_method,
            output_path,
            ema_decay=cfg.variance_reduction_ema_decay,
            swa_start_epoch=swa_start_epoch,
            save_every_epochs=cfg.variance_reduction_save_every_epochs,
            resume=cfg.variance_reduction_resume,
        )
        logger.info(
            "Weight averaging enabled (%s) -> %s",
            cfg.variance_reduction_method,
            output_path,
        )

    progress_bar_cb = None
    enable_progress_bar = cfg.progress_bar_refresh_rate > 0
    if enable_progress_bar:
        progress_bar_cb = TQDMProgressBar(refresh_rate=cfg.progress_bar_refresh_rate)

    callback_list: List[Callback] = [selection_callback, checkpoint_cb, early_stop_cb, *lr_callbacks]
    if tuning_callback is not None:
        callback_list.append(tuning_callback)
    if weight_avg_callback is not None:
        callback_list.append(weight_avg_callback)
    if alternate_checkpoint_cb is not None:
        callback_list.append(alternate_checkpoint_cb)
    if progress_bar_cb is not None:
        callback_list.append(progress_bar_cb)
    callback_list.append(TrainingProgressLogger(logger))
    callback_list.append(EpochSummaryLogger(logger))
    timing_callback = EpochTimingCallback()
    callback_list.append(timing_callback)

    csv_logger = CSVLogger(str(cfg.save_dir), name="cpu_training", flush_logs_every_n_steps=1)

    loggers: List = [csv_logger]
    mlflow_logger = None
    if cfg.mlflow.enabled:
        if MLFlowLogger is None:
            formatted = (
                "MLflow logging requested but the dependency could not be imported: %s"
                % (_MLFLOW_IMPORT_ERROR or "unknown error")
            )
            logger.error(formatted)
            raise RuntimeError(formatted)
        else:
            try:
                run_name = cfg.mlflow.run_name or (args.trial_label or (run_dir.name if run_dir else cfg.save_dir.name))
                mlflow_logger = MLFlowLogger(
                    experiment_name=cfg.mlflow.experiment,
                    tracking_uri=cfg.mlflow.tracking_uri,
                    run_name=run_name,
                )
                if cfg.mlflow.tags:
                    mlflow_logger.experiment.set_tags(mlflow_logger.run_id, cfg.mlflow.tags)
                hparams = {
                    "paths.graph": str(cfg.graph_dir),
                    "paths.train_labels": str(cfg.train_label_file),
                    "paths.val_labels": str(cfg.val_label_file),
                    "model.pooling_type": cfg.pooling_type,
                    "model.attention_head": cfg.attention_head,
                    "trainer.accelerator": cfg.accelerator,
                    "trainer.devices": cfg.devices,
                    "trainer.precision": cfg.precision,
                    "trainer.num_epochs": cfg.num_epochs,
                    "trainer.accumulate_grad_batches": cfg.accumulate_grad_batches,
                    "dataloader.batch_size": cfg.batch_size,
                    "dataloader.num_workers": cfg.num_workers,
                    "optimizer.learning_rate": cfg.learning_rate,
                    "scheduler.type": cfg.lr_scheduler_type,
                    "scheduler.factor": cfg.lr_scheduler_factor,
                    "scheduler.patience": cfg.lr_scheduler_patience,
                    "selection.use_val_spearman": cfg.use_val_spearman_as_secondary,
                    "selection.spearman_weight": cfg.spearman_secondary_weight,
                    "selection.spearman_min_delta": cfg.spearman_secondary_min_delta,
                }
                mlflow_logger.log_hyperparams(hparams)
                logger.info(
                    "MLflow logging enabled (experiment=%s, run_id=%s)",
                    cfg.mlflow.experiment,
                    mlflow_logger.run_id,
                )
                loggers.append(mlflow_logger)
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Unable to initialise MLflow logging: %s", exc)
                mlflow_logger = None

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callback_list,
        logger=loggers if len(loggers) > 1 else loggers[0],
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=max(1, cfg.log_every_n_steps),
        deterministic=True,
        default_root_dir=str(cfg.save_dir),
        fast_dev_run=cfg.fast_dev_run,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    logger.info("Starting training (fast_dev_run=%s)", cfg.fast_dev_run)
    fit_start_time = time.perf_counter()
    if cfg.accelerator == "cpu" and (
        torch.cuda.is_available()
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    ):
        logger.info(
            "Detected GPU/MPS hardware. Current run uses accelerator='cpu' "
            "(configurable via config.yaml -> accelerator)."
        )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt_path,
    )
    fit_duration = max(0.0, time.perf_counter() - fit_start_time)

    renamed_checkpoints: List[Path] = []
    ckpt_dir = Path(checkpoint_cb.dirpath) if checkpoint_cb.dirpath else checkpoint_dir
    if ckpt_dir.exists():
        updated_best_k_models: Dict[str, float] = {}
        renamed_lookup: Dict[str, Path] = {}
        for ckpt_file in sorted(ckpt_dir.glob("*.ckpt")):
            stem = _format_checkpoint_stem(ckpt_file)
            candidate = ckpt_dir / f"{stem}.chkpt"
            counter = 1
            while candidate.exists():
                candidate = ckpt_dir / f"{stem}_dup{counter}.chkpt"
                counter += 1
            ckpt_file.rename(candidate)
            renamed_checkpoints.append(candidate)
            renamed_lookup[str(ckpt_file)] = candidate
            if checkpoint_cb.best_model_path and Path(checkpoint_cb.best_model_path) == ckpt_file:
                checkpoint_cb.best_model_path = str(candidate)
        if checkpoint_cb.best_k_models:
            for old_path, score in checkpoint_cb.best_k_models.items():
                old_path_obj = Path(old_path)
                new_path = renamed_lookup.get(str(old_path_obj))
                if new_path is None:
                    stem = _format_checkpoint_stem(old_path_obj)
                    tentative = ckpt_dir / f"{stem}.chkpt"
                    new_path = tentative if tentative.exists() else old_path_obj.with_suffix(".chkpt")
                updated_best_k_models[str(new_path)] = float(score)
            checkpoint_cb.best_k_models = updated_best_k_models
        if renamed_checkpoints:
            logger.info(
                "Renamed checkpoint files: %s",
                ", ".join(p.name for p in renamed_checkpoints),
            )

    if checkpoint_cb.best_model_path:
        logger.info("Training completed. Best checkpoint: %s", checkpoint_cb.best_model_path)
        logger.info("Best validation loss: %s", checkpoint_cb.best_model_score)
    else:
        logger.info("Training completed. No checkpoints were saved (likely fast_dev_run).")

    best_ckpt_path: Optional[str] = checkpoint_cb.best_model_path or None
    best_path_obj: Optional[Path] = resolve_checkpoint_path(best_ckpt_path)

    ranked_checkpoints: List[Path] = []
    val_metrics: List[Dict[str, object]] = []
    ema_metrics: Optional[Dict[str, object]] = None
    ema_checkpoint_path: Optional[Path] = None

    epoch_durations = timing_callback.epoch_durations[:]
    runtime_summary: Dict[str, Any] = {
        "training_seconds": int(round(fit_duration)),
        "epochs_completed": len(epoch_durations),
        "epochs_total": cfg.num_epochs,
    }
    if epoch_durations:
        runtime_summary["epoch_seconds"] = [int(round(max(0.0, d))) for d in epoch_durations]
        runtime_summary["epoch_seconds_stats"] = {
            "mean": float(statistics.mean(epoch_durations)),
            "median": float(statistics.median(epoch_durations)),
            "min": float(min(epoch_durations)),
            "max": float(max(epoch_durations)),
        }
    try:
        update_run_metadata(cfg.save_dir, lambda metadata: metadata.setdefault("runtime", {}).update(runtime_summary))
    except Exception as runtime_exc:  # pragma: no cover
        logger.warning("Unable to record runtime metadata: %s", runtime_exc)

    if cfg.fast_dev_run:
        logger.info("Fast dev run enabled; skipping full validation sweep.")
    else:
        logger.info("Running final validation/evaluation pass on full validation set.")
        val_metrics = trainer.validate(
            model,
            dataloaders=val_loader,
            ckpt_path=best_ckpt_path,
        )
        logger.info("Validation metrics: %s", val_metrics)
        if weight_avg_callback is not None:
            ema_checkpoint_path = weight_avg_callback.averaged_path or weight_avg_callback.output_path
            if ema_checkpoint_path and ema_checkpoint_path.exists():
                logger.info(
                    "Running %s validation using averaged checkpoint %s",
                    cfg.variance_reduction_method,
                    ema_checkpoint_path,
                )
                try:
                    ema_val_metrics = trainer.validate(
                        model,
                        dataloaders=val_loader,
                        ckpt_path=str(ema_checkpoint_path),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to compute %s validation metrics: %s",
                        cfg.variance_reduction_method,
                        exc,
                    )
                else:
                    logger.info(
                        "%s validation metrics: %s",
                        cfg.variance_reduction_method,
                        ema_val_metrics,
                    )
                    if ema_val_metrics:
                        ema_metrics = ema_val_metrics[0]
            else:
                logger.warning(
                    "Averaged checkpoint not found; skipping %s validation metrics.",
                    cfg.variance_reduction_method,
                )

    best_ckpt_path = checkpoint_cb.best_model_path or best_ckpt_path
    best_path_obj = resolve_checkpoint_path(best_ckpt_path)
    if ema_metrics is not None and ema_checkpoint_path is not None:
        def _update_ema_metadata(metadata: Dict[str, object]) -> None:
            ema_block = metadata.setdefault("ema_metrics", {})
            if isinstance(ema_block, dict):
                ema_block["checkpoint"] = str(ema_checkpoint_path)
                ema_block["method"] = cfg.variance_reduction_method
                for key in (
                    "val_loss",
                    "val_pearson_corr",
                    "val_spearman_corr",
                    "val_rank_spearman",
                    "val_rank_regret",
                    "tuning_rank_spearman",
                    "tuning_rank_regret",
                ):
                    value = _safe_scalar(ema_metrics.get(key))
                    if value is not None:
                        ema_block[key] = value

        try:
            update_run_metadata(cfg.save_dir, _update_ema_metadata)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Unable to record EMA validation metrics: %s", exc)

    if checkpoint_dir.exists():
        ranked_checkpoints = _rank_checkpoints(
            checkpoint_dir,
            scores={k: v for k, v in checkpoint_cb.best_k_models.items()} if checkpoint_cb.best_k_models else None,
            mode=monitor_mode,
        )
        if not ranked_checkpoints:
            # Fall back to filesystem order if Lightning did not populate best_k_models.
            ranked_checkpoints = sorted(
                checkpoint_dir.glob("*.chkpt"),
                key=lambda p: p.stat().st_mtime,
            )
        if best_path_obj and best_path_obj.exists():
            best_path_resolved = best_path_obj.resolve()
            if ranked_checkpoints:
                ranked_checkpoints = [best_path_resolved] + [
                    path for path in ranked_checkpoints if path != best_path_resolved
                ]
            else:
                ranked_checkpoints = [best_path_resolved]
        elif best_ckpt_path:
            candidate_path = Path(best_ckpt_path)
            if candidate_path.exists():
                ranked_checkpoints = [candidate_path.resolve()] + [
                    path for path in ranked_checkpoints if path != candidate_path.resolve()
                ]
        if ranked_checkpoints:
            best_ckpt_path = str(ranked_checkpoints[0])
            checkpoint_cb.best_model_path = best_ckpt_path
            best_path_obj = resolve_checkpoint_path(best_ckpt_path)
        _create_checkpoint_symlinks(checkpoint_dir, ranked_checkpoints, logger)

    alternate_checkpoint_path: Optional[Path] = None
    variance_checkpoint_path: Optional[Path] = None
    weight_avg_path: Optional[Path] = None
    if cfg.variance_reduction_enabled and cfg.variance_reduction_method == "topk_avg":
        top_k = max(1, int(cfg.variance_reduction_top_k))
        if ranked_checkpoints:
            variance_checkpoint_path = _average_checkpoints(
                ranked_checkpoints[:top_k],
                checkpoint_dir,
                logger,
            )
            if variance_checkpoint_path:
                _create_named_symlink(checkpoint_dir / "averaged_topk.ckpt", variance_checkpoint_path, logger)
    if weight_avg_callback is not None and weight_avg_callback.averaged_path is not None:
        weight_avg_path = weight_avg_callback.averaged_path
    if alternate_checkpoint_cb is not None:
        best_alt_path = alternate_checkpoint_cb.best_model_path
        if best_alt_path:
            alt_path_obj = Path(best_alt_path)
            alt_path_obj = _normalise_checkpoint_file(alt_path_obj, logger)
            alternate_checkpoint_cb.best_model_path = str(alt_path_obj)
            alternate_checkpoint_path = resolve_checkpoint_path(str(alt_path_obj))
            if alternate_symlink_name:
                _create_named_symlink(checkpoint_dir / alternate_symlink_name, alternate_checkpoint_path, logger)
        elif alternate_symlink_name:
            stale = checkpoint_dir / alternate_symlink_name
            if stale.exists() or stale.is_symlink():
                try:
                    stale.unlink()
                except OSError:
                    pass

    alternates_meta: Dict[str, Optional[Path]] = {}
    if alternate_metric_name is not None:
        alternates_meta[alternate_metric_name] = alternate_checkpoint_path
    if variance_checkpoint_path is not None:
        alternates_meta["avg_topk"] = variance_checkpoint_path
    if weight_avg_path is not None:
        alternates_meta[f"avg_{cfg.variance_reduction_method}"] = weight_avg_path
    try:
        record_checkpoint_paths(
            cfg.save_dir,
            primary_metric=primary_metric,
            primary_path=best_path_obj,
            alternates=alternates_meta if alternates_meta else None,
        )
    except Exception as exc:  # pragma: no cover - metadata recording best effort
        logger.warning("Unable to record checkpoint metadata: %s", exc)

    if load_profiler.enabled:
        load_profile_summary = load_profiler.finalise(cfg.save_dir, logger)

    missing_total = sum(len(stats["missing"]) for stats in coverage.values())
    if missing_total:
        logger.warning(
            "Training proceeded with missing graphs. Consider regenerating .pt files "
            "to close the gap and potentially improve MSE."
        )

    if mlflow_logger is not None:
        try:
            metrics_to_log: Dict[str, float] = {}
            best_score = _safe_scalar(getattr(checkpoint_cb, "best_model_score", None))
            if best_score is not None:
                metrics_to_log["best_val_loss"] = best_score

            final_selection = _safe_scalar(trainer.callback_metrics.get("selection_metric"))
            if final_selection is not None:
                metrics_to_log["selection_metric"] = final_selection
            for metric_name in (
                "val_rank_spearman",
                "val_rank_regret",
                "tuning_rank_spearman",
                "tuning_rank_regret",
            ):
                metric_value = _safe_scalar(trainer.callback_metrics.get(metric_name))
                if metric_value is not None:
                    metrics_to_log[metric_name] = metric_value

            if val_metrics:
                first_metrics = val_metrics[0]
                for key in ("val_loss", "val_pearson_corr", "val_spearman_corr"):
                    if key in first_metrics:
                        value = _safe_scalar(first_metrics[key])
                        if value is not None:
                            metrics_to_log[key] = value
            if load_profile_summary:
                mean_ms = load_profile_summary.get("mean_ms")
                if mean_ms is not None:
                    metrics_to_log["graph_load_mean_ms"] = float(mean_ms)
                p95_ms = load_profile_summary.get("p95_ms")
                if p95_ms is not None:
                    metrics_to_log["graph_load_p95_ms"] = float(p95_ms)
                max_ms = load_profile_summary.get("max_ms")
                if max_ms is not None:
                    metrics_to_log["graph_load_max_ms"] = float(max_ms)
                count = load_profile_summary.get("count")
                if count is not None:
                    metrics_to_log["graph_load_samples"] = float(count)

            if metrics_to_log:
                mlflow_logger.log_metrics(metrics_to_log)

            if cfg.mlflow.log_artifacts:
                if coverage_path.exists():
                    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(coverage_path))
                if metadata_path.exists():
                    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(metadata_path))
                if schema_summary_path and schema_summary_path.exists():
                    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(schema_summary_path))
                profile_path = load_profile_summary.get("output_path") if load_profile_summary else None
                if profile_path:
                    try:
                        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(profile_path))
                    except Exception as artifact_exc:  # pragma: no cover
                        logger.warning("Unable to log graph load profiling summary to MLflow: %s", artifact_exc)
                for artifact_path in ranked_checkpoints[:3]:
                    if artifact_path.exists():
                        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(artifact_path))
            try:
                experiment = mlflow_logger.experiment
                tracking_uri = getattr(mlflow_logger, "tracking_uri", None)
                if not tracking_uri and hasattr(experiment, "get_tracking_uri"):
                    try:
                        tracking_uri = experiment.get_tracking_uri()
                    except Exception:
                        tracking_uri = None
                if not tracking_uri and hasattr(experiment, "tracking_uri"):
                    tracking_uri = getattr(experiment, "tracking_uri", None)

                experiment_name = getattr(mlflow_logger, "experiment_name", None)
                if not experiment_name:
                    experiment_name = getattr(mlflow_logger, "_experiment_name", None)
                experiment_id = getattr(mlflow_logger, "experiment_id", None)
                if not experiment_name and experiment_id and experiment is not None:
                    try:
                        exp_info = experiment.get_experiment(experiment_id)
                        if exp_info is not None:
                            experiment_name = getattr(exp_info, "name", experiment_name)
                    except Exception:
                        pass

                run_name = getattr(mlflow_logger, "run_name", None) or getattr(mlflow_logger, "_run_name", None)

                if not tracking_uri:
                    try:  # pragma: no cover - mlflow optional
                        import mlflow  # type: ignore

                        tracking_uri = mlflow.get_tracking_uri()
                    except Exception:
                        tracking_uri = None

                def _update_mlflow_metadata(metadata: Dict[str, Any]) -> None:
                    mlflow_section = metadata.setdefault("mlflow", {})
                    mlflow_section.update({"run_id": mlflow_logger.run_id})
                    if experiment_name:
                        mlflow_section["experiment"] = experiment_name
                    if experiment_id and "experiment_id" not in mlflow_section:
                        mlflow_section["experiment_id"] = experiment_id
                    if run_name:
                        mlflow_section.setdefault("run_name", run_name)
                    if tracking_uri:
                        mlflow_section["tracking_uri"] = tracking_uri

                update_run_metadata(cfg.save_dir, _update_mlflow_metadata)
            except Exception as mlflow_meta_exc:  # pragma: no cover
                logger.warning("Unable to record MLflow metadata: %s", mlflow_meta_exc)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Unable to log metrics/artifacts to MLflow: %s", exc)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
