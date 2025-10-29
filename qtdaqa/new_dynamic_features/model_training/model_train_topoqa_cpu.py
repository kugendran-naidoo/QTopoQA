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
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import platform

import numpy as np
import pandas as pd
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
from gat_5_edge1 import GNN_edge1_edgepooling  # type: ignore  # noqa: E402


@dataclasses.dataclass
class TrainingConfig:
    graph_dir: Path
    train_label_file: Path
    val_label_file: Path
    save_dir: Path
    metadata_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    accelerator: str = "cpu"
    attention_head: int = 8
    pooling_type: str = "mean"
    batch_size: int = 16
    learning_rate: float = 5e-3
    num_epochs: int = 200
    accumulate_grad_batches: int = 32
    seed: int = 222
    num_workers: int = 0
    precision: int = 32
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    fast_dev_run: bool = False
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None
    log_lr: bool = False
    progress_bar_refresh_rate: int = 0
    log_every_n_steps: int = 100
    edge_schema: dict = dataclasses.field(default_factory=dict)
    topology_schema: dict = dataclasses.field(default_factory=dict)


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

    base_dir = path.parent.resolve()

    def _get(name: str, default=None):
        return data.get(name, default)

    graph_dir = _resolve_path(_get("graph_dir"), base_dir, fallbacks=(SCRIPT_DIR,))
    train_label_file = _resolve_path(_get("train_label_file"), base_dir, fallbacks=(SCRIPT_DIR,))
    val_label_file = _resolve_path(_get("val_label_file"), base_dir, fallbacks=(SCRIPT_DIR,))
    save_dir = _resolve_path(_get("save_dir"), base_dir, fallbacks=(SCRIPT_DIR,), allow_missing=True)

    metadata_path_raw = _get("metadata_path")
    summary_path_raw = _get("summary_path")
    metadata_path = (
        _resolve_path(metadata_path_raw, base_dir, fallbacks=(SCRIPT_DIR,))
        if metadata_path_raw
        else None
    )
    summary_path = (
        _resolve_path(summary_path_raw, base_dir, fallbacks=(SCRIPT_DIR,))
        if summary_path_raw
        else None
    )

    cfg = TrainingConfig(
        graph_dir=graph_dir,
        train_label_file=train_label_file,
        val_label_file=val_label_file,
        save_dir=save_dir,
        metadata_path=metadata_path,
        summary_path=summary_path,
        accelerator=str(_get("accelerator", "cpu")),
        attention_head=int(_get("attention_head", 8)),
        pooling_type=str(_get("pooling_type", "mean")),
        batch_size=int(_get("batch_size", 16)),
        learning_rate=float(_get("learning_rate", 5e-3)),
        num_epochs=int(_get("num_epochs", 200)),
        accumulate_grad_batches=int(_get("accumulate_grad_batches", 32)),
        seed=int(_get("seed", 222)),
        num_workers=int(_get("num_workers", 0)),
        precision=int(_get("precision", 32)),
        early_stopping_patience=int(_get("early_stopping_patience", 20)),
        lr_scheduler_patience=int(_get("lr_scheduler_patience", 10)),
        lr_scheduler_factor=float(_get("lr_scheduler_factor", 0.5)),
        fast_dev_run=bool(_get("fast_dev_run", False)),
        limit_train_batches=_get("limit_train_batches"),
        limit_val_batches=_get("limit_val_batches"),
        log_lr=bool(_get("log_lr", False)),
        progress_bar_refresh_rate=int(_get("progress_bar_refresh_rate", 0)),
        log_every_n_steps=int(_get("log_every_n_steps", 100)),
        edge_schema=dict(_get("edge_schema", {})),
        topology_schema=dict(_get("topology_schema", {})),
    )
    return cfg


def load_label_map(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
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


class GraphRegressionDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, Path, torch.Tensor]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        model, path, label = self.samples[idx]
        data = torch.load(path)

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
    return [merged]


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
        **kwargs,
    ):
        super().__init__(init_lr=lr, pooling_type=pooling_type, mode="zuhe", edge_schema=edge_schema, **kwargs)
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.feature_metadata = feature_metadata or {}

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

    def on_load_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        self.feature_metadata = checkpoint.get("feature_metadata", {})



def _setup_logging(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_cpu_topoqa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

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
        logger.warning(
            "[%s] %d label entries missing matching .pt graphs (first examples: %s)",
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
        report[split] = {
            "total": total,
            "usable": usable,
            "coverage": (usable / total) if total else 0.0,
            "missing": missing,
        }
    path = save_dir / "dataset_coverage.json"
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    except OSError as exc:
        raise RuntimeError(f"Unable to write dataset coverage report: {exc}")
    return path


def _log_feature_summary(logger: logging.Logger, metadata: GraphFeatureMetadata) -> None:
    node_dim = metadata.node_schema.get("dim")
    edge_dim = metadata.edge_schema.get("dim")
    sample_nodes = metadata.sample_node_count
    sample_edges = metadata.sample_edge_count

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


_CHECKPOINT_SCORE_RE = re.compile(r"model\.([0-9eE+\-.]+)")


def _parse_checkpoint_score(path: Path) -> Optional[float]:
    match = _CHECKPOINT_SCORE_RE.search(path.name)
    if not match:
        return None
    token = match.group(1).split("-", 1)[0]
    token = token.rstrip(".")
    try:
        return float(token)
    except ValueError:
        return None


def _rank_checkpoints(checkpoint_dir: Path, limit: int = 3) -> List[Path]:
    ranked: List[Tuple[float, Path]] = []
    for candidate in checkpoint_dir.glob("*.chkpt"):
        score = _parse_checkpoint_score(candidate)
        if score is None:
            continue
        ranked.append((score, candidate.resolve()))
    ranked.sort(key=lambda item: item[0])
    return [path for _, path in ranked[:limit]]


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

    graph_lookup = _build_graph_lookup(cfg.graph_dir)

    train_samples, train_missing = _gather_samples(cfg.graph_dir, train_labels, graph_lookup)
    val_samples, val_missing = _gather_samples(cfg.graph_dir, val_labels, graph_lookup)

    if not train_samples:
        raise RuntimeError("No training graphs found; aborting.")
    if not val_samples:
        raise RuntimeError("No validation graphs found; aborting.")

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

    train_loader = DataLoader(
        GraphRegressionDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        GraphRegressionDataset(val_samples),
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
        },
        "val": {
            "total": len(val_labels),
            "usable": len(val_samples),
            "missing": list(val_missing),
        },
    }
    return train_loader, val_loader, coverage, feature_metadata


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

    for path in (
        cfg.graph_dir,
        cfg.train_label_file,
        cfg.val_label_file,
    ):
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

    train_loader, val_loader, coverage, feature_metadata = build_dataloaders(cfg, logger)

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

    cfg.edge_schema = resolved_edge_schema
    feature_metadata.edge_schema = dict(resolved_edge_schema)
    feature_metadata_dict = feature_metadata.to_dict()
    feature_metadata_dict["topology_schema"] = cfg.topology_schema

    _log_feature_summary(logger, feature_metadata)

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
        edge_schema=cfg.edge_schema,
        feature_metadata=feature_metadata_dict,
    )

    checkpoint_dir = cfg.save_dir / "model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="model.{val_loss:.5f}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.early_stopping_patience,
        verbose=True,
    )
    lr_callbacks = []
    if args.log_lr:
        lr_callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    progress_bar_cb = None
    enable_progress_bar = cfg.progress_bar_refresh_rate > 0
    if enable_progress_bar:
        progress_bar_cb = TQDMProgressBar(refresh_rate=cfg.progress_bar_refresh_rate)

    callback_list: List = [checkpoint_cb, early_stop_cb, *lr_callbacks]
    if progress_bar_cb is not None:
        callback_list.append(progress_bar_cb)

    callback_list.append(TrainingProgressLogger(logger))
    callback_list.append(EpochSummaryLogger(logger))

    csv_logger = CSVLogger(str(cfg.save_dir), name="cpu_training", flush_logs_every_n_steps=1)

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=1,
        precision=cfg.precision,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callback_list,
        logger=csv_logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=max(1, cfg.log_every_n_steps),
        deterministic=True,
        default_root_dir=str(cfg.save_dir),
        fast_dev_run=cfg.fast_dev_run,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    logger.info("Starting training (fast_dev_run=%s)", cfg.fast_dev_run)
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
        ckpt_path=args.resume_from,
    )

    renamed_checkpoints: List[Path] = []
    ckpt_dir = Path(checkpoint_cb.dirpath) if checkpoint_cb.dirpath else checkpoint_dir
    if ckpt_dir.exists():
        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            new_path = ckpt_file.with_suffix(".chkpt")
            ckpt_file.rename(new_path)
            renamed_checkpoints.append(new_path)
            if checkpoint_cb.best_model_path and Path(checkpoint_cb.best_model_path) == ckpt_file:
                checkpoint_cb.best_model_path = str(new_path)
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

    ranked_checkpoints: List[Path] = []

    if cfg.fast_dev_run:
        logger.info("Fast dev run enabled; skipping full validation sweep.")
        best_ckpt_path = checkpoint_cb.best_model_path or None
    else:
        logger.info("Running final validation/evaluation pass on full validation set.")
        best_ckpt_path = checkpoint_cb.best_model_path or None
        val_metrics = trainer.validate(
            model,
            dataloaders=val_loader,
            ckpt_path=best_ckpt_path,
        )
        logger.info("Validation metrics: %s", val_metrics)

    if checkpoint_dir.exists():
        ranked_checkpoints = _rank_checkpoints(checkpoint_dir)
        if ranked_checkpoints:
            best_ckpt_path = str(ranked_checkpoints[0])
        _create_checkpoint_symlinks(checkpoint_dir, ranked_checkpoints, logger)

    missing_total = sum(len(stats["missing"]) for stats in coverage.values())
    if missing_total:
        logger.warning(
            "Training proceeded with missing graphs. Consider regenerating .pt files "
            "to close the gap and potentially improve MSE."
        )

    metadata_path = cfg.save_dir / "feature_metadata.json"
    try:
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(feature_metadata_dict, handle, indent=2)
        logger.info("Feature metadata written to %s", metadata_path)
    except OSError as exc:
        logger.warning("Unable to write feature metadata file: %s", exc)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
