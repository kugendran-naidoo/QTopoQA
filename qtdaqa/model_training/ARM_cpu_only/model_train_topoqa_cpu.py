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
import logging
import os
import random
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
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(REPO_ROOT / "topoqa_train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "topoqa_train"))

from gat_5_edge1 import GNN_edge1_edgepooling  # type: ignore  # noqa: E402


@dataclasses.dataclass
class TrainingConfig:
    graph_dir: Path
    train_label_file: Path
    val_label_file: Path
    save_dir: Path
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


def _resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base / raw).resolve()
    return path


def load_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    def _get(name: str, default=None):
        return data.get(name, default)

    graph_dir = _resolve_path(_get("graph_dir"), SCRIPT_DIR)
    train_label_file = _resolve_path(_get("train_label_file"), SCRIPT_DIR)
    val_label_file = _resolve_path(_get("val_label_file"), SCRIPT_DIR)
    save_dir = _resolve_path(_get("save_dir"), SCRIPT_DIR)

    cfg = TrainingConfig(
        graph_dir=graph_dir,
        train_label_file=train_label_file,
        val_label_file=val_label_file,
        save_dir=save_dir,
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


def _gather_samples(
    graph_dir: Path,
    label_map: Dict[str, float],
) -> Tuple[List[Tuple[str, Path, torch.Tensor]], List[str]]:
    samples: List[Tuple[str, Path, torch.Tensor]] = []
    missing: List[str] = []
    for model, label in label_map.items():
        candidate = graph_dir / f"{model}.pt"
        if candidate.exists():
            tensor = torch.tensor([label], dtype=torch.float32)
            samples.append((model, candidate, tensor))
        else:
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
        **kwargs,
    ):
        super().__init__(init_lr=lr, pooling_type=pooling_type, mode="zuhe", **kwargs)
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

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


def build_dataloaders(cfg: TrainingConfig, logger: logging.Logger):
    train_labels = load_label_map(cfg.train_label_file)
    val_labels = load_label_map(cfg.val_label_file)

    train_samples, train_missing = _gather_samples(cfg.graph_dir, train_labels)
    val_samples, val_missing = _gather_samples(cfg.graph_dir, val_labels)

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
    return train_loader, val_loader, {"train_missing": train_missing, "val_missing": val_missing}


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

    logger.info("Repository root: %s", REPO_ROOT)
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

    train_loader, val_loader, coverage = build_dataloaders(cfg, logger)

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
        edge_dim=11,
        heads=cfg.attention_head,
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

    csv_logger = CSVLogger(str(cfg.save_dir), name="cpu_training", flush_logs_every_n_steps=1)

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=1,
        precision=cfg.precision,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb, early_stop_cb, *lr_callbacks],
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
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

    if cfg.fast_dev_run:
        logger.info("Fast dev run enabled; skipping full validation sweep.")
    else:
        logger.info("Running final validation/evaluation pass on full validation set.")
        best_ckpt_path = checkpoint_cb.best_model_path or None
        val_metrics = trainer.validate(
            model,
            dataloaders=val_loader,
            ckpt_path=best_ckpt_path,
        )
        logger.info("Validation metrics: %s", val_metrics)

    if coverage["train_missing"] or coverage["val_missing"]:
        logger.warning(
            "Training proceeded with missing graphs. Consider regenerating .pt files "
            "to close the gap and potentially improve MSE."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
