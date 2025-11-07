"""
Load TopoQA checkpoints and instantiate Lightning modules for saliency analysis.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

import torch
from pytorch_lightning import LightningModule

from .config import CheckpointConfig


def _ensure_sys_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _import_training_module() -> ModuleType:
    return importlib.import_module("qtdaqa.new_dynamic_features.model_training.model_train_topoqa_cpu")


def load_lightning_module(cfg: CheckpointConfig) -> LightningModule:
    cfg.validate()
    _ensure_sys_path(cfg.repo_root)
    module = _import_training_module()
    model_cls = getattr(module, "CpuTopoQAModule")
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    hparams = checkpoint.get("hyper_parameters", {})
    feature_metadata = checkpoint.get("feature_metadata")
    model: LightningModule = model_cls(**hparams)
    if feature_metadata:
        model.feature_metadata = feature_metadata
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'state_dict'")
    model.load_state_dict(state_dict, strict=cfg.strict)
    model.to(cfg.device)
    model.eval()
    return model


@contextmanager
def model_context(cfg: CheckpointConfig) -> Iterator[LightningModule]:
    model = load_lightning_module(cfg)
    try:
        yield model
    finally:
        del model
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
