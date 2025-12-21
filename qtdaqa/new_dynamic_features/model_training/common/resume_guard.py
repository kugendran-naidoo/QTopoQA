from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch


def load_checkpoint_epoch(path: Path) -> Optional[int]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    for key in ("epoch", "current_epoch"):
        if key in payload:
            try:
                return int(payload[key])
            except Exception:
                continue
    return None


def discover_checkpoints(checkpoint_dir: Path) -> List[Path]:
    patterns = ("*.ckpt", "*.chkpt")
    candidates: List[Path] = []
    if checkpoint_dir.exists():
        for pattern in patterns:
            candidates.extend(checkpoint_dir.glob(pattern))
        for subdir in checkpoint_dir.glob("*_checkpoints"):
            if subdir.is_dir():
                for pattern in patterns:
                    candidates.extend(subdir.glob(pattern))
    return sorted({p.resolve() for p in candidates if p.is_file()})


def validate_resume_checkpoint(
    run_dir: Path,
    checkpoint_dir: Path,
    ckpt_path_raw: Optional[str],
    max_epochs: Optional[int],
    logger: logging.Logger,
) -> Optional[str]:
    """Guard against accidental resume from stale/overlong checkpoints or dirty run dirs."""
    if ckpt_path_raw:
        ckpt_path = Path(ckpt_path_raw).expanduser().resolve()
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint path not found: {ckpt_path}")
        ckpt_epoch = load_checkpoint_epoch(ckpt_path)
        if max_epochs is not None and ckpt_epoch is not None and ckpt_epoch >= max_epochs:
            raise RuntimeError(
                f"Refusing to resume from checkpoint {ckpt_path} (epoch={ckpt_epoch}) "
                f"with trainer.max_epochs={max_epochs}. "
                "To resume, either increase trainer.num_epochs/max_epochs to exceed the checkpoint epoch "
                f"(e.g., set trainer.num_epochs >= {ckpt_epoch + 1}) or choose an earlier checkpoint. "
                f"Example: --resume-from {ckpt_path}. "
                "If no suitable checkpoint exists or you want a clean run, empty/rename the run directory and restart."
            )
        logger.info("Resuming from checkpoint: %s (epoch=%s)", ckpt_path, ckpt_epoch)
        return str(ckpt_path)

    existing = discover_checkpoints(checkpoint_dir)
    if existing:
        newest = existing[-1]
        ckpt_epoch = load_checkpoint_epoch(newest)
        epoch_info = f" (epoch={ckpt_epoch})" if ckpt_epoch is not None else ""
        raise RuntimeError(
            "Run directory already contains checkpoints but no --resume-from was provided. "
            f"Example checkpoint: {newest}{epoch_info}. "
            "Clean or rename the run directory before starting a new run, "
            "or rerun with --resume-from <checkpoint> and ensure trainer.num_epochs is >= the checkpoint epoch. "
            f"Example: --resume-from {newest}. "
            "Best-effort guidance: prefer resuming to avoid wasted training time; if the stored checkpoints are unusable "
            "(e.g., epochs beyond your planned schedule), pick an earlier checkpoint or start clean by emptying the run directory."
        )
    return None
