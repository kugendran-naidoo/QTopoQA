from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def _normalise_mask_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Feature mask is empty.")
    return arr


def load_feature_mask(path: Path) -> np.ndarray:
    """
    Load a feature mask from JSON/YAML/NPY.

    Accepted formats:
    - JSON list: [0, 1, 1, ...]
    - JSON object: {"mask": [..], ...}
    - NPY: numpy 1D array
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
        return _normalise_mask_values(mask)

    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unsupported feature mask format for {path}") from exc

    if isinstance(payload, dict):
        if "mask" not in payload:
            raise ValueError(f"Feature mask dict missing 'mask' key: {path}")
        return _normalise_mask_values(payload["mask"])
    if isinstance(payload, list):
        return _normalise_mask_values(payload)

    raise ValueError(f"Unsupported feature mask payload type: {type(payload)}")


def summarise_feature_mask(mask: Iterable[float]) -> dict:
    arr = np.asarray(list(mask), dtype=np.float32).reshape(-1)
    kept = float(np.count_nonzero(arr))
    total = float(arr.size)
    return {
        "length": int(total),
        "kept": int(kept),
        "kept_fraction": float(kept / total) if total else 0.0,
    }
