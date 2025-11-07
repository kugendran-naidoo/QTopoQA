"""
Internal implementation package for the saliency toolkit.

The module avoids importing heavyweight dependencies (e.g. Torch) at import time
so helpers like ``python saliency_cli.py --help`` can run in lightweight environments.
"""
from __future__ import annotations

from typing import Any

from .config import CheckpointConfig, GraphSelection, SaliencyRequest, default_output_dir  # noqa: F401

__all__ = [
    "CheckpointConfig",
    "GraphSelection",
    "SaliencyRequest",
    "default_output_dir",
    "run_saliency",
]


def run_saliency(*args: Any, **kwargs: Any):
    from .runner import run_saliency as _run_saliency

    return _run_saliency(*args, **kwargs)
