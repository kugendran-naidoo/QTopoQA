"""
Logging and profiling utilities for the graph builder.

Provides:
- setup_logger: configure a rotating file + console logger. The caller supplies
  the logger "name" (often with a timestamp) so each execution writes to a new
  log file, while the console stream prints the same lines. Timestamps are
  compact (YYYY-MM-DD HH:MM:SS).
- timeit: decorator to measure execution time of functions/methods (optional).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional


def setup_logger(log_dir: str, name: str = "qtdaqa") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers when reusing in subprocesses/tests
    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    # Also wire a base 'qtdaqa' logger so internal modules logging to this name reuse the same handlers.
    base = logging.getLogger("qtdaqa")
    if not base.handlers:
        base.setLevel(logging.INFO)
        base.addHandler(file_handler)
        base.addHandler(stream_handler)
        base.propagate = False
    return logger


def timeit(label: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to log execution time of large running components.
    """
    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            _label = label or func.__qualname__
            logger = logging.getLogger("qtdaqa")
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                logger.info(f"[time] {_label} took {dt:.3f}s")
        return _wrapper
    return _decorator
