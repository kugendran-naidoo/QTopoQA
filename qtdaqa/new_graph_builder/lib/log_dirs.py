"""Helpers for preparing per-run log directories.

The functions here mirror the behaviour of the original `graph_builder` tool:
  * ensure the root logging directory exists;
  * move any existing files or folders into a persistent ``history`` folder;
  * create a fresh run directory named with a prefix and timestamp.

The helpers are self-contained so they can be reused by other scripts that need
simple run-based logging directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional


@dataclass(frozen=True)
class LogDirectoryInfo:
    """Paths associated with the active logging run."""

    root_dir: Path
    history_dir: Path
    run_dir: Path
    run_name: str


def _move_entry(source: Path, destination: Path) -> None:
    """Atomically move *source* into *destination*, replacing if necessary."""
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        source.replace(destination)
    except Exception:
        shutil.move(str(source), str(destination))


def prepare_log_directory(
    log_root: Path,
    *,
    run_prefix: str,
    timestamp: Optional[datetime] = None,
) -> LogDirectoryInfo:
    """Ensure ``log_root`` is ready for a new run and return useful paths.

    Parameters
    ----------
    log_root:
        Directory that should hold per-run logs.
    run_prefix:
        String prepended to the timestamp when naming the new run directory.
    timestamp:
        Optional datetime to use when naming the run directory.  Defaults to
        ``datetime.now()``; useful for deterministic testing.

    Any existing content directly inside ``log_root`` is moved to the
    ``history`` subdirectory before a new run directory (``<prefix>.<ts>``) is
    created.  The returned :class:`LogDirectoryInfo` instance can be used by
    calling code to wire up log handlers or worker log directories.
    """

    log_root.mkdir(parents=True, exist_ok=True)

    history_dir = log_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    for entry in list(log_root.iterdir()):
        if entry == history_dir:
            continue
        _move_entry(entry, history_dir / entry.name)

    ts = timestamp or datetime.now()
    run_name = f"{run_prefix}.{ts.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return LogDirectoryInfo(
        root_dir=log_root,
        history_dir=history_dir,
        run_dir=run_dir,
        run_name=run_name,
    )
