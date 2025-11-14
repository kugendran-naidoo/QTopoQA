"""Lightweight progress logging helpers for long-running stages."""

from __future__ import annotations

import logging
from typing import Optional


class StageProgress:
    """Emit periodic percentage updates for batch-oriented stages."""

    def __init__(
        self,
        stage_name: str,
        total: int,
        logger: Optional[logging.Logger] = None,
        steps: int = 20,
    ) -> None:
        self.stage_name = stage_name
        self.total = total
        self.logger = logger or logging.getLogger("graph_builder")
        self.completed = 0
        if total <= 0:
            self._step = None
        else:
            # Aim for ~`steps` updates; fall back to 1 when dataset is tiny.
            raw_step = max(1, total // max(1, steps))
            self._step = raw_step
        self._next_log = self._step

    def increment(self, count: int = 1) -> None:
        if self._step is None:
            return
        self.completed += count
        if self.completed >= self.total:
            self._log(force=True)
            self._step = None
            return
        if self.completed >= self._next_log:
            self._log()
            self._next_log = min(self.total, self.completed + self._step)

    def _log(self, force: bool = False) -> None:
        if self.total <= 0:
            return
        if not force and self.completed < self.total and self.completed <= 0:
            return
        percent = (self.completed / self.total) * 100
        self.logger.info(
            "[%s] %.1f%% complete (%s/%s)",
            self.stage_name,
            percent,
            self.completed,
            self.total,
        )
