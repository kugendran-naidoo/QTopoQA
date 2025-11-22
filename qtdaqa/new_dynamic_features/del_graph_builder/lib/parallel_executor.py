"""Lightweight helpers for parallel task execution.

These utilities encapsulate the process-pool logic used in the original
``graph_builder`` so ``graph_builder2`` (and future scripts) can submit generic
work items without duplicating the boilerplate.  The helpers deliberately avoid
reference to "targets" or "decoys" so they can be reused in other contexts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar

Task = TypeVar("Task")
Result = TypeVar("Result")


@dataclass(frozen=True)
class ParallelConfig:
    """Normalised worker configuration."""

    workers: Optional[int]


def normalise_worker_count(cli_workers: Optional[int], default_workers: Optional[int]) -> ParallelConfig:
    """Return a :class:`ParallelConfig` with a sanitised worker count.

    ``cli_workers`` is the explicit value supplied by the user (e.g. ``--parallel``);
    ``default_workers`` may come from configuration files.  ``None`` means "use
    sequential execution".  Any positive integer is accepted; non-positive inputs
    are treated as sequential.
    """

    if cli_workers is not None:
        workers = cli_workers if cli_workers and cli_workers > 1 else None
    elif default_workers is not None:
        workers = default_workers if default_workers and default_workers > 1 else None
    else:
        workers = None
    return ParallelConfig(workers=workers)


def run_tasks(
    tasks: Sequence[Task],
    worker_fn: Callable[[Task], Result],
    *,
    workers: Optional[int],
) -> Iterator[Result]:
    """Execute *tasks* sequentially or in a process pool.

    When ``workers`` is ``None`` or ``<= 1`` the tasks run sequentially,
    preserving behaviour identical to the original script.  Otherwise tasks are
    dispatched to a :class:`~concurrent.futures.ProcessPoolExecutor` with the
    requested number of workers and results are yielded in submission order.
    """

    if not workers or workers <= 1:
        for task in tasks:
            yield worker_fn(task)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(worker_fn, tasks):
            yield result
