"""Filesystem permission helpers for graph_builder2.

These utilities validate that a directory tree is readable (and optionally
writeable) before the main script attempts to operate on it.  They raise
``PermissionError`` with descriptive messages so callers can surface the issue
to end users.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple


def _describe_failures(failures: Iterable[Tuple[Path, str]]) -> str:
    """Format a short human-readable description of permission failures."""
    parts = [f"{path}: {reason}" for path, reason in failures]
    return "; ".join(parts)


def _verify_directory_tree(root: Path, *, require_write: bool) -> None:
    """Ensure ``root`` exists, is a directory, and is accessible as required."""
    failures: List[Tuple[Path, str]] = []

    if not root.exists():
        failures.append((root, "does not exist"))
    elif not root.is_dir():
        failures.append((root, "is not a directory"))
    else:
        dir_mode = os.R_OK | os.X_OK
        file_mode = os.R_OK
        if require_write:
            dir_mode |= os.W_OK
            file_mode |= os.W_OK

        # Walk the tree; os.walk will raise if we lack execute permission so we
        # use onerror to capture such exceptions instead of aborting early.
        def _on_error(err: OSError) -> None:
            failures.append((Path(err.filename), err.strerror or "access failed"))

        for dirpath, dirnames, filenames in os.walk(root, onerror=_on_error, followlinks=False):
            current_dir = Path(dirpath)
            if not os.access(current_dir, dir_mode):
                failures.append((current_dir, "insufficient directory permissions"))

            for dirname in dirnames:
                subdir = current_dir / dirname
                if not os.access(subdir, dir_mode):
                    failures.append((subdir, "insufficient directory permissions"))

            for filename in filenames:
                file_path = current_dir / filename
                if not os.access(file_path, file_mode):
                    failures.append((file_path, "insufficient file permissions"))

    if failures:
        raise PermissionError(_describe_failures(failures))


def ensure_tree_readable(root: Path) -> None:
    """Validate that ``root`` and all descendants are readable.

    Raises
    ------
    PermissionError
        When the directory tree cannot be fully read.
    """

    _verify_directory_tree(root, require_write=False)


def ensure_tree_readwrite(root: Path) -> None:
    """Validate that ``root`` and all descendants are readable and writeable.

    Raises
    ------
    PermissionError
        When the directory tree is not fully read/write accessible.
    """

    _verify_directory_tree(root, require_write=True)
