"""Compatibility wrapper around the legacy interface extractor.

graph_builder2 originally shipped a rewritten interface pipeline that
stabilised ordering and deduplication.  However, the original inference
code (``topoqa/src/get_interface.py``) determines the node ordering used
throughout the historic training runs.  To reproduce the exact `.pt`
files emitted by that pipeline we proxy back to the legacy ``cal_interface``.
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Protocol


def _locate_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        candidate = parent / "topoqa" / "src"
        if candidate.exists():
            return parent
    raise RuntimeError("Unable to locate repo root containing 'topoqa/src'.")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
TOPOQA_SRC = REPO_ROOT / "topoqa" / "src"
if str(TOPOQA_SRC) not in sys.path:
    sys.path.insert(0, str(TOPOQA_SRC))

from get_interface import cal_interface  # type: ignore  # noqa: E402


class LoggingLike(Protocol):
    """Minimal protocol for objects that mirror ``logging.Logger``."""

    def debug(self, message: str, *args, **kwargs) -> None: ...


class NullLogger:
    """Fallback logger that ignores all messages."""

    def debug(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - trivial
        pass


def process_pdb_file(
    pdb_path: Path,
    output_path: Path,
    cutoff: float = 10.0,
    *,
    logger: Optional[LoggingLike] = None,
) -> int:
    """Process a single PDB file and write its interface information."""

    log = logger or NullLogger()
    log.debug(f"Processing PDB file: {pdb_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    interface_calc = cal_interface(str(pdb_path), cut=cutoff)
    interface_calc.find_and_write(str(output_path))

    try:
        with output_path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        count = 0

    log.debug(f"Wrote interface data to {output_path} ({count} residue(s))")
    return count


def process_directory(
    pdb_dir: Path,
    output_dir: Path,
    cutoff: float = 10.0,
    *,
    workers: int | None = None,
    output_suffix: str = "",
    output_extension: str = "txt",
) -> Mapping[Path, int]:
    """Process every ``*.pdb`` in *pdb_dir* and write output files under *output_dir*."""

    pdb_dir = Path(pdb_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not pdb_dir.is_dir():
        raise NotADirectoryError(f"PDB directory does not exist: {pdb_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_suffix:
        output_suffix = f".{output_suffix.strip('.')}"
    output_extension = output_extension.strip(".") or "txt"

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        return {}

    results: MutableMapping[Path, int] = {}
    failures: MutableMapping[Path, Exception] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(
                process_pdb_file,
                pdb_path,
                output_dir / f"{pdb_path.stem}{output_suffix}.{output_extension}",
                cutoff,
            ): pdb_path
            for pdb_path in pdb_files
        }

        for future in concurrent.futures.as_completed(future_to_path):
            pdb_path = future_to_path[future]
            try:
                results[pdb_path] = future.result()
            except Exception as exc:
                failures[pdb_path] = exc

    if failures:
        failure_messages = ", ".join(f"{path.name}: {exc}" for path, exc in failures.items())
        raise RuntimeError(f"Failed to process one or more PDB files: {failure_messages}")

    return results


__all__ = [
    "process_pdb_file",
    "process_directory",
    "LoggingLike",
    "NullLogger",
]
