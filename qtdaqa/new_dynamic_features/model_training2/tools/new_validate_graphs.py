"""Validate graph_dir files with optional progress/ETA and parallel workers.

This mirrors the original validate_graphs behavior by default (serial execution,
same hashing and manifest semantics) but adds opt-in progress output and worker
parallelism so long-running runs can report status without touching the original
entrypoint.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

from .validate_graphs import canonicalize_graph, hash_canonical_graph, walk_graphs


def _hash_single(
    pt_path: Path, graph_dir: Path, ignore_metadata: bool
) -> Tuple[str, Optional[Dict[str, str]], Optional[str]]:
    """Load, canonicalize, and hash a single graph file."""
    rel = str(pt_path.relative_to(graph_dir))
    try:
        data = torch.load(pt_path, map_location="cpu")
        canonical = canonicalize_graph(data)
        hashes = hash_canonical_graph(canonical, ignore_metadata=ignore_metadata)
        return rel, hashes, None
    except Exception as exc:  # noqa: BLE001
        return rel, None, f"load error: {exc}"


def _log_progress(
    *,
    start_time: float,
    last_log: float,
    processed: int,
    total: int,
    mismatched: int,
    missing: int,
    interval: float,
    force: bool = False,
) -> float:
    """Emit periodic progress with ETA; returns updated last_log."""
    now = time.time()
    if not force and interval > 0 and (now - last_log) < interval:
        return last_log

    elapsed = now - start_time
    pct = (processed / total * 100.0) if total else 100.0
    rate = processed / elapsed if elapsed > 0 and processed > 0 else None
    remaining = total - processed
    eta = (remaining / rate) if rate else None
    eta_str = f"{eta:0.1f}s" if eta is not None else "n/a"
    print(
        f"[validate_graphs] PROGRESS: {processed}/{total} ({pct:0.1f}%) "
        f"elapsed {elapsed:0.1f}s eta {eta_str} mismatched={mismatched} missing={missing}",
        flush=True,
    )
    return now


def validate(
    graph_dir: Path,
    manifest: Optional[Path],
    create_manifest: bool,
    sample: Optional[int],
    ignore_metadata: bool,
    *,
    workers: int = 1,
    progress_interval: float = 15.0,
) -> int:
    graph_dir = graph_dir.resolve()
    if not graph_dir.is_dir():
        print(f"[validate_graphs] ERROR: graph_dir not found: {graph_dir}", file=sys.stderr, flush=True)
        return 2

    manifest_data: Dict[str, Dict[str, str]] = {}
    if manifest and manifest.exists():
        try:
            manifest_data = json.loads(manifest.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[validate_graphs] ERROR: failed to read manifest: {exc}", file=sys.stderr, flush=True)
            return 2

    files = list(walk_graphs(graph_dir))
    if sample:
        files = files[:sample]
    total = len(files)

    missing = []
    mismatched = []
    calculated: Dict[str, Dict[str, str]] = {}

    start = time.time()
    last_log = start

    stop_event: Optional[threading.Event] = None
    ticker: Optional[threading.Thread] = None

    def current_counts() -> Tuple[int, int, int]:
        return len(calculated) + len(mismatched), len(mismatched), len(missing)

    if progress_interval > 0:
        stop_event = threading.Event()

        def _ticker() -> None:
            while not stop_event.wait(progress_interval):
                processed, mismatched_count, missing_count = current_counts()
                _log_progress(
                    start_time=start,
                    last_log=time.time(),  # tickers always force log
                    processed=processed,
                    total=total,
                    mismatched=mismatched_count,
                    missing=missing_count,
                    interval=0,
                    force=True,
                )

        ticker = threading.Thread(target=_ticker, daemon=True)
        ticker.start()

    def handle_result(rel: str, hashes: Optional[Dict[str, str]], error: Optional[str]) -> None:
        nonlocal last_log
        if error:
            mismatched.append((rel, error))
        elif hashes is not None:
            calculated[rel] = hashes
            if manifest_data:
                expected = manifest_data.get(rel)
                if expected is not None and ignore_metadata and "metadata" in expected:
                    expected = {k: v for k, v in expected.items() if k != "metadata"}
                if expected is None:
                    missing.append(rel)
                elif expected != hashes:
                    mismatched.append((rel, "hash mismatch"))
        processed = len(calculated) + len(mismatched)
        last_log = _log_progress(
            start_time=start,
            last_log=last_log,
            processed=processed,
            total=total,
            mismatched=len(mismatched),
            missing=len(missing),
            interval=progress_interval,
        )

    max_workers = None
    if workers == 0:
        max_workers = os.cpu_count() or 1
    else:
        max_workers = max(1, workers)

    if total == 0:
        _log_progress(
            start_time=start,
            last_log=last_log,
            processed=0,
            total=0,
            mismatched=0,
            missing=0,
            interval=progress_interval,
            force=True,
        )
    elif max_workers == 1:
        for pt_path in files:
            rel, hashes, error = _hash_single(pt_path, graph_dir, ignore_metadata)
            handle_result(rel, hashes, error)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_hash_single, pt_path, graph_dir, ignore_metadata): pt_path for pt_path in files
            }
            for future in concurrent.futures.as_completed(future_map):
                try:
                    rel, hashes, error = future.result()
                except Exception as exc:  # noqa: BLE001
                    rel = str(future_map[future].relative_to(graph_dir))
                    handle_result(rel, None, f"worker error: {exc}")
                else:
                    handle_result(rel, hashes, error)

    # Final progress line.
    _log_progress(
        start_time=start,
        last_log=last_log,
        processed=len(calculated) + len(mismatched),
        total=total,
        mismatched=len(mismatched),
        missing=len(missing),
        interval=0,
        force=True,
    )

    if manifest and create_manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(calculated, indent=2, sort_keys=True))
        print(f"[validate_graphs] Manifest written: {manifest}", flush=True)
        manifest_data = calculated
        mismatched = []
        missing = []

    if stop_event:
        stop_event.set()
    if ticker:
        ticker.join()

    if mismatched or (manifest_data and missing):
        if mismatched:
            print(f"[validate_graphs] MISMATCHED: {len(mismatched)} (showing first 10)", flush=True)
            for rel, reason in mismatched[:10]:
                print(f"  {rel}: {reason}", flush=True)
        if manifest_data and missing:
            print(f"[validate_graphs] MISSING in manifest: {len(missing)} (showing first 10)", flush=True)
            for rel in missing[:10]:
                print(f"  {rel}", flush=True)
        return 1

    print(f"[validate_graphs] OK: validated {len(calculated)} graph(s) in {graph_dir}", flush=True)
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", required=True, type=Path, help="Path to graph_dir containing .pt files")
    parser.add_argument("--manifest", type=Path, help="Path to manifest of canonical hashes for comparison")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write (or overwrite) manifest with current canonical hashes instead of comparing",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Validate only the first N graphs (for quick spot checks). Omit to validate all.",
    )
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Ignore the 'metadata' field when hashing (compare tensors only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes; 1 keeps serial behavior, 0 uses all available CPUs.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=15.0,
        help="Seconds between progress/ETA updates (set 0 to disable periodic logs).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    return validate(
        args.graph_dir,
        args.manifest,
        args.write_manifest,
        args.sample,
        args.ignore_metadata,
        workers=args.workers,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
