#!/usr/bin/env python3
"""Remove all run directories under training_runs/ (including history)."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUNS = ROOT / "training_runs"


def main() -> int:
    if not TRAINING_RUNS.exists():
        print(f"[cleanup_training_runs] No training_runs directory found at {TRAINING_RUNS}")
        return 0

    for entry in TRAINING_RUNS.iterdir():
        if entry.is_dir() or entry.is_symlink():
            print(f"[cleanup_training_runs] Removing {entry}")
            shutil.rmtree(entry, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
