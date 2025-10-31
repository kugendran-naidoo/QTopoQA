#!/usr/bin/env python3
"""Remove all MLflow runs stored under ml_runs/."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ML_RUNS = ROOT / "ml_runs"


def main() -> int:
    if not ML_RUNS.exists():
        print(f"[cleanup_ml_runs] No ml_runs directory found at {ML_RUNS}")
        return 0

    for entry in ML_RUNS.iterdir():
        if entry.is_dir() or entry.is_symlink():
            print(f"[cleanup_ml_runs] Removing {entry}")
            shutil.rmtree(entry, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
