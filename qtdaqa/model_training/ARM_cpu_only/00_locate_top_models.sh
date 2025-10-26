#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
RUN_ROOT="${SCRIPT_DIR}/training_runs"
HISTORY_ROOT="${RUN_ROOT}/history"

# shellcheck disable=SC2016
python - "$RUN_ROOT" "$HISTORY_ROOT" <<'PY'
import sys
import re
from pathlib import Path

run_root = Path(sys.argv[1])
history_root = Path(sys.argv[2])
pattern = re.compile(r"model\.([0-9]+\.[0-9]+)\.chkpt$", re.IGNORECASE)


def resolve_checkpoint(path_str: str, log_path: Path) -> Path | None:
    raw_path = Path(path_str)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((log_path.parent / raw_path).resolve())
    candidates.append((log_path.parent / "model_checkpoints" / raw_path.name).resolve())
    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    for candidate in seen:
        if candidate.exists():
            return candidate
    return None


def extract_value_from_filename(path: Path) -> float | None:
    match = pattern.search(path.name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract(log_path: Path):
    best_ckpt = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "Best checkpoint:" in line:
                best_ckpt = line.split("Best checkpoint:", 1)[1].strip()
    if best_ckpt:
        resolved = resolve_checkpoint(best_ckpt, log_path)
        if resolved:
            loss = extract_value_from_filename(resolved)
            if loss is not None:
                return loss, resolved
    return None


def gather(root: Path):
    for run_dir in root.glob("training_run_*"):
        log_path = run_dir / "training.log"
        if log_path.exists():
            result = extract(log_path)
            if result:
                yield result

candidates = []
if run_root.exists():
    candidates.extend(gather(run_root))
if history_root.exists():
    candidates.extend(gather(history_root))

if not candidates:
    print("No completed training runs found.")
    sys.exit(0)

candidates.sort(key=lambda item: item[0])
print("Top checkpoints (lowest validation MSE):")
for rank, (loss, path) in enumerate(candidates[:3], start=1):
    print(f"{rank}. val_loss={loss} -> {path}")
PY
