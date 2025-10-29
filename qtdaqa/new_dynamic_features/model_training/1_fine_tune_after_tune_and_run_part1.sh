#!/usr/bin/env bash
# set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

cfg_path="configs/sched_boost_finetune.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

locate_best_checkpoint() {
  python - "$SCRIPT_DIR" <<'PY'
import sys
import re
from pathlib import Path

script_dir = Path(sys.argv[1])
run_root = script_dir / "training_runs"
history_root = run_root / "history"

def resolve_checkpoint(path_str: str, log_path: Path) -> Path | None:
    raw_path = Path(path_str)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((log_path.parent / raw_path).resolve())
    # Always try the current run directory in case the run was moved to history.
    candidates.append((log_path.parent / "model_checkpoints" / raw_path.name).resolve())
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    for candidate in ordered:
        if candidate.exists():
            return candidate
    return None

def extract_best(log_path: Path):
    best_ckpt = None
    best_loss = None
    pattern = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "Best checkpoint:" in line:
                best_ckpt = line.split("Best checkpoint:", 1)[1].strip()
            elif "Best validation loss:" in line:
                candidate = line.split("Best validation loss:", 1)[1]
                match = pattern.search(candidate)
                if match:
                    best_loss = float(match.group(0))
    if best_ckpt and best_loss is not None:
        ckpt_path = resolve_checkpoint(best_ckpt, log_path)
        if ckpt_path is not None:
            return best_loss, ckpt_path
    return None

def gather(root: Path):
    for run_dir in root.glob("training_run_*"):
        log_path = run_dir / "training.log"
        if log_path.exists():
            result = extract_best(log_path)
            if result:
                yield result

candidates = []
if run_root.exists():
    candidates.extend(gather(run_root))
if history_root.exists():
    candidates.extend(gather(history_root))

if not candidates:
    print("No completed training runs with checkpoints found under training_runs/.", file=sys.stderr)
    sys.exit(1)

candidates.sort(key=lambda item: item[0])
print(candidates[0][1])
PY
}

CKPT_PATH="$(locate_best_checkpoint)" || exit 1

CACHE_DIR="${SCRIPT_DIR}/fine_tune_cache"
mkdir -p "${CACHE_DIR}"
CKPT_BASENAME="$(basename "${CKPT_PATH}")"
STABLE_CKPT_PATH="${CACHE_DIR}/phase1_${CKPT_BASENAME}"
cp "${CKPT_PATH}" "${STABLE_CKPT_PATH}"
CKPT_PATH="${STABLE_CKPT_PATH}"

printf "Selected checkpoint (cached) = %s\n" "${CKPT_PATH}"

printf "=== Fine-tuning Phase 1 ===\n"
CONFIG_ABS="${SCRIPT_DIR}/${cfg_path}"
if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "Config not found: ${CONFIG_ABS}" >&2
  exit 1
fi
exec "${PYTHON_BIN}" -m train_cli run --config "${CONFIG_ABS}" --resume-from "${CKPT_PATH}" "$@"
