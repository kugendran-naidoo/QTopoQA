#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

RUN_ROOT="${SCRIPT_DIR}/training_runs"
PHASE1_CONFIG="configs/sched_boost_finetune.yaml"
PHASE2_SEEDS=(101 555 888)

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage: ./run_fine_tune_only.sh [--checkpoint PATH] [--run-dir DIR] [--help]

Without arguments, the script scans training_runs/ (and training_runs/history/)
for the run with the lowest validation loss and uses its best checkpoint.

Optional arguments:
  --checkpoint PATH   Explicit checkpoint to resume from; skips auto-discovery.
  --run-dir DIR       Explicit run directory name (under training_runs/) to
                      label the fine-tune jobs. Required if --checkpoint is used.
  --help              Show this message.
EOF
}

BEST_CKPT=""
BEST_RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      BEST_CKPT="$2"
      shift 2
      ;;
    --run-dir)
      BEST_RUN_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${BEST_CKPT}" ]]; then
  BEST_INFO="$(python - <<'PY' "${RUN_ROOT}" "${REPO_ROOT}"
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(str(repo_root))))

from qtdaqa.new_dynamic_features.model_training import train_cli

search_dirs = []
if run_root.exists():
    search_dirs.extend(p for p in run_root.iterdir() if p.is_dir())
history_root = run_root / "history"
if history_root.exists():
    search_dirs.extend(p for p in history_root.iterdir() if p.is_dir())

candidates = []
for directory in search_dirs:
    meta_path = directory / "run_metadata.json"
    if not meta_path.exists():
        continue
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    summary = train_cli._summarise_run(directory)
    best_loss = summary.get("best_val_loss")
    best_ckpt = summary.get("best_checkpoint")
    if best_loss is None or best_ckpt is None:
        continue
    candidates.append((float(best_loss), best_ckpt, str(directory)))

if not candidates:
    print("ERROR: No eligible runs found in training_runs/.", file=sys.stderr)
    sys.exit(1)

candidates.sort()
best_loss, best_ckpt, best_run = candidates[0]
print(best_ckpt)
print(best_run)
print(best_loss)
PY
)"
  if [[ -z "${BEST_INFO}" ]]; then
    echo "[run_fine_tune_only] Unable to identify best checkpoint." >&2
    exit 1
  fi

  BEST_CKPT="$(echo "${BEST_INFO}" | sed -n '1p')"
  BEST_RUN_DIR="$(echo "${BEST_INFO}" | sed -n '2p')"
  BEST_LOSS="$(echo "${BEST_INFO}" | sed -n '3p')"
  echo "[run_fine_tune_only] Auto-selected best run ${BEST_RUN_DIR} (val_loss=${BEST_LOSS})"
  echo "[run_fine_tune_only] Best checkpoint: ${BEST_CKPT}"
else
  if [[ -z "${BEST_RUN_DIR}" ]]; then
    echo "[run_fine_tune_only] --run-dir is required when --checkpoint is supplied." >&2
    exit 1
  fi
  if [[ ! -d "${BEST_RUN_DIR}" ]]; then
    if [[ -d "${RUN_ROOT}/${BEST_RUN_DIR}" ]]; then
      BEST_RUN_DIR="${RUN_ROOT}/${BEST_RUN_DIR}"
    elif [[ -d "${RUN_ROOT}/history/${BEST_RUN_DIR}" ]]; then
      BEST_RUN_DIR="${RUN_ROOT}/history/${BEST_RUN_DIR}"
    else
      echo "[run_fine_tune_only] Run directory ${BEST_RUN_DIR} not found." >&2
      exit 1
    fi
  fi
  echo "[run_fine_tune_only] Using user-provided checkpoint: ${BEST_CKPT}"
  echo "[run_fine_tune_only] Run directory label: ${BEST_RUN_DIR}"
fi

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "[run_fine_tune_only] Checkpoint not found at ${BEST_CKPT}" >&2
  exit 1
fi

BEST_RUN_NAME="$(basename "${BEST_RUN_DIR}")"
PHASE1_RUN_NAME="${BEST_RUN_NAME}_finetune_phase1"

echo "[run_fine_tune_only] Launching Phase 1 fine-tune -> ${PHASE1_RUN_NAME}"
python -m train_cli run \
  --config "${PHASE1_CONFIG}" \
  --run-name "${PHASE1_RUN_NAME}" \
  --resume-from "${BEST_CKPT}"

for seed in "${PHASE2_SEEDS[@]}"; do
  PHASE2_CONFIG="${SCRIPT_DIR}/configs/sched_boost_finetune_seed${seed}.yaml"
  if [[ ! -f "${PHASE2_CONFIG}" ]]; then
    echo "[run_fine_tune_only] Skipping seed ${seed} (missing config: ${PHASE2_CONFIG})" >&2
    continue
  fi
  RUN_NAME="${PHASE1_RUN_NAME}_seed${seed}"
  echo "[run_fine_tune_only] Launching Phase 2 fine-tune (seed ${seed}) -> ${RUN_NAME}"
  python -m train_cli run \
    --config "${PHASE2_CONFIG}" \
    --run-name "${RUN_NAME}" \
    --resume-from "${BEST_CKPT}"
done

ISO_TS="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date '+%Y-%m-%dT%H:%M:%SZ')"
echo "[run_fine_tune_only] Fine-tuning stages completed at ${ISO_TS}"
