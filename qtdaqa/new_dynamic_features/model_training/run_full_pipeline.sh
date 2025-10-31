#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

RUN_ROOT="${SCRIPT_DIR}/training_runs"
MANIFEST="manifests/run_all.yaml"
PHASE1_CONFIG="configs/sched_boost_finetune.yaml"
PHASE2_SEEDS=(101 555 888)

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found at ${MANIFEST}" >&2
  exit 1
fi

START_DATA="$(python - <<'PY'
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
print(now.isoformat(), int(now.timestamp()))
PY
)"
read -r START_ISO START_TS <<< "${START_DATA}"

echo "[run_full_pipeline] Starting sweep at ${START_ISO}"
python -m train_cli batch --manifest "${MANIFEST}"

BEST_INFO="$(python - <<'PY' "${START_TS}" "${RUN_ROOT}"
import json
import sys
from pathlib import Path
from datetime import datetime

from qtdaqa.new_dynamic_features.model_training import train_cli

start_ts = int(sys.argv[1])
run_root = Path(sys.argv[2])
start_dt = datetime.fromtimestamp(start_ts)

def iter_runs(root: Path):
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir() and (p / "run_metadata.json").exists()]

candidates = []
for directory in iter_runs(run_root) + iter_runs(run_root / "history"):
    meta_path = directory / "run_metadata.json"
    if not meta_path.exists():
        continue
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    created = meta.get("created")
    if not created:
        continue
    try:
        created_dt = datetime.fromisoformat(created)
    except ValueError:
        continue
    if created_dt < start_dt:
        continue
    summary = train_cli._summarise_run(directory)
    best_loss = summary.get("best_val_loss")
    best_ckpt = summary.get("best_checkpoint")
    if best_loss is None or best_ckpt is None:
        continue
    candidates.append((float(best_loss), best_ckpt, str(directory)))

if not candidates:
    print("ERROR: No new training runs detected after sweep.", file=sys.stderr)
    sys.exit(1)

candidates.sort()
best_loss, best_ckpt, best_run = candidates[0]
print(best_ckpt)
print(best_run)
print(best_loss)
PY
)"

if [[ -z "${BEST_INFO}" ]]; then
  echo "[run_full_pipeline] Unable to identify best checkpoint." >&2
  exit 1
fi

BEST_CKPT="$(echo "${BEST_INFO}" | sed -n '1p')"
BEST_RUN_DIR="$(echo "${BEST_INFO}" | sed -n '2p')"
BEST_LOSS="$(echo "${BEST_INFO}" | sed -n '3p')"

echo "[run_full_pipeline] Best run: ${BEST_RUN_DIR} (val_loss=${BEST_LOSS})"
echo "[run_full_pipeline] Best checkpoint: ${BEST_CKPT}"

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "[run_full_pipeline] Best checkpoint path not found: ${BEST_CKPT}" >&2
  exit 1
fi

PHASE1_RUN_NAME="$(basename "${BEST_RUN_DIR}")_finetune_phase1"

echo "[run_full_pipeline] Phase 1 fine-tuning -> ${PHASE1_RUN_NAME}"
python -m train_cli run \
  --config "${PHASE1_CONFIG}" \
  --run-name "${PHASE1_RUN_NAME}" \
  --resume-from "${BEST_CKPT}"

for seed in "${PHASE2_SEEDS[@]}"; do
  PHASE2_CONFIG="${SCRIPT_DIR}/configs/sched_boost_finetune_seed${seed}.yaml"
  if [[ ! -f "${PHASE2_CONFIG}" ]]; then
    echo "[run_full_pipeline] Skipping seed ${seed} (config not found: ${PHASE2_CONFIG})" >&2
    continue
  fi
  RUN_NAME="${PHASE1_RUN_NAME}_seed${seed}"
  echo "[run_full_pipeline] Phase 2 fine-tuning (seed ${seed}) -> ${RUN_NAME}"
  python -m train_cli run \
    --config "${PHASE2_CONFIG}" \
    --run-name "${RUN_NAME}" \
    --resume-from "${BEST_CKPT}"
done

echo "[run_full_pipeline] Pipeline complete at $(date --iso-8601=seconds)"
