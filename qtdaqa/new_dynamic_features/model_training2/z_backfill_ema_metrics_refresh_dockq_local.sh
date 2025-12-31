#!/usr/bin/env bash
set -euo pipefail

# Forced EMA backfill (refresh) to add tuning_dockq_mae + tuning_hit_rate_023.
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_ROOT="/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/model_training2/training_runs2"
export PYTHONPATH="/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/model_training2:/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA"

echo "[ema_backfill] refresh (force) -> ${TRAINING_ROOT}"
"${PYTHON_BIN}" -m qtdaqa.new_dynamic_features.model_training2.tools.ema_backfill_eval \
  --training-root "${TRAINING_ROOT}" \
  --force

# One-line verify: print first run + ema_metrics keys (if any).
PYTHON_BIN="${PYTHON_BIN}" TRAINING_ROOT="${TRAINING_ROOT}" \
  "${PYTHON_BIN}" -c 'import json,os;from pathlib import Path;root=Path(os.environ["TRAINING_ROOT"]);runs=[p for p in root.iterdir() if p.is_dir() and (p/"run_metadata.json").exists()];run=sorted(runs)[0] if runs else None;print("verify:", run or "no runs");print(sorted((json.loads((run/"run_metadata.json").read_text()).get("ema_metrics") or {}).keys()) if run else "no runs")'
