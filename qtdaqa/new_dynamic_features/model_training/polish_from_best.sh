#!/usr/bin/env bash
# Helper to pick the best sweep run and generate a polish manifest with heavier settings.
# Usage:
#   RUN_ROOT=/path/to/training_runs \  # optional (defaults to ./training_runs)
#   MANIFEST_PATH=/path/to/run_polish_heavy.yaml \  # optional
#   LAUNCH=1 \  # optional; if set, will invoke run_full_pipeline.sh with the manifest
#   ./polish_from_best.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/training_runs}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/manifests/run_polish_heavy.yaml}"
LAUNCH="${LAUNCH:-0}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export RUN_ROOT

BEST_INFO=$(python - <<'PY'
import json
import os
from pathlib import Path

from qtdaqa.new_dynamic_features.model_training import train_cli

run_root = Path(os.environ.get("RUN_ROOT", "")).resolve()
if not run_root.exists():
    print(json.dumps({"error": f"run root not found: {run_root}"}))
    raise SystemExit(1)

ranked = train_cli.rank_runs(run_root)
if not ranked:
    print(json.dumps({"error": f"no runs found under {run_root}"}))
    raise SystemExit(1)

_metric, _value, summary = ranked[0]
run_dir = Path(summary["run_dir"]).resolve()
meta_path = run_dir / "run_metadata.json"
source_config = None
seed = None
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_config = meta.get("source_config")
    tp = meta.get("training_parameters")
    if isinstance(tp, dict):
        seed = tp.get("seed")
if source_config is None:
    source_config = summary.get("run_metadata", {}).get("source_config")

print(json.dumps({
    "run_dir": str(run_dir),
    "source_config": source_config,
    "seed": seed
}))
PY
)
export BEST_INFO

if echo "${BEST_INFO}" | grep -q '"error"'; then
  echo "[polish_from_best] ${BEST_INFO}" >&2
  exit 1
fi

# Extract fields
BEST_RUN_DIR=$(python - <<'PY'
import json, os
data = json.loads(os.environ.get("BEST_INFO", "{}"))
print(data.get("run_dir",""))
PY)
BEST_CONFIG=$(python - <<'PY'
import json, os
data = json.loads(os.environ.get("BEST_INFO", "{}"))
print(data.get("source_config",""))
PY)
BEST_SEED=$(python - <<'PY'
import json, os
data = json.loads(os.environ.get("BEST_INFO", "{}"))
seed = data.get("seed")
print("" if seed is None else seed)
PY)

if [[ -z "${BEST_CONFIG}" ]]; then
  echo "[polish_from_best] ERROR: could not determine source_config for best run (${BEST_RUN_DIR})" >&2
  exit 1
fi

# Resolve config to absolute path for safety
if [[ "${BEST_CONFIG}" != /* ]]; then
  BEST_CONFIG="${SCRIPT_DIR}/${BEST_CONFIG}"
fi
if [[ ! -f "${BEST_CONFIG}" ]]; then
  echo "[polish_from_best] ERROR: source_config not found: ${BEST_CONFIG}" >&2
  exit 1
fi

# Assemble seeds (best seed + extras)
SEEDS=()
if [[ -n "${BEST_SEED}" ]]; then
  SEEDS+=("${BEST_SEED}")
fi
for s in 101 555 888; do
  SEEDS+=("${s}")
done
# de-duplicate
SEEDS=($(printf "%s\n" "${SEEDS[@]}" | awk '!x[$0]++'))

mkdir -p "$(dirname "${MANIFEST_PATH}")"

cat > "${MANIFEST_PATH}" <<EOF
shared:
  notes: "Auto-generated polish manifest from best run: ${BEST_RUN_DIR}"
  log_lr: false
  fast_dev_run: false

jobs:
EOF

for seed in "${SEEDS[@]}"; do
  cat >> "${MANIFEST_PATH}" <<EOF
  - name: polish_seed${seed}
    config: ${BEST_CONFIG}
    overrides:
      trainer.num_epochs: 320
      early_stopping.patience: 40
      scheduler.patience: 20
      scheduler.factor: 0.3
      optimizer.learning_rate: 3e-3
      dataloader.seed: ${seed}

  - name: polish_lr_low_seed${seed}
    config: ${BEST_CONFIG}
    overrides:
      trainer.num_epochs: 360
      early_stopping.patience: 50
      scheduler.patience: 25
      scheduler.factor: 0.3
      optimizer.learning_rate: 2e-3
      dataloader.batch_size: 12
      trainer.accumulate_grad_batches: 48
      dataloader.seed: ${seed}
EOF
done

cat >> "${MANIFEST_PATH}" <<'EOF'

continue_on_error: false
EOF

echo "[polish_from_best] Manifest written to ${MANIFEST_PATH}"
echo "[polish_from_best] Best run: ${BEST_RUN_DIR}"
echo "[polish_from_best] Base config: ${BEST_CONFIG}"
echo "[polish_from_best] Seeds: ${SEEDS[*]}"

if [[ "${LAUNCH}" != "0" ]]; then
  echo "[polish_from_best] Launching run_full_pipeline.sh with ${MANIFEST_PATH}"
  GRAPH_DIR="${GRAPH_DIR:-}" TMPDIR="${TMPDIR:-}" TEMP="${TEMP:-}" TMP="${TMP:-}" \
    "${SCRIPT_DIR}/run_full_pipeline.sh" --manifest "${MANIFEST_PATH}"
fi
