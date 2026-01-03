#!/usr/bin/env bash
# Run inference for Option B + EMA checkpoint.
set -euo pipefail
trap 'echo "Interrupted; aborting." >&2; exit 1' INT TERM

DATASETS=("BM55-AF2" "HAF2" "ABAG-AF3")
TOP_K="${TOP_K:-10}"
# Options: best_val_loss, ema_val_loss
SHORTLIST_METRIC="${SHORTLIST_METRIC:-ema_val_loss}"
# TUNING_METRIC="${TUNING_METRIC:-ema_tuning_rank_spearman}"
# TUNING_METRIC="${TUNING_METRIC:-ema_tuning_dockq_mae}"
TUNING_METRIC="${TUNING_METRIC:-ema_tuning_hit_rate_023}"
WORK_DIR="${WORK_DIR:-null_topology_10A_ARM_optionB_hit_rate_023}"
REUSE_ONLY="${REUSE_ONLY:-true}"
export QTOPO_REUSE_ONLY="${REUSE_ONLY}"
ZERO_EDGE_OK="${ZERO_EDGE_OK:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_NAME="$("${PYTHON_BIN}" -m qtdaqa.new_dynamic_features.model_training2.tools.option_b_select \
  --top-k "${TOP_K}" \
  --shortlist-metric "${SHORTLIST_METRIC}" \
  --tuning-metric "${TUNING_METRIC}" | sed -n 's/^run=//p' | tail -n 1)"

if [[ -z "${RUN_NAME}" ]]; then
  echo "Error: option_b_select returned empty run name." >&2
  exit 1
fi

CKPT_PATH="${REPO_ROOT}/qtdaqa/new_dynamic_features/model_training2/training_runs2/${RUN_NAME}/model_checkpoints/averaged_ema.chkpt"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Error: EMA checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

CKPT_ID="ema_${RUN_NAME}"

OUT_ROOT="${SCRIPT_DIR}/output/${WORK_DIR}"
WORK_ROOT="${OUT_ROOT}/work"
RESULTS_ROOT="${OUT_ROOT}/results/${CKPT_ID}"
mkdir -p "${WORK_ROOT}" "${RESULTS_ROOT}"

echo "Option B EMA checkpoint: ${CKPT_PATH}"

run_schema_preflight() {
  local dataset="$1"
  local ckpt_path="$2"
  local ckpt_id="$3"
  local log_file="${dataset}_${ckpt_id}_${WORK_DIR}_preflight.log"
  echo "Preflight (schema/build) for ${dataset} with ${ckpt_id}"
  set -o pipefail
  if ! env QTOPO_REUSE_ONLY="${REUSE_ONLY}" "${SCRIPT_DIR}/run_model_inference.sh" \
    --dataset-name "${dataset}" \
    --config "${SCRIPT_DIR}/config.yaml.${dataset}" \
    --checkpoint-path "${ckpt_path}" \
    --work-dir "${WORK_ROOT}" \
    --results-dir "${RESULTS_ROOT}" \
    --reuse-existing-graphs \
    --check-schema \
    --log-level INFO \
    > "${log_file}" 2>&1; then
    echo "Preflight failed for ${dataset} with ${ckpt_id}. See ${log_file}" >&2
    set +o pipefail
    return 1
  fi
  set +o pipefail
}

check_zero_edges() {
  local dataset="$1"
  local graph_meta="${WORK_ROOT}/${dataset}/graph_data/graph_metadata.json"
  local allow_zero="${ZERO_EDGE_OK:-}"
  "${PYTHON_BIN}" - <<'PY' "${graph_meta}" "${dataset}" "${allow_zero}"
import json, sys
from pathlib import Path
graph_meta_path = Path(sys.argv[1])
dataset = sys.argv[2]
allow = sys.argv[3].strip().lower() in {"1", "true", "yes"}
if not graph_meta_path.exists():
    print(f"[zero-edge-check] graph_metadata.json not found for {dataset}: {graph_meta_path}")
    sys.exit(1)
try:
    data = json.loads(graph_meta_path.read_text())
except Exception as exc:
    print(f"[zero-edge-check] failed to read {graph_meta_path}: {exc}")
    sys.exit(1)
zero = 0
total = 0
min_edges = None
max_edges = None
for key, entry in data.items():
    if key.startswith("_") or not isinstance(entry, dict) or "edge_metadata" not in entry:
        continue
    total += 1
    cnt = (entry.get("edge_metadata") or {}).get("edge_count", 0)
    if min_edges is None or cnt < min_edges:
        min_edges = cnt
    if max_edges is None or cnt > max_edges:
        max_edges = cnt
    if cnt == 0:
        zero += 1
print(f"[zero-edge-check] {dataset}: graphs={total} zero_edge={zero} min_edges={min_edges} max_edges={max_edges}")
if zero > 0 and not allow:
    print("[zero-edge-check] zero-edge graphs detected; set ZERO_EDGE_OK=1 to continue anyway.")
    sys.exit(2)
sys.exit(0)
PY
}

echo "Running schema preflight (no graph build; checks compatibility only)..."
for DATASET in "${DATASETS[@]}"; do
  if ! run_schema_preflight "${DATASET}" "${CKPT_PATH}" "${CKPT_ID}"; then
    exit 1
  fi
done
echo "Schema preflight complete."

PIDS_STAGE1=()
STATUS_STAGE1=()

for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  LOG_FILE="${DATASET}_${CKPT_ID}_${WORK_DIR}_$(date +%Y%m%d_%H%M%S).log"
  echo "Running ${DATASET} -> ${LOG_FILE}"
  set -o pipefail
  env QTOPO_REUSE_ONLY="${REUSE_ONLY}" \
  time "${SCRIPT_DIR}/run_model_inference.sh" \
    --dataset-name "${DATASET}" \
    --config "${SCRIPT_DIR}/config.yaml.${DATASET}" \
    --checkpoint-path "${CKPT_PATH}" \
    --work-dir "${WORK_ROOT}" \
    --results-dir "${RESULTS_ROOT}" \
    --reuse-existing-graphs \
    --log-level INFO \
    2>&1 | tee "${LOG_FILE}" &
  PIDS_STAGE1[idx_ds]=$!
  STATUS_STAGE1[idx_ds]="pending"
  set +o pipefail
done

echo "Waiting for dataset runs to complete ..."
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  PID=${PIDS_STAGE1[idx_ds]}
  if wait "${PID}"; then
    STATUS_STAGE1[idx_ds]="success"
  else
    STATUS_STAGE1[idx_ds]="failed"
    echo "Run failed for ${DATASET} (PID ${PID})." >&2
  fi
done

echo "Running zero-edge checks on built graphs..."
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  [[ ${STATUS_STAGE1[idx_ds]} != "success" ]] && continue
  if ! check_zero_edges "${DATASET}"; then
    echo "Zero-edge check failed for ${DATASET}; aborting." >&2
    exit 1
  fi
done
echo "Zero-edge checks complete."

echo "Inference complete - Running final results ..."
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  [[ ${STATUS_STAGE1[idx_ds]} != "success" ]] && continue
  RESULTS_DIR="${RESULTS_ROOT}/${DATASET}"
  if [[ -d "${RESULTS_DIR}" ]]; then
    "${SCRIPT_DIR}/run_results_summary.sh" --results-dir "${RESULTS_DIR}"
  else
    echo "Warning: results dir not found for ${DATASET}: ${RESULTS_DIR}" >&2
  fi
done

echo "Inference complete; results under ${RESULTS_ROOT}"
