#!/usr/bin/env bash
# Robust wrapper to run edge_bal_heavy_10A inference across datasets for top-3 checkpoints.
# Graphs are built once per dataset (first checkpoint), then reused for the remaining checkpoints.

set -euo pipefail
trap 'echo "Interrupted; aborting." >&2; exit 1' INT TERM

DATASETS=("BM55-AF2" "HAF2" "ABAG-AF3")
TOP_K=3                     # number of top checkpoints to use
export TOP_K
# RANK_MODE controls how top-K runs are chosen:
#   primary   = use each run's selection.primary_metric (default in training)
#   val_loss  = sort by best validation loss (lowest)
#   secondary = sort by val_spearman_corr (highest; see SECONDARY_METRIC below)
RANK_MODE="primary"
# When RANK_MODE=secondary, which metric to use:
#   val_spearman_corr (recommended; uses best val Spearman from selection history)
SECONDARY_METRIC="val_spearman_corr"
export RANK_MODE SECONDARY_METRIC
WORK_DIR="advanced_lean_10A_ARM_training2_inf_same"
APPEND_TIMESTAMP=false
REUSE_ONLY=true
export QTOPO_REUSE_ONLY="${REUSE_ONLY}"
ZERO_EDGE_OK=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
derive_work_dir() {
  local ckpt_path="$1"
  local base="$("${PYTHON_BIN}" - <<'PY' "$ckpt_path"
import pathlib, sys, torch
ckpt = sys.argv[1]
data = torch.load(ckpt, map_location="cpu")
meta = data.get("feature_metadata", {}) or {}
edge = meta.get("edge_schema", {}) or {}
src = edge.get("source")
name = None
if src:
    p = pathlib.Path(src)
    if p.name == "graph_metadata.json":
        name = p.parent.name  # graph_data
        if p.parent.parent:
            name = p.parent.parent.name  # builder output name
if not name:
    name = "inference_run"
print(name)
PY
)"
  if [[ "${APPEND_TIMESTAMP}" == "true" ]]; then
    local now
    now="$(date +%Y-%m-%d_%H-%M-%S)"
    echo "${base}_${now}"
  else
    echo "${base}"
  fi
}

# Detect python binary
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: PYTHON='${PYTHON}' does not point to an executable interpreter." >&2
    exit 1
  fi
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: could not locate a python interpreter (python3 or python)." >&2
    exit 1
  fi
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Derive WORK_DIR if not manually set
if [[ -z "${WORK_DIR}" ]]; then
  # Need PYTHON_BIN resolved and at least the top checkpoint path
  :
fi

# Fetch top checkpoints (score, path, run_name)
CHECKPOINT_ROWS=()
while IFS= read -r line; do
  CHECKPOINT_ROWS+=("$line")
done < <("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from qtdaqa.new_dynamic_features.model_training2 import train_cli

import os
top_k = int(os.environ.get("TOP_K", "3"))
rank_mode = os.environ.get("RANK_MODE", "primary").strip().lower()
secondary_metric = os.environ.get("SECONDARY_METRIC", "val_spearman_corr").strip().lower()

def _metric_direction(metric_name: str) -> str:
    return "max" if metric_name in train_cli.MAXIMIZE_METRICS else "min"

def _pick_alt_checkpoint(summary, run_dir: Path, metric_name: str):
    alt_map = summary.get("alternate_checkpoints") or {}
    alt_path = alt_map.get(metric_name)
    if alt_path:
        candidate = Path(alt_path)
        if candidate.exists():
            return str(candidate)
    if metric_name == "val_loss":
        best_link = run_dir / "model_checkpoints" / "val_loss_best.ckpt"
        if best_link.exists():
            return str(best_link.resolve())
        alt_dir = run_dir / "model_checkpoints" / "val_loss_checkpoints"
        if alt_dir.exists():
            candidates = sorted(alt_dir.glob("*.ckpt")) + sorted(alt_dir.glob("*.chkpt"))
            if candidates:
                best = None
                for path in candidates:
                    name = path.name
                    val = None
                    try:
                        marker = "val-"
                        if marker in name:
                            tail = name.split(marker, 1)[1]
                            val_str = tail.split("_", 1)[0]
                            val = float(val_str)
                    except Exception:
                        val = None
                    if val is None:
                        best = best or (float("inf"), path)
                        continue
                    if best is None or val < best[0]:
                        best = (val, path)
                if best:
                    return str(best[1])
    return None

ranked = []
for run_dir in sorted(train_cli.RUN_ROOT.iterdir()):
    if run_dir.is_symlink():
        continue
    if not run_dir.is_dir() or not (run_dir / "run_metadata.json").exists():
        continue
    summary = train_cli._summarise_run(run_dir)
    metric_name = None
    metric_value = None
    ckpt = None

    if rank_mode == "primary":
        metric_name, metric_value = train_cli._resolve_primary_metric_value(summary)
        ckpt = summary.get("best_checkpoint")
    elif rank_mode == "val_loss":
        metric_name = "val_loss"
        metric_value = summary.get("best_val_loss")
        ckpt = _pick_alt_checkpoint(summary, run_dir, "val_loss") or summary.get("best_checkpoint")
    elif rank_mode == "secondary":
        metric_name = secondary_metric
        if secondary_metric == "val_spearman_corr":
            metric_value = summary.get("best_selection_val_spearman")
        else:
            metric_value = summary.get(f"best_{secondary_metric}")
        ckpt = summary.get("best_checkpoint")
    else:
        metric_name, metric_value = train_cli._resolve_primary_metric_value(summary)
        ckpt = summary.get("best_checkpoint")

    if metric_value is None or not ckpt:
        continue
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        if rank_mode == "val_loss":
            ckpt = _pick_alt_checkpoint(summary, run_dir, "val_loss")
            ckpt_path = Path(ckpt) if ckpt else None
        if not ckpt_path or not ckpt_path.exists():
            continue
    ranked.append((metric_name, float(metric_value), summary, ckpt))

if not ranked:
    raise SystemExit("No runs with usable metrics found.")

direction = _metric_direction(ranked[0][0]) if ranked else "min"
ranked.sort(key=lambda item: (-item[1] if direction == "max" else item[1]))

for metric_name, score, summary, ckpt in ranked[:top_k]:
    run_name = summary.get("run_name") or ""
    print(f"{score}\t{ckpt}\t{run_name}")
PY
)

if [[ ${#CHECKPOINT_ROWS[@]} -eq 0 ]]; then
  echo "Error: no checkpoints found from leaderboard." >&2
  exit 1
fi

# Parse checkpoints into arrays
CHECKPOINT_PATHS=()
CHECKPOINT_IDS=()
for ROW in "${CHECKPOINT_ROWS[@]}"; do
  score="${ROW%%$'\t'*}"
  rest="${ROW#*$'\t'}"
  ckpt_path="${rest%%$'\t'*}"
  [[ -z "${ckpt_path}" ]] && continue
  CHECKPOINT_PATHS+=("${ckpt_path}")
  base="$(basename "${ckpt_path}")"
  id="${base%.chkpt}"
  id="${id#checkpoint.}"
  CHECKPOINT_IDS+=("${id}")
done

if [[ ${#CHECKPOINT_PATHS[@]} -eq 0 ]]; then
  echo "Error: no valid checkpoint paths parsed." >&2
  exit 1
fi

if [[ -z "${WORK_DIR}" ]]; then
  WORK_DIR="$(derive_work_dir "${CHECKPOINT_PATHS[0]}")"
fi

# Build output roots. If WORK_DIR is absolute, use it as-is; otherwise root under script/output.
if [[ "${WORK_DIR}" = /* ]]; then
  OUT_ROOT="${WORK_DIR}"
else
  OUT_ROOT="${SCRIPT_DIR}/output/${WORK_DIR}"
fi
WORK_ROOT="${OUT_ROOT}/work"       # shared per dataset to reuse graphs
RESULTS_ROOT="${OUT_ROOT}/results" # checkpoint-specific subdirs live under here

mkdir -p "${WORK_ROOT}" "${RESULTS_ROOT}"

echo "Using checkpoints:"
for idx in "${!CHECKPOINT_PATHS[@]}"; do
  echo "  [$((idx+1))] ${CHECKPOINT_IDS[$idx]} -> ${CHECKPOINT_PATHS[$idx]}"
done

# Helper: preflight reuse check for a dataset/ckpt
precheck_reuse() {
  local dataset="$1"
  local ckpt_path="$2"
  local graph_dir="${WORK_ROOT}/${dataset}/graph_data"
  "${PYTHON_BIN}" - <<'PY' "${graph_dir}" "${ckpt_path}"
import os, sys, torch
from pathlib import Path
from qtdaqa.new_dynamic_features.model_inference import builder_runner as br

graph_dir = Path(sys.argv[1])
ckpt_path = Path(sys.argv[2])
if not graph_dir.exists():
    print(f"Graph dir missing: {graph_dir}")
    sys.exit(1)
# Look recursively for .pt files (graph_builder writes per-model subdirs)
if not any(graph_dir.rglob("*.pt")):
    print(f"No .pt files in graph dir: {graph_dir}")
    sys.exit(1)

# If reuse-only is enabled, skip schema check and trust existing graphs.
reuse_only = os.environ.get("QTOPO_REUSE_ONLY", "").strip().lower() in {"1", "true", "yes"}
if reuse_only:
    print("REUSE_OK (reuse-only mode)")
    sys.exit(0)

ckpt = torch.load(ckpt_path, map_location="cpu")
meta = ckpt.get("feature_metadata") or {}
edge = meta.get("edge_schema") or {}
topo = meta.get("topology_schema") or {}
node = meta.get("node_schema") or {}
final = {"edge_schema": edge, "topology_schema": topo, "node_schema": node}

ok, reason = br._graph_metadata_matches(graph_dir, final)  # reuse-friendly check
if ok:
    print("REUSE_OK")
    sys.exit(0)
print(f"REUSE_FAIL: {reason}")
sys.exit(1)
PY
}

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
    --results-dir "${RESULTS_ROOT}/${ckpt_id}" \
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
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  CKPT_PATH="${CHECKPOINT_PATHS[0]}"
  CKPT_ID="${CHECKPOINT_IDS[0]}"
  if ! run_schema_preflight "${DATASET}" "${CKPT_PATH}" "${CKPT_ID}"; then
    exit 1
  fi
done
echo "Schema preflight complete."

PIDS_STAGE1=()
STATUS_STAGE1=()

# Stage 1: one checkpoint per dataset to build graphs (run #0 for each)
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  CKPT_PATH="${CHECKPOINT_PATHS[0]}"
  CKPT_ID="${CHECKPOINT_IDS[0]}"
  LOG_FILE="${DATASET}_${CKPT_ID}_${WORK_DIR}_$(date +%Y%m%d_%H%M%S)_stage1.log"

  echo "Stage1: ${DATASET} using ${CKPT_ID} (build graphs + inference)"
  mkdir -p "${RESULTS_ROOT}/${CKPT_ID}"
  set -o pipefail
  env QTOPO_REUSE_ONLY="${REUSE_ONLY}" \
  time "${SCRIPT_DIR}/run_model_inference.sh" \
    --dataset-name "${DATASET}" \
    --config "${SCRIPT_DIR}/config.yaml.${DATASET}" \
    --checkpoint-path "${CKPT_PATH}" \
    --work-dir "${WORK_ROOT}" \
    --results-dir "${RESULTS_ROOT}/${CKPT_ID}" \
    --reuse-existing-graphs \
    --log-level INFO \
    2>&1 | tee "${LOG_FILE}" &
  PIDS_STAGE1[idx_ds]=$!
  STATUS_STAGE1[idx_ds]="pending"
  set +o pipefail
done

echo "Waiting for Stage1 (graph build) to complete ..."
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  PID=${PIDS_STAGE1[idx_ds]}
  if wait "${PID}"; then
    STATUS_STAGE1[idx_ds]="success"
  else
    STATUS_STAGE1[idx_ds]="failed"
    echo "Stage1 failed for ${DATASET} (PID ${PID}); skipping remaining checkpoints." >&2
  fi
done

# After Stage1, run zero-edge checks on built graphs (requires graph_metadata.json)
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

# Stage 2: remaining checkpoints per dataset, serialized per dataset, reuse only
STATUS_STAGE2_KEYS=()
STATUS_STAGE2_VALUES=()

for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  [[ ${STATUS_STAGE1[idx_ds]} != "success" ]] && continue
  for idx in "${!CHECKPOINT_PATHS[@]}"; do
    [[ ${idx} -eq 0 ]] && continue  # already ran
    CKPT_PATH="${CHECKPOINT_PATHS[$idx]}"
    CKPT_ID="${CHECKPOINT_IDS[$idx]}"

    echo "Stage2 precheck: ${DATASET} with ${CKPT_ID}"
    if ! precheck_reuse "${DATASET}" "${CKPT_PATH}"; then
      echo "Stage2 blocked for ${DATASET} with ${CKPT_ID}: reuse precheck failed (would rebuild). Fix schema/match before rerunning." >&2
      exit 1
    fi

    LOG_FILE="${DATASET}_${CKPT_ID}_${WORK_DIR}_$(date +%Y%m%d_%H%M%S)_stage2.log"
    echo "Stage2: ${DATASET} using ${CKPT_ID} (reuse graphs, serialized)"
    mkdir -p "${RESULTS_ROOT}/${CKPT_ID}"
    set -o pipefail
    if env QTOPO_REUSE_ONLY="${REUSE_ONLY}" time "${SCRIPT_DIR}/run_model_inference.sh" \
      --dataset-name "${DATASET}" \
      --config "${SCRIPT_DIR}/config.yaml.${DATASET}" \
      --checkpoint-path "${CKPT_PATH}" \
      --work-dir "${WORK_ROOT}" \
      --results-dir "${RESULTS_ROOT}/${CKPT_ID}" \
      --reuse-existing-graphs \
      --log-level INFO \
      2>&1 | tee "${LOG_FILE}"; then
      STATUS_STAGE2_KEYS+=("${DATASET}|${CKPT_ID}")
      STATUS_STAGE2_VALUES+=("success")
    else
      STATUS_STAGE2_KEYS+=("${DATASET}|${CKPT_ID}")
      STATUS_STAGE2_VALUES+=("failed")
      echo "Stage2 failed for ${DATASET} with ${CKPT_ID}; stopping." >&2
      exit 1
    fi
    set +o pipefail
  done
done

echo "Inference complete - Running final results ..."
for idx_ds in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx_ds]}"
  for idx in "${!CHECKPOINT_PATHS[@]}"; do
    # stage1 status for idx=0, stage2 status for idx>0
    status="failed"
    if [[ ${idx} -eq 0 ]]; then
      status="${STATUS_STAGE1[$idx_ds]:-failed}"
    else
      status="failed"
      key="${DATASET}|${CHECKPOINT_IDS[$idx]}"
      for i in "${!STATUS_STAGE2_KEYS[@]}"; do
        if [[ "${STATUS_STAGE2_KEYS[$i]}" == "${key}" ]]; then
          status="${STATUS_STAGE2_VALUES[$i]}"
          break
        fi
      done
    fi
    [[ "${status}" != "success" ]] && continue

    CKPT_ID="${CHECKPOINT_IDS[$idx]}"
    RESULTS_DIR="${RESULTS_ROOT}/${CKPT_ID}/${DATASET}"
    "${SCRIPT_DIR}/run_results_summary.sh" --results-dir "${RESULTS_DIR}"
  done
done
