#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# TopoQA batch runner (portable bash version)
# -----------------------------------------------------------------------------
set -euo pipefail
trap 'rc=$?; echo "[debug] EXIT trap: status=$rc processed=${processed:-0} total=${total:-0}" >&2' EXIT
shopt -s nullglob

usage() {
  cat <<'USAGE'
Usage: graphs_training_MAF2.bash -n DATASET_NAME -d DATASET_DIR -g GROUND_TRUTH_FILE
   or: graphs_training_MAF2.bash --dataset-name NAME --dataset-dir DIR --ground-truth-file FILE
All three parameters are mandatory (no defaults).
USAGE
}

DATASET_NAME="${DATASET_NAME:-Dockground_MAF2}"
DATASET_DIR="${DATASET_DIR:-../../../../datasets/training/Dockground_MAF2}"
GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-../../../../datasets/training/Dockground_MAF2/label_info_Dockground_MAF2.csv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dataset-name)
      [[ $# -ge 2 ]] || { echo "ERROR: missing argument for $1" >&2; exit 64; }
      DATASET_NAME="$2"; shift 2 ;;
    -d|--dataset-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: missing argument for $1" >&2; exit 64; }
      DATASET_DIR="$2"; shift 2 ;;
    -g|--ground-truth-file)
      [[ $# -ge 2 ]] || { echo "ERROR: missing argument for $1" >&2; exit 64; }
      GROUND_TRUTH_FILE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    -*)
      printf 'Unknown option: %s\n' "$1" >&2
      usage
      exit 64 ;;
    *)
      break ;;
  esac
done

: "${DATASET_NAME:?ERROR: DATASET_NAME is required (-n)}"
: "${DATASET_DIR:?ERROR: DATASET_DIR is required (-d)}"
: "${GROUND_TRUTH_FILE:?ERROR: GROUND_TRUTH_FILE is required (-g)}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
INF_SCRIPT="${INF_SCRIPT:-${script_dir}/lib/lib_mac_run_inference.sh}"
RESULTS_SCRIPT="${RESULTS_SCRIPT:-${script_dir}/lib/lib_mac_run_results_target.sh}"

if [[ ! -x "$INF_SCRIPT" ]]; then
  echo "ERROR: INFERENCE script not found or not executable: $INF_SCRIPT" >&2
  exit 3
fi

if [[ ! -r "$GROUND_TRUTH_FILE" ]]; then
  echo "ERROR: GROUND TRUTH FILE not found or not readable: $GROUND_TRUTH_FILE" >&2
  exit 3
fi

if [[ ! -x "$RESULTS_SCRIPT" ]]; then
  echo "ERROR: RESULTS script not found or not executable: $RESULTS_SCRIPT" >&2
  exit 3
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: DATASET_DIR does not exist or is not a directory: $DATASET_DIR" >&2
  exit 2
fi

mkdir -p logs
batch_log="logs/infer_batch_${DATASET_NAME}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$batch_log") 2>&1

echo "== TopoQA Inference (Batch) =="
echo "Script dir     : $script_dir"
echo "Dataset dir    : $DATASET_DIR"
echo "Dataset name   : $DATASET_NAME"
echo "INF script     : $INF_SCRIPT"
echo "RESULTS script : $RESULTS_SCRIPT"
echo "Log file       : $batch_log"
echo

targets=()
while IFS= read -r dir; do
  targets+=("$dir")
done < <(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d -print | sort)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "ERROR: No target directories found under: $DATASET_DIR"
  echo "Hint: Expecting layout like: $DATASET_DIR/<TARGET_DIR>/"
  exit 4
fi

echo "Found ${#targets[@]} targets under $DATASET_DIR:"
for t in "${targets[@]}"; do
  echo " - $(basename "$t")"
done
echo

processed=0
ok=0
ko=0
declare -a failed=()
total=${#targets[@]}
echo "[debug] total targets queued: $total"

for target_path in "${targets[@]}"; do
  target_name="$(basename "$target_path")"
  current=$((processed + 1))
  echo "------------------------------------------------------------"
  echo ">> [$current/$total] Processing target: $target_name"
  echo "   Target path : $target_path"
  echo "   Started     : $(date)"
  echo "------------------------------------------------------------"

  ts="$(date +%Y%m%dT%H%M%S)"
  target_log="logs/infer_${DATASET_NAME}_${target_name}_${ts}.log"
  TOPO_WORK_DIR="logs/output/work/${DATASET_NAME}/${target_name}"
  TOPO_RESULTS_DIR="logs/output/results/${DATASET_NAME}/${target_name}"

  DATASET_NAME="$DATASET_NAME" \
  DATASET_DIR="$DATASET_DIR" \
  TARGET_DIR="$target_name" \
  TOPO_WORK_DIR="$TOPO_WORK_DIR" \
  TOPO_RESULTS_DIR="$TOPO_RESULTS_DIR" \
  LOG_FILE="$target_log" \
  "$INF_SCRIPT"
  inf_status=$?
  echo "[debug] inference status for $target_name: $inf_status"

  if [[ $inf_status -eq 0 ]]; then
    echo ">> SUCCESS: $target_name"
    ((++ok))
  else
    echo ">> FAILURE: $target_name (status: $inf_status)"
    failed+=("$target_name")
    ((++ko))
  fi

  echo "   Finished    : $(date)"
  echo
  ((++processed))
done

echo "== Batch complete =="
echo "Processed : $processed"
echo "Succeeded : $ok"
echo "Failed    : $ko"
echo "[debug] failed targets list: ${failed[*]:-<none>}"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failures : ${failed[*]}"
fi

exit $([[ ${#failed[@]} -eq 0 ]] && echo 0 || echo 5)
