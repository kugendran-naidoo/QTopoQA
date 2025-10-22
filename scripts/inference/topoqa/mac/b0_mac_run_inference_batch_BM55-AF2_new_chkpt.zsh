#!/usr/bin/env zsh
# -----------------------------------------------------------------------------
# TopoQA batch runner (Mac, zsh) — fixed for set -e with arithmetic increments
# -----------------------------------------------------------------------------

emulate -L zsh
set -e              # exit on error
set -u              # error on unset vars
set -o pipefail     # fail a pipeline if any command fails
setopt null_glob    # unmatched globs expand to nothing
setopt extendedglob

# ---- Required parameters (CLI or env) --------------------------------------
usage() {
  cat <<USAGE
Usage: $0 -n DATASET_NAME -d DATASET_DIR -g GROUND_TRUTH_FILE 
  or   $0 --dataset-name NAME --dataset-dir DIR --ground-truth-file FILE 
All three parameters are mandatory (no defaults).
USAGE
}

# Allow env overrides but do NOT set defaults - EXCEPT for run below
DATASET_NAME="${DATASET_NAME:-BM55-AF2}"
DATASET_DIR="${DATASET_DIR:-../../../../datasets/evaluation/BM55-AF2/decoy}"
GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-../../../../datasets/evaluation/BM55-AF2/label_info.csv}"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dataset-name)      DATASET_NAME="$2"; shift 2 ;;
    -d|--dataset-dir)       DATASET_DIR="$2"; shift 2 ;;
    -g|--ground-truth-file) GROUND_TRUTH_FILE="$2"; shift 2 ;;
    -h|--help)              usage; exit 0 ;;
    --)                     shift; break ;;
    -*)                     printf 'Unknown option: %s\n' "$1" >&2; usage; exit 64 ;;
    *)                      break ;;
  esac
done

: "${DATASET_NAME:?ERROR: DATASET_NAME is required (-n)}"
: "${DATASET_DIR:?ERROR: DATASET_DIR is required (-d)}"
: "${GROUND_TRUTH_FILE:?ERROR: GROUND_TRUTH_FILE is required (-g)}"

# ---- Paths ------------------------------------------------------------------
script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
INF_SCRIPT="${INF_SCRIPT:-${script_dir}/lib/lib_mac_run_inference_new_chkpt.zsh}"
RESULTS_SCRIPT="${RESULTS_SCRIPT:-${script_dir}/lib/lib_mac_run_results_target.zsh}"

# ---- Validation --------------------------------------------------------------
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

# ---- Logging (match INF's 'log/' directory) ----------------------------------
mkdir -p logs
_batch_log="logs/infer_batch_${DATASET_NAME}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$_batch_log") 2>&1

echo "== TopoQA Inference (Batch) =="
echo "Script dir     : $script_dir"
echo "Dataset dir    : $DATASET_DIR"
echo "Dataset name   : $DATASET_NAME"
echo "INF script     : $INF_SCRIPT"
echo "RESULTS script : $RESULTS_SCRIPT"
echo "Log file       : $_batch_log"
echo

# ---- Discover targets: immediate subdirectories -----------------------------
typeset -a targets
targets=("$DATASET_DIR"/*(/N))   # dirs only; N avoids literal '*'

if (( ${#targets[@]} == 0 )); then
  echo "ERROR: No target directories found under: $DATASET_DIR"
  echo "Hint: Expecting layout like: $DATASET_DIR/<TARGET_DIR>/"
  exit 4
fi

# Stable order: prefer gsort -V (GNU), fallback to BSD sort
if command -v gsort >/dev/null 2>&1; then
  IFS=$'\n' targets=($(printf '%s\n' "${targets[@]}" | gsort -V)); unset IFS
else
  IFS=$'\n' targets=($(printf '%s\n' "${targets[@]}" | sort));     unset IFS
fi

printf 'Found %d targets under %s:\n' ${#targets[@]} "$DATASET_DIR"
for t in "${targets[@]}"; do
  echo " - ${t:t}"
done
echo

# ---- Process targets (index-based while; prefix increments) -----------------
typeset -i i=1 total=${#targets[@]} ok=0 ko=0
typeset -a failed
failed=()

while (( i <= total )); do
  target_path="${targets[i]}"
  target="${target_path:t}"

  echo "------------------------------------------------------------"
  echo ">> [$i/$total] Processing target: $target"
  echo "   Target path : $target_path"
  echo "   Started     : $(date)"
  echo "------------------------------------------------------------"

  if DATASET_NAME="$DATASET_NAME" DATASET_DIR="$DATASET_DIR" TARGET_DIR="$target" \
  
    ts="$(date +%Y%m%dT%H%M%S)"
    LOG_FILE="logs/infer_${DATASET_NAME}_${target}_${ts}.log"      # INF's own per-target log
    TOPO_WORK_DIR="logs/output/work/${DATASET_NAME}/${target}"
    TOPO_RESULTS_DIR="logs/output/results/${DATASET_NAME}/${target}"

    # Call inference
    DATASET_NAME="$DATASET_NAME" DATASET_DIR="$DATASET_DIR" TARGET_DIR="$target" \
    TOPO_WORK_DIR="$TOPO_WORK_DIR" TOPO_RESULTS_DIR="$TOPO_RESULTS_DIR" LOG_FILE="$LOG_FILE" \
    "$INF_SCRIPT"

    # quick and dirty absolute path for GROUND_TRUTH_FILE
    full_ground_truth_file_path=$(cd "$(dirname "$GROUND_TRUTH_FILE")" && pwd)/$(basename "$GROUND_TRUTH_FILE")

    # Call results - pass absolute paths for lib subdirectory
    DATASET_NAME="$DATASET_NAME" TARGET="$target" \
    GROUND_TRUTH_FILE="$full_ground_truth_file_path" \
    TOPO_RESULTS_DIR="${script_dir}/logs/output/results" TOPO_RESULT_FILE="result.csv" \
    LOG_FILE="${script_dir}/logs/results_${DATASET_NAME}_${target}_${ts}.log" \
    "$RESULTS_SCRIPT"

  then
    echo ">> SUCCESS: $target"
    (( ++ok ))            # prefix increment: safe with set -e
  else
    echo ">> FAILURE: $target"
    failed+=("$target")
    (( ++ko ))            # prefix increment: safe with set -e
  fi

  echo "   Finished    : $(date)"
  echo
  (( ++i ))               # prefix increment: safe with set -e

done

# ---- Summary (use if, not short-circuit, to be set -e safe) -----------------
echo "== Batch complete =="
echo "Targets discovered : $total"
echo "Succeeded          : $ok"
echo "Failed             : $ko"
if (( ko > 0 )); then
  echo "Failures:"
  for f in "${failed[@]}"; do
    echo " - $f"
  done
fi

echo
echo "== All done at $(date) =="

