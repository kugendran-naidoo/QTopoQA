#!/usr/bin/env zsh
# -----------------------------------------------------------------------------
# TopoQA Inference Runner (zsh) — library version (placed in mac/lib/)
# - Mandatory inputs via environment (no defaults):
#   DATASET_NAME, DATASET_DIR, TARGET_DIR, TOPO_WORK_DIR, TOPO_RESULTS_DIR, LOG_FILE
# - Keeps original logging style, pre-flight checks, helper structure, and main flow
# - Relative paths adjusted for mac/lib/ location (one extra ..)
# -----------------------------------------------------------------------------

emulate -L zsh
set -e              # exit on error
set -u              # error on unset vars
set -o pipefail     # fail a pipeline if any command fails
setopt null_glob

# ---- Mandatory inputs --------------------------------------------------------
: "${DATASET_NAME:?ERROR: DATASET_NAME is required}"
: "${DATASET_DIR:?ERROR: DATASET_DIR is required}"
: "${TARGET_DIR:?ERROR: TARGET_DIR is required}"
: "${TOPO_WORK_DIR:?ERROR: TOPO_WORK_DIR is required}"
: "${TOPO_RESULTS_DIR:?ERROR: TOPO_RESULTS_DIR is required}"
: "${LOG_FILE:?ERROR: LOG_FILE is required}"

# Optional knobs (preserve original behavior if you had them)
TOPO_DEVICE="${TOPO_DEVICE:-cpu}"
TOPO_JOBS="${TOPO_JOBS:-4}"
TOPO_NUM_WORKERS="${TOPO_NUM_WORKERS:-2}"
TOPO_BATCH_SIZE="${TOPO_BATCH_SIZE:-8}"
TOPO_CUTOFF="${TOPO_CUTOFF:-10.0}"
TOPO_OVERWRITE="${TOPO_OVERWRITE:-yes}"  # informational; we still pass --overwrite

# Optional external tool (e.g., mkdssp); if you rely on it, set DSSP to a RELATIVE path
export DSSP="${DSSP:-}"   # e.g., ./bin/mkdssp (if required by your pipeline)

# ---- Paths -------------------------------------------------------------------
script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"     # .../scripts/inference/topoqa/mac/lib
# From mac/lib to repo-top/topoqa is one extra '..' vs previous A0:
#   mac/lib -> mac -> topoqa -> inference -> scripts -> (repo root) -> topoqa
py_rel='../../../../../topoqa'
ckpt_rel="${py_rel}/model/topoqa.ckpt"
py_script_rel="${py_rel}/k_mac_inference_pca_tsne4.py"

# Resolve absolute paths for reliability
topoqa_dir="$(cd -- "$script_dir/$py_rel" && pwd -P)"
ckpt_path="$(cd -- "$script_dir" && cd -- "$ckpt_rel:h" && pwd -P)/${ckpt_rel:t}"
py_script="$(cd -- "$script_dir" && cd -- "$py_script_rel:h" && pwd -P)/${py_script_rel:t}"

# Normalize dataset target path to absolute (for printing & subprocesses)
dataset_dir_abs="$(cd -- "$DATASET_DIR" && pwd -P)"
target_path="${dataset_dir_abs%/}/${TARGET_DIR}"

# ---- Logging -----------------------------------------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1

echo "== TopoQA Inference Runner (zsh) =="
echo "  Start time : $(date)"
echo "  Log file   : $LOG_FILE"
echo
echo "Script directory: $script_dir"
echo
# Keep the familiar PYTHONPATH prefix message (still using a relative path string)
echo "PYTHONPATH prefixed with: $py_rel"
export PYTHONPATH="${topoqa_dir}${PYTHONPATH:+":$PYTHONPATH"}"
echo
echo "DATASET_NAME : $DATASET_NAME"
echo "DATASET_DIR  : $DATASET_DIR"
echo "TARGET_DIR   : $TARGET_DIR"
echo
echo "------------------------------------------------------------"
echo "Target        : $target_path"
echo "Work dir      : $TOPO_WORK_DIR"
echo "Results dir   : $TOPO_RESULTS_DIR"
echo "Device        : $TOPO_DEVICE"
echo "Jobs/Workers  : $TOPO_JOBS/$TOPO_NUM_WORKERS"
echo "Batch/Cutoff  : $TOPO_BATCH_SIZE/$TOPO_CUTOFF"
echo "Overwrite     : ${TOPO_OVERWRITE}--overwrite"
echo "Checkpoint    : $ckpt_path"
echo "Script        : $py_script"
echo "------------------------------------------------------------"

# We validate existence, but we DO NOT rewrite arguments to absolute paths.
# This avoids brittle late failures and preserves your relative-path design.

# python
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: 'python' not found on PATH." >&2
  exit 127
fi

# dataset dir
if [[ ! -d "$dataset_dir_abs" ]]; then
  echo "ERROR: Dataset directory missing: ${DATASET_DIR}" >&2
  exit 2
fi

# optional DSSP validation (only if user set it)
if [[ -n "$DSSP" && ! -x "$DSSP" ]]; then
  echo "ERROR: DSSP set but not executable: ${DSSP}" >&2
  exit 2
fi

# ---- Pre-flight checks (kept as in spirit of original) ----------------------
if [[ ! -d "$target_path" ]]; then
  echo "ERROR: Target directory does not exist: $target_path" >&2
  exit 2
fi
if [[ ! -f "$py_script" ]]; then
  echo "ERROR: Inference Python script not found: $py_script" >&2
  exit 3
fi
if [[ ! -f "$ckpt_path" ]]; then
  echo "ERROR: Checkpoint not found: $ckpt_path" >&2
  exit 4
fi

# Ensure output directories exist
mkdir -p -- "$TOPO_WORK_DIR" "$TOPO_RESULTS_DIR"

# ---- Helper ------------------------------------------------------------------
exec_inference() {
  # Build and run the python command exactly like before (with --overwrite)
  local -a cmd
  cmd=(
    python "$py_script"
    --complex-folder "$target_path"
    --work-dir "$TOPO_WORK_DIR"
    --results-dir "$TOPO_RESULTS_DIR"
    --checkpoint "$ckpt_path"
    --device "$TOPO_DEVICE"
    --jobs "$TOPO_JOBS"
    --num-workers "$TOPO_NUM_WORKERS"
    --batch-size "$TOPO_BATCH_SIZE"
    --cutoff "$TOPO_CUTOFF"
    --overwrite
  )

  # Echo the exact command line (as before)
  echo "${(j: :)cmd}"
  "${cmd[@]}"
}

# ---- Main --------------------------------------------------------------------
exec_inference

echo
echo "== All done at $(date) =="

