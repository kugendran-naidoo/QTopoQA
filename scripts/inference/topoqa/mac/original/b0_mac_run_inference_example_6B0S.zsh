#!/usr/bin/env zsh
# -----------------------------------------------------------------------------
# TopoQA inference runner (Mac, zsh) — robust, path-safe, and fully relative
# -----------------------------------------------------------------------------
# What this script does:
#   1) Searches a dataset directory for the requested TARGET_DIR (e.g., 3SE8)
#   2) For each matching target directory, runs the TopoQA inference Python
#      program using ONLY relative paths (per your requirement)
#   3) Writes inference results to: ${TOPO_RESULTS_DIR}/${DATASET_NAME}/${TARGET}
#
# Notes:
# - All paths remain RELATIVE to this script or to the working dir you launch
#   it from. The script never expands them to absolute paths for command args.
# - It validates the presence of key files/dirs, quotes all paths, and logs
#   the exact Python command executed for reproducibility.
# -----------------------------------------------------------------------------

# ---- Safe zsh mode -----------------------------------------------------------
emulate -L zsh
set -euo pipefail
setopt NO_NOMATCH
IFS=$'\n\t'

# ---- User-configurable variables (remain RELATIVE) ---------------------------
# (Keep/adjust these as you already had in I0. They stay relative.)
export DATASET_NAME="${DATASET_NAME:-example}"
export DATASET_DIR="${DATASET_DIR:-../../../../datasets/examples/BM55-AF2/decoy}"
export TARGET_DIR="${TARGET_DIR:-6B0S}"
export TOPO_WORK_DIR="${TOPO_WORK_DIR:-logs/output/work}"
export TOPO_RESULTS_DIR="${TOPO_RESULTS_DIR:-logs/output/results}"

# Inference script & model checkpoint (RELATIVE paths expected)
export TOPO_INF_SCRIPT="${TOPO_INF_SCRIPT:-../../../../topoqa/k_mac_inference_pca_tsne4.py}"
export TOPO_INF_MODEL="${TOPO_INF_MODEL:-../../../../topoqa/model/topoqa.ckpt}"

# Optional knobs (will be passed only if set)
export TOPO_DEVICE="${TOPO_DEVICE:-cpu}"          # e.g., "cpu" or "mps"
export TOPO_JOBS="${TOPO_JOBS:-4}"                # e.g., number of jobs
export TOPO_NUM_WORKERS="${TOPO_NUM_WORKERS:-2}"  # e.g., dataloader workers
export TOPO_BATCH_SIZE="${TOPO_BATCH_SIZE:-8}"
export TOPO_CUTOFF="${TOPO_CUTOFF:-10.0}"         # Å cutoff (example)
export TOPO_RESULTS_OVERWRITE="${TOPO_RESULTS_OVERWRITE:-YES}"  # YES/TRUE to enable

# Optional external tool (e.g., mkdssp); if you rely on it, set DSSP to a RELATIVE path
export DSSP="${DSSP:-}"   # e.g., ./bin/mkdssp (if required by your pipeline)

# ---- Logging (relative) ------------------------------------------------------
mkdir -p logs
_log_file="logs/infer_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$_log_file") 2>&1

echo "== TopoQA Inference Runner (zsh) =="
echo "  Start time : $(date)"
echo "  Log file   : ${_log_file}"
echo

# ---- Remember & auto-restore the caller working directory -------------------
current_dir="$(pwd -P)"
trap 'cd "$current_dir" >/dev/null 2>&1 || true' EXIT

# ---- Move to the script's directory so all RELATIVE paths are stable --------
# (Keeps the 'relative to script' design consistent across calls.)
# Using %N to get the current script path is the most reliable in zsh.
script_path="${(%):-%N}"
if [[ -z "$script_path" || "$script_path" == "zsh" ]]; then
  echo "WARNING: Could not resolve script path via %N; continuing with CWD '$PWD'" >&2
  exec_dir="$PWD"
else
  exec_dir="$(cd "$(dirname "$script_path")" && pwd -P)"
  cd "$exec_dir"
fi

echo "Script directory: $exec_dir"
echo

# ---- Preflight checks (still honoring relative paths) ------------------------
# We validate existence, but we DO NOT rewrite arguments to absolute paths.
# This avoids brittle late failures and preserves your relative-path design.

# python
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: 'python' not found on PATH." >&2
  exit 127
fi

# inference script
if [[ ! -f "$TOPO_INF_SCRIPT" ]]; then
  echo "ERROR: Inference script missing: ${TOPO_INF_SCRIPT}" >&2
  exit 2
fi

# checkpoint
if [[ ! -f "$TOPO_INF_MODEL" ]]; then
  echo "ERROR: Checkpoint missing: ${TOPO_INF_MODEL}" >&2
  exit 2
fi

# dataset dir
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: Dataset directory missing: ${DATASET_DIR}" >&2
  exit 2
fi

# optional DSSP validation (only if user set it)
if [[ -n "$DSSP" && ! -x "$DSSP" ]]; then
  echo "ERROR: DSSP set but not executable: ${DSSP}" >&2
  exit 2
fi

# ---- Ensure top-level output roots exist (RELATIVE) --------------------------
mkdir -p "$TOPO_WORK_DIR" "$TOPO_RESULTS_DIR"

# ---- Ensure PYTHONPATH includes the directory of the inference script -------
inf_dir="$(dirname "$TOPO_INF_SCRIPT")"
if [[ ! -d "$inf_dir" ]]; then
  echo "ERROR: Inference script directory not found: ${inf_dir}" >&2
  exit 2
fi
export PYTHONPATH="$(cd "$inf_dir" && pwd -P):${PYTHONPATH:-}"
echo "PYTHONPATH prefixed with: $inf_dir"
echo

# ---- Helper: find target directories safely (returns 0+ lines) --------------
# We keep outputs as RELATIVE (from where the script runs), not absolute.
search_decoys() {
  # We only want directories whose basename equals TARGET_DIR under DATASET_DIR.
  # This avoids over-matching and handles spaces safely.
  # Using -name for an exact match on basename; paths remain relative.
  (
    cd "$DATASET_DIR"
    # Print relative paths (from DATASET_DIR) then prefix back DATASET_DIR
    # to remain consistent with the script’s relative-call pattern.
    # -maxdepth can be adjusted; here we allow any depth under DATASET_DIR.
    find . -type d -name "$TARGET_DIR" -print0 | while IFS= read -r -d '' rel; do
      # Join DATASET_DIR + rel (still relative overall to script dir)
      printf '%s\n' "${DATASET_DIR}/${rel#./}"
    done
  )
}

# ---- Helper: run one inference given a single target directory ---------------
exec_inference() {
  local target_dir="$1"

  # Normalize target name for output subfolders (basename of the target dir)
  local target_name
  target_name="$(basename "$target_dir")"

  # Create per-target work/results dirs (RELATIVE)
  local work_dir="${TOPO_WORK_DIR}/${DATASET_NAME}/${target_name}"
  local results_dir="${TOPO_RESULTS_DIR}/${DATASET_NAME}/${target_name}"
  mkdir -p "$work_dir" "$results_dir"

  # Overwrite flag (uppercased parsing)
  local overwrite_flag=()
  local _ow="${TOPO_RESULTS_OVERWRITE:u}"
  if [[ "$_ow" == "YES" || "$_ow" == "TRUE" ]]; then
    overwrite_flag=(--overwrite)
  fi

  # Optional arguments: only include if non-empty (keeps CLI clean)
  local opt_args=()
  [[ -n "${TOPO_DEVICE:-}" ]]       && opt_args+=("--device" "$TOPO_DEVICE")
  [[ -n "${TOPO_JOBS:-}" ]]         && opt_args+=("--jobs" "$TOPO_JOBS")
  [[ -n "${TOPO_NUM_WORKERS:-}" ]]  && opt_args+=("--num-workers" "$TOPO_NUM_WORKERS")
  [[ -n "${TOPO_BATCH_SIZE:-}" ]]   && opt_args+=("--batch-size" "$TOPO_BATCH_SIZE")
  [[ -n "${TOPO_CUTOFF:-}" ]]       && opt_args+=("--cutoff" "$TOPO_CUTOFF")

  # If DSSP is required downstream, keep it exported as is (RELATIVE ok)
  export DSSP

  echo "------------------------------------------------------------"
  echo "Target        : $target_dir"
  echo "Work dir      : $work_dir"
  echo "Results dir   : $results_dir"
  echo "Device        : ${TOPO_DEVICE:-}"
  echo "Jobs/Workers  : ${TOPO_JOBS:-}/${TOPO_NUM_WORKERS:-}"
  echo "Batch/Cutoff  : ${TOPO_BATCH_SIZE:-}/${TOPO_CUTOFF:-}"
  echo "Overwrite     : ${overwrite_flag:+yes}${overwrite_flag:-no}"
  echo "Checkpoint    : $TOPO_INF_MODEL"
  echo "Script        : $TOPO_INF_SCRIPT"
  echo "------------------------------------------------------------"

  # Print the exact Python command we will run (quoted, RELATIVE paths preserved)
  {
    printf 'python %q ' "$TOPO_INF_SCRIPT"
    printf -- '--complex-folder %q ' "$target_dir"
    printf -- '--work-dir %q ' "$work_dir"
    printf -- '--results-dir %q ' "$results_dir"
    printf -- '--checkpoint %q ' "$TOPO_INF_MODEL"
    for a in "${opt_args[@]}"; do printf '%q ' "$a"; done
    for a in "${overwrite_flag[@]}"; do printf '%q ' "$a"; done
    printf '\n'
  } | sed 's/ \+/ /g'

  # Execute the same command (RELATIVE paths; robust quoting)
  python "$TOPO_INF_SCRIPT" \
    --complex-folder "$target_dir" \
    --work-dir "$work_dir" \
    --results-dir "$results_dir" \
    --checkpoint "$TOPO_INF_MODEL" \
    "${opt_args[@]}" \
    "${overwrite_flag[@]}"
}

# ---- Main: discover and iterate targets -------------------------------------
echo "DATASET_NAME : $DATASET_NAME"
echo "DATASET_DIR  : $DATASET_DIR"
echo "TARGET_DIR   : $TARGET_DIR"
echo

targets_found=0
search_decoys | while IFS= read -r target_path; do
  [[ -z "$target_path" ]] && continue
  targets_found=1
  exec_inference "$target_path"
done

if [[ $targets_found -eq 0 ]]; then
  echo "ERROR: No target directories named '${TARGET_DIR}' found under '${DATASET_DIR}'." >&2
  exit 3
fi

echo
echo "== All done at $(date) =="

