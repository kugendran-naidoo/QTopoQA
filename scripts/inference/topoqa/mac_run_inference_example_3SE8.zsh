# K modified TopoQA inference - ported to Mac

# 1.) searches a particular dataset directory
# 2.) extracts list of targets (example 3SE8, 3WD5)
# 3.) submits targets to TopoQA inference
# 4.) write inference results to TOPO_RESULTS_DIR/DATASET_NAME/TARGET

export DATASET_NAME="BM55-AF2"
# relative path to script
export DATASET_DIR="../../../datasets/examples/BM55-AF2/decoy"
# relative path to script
export TARGET_DIR="3SE8"
# relative path to script
export TOPO_WORK_DIR="output/work"
# relative path to script
export TOPO_RESULTS_DIR="output/results"
# relative path to script
export TOPO_INF_MODEL="../../../topoqa/model/topoqa.ckpt"
# relative path to script
export TOPO_INF_SCRIPT="../../../topoqa/k_mac_inference_pca_tsne4.py"

# inference configuration
export TOPO_JOBS=8
export TOPO_NUM_WORKERS=2
export TOPO_BATCH_SIZE=4
export TOPO_DEVICE="cpu"
# CUTOFF 10 angstroms for contact metric
export TOPO_CUTOFF=10                  
# accept true or yes - case insensitive
export TOPO_RESULTS_OVERWRITE="yes"
export DSSP=/opt/homebrew/bin/mkdssp

# Robust path to the *current script* across sh/bash/zsh/ksh (exec or source)
_get_current_script_path() {
  if [ -n "$BASH_SOURCE" ]; then
    printf '%s\n' "${BASH_SOURCE[0]}"
  elif [ -n "$ZSH_VERSION" ]; then
    # Prefer zsh's call stack to get the file even inside functions
    if [ -n "${funcfiletrace[1]}" ]; then
      # funcfiletrace[1] looks like: /abs/or/rel/path/to/file:LINE:function
      # Use parameter expansion in a POSIX-portable way (no zsh-only ${var%:*:*})
      _fft=${funcfiletrace[1]}
      # Strip the last two :segments (line and function)
      _file=${_fft%:*}       # remove :function
      _file=${_file%:*}      # remove :LINE
      printf '%s\n' "$(_printf_file_path "$_file")"
    else
      # Fallback: %N is the current script (works for exec/source in most cases)
      # shellcheck disable=SC2296
      printf '%s\n' "${(%):-%N}"
    fi
  elif [ -n "$KSH_VERSION" ] && [ -n "${.sh.file}" ] 2>/dev/null; then
    printf '%s\n' "${.sh.file}"
  else
    printf '%s\n' "$0"   # POSIX fallback (exec only)
  fi
}

# helper to normalize any empty/relative path value
_printf_file_path() {
  # If empty, print a placeholder so callers can handle it
  if [ -z "$1" ]; then
    printf '%s\n' ""
  else
    printf '%s\n' "$1"
  fi
}

# Your algorithm: cd → pwd -P → cd -
get_script_dir() {
  _script=$(_get_current_script_path) || return 1

  # If _script is empty (very rare), bail
  [ -n "$_script" ] || return 1

  # If _script has no slash (invoked via PATH), resolve it so dirname works
  case $_script in
    */*) _target=$_script ;;
    *)   _resolved=$(command -v -- "$_script" 2>/dev/null) && _target=$_resolved || _target=$_script ;;
  esac

  # Save caller's dir so cd - returns here; also keep a fallback
  _oldpwd=${PWD:-$(pwd -P)}

  # cd into the script's directory (relative or absolute is fine), print its physical path
  cd -P -- "$(dirname -- "$_target")" || return 1
  _dir=$(pwd -P)

  # return to caller's dir quietly
  cd - >/dev/null 2>&1 || cd -- "$_oldpwd" || return 1

  printf '%s\n' "$_dir"
}

# Convenience: (if you want .../output, append below; for now, just the dir)
get_exec_dir() {
  _dir=$(get_script_dir) || return 1
  printf '%s\n' "$_dir"
}

# ----- usage -----
exec_dir=$(get_exec_dir) || {
  echo "Failed to determine script directory." >&2
  exit 1
}

# search topo_results directory for target results
search_decoys() {

   cd ${DATASET_DIR}
   absolute_dataset_dir=$(pwd)
   cd -

   find ${absolute_dataset_dir}/${TARGET_DIR} -type d |
   sort |
   grep "/${TARGET_DIR}"

}

exec_inference() {

   # timer
   start=$(date +%s)

   # target_name
   target_name="${1##*/}"

   printf "\nInference Target: ${target_name}\n"

   # run TopoQA inference

   # Macbook Pro M1, 64GB memory
   # jobs 8, num-workers 2, batch-size 4, device cpu (tried, but mps does not work)
   # checkpoint model/topoqa.ckpt
   # cutoff must = 10 - 10^(-9) metres for residue distances in paper

   # print executed statement
   printf '%s\n' \
   "python ${TOPO_INF_SCRIPT} \\" \
   "    --complex-folder \\" \
   "    ${1} \\" \
   "    --work-dir \\" \
   "    ${TOPO_WORK_DIR}/${DATASET_NAME}/${target_name} \\" \
   "    --results-dir \\" \
   "    ${TOPO_RESULTS_DIR}/${DATASET_NAME}/${target_name} \\" \
   "    --checkpoint \\" \
   "    ${TOPO_INF_MODEL} \\" \
   "    --jobs ${TOPO_JOBS} \\" \
   "    --num-workers ${TOPO_NUM_WORKERS} \\" \
   "    --batch-size ${TOPO_BATCH_SIZE} \\" \
   "    --device ${TOPO_DEVICE} \\" \
   "    --cutoff ${TOPO_CUTOFF} \\" \
   "    ${overwrite:+--overwrite}"

   python ${TOPO_INF_SCRIPT} \
   --complex-folder \
   ${1} \
   --work-dir \
   ${TOPO_WORK_DIR}/${DATASET_NAME}/${target_name} \
   --results-dir \
   ${TOPO_RESULTS_DIR}/${DATASET_NAME}/${target_name} \
   --checkpoint \
   ${TOPO_INF_MODEL} \
   --jobs ${TOPO_JOBS} \
   --num-workers ${TOPO_NUM_WORKERS} \
   --batch-size ${TOPO_BATCH_SIZE} \
   --device ${TOPO_DEVICE} \
   --cutoff ${TOPO_CUTOFF} \
   ${overwrite:+--overwrite}

   # timer
   end=$(date +%s)

   printf 'Elapsed: %d s\n' "$(( end - start ))"

}

# Main

typeset -u is_overwrite_set

# used for inference execution later
is_overwrite_set=${TOPO_RESULTS_OVERWRITE}

if [[ ${is_overwrite_set} == TRUE || ${is_overwrite_set} == YES ]]; then
  overwrite=true
else
  unset overwrite
fi

# save current dir
# restore it later
current_dir=$(pwd)

# change to script dir for execution
cd ${exec_dir}

printf "Running inference for dataset = ${DATASET_NAME}, target = ${TARGET_DIR}\n"

# list all available prediction result files - TOPO_RESULT_FILE
target_location=$(search_decoys)

printf "target_location: ${target_location}\n"

# set PYTHONPATH
SCRIPT=${TOPO_INF_SCRIPT}
export PYTHONPATH="$(cd "$(dirname "$SCRIPT")" && pwd -P):${PYTHONPATH:-}"

exec_inference "${target_location}"

# restore to current dir
cd ${current_dir}
