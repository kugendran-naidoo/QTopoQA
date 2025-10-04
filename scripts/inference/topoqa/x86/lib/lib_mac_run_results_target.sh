#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# TopoQA post-processing (bash version)
# Required inputs (via CLI or environment):
#   DATASET_NAME, TARGET, GROUND_TRUTH_FILE, TOPO_RESULTS_DIR,
#   TOPO_RESULT_FILE, LOG_FILE
# Produces per-target summary/metrics identical to the original zsh script.
# -----------------------------------------------------------------------------
set -uo pipefail
IFS=$'
	'

usage() {
  cat <<'USAGE'
Usage: lib_mac_run_results_target.sh -d DATASET_NAME -t TARGET \
       -g GROUND_TRUTH_FILE -r TOPO_RESULTS_DIR -f TOPO_RESULT_FILE -l LOG_FILE
USAGE
}

DATASET_NAME="${DATASET_NAME:-}"
TARGET="${TARGET:-}"
GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-}"
TOPO_RESULTS_DIR="${TOPO_RESULTS_DIR:-}"
TOPO_RESULT_FILE="${TOPO_RESULT_FILE:-}"
LOG_FILE="${LOG_FILE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset-name)      DATASET_NAME="$2"; shift 2 ;;
    -t|--target)            TARGET="$2"; shift 2 ;;
    -g|--ground-truth-file) GROUND_TRUTH_FILE="$2"; shift 2 ;;
    -r|--topo-results-dir)  TOPO_RESULTS_DIR="$2"; shift 2 ;;
    -f|--topo-result-file)  TOPO_RESULT_FILE="$2"; shift 2 ;;
    -l|--log-file)          LOG_FILE="$2"; shift 2 ;;
    -h|--help)              usage; exit 0 ;;
    --)                     shift; break ;;
    -*)                     printf 'Unknown option: %s
' "$1" >&2; usage; exit 64 ;;
    *)                      break ;;
  esac
done

: "${DATASET_NAME:?ERROR: DATASET_NAME is required (-d)}"
: "${TARGET:?ERROR: TARGET is required (-t)}"
: "${GROUND_TRUTH_FILE:?ERROR: GROUND_TRUTH_FILE is required (-g)}"
: "${TOPO_RESULTS_DIR:?ERROR: TOPO_RESULTS_DIR is required (-r)}"
: "${TOPO_RESULT_FILE:?ERROR: TOPO_RESULT_FILE is required (-f)}"
: "${LOG_FILE:?ERROR: LOG_FILE is required (-l)}"

log_dir="${LOG_FILE%/*}"
if [[ "$log_dir" != "$LOG_FILE" ]]; then
  mkdir -p -- "$log_dir"
fi
exec > >(tee -a "$LOG_FILE") 2>&1
printf '== TopoQA Post-Processing ==
Dataset : %s
Target  : %s
Log     : %s

' "$DATASET_NAME" "$TARGET" "$LOG_FILE"

search_topo_results() {
  find "$TOPO_RESULTS_DIR/$DATASET_NAME" -type f -name "$TOPO_RESULT_FILE" \
       -path "*/${TARGET}/${TOPO_RESULT_FILE}" \
  | LC_ALL=C sort
}

join_pred_truth() {
  local pred_file=$1
  local truth_file=$2
  awk -F, '
    FNR==NR { truth[$1 FS $2]=$3; next }
             { key=$1 FS $2; if (key in truth) print $1","$2","$3","truth[key]; }
  ' "$truth_file" "$pred_file"
}

num() { printf '%s' "$1" | tr ',' '.'; }

_tmp_pred="$(mktemp -t topo_pred.XXXXXX)"
_tmp_truth="$(mktemp -t topo_truth.XXXXXX)"
trap 'rm -f "$_tmp_pred" "$_tmp_truth"' EXIT

results_found=0

while IFS= read -r result_path; do
  [[ -z "$result_path" ]] && continue
  results_found=1

  target_path="$(dirname "$result_path")"
  target_name="$(basename "$target_path")"

  printf '>> Processing: %s
' "$result_path"

  LC_ALL=C tail -n +2 -- "$result_path" \
  | sed "s|^|${target_name},|" \
  | sed 's/_tidy,/,/' \
  > "$_tmp_pred"

  LC_ALL=C grep -E "^${target_name}," -- "$GROUND_TRUTH_FILE" \
  | LC_ALL=C sort \
  > "$_tmp_truth"

  unified_csv="${target_path}/${target_name}.unified_result.csv"
  {
    printf 'TARGET,MODEL,PRED_DOCKQ,TRUE_DOCKQ
'
    join_pred_truth "$_tmp_pred" "$_tmp_truth"
  } > "$unified_csv"

  best_true_row=$(
    LC_ALL=C tail -n +2 -- "$unified_csv" \
    | LC_ALL=C sort -t, -k4,4nr | head -n 1
  )
  m_star_best_true="$(printf '%s
' "$best_true_row" | cut -d',' -f4)"

  top_row=$(
    LC_ALL=C tail -n +2 -- "$unified_csv" \
    | LC_ALL=C sort -t, -k3,3nr | head -n 1
  )
  top_pred_pred="$(printf '%s
' "$top_row" | cut -d',' -f3)"
  top_pred_true="$(printf '%s
' "$top_row" | cut -d',' -f4)"

  ranking_loss=$(LC_NUMERIC=C awk -v a="$(num "$m_star_best_true")" -v b="$(num "$top_pred_true")" \
                     'BEGIN{printf "%.6f",(a+0)-(b+0)}')

  top10_rows=$(
    LC_ALL=C tail -n +2 -- "$unified_csv" \
    | LC_ALL=C sort -t, -k3,3nr | head -n 10
  )

  abc_count=$(
    printf '%s
' "$top10_rows" \
    | sed $'s/$//' \
    | cut -d',' -f4 \
    | LC_ALL=C awk '
        NF {
          v=$1; gsub(",", ".", v); x=v+0;
          if (x >= 0.23) A++;
          if (x >= 0.49) B++;
          if (x >= 0.80) C++;
        }
        END { printf "%d/%d/%d", A+0,B+0,C+0 }'
  )
  IFS=/ read -r top10_A top10_B top10_C <<<"$abc_count"

  metrics_csv="${target_path}/${target_name}.summary_metrics.csv"
  printf 'Note last 4 columns ranking_loss, top10_A, top10_B, top10_C

' > "$metrics_csv"
  printf 'TARGET,m_star_best_true,top_pred_true,top_pred_pred,ranking_loss,top10_A,top10_B,top10_C
' >> "$metrics_csv"

  LC_ALL=C printf '%s,%.6f,%.6f,%.6f,%.6f,%d,%d,%d
' \
         "$target_name" \
         "$(num "$m_star_best_true")" \
         "$(num "$top_pred_true")" \
         "$(num "$top_pred_pred")" \
         "$ranking_loss" \
         "$top10_A" "$top10_B" "$top10_C" \
  >> "$metrics_csv"

  hrr_csv="${target_path}/${target_name}.hit.rate_result.csv"
  {
    printf 'Acceptable-or-better (a): count DockQ ≥ 0.23
'
    printf 'Medium-or-better (b): count DockQ ≥ 0.49
'
    printf 'High (c): count DockQ ≥ 0.80

'
    printf 'hit rate = a/b/c 

'
    printf '%s hit rate =  %d/%d/%d

' "$target_name" "$top10_A" "$top10_B" "$top10_C"
    printf '%s
' "$top10_rows"
  } > "$hrr_csv"

  rlr_csv="${target_path}/${target_name}.ranking_loss_result.csv"
  m_star_disp=$(LC_NUMERIC=C awk -v a="$(num "$m_star_best_true")" 'BEGIN{printf "%.3f", a+0}')
  m_hat_disp=$(LC_NUMERIC=C awk -v b="$(num "$top_pred_true")"     'BEGIN{printf "%.3f", b+0}')
  rl_disp=$(LC_NUMERIC=C awk -v a="$(num "$m_star_best_true")" -v b="$(num "$top_pred_true")" \
                  'BEGIN{printf "%.3f", (a+0)-(b+0)}')
  {
    printf 'ranking loss = m* - m^
'
    printf '%s m* = %s
' "$target_name" "$m_star_disp"
    printf '%s m^ = %s
' "$target_name" "$m_hat_disp"
    printf '%s ranking loss = %s

' "$target_name" "$rl_disp"
    printf '%s
' "$best_true_row"
    printf '%s
' "$top_row"
  } > "$rlr_csv"

  read_back=$(
    LC_ALL=C grep -E "^${target_name} hit rate = " -- "$hrr_csv" \
    | sed -E 's/.*=\s*([0-9]+\/[0-9]+\/[0-9]+).*/\1/' | head -n 1
  )

  printf '
Summary for %s
' "$target_name"
  printf '  Summary Metrics CSV : %s
' "$metrics_csv"
  printf '  Unified Results CSV : %s
' "$unified_csv"
  LC_ALL=C printf '  m* (best TRUE)            : %.6f
' "$(num "$m_star_best_true")"
  LC_ALL=C printf '  m^ TRUE at top predicted  : %.6f
' "$(num "$top_pred_true")"
  LC_ALL=C printf '  PRED score (top predicted): %.6f
' "$(num "$top_pred_pred")"
  LC_ALL=C printf '  Ranking loss (m* - m^)    : %.6f
' "$ranking_loss"
  printf '  Top-10 hits (A>=0.23, B>=0.49, C>=0.80): %s
' "${read_back:-${top10_A}/${top10_B}/${top10_C}}"
  printf '  Files : %s,
          %s

' "$hrr_csv" "$rlr_csv"

done < <(search_topo_results)

if [[ $results_found -eq 0 ]]; then
  printf 'ERROR: No "%s" files found under %s/%s for target "%s".
' \
         "$TOPO_RESULT_FILE" "$TOPO_RESULTS_DIR" "$DATASET_NAME" "$TARGET" >&2
  exit 3
fi

printf '== Done: %s ==
' "$(date)"
