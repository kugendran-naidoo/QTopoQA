#!/usr/bin/env zsh
# -----------------------------------------------------------------------------
# TopoQA post-processing (Mac, zsh) — robust + relative-paths preserved
# Produces per-target files next to the inference output:
#   - <TARGET>.unified_result.csv
#   - <TARGET>.metrics.csv
#   - <TARGET>.hit.rate_result.csv        # HRR0 (Top-10 hit rate + Top-10 rows)
#   - <TARGET>.ranking_loss_result.csv    # RLR0
#
# Hit-rate definition (cumulative over Top-10 by predicted score):
#   a := count(TRUE_DOCKQ >= 0.23)
#   b := count(TRUE_DOCKQ >= 0.49)
#   c := count(TRUE_DOCKQ >= 0.80)
# Printed as "a/b/c"
# -----------------------------------------------------------------------------

set -o errexit -o nounset -o pipefail
setopt NO_NOMATCH
IFS=$'\n\t'

# Anchor to this script’s directory so relative paths are stable
SCRIPT_DIR=${0:A:h}
cd "$SCRIPT_DIR"

# ---- User-configurable (RELATIVE defaults; override via env) -----------------
export DATASET_NAME="${DATASET_NAME:-example}"            # as requested
export TARGET_DIR="${TARGET_DIR:-6B0S}"

export TOPO_RESULTS_DIR="${TOPO_RESULTS_DIR:-logs/output/results}"
export TOPO_RESULT_FILE="${TOPO_RESULT_FILE:-result.csv}"

# Ground truth default (kept pointing at BM55-AF2 unless you override)
export GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-../../../../datasets/examples/BM55-AF2/label_info.csv}"

# ---- Tool checks -------------------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { printf 'Missing tool: %s\n' "$1" >&2; exit 127; }; }
for bin in find awk sort grep cut sed head tail mktemp; do need "$bin"; done

# ---- Validate inputs ---------------------------------------------------------
[[ -d "$TOPO_RESULTS_DIR" ]]  || { printf 'ERROR: Results dir missing: %s\n' "$TOPO_RESULTS_DIR" >&2; exit 66; }
[[ -f "$GROUND_TRUTH_FILE" ]] || { printf 'ERROR: Ground truth missing: %s\n' "$GROUND_TRUTH_FILE" >&2; exit 66; }

case "$GROUND_TRUTH_FILE" in
  *"/${DATASET_NAME}/label_info.csv") ;;
  *) printf 'WARNING: DATASET_NAME is "%s" but GROUND_TRUTH_FILE points to "%s".\n' "$DATASET_NAME" "$GROUND_TRUTH_FILE" >&2 ;;
esac

# ---- Logging -----------------------------------------------------------------
mkdir -p logs
_log_file="logs/test_${DATASET_NAME}_${TARGET_DIR}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$_log_file") 2>&1
printf '== TopoQA Post-Processing ==\nDataset : %s\nTarget  : %s\nLog     : %s\n\n' "$DATASET_NAME" "$TARGET_DIR" "$_log_file"

# ---- Helpers -----------------------------------------------------------------
search_topo_results() {
  find "$TOPO_RESULTS_DIR/$DATASET_NAME" -type f -name "$TOPO_RESULT_FILE" \
       -path "*/${TARGET_DIR}/${TOPO_RESULT_FILE}" \
  | LC_ALL=C sort
}

# Join predictions and truth -> TARGET,MODEL,PRED_DOCKQ,TRUE_DOCKQ
join_pred_truth() {
  # $1: TARGET,MODEL,PRED (no header)
  # $2: TARGET,MODEL,TRUE  (no header)
  awk -F, '
    FNR==NR { truth[$1 FS $2]=$3; next }
             { key=$1 FS $2; if (key in truth) print $1","$2","$3","truth[key]; }
  ' "$2" "$1"
}

# normalize numbers: decimal comma -> dot
num() { printf '%s' "$1" | tr ',' '.'; }

# ---- Temps -------------------------------------------------------------------
_tmp_pred="$(mktemp -t topo_pred.XXXXXX)"
_tmp_truth="$(mktemp -t topo_truth.XXXXXX)"
trap 'rm -f "$_tmp_pred" "$_tmp_truth"' EXIT

# ---- Main --------------------------------------------------------------------
results_found=0

while IFS= read -r result_path; do
  [[ -z "$result_path" ]] && continue
  results_found=1

  target_path="$(dirname "$result_path")"   # .../results/<DATASET_NAME>/<TARGET>
  target_name="$(basename "$target_path")"  # e.g., 3SE8

  printf '>> Processing: %s\n' "$result_path"

  # Predictions -> TARGET,MODEL,PRED (no header)
  LC_ALL=C tail -n +2 -- "$result_path" \
  | sed "s|^|${target_name},|" \
  | sed 's/_tidy,/,/' \
  > "$_tmp_pred"

  # Ground truth (this target) -> TARGET,MODEL,TRUE (no header)
  LC_ALL=C grep -E "^${target_name}," -- "$GROUND_TRUTH_FILE" \
  | LC_ALL=C sort \
  > "$_tmp_truth"

  # Unified CSV
  unified_csv="${target_path}/${target_name}.unified_result.csv"
  {
    printf 'TARGET,MODEL,PRED_DOCKQ,TRUE_DOCKQ\n'
    join_pred_truth "$_tmp_pred" "$_tmp_truth"
  } > "$unified_csv"

  # ---- Metrics ---------------------------------------------------------------
  best_true_row=$(
    LC_ALL=C tail -n +2 -- "$unified_csv" \
    | LC_ALL=C sort -t, -k4,4nr | head -n 1
  )
  m_star_best_true="$(printf '%s\n' "$best_true_row" | cut -d',' -f4)"

  top_row=$(
    LC_ALL=C tail -n +2 -- "$unified_csv" \
    | LC_ALL=C sort -t, -k3,3nr | head -n 1
  )
  top_pred_pred="$(printf '%s\n' "$top_row" | cut -d',' -f3)"
  top_pred_true="$(printf '%s\n' "$top_row" | cut -d',' -f4)"

  ranking_loss=$(awk -v a="$(num "$m_star_best_true")" -v b="$(num "$top_pred_true")" \
                     'BEGIN{printf "%.6f",(a+0)-(b+0)}')

  # Top-10 by predicted (rows printed into HRR0)
  top10_rows=$(
    LC_ALL=C tail -n +2 -- "$unified_csv" \
    | LC_ALL=C sort -t, -k3,3nr | head -n 10
  )

  # ====== HIT-RATE CALCULATION (the critical part) ===========================
  # EXACTLY as specified: take Top-10 by predicted; count cumulative thresholds
  # We extract TRUE_DOCKQ (col 4), normalize decimal commas, then count.
  abc_count=$(
    printf '%s\n' "$top10_rows" \
    | sed $'s/\r$//' \
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
  # ===========================================================================

  # Compact metrics CSV (for aggregation)
  metrics_csv="${target_path}/${target_name}.metrics.csv"
  if [[ ! -f "$metrics_csv" ]]; then
    printf 'TARGET,m_star_best_true,top_pred_true,top_pred_pred,ranking_loss,top10_A,top10_B,top10_C\n' > "$metrics_csv"
  fi
  LC_ALL=C printf '%s,%.6f,%.6f,%.6f,%.6f,%d,%d,%d\n' \
         "$target_name" \
         "$(num "$m_star_best_true")" \
         "$(num "$top_pred_true")" \
         "$(num "$top_pred_pred")" \
         "$ranking_loss" \
         "$top10_A" "$top10_B" "$top10_C" \
  >> "$metrics_csv"

  # HRR0: <TARGET>.hit.rate_result.csv (exact format)
  hrr_csv="${target_path}/${target_name}.hit.rate_result.csv"
  {
    printf 'Acceptable-or-better (a): count DockQ ≥ 0.23\n'
    printf 'Medium-or-better (b): count DockQ ≥ 0.49\n'
    printf 'High (c): count DockQ ≥ 0.80\n\n'
    printf 'hit rate = a/b/c \n\n'
    printf '%s hit rate =  %d/%d/%d\n\n' "$target_name" "$top10_A" "$top10_B" "$top10_C"
    printf '%s\n' "$top10_rows"
  } > "$hrr_csv"

  # RLR0: <TARGET>.ranking_loss_result.csv
  rlr_csv="${target_path}/${target_name}.ranking_loss_result.csv"
  m_star_disp=$(awk -v a="$(num "$m_star_best_true")" 'BEGIN{printf "%.3f", a+0}')
  m_hat_disp=$(awk -v b="$(num "$top_pred_true")"     'BEGIN{printf "%.3f", b+0}')
  rl_disp=$(awk -v a="$(num "$m_star_best_true")" -v b="$(num "$top_pred_true")" \
                  'BEGIN{printf "%.3f", (a+0)-(b+0)}')
  {
    printf 'ranking loss = m* - m^\n'
    printf '%s m* = %s\n' "$target_name" "$m_star_disp"
    printf '%s m^ = %s\n' "$target_name" "$m_hat_disp"
    printf '%s ranking loss = %s\n\n' "$target_name" "$rl_disp"
    printf '%s\n' "$best_true_row"
    printf '%s\n' "$top_row"
  } > "$rlr_csv"

  # Console summary — read back triple from HRR0 for authoritative display
  read_back=$(
    LC_ALL=C grep -E "^${target_name} hit rate = " -- "$hrr_csv" \
    | sed -E 's/.*=\s*([0-9]+\/[0-9]+\/[0-9]+).*/\1/' | head -n 1
  )

  printf '\nSummary for %s\n' "$target_name"
  printf '  Unified CSV : %s\n' "$unified_csv"
  printf '  Metrics CSV : %s\n' "$metrics_csv"
  printf '  Top-10 hits (A>=0.23, B>=0.49, C>=0.80): %s\n' "${read_back:-${top10_A}/${top10_B}/${top10_C}}"
  LC_ALL=C printf '  m* (best TRUE)           : %.6f\n' "$(num "$m_star_best_true")"
  LC_ALL=C printf '  m^ TRUE at top predicted     : %.6f\n' "$(num "$top_pred_true")"
  LC_ALL=C printf '  PRED score (top predicted): %.6f\n' "$(num "$top_pred_pred")"
  LC_ALL=C printf '  Ranking loss (m* - TRUE@top): %s\n' "$ranking_loss"
  printf '  Files       : %s, %s\n\n' "$hrr_csv" "$rlr_csv"

done < <(search_topo_results)

if [[ $results_found -eq 0 ]]; then
  printf 'ERROR: No "%s" files found under %s/%s for target "%s".\n' \
         "$TOPO_RESULT_FILE" "$TOPO_RESULTS_DIR" "$DATASET_NAME" "$TARGET_DIR" >&2
  exit 3
fi

printf '== Done: %s ==\n' "$(date)"

