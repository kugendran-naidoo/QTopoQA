#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TOP_K="${TOP_K:-3}"
TRAINING_ROOT="${TRAINING_ROOT:-}"
SHORTLIST_METRIC="${SHORTLIST_METRIC:-best_val_loss}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

METRICS=(
  "best_val_tuning_rank_spearman"
  "best_val_tuning_rank_regret"
  "best_tuning_rank_spearman"
  "best_tuning_rank_regret"
)

echo "Option B selection (top-k=${TOP_K})"
if [[ -n "${TRAINING_ROOT}" ]]; then
  echo "Training root override: ${TRAINING_ROOT}"
fi

for metric in "${METRICS[@]}"; do
  echo
  echo "=== ${metric} ==="
  if [[ -n "${TRAINING_ROOT}" ]]; then
    output="$(python "${SCRIPT_DIR}/tools/option_b_select.py" \
      --top-k "${TOP_K}" \
      --shortlist-metric "${SHORTLIST_METRIC}" \
      --tuning-metric "${metric}" \
      --training-root "${TRAINING_ROOT}")"
  else
    output="$(python "${SCRIPT_DIR}/tools/option_b_select.py" \
      --top-k "${TOP_K}" \
      --shortlist-metric "${SHORTLIST_METRIC}" \
      --tuning-metric "${metric}")"
  fi
  echo "${output}"
  ckpt_path="$(printf "%s\n" "${output}" | sed -n 's/^checkpoint=//p' | tail -n 1)"
  if [[ -n "${ckpt_path}" ]]; then
    echo "checkpoint=$(basename "${ckpt_path}")"
  fi
done
