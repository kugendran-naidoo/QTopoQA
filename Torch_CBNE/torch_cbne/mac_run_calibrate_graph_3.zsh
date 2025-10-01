#!/usr/bin/env zsh
set -euo pipefail

graph="Graph-3"
base_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="${base_dir}/../.."
repo_root="${project_root}/.."
venv="${project_root}/venv_tcbne/bin/activate"

if [[ -f "${venv}" ]]; then
  source "${venv}"
else
  printf "Warning: virtualenv not found at %s\n" "${venv}"
fi

file="${repo_root}/quantinuum/CBNE/graphs/${graph}.graphml"
if [[ ! -f "${file}" ]]; then
  printf "Graph file not found: %s\n" "${file}" >&2
  exit 1
fi

ground_truth=$(grep "<data key='v_betti'>" "${file}" | cut -d ">" -f 2 | cut -d "<" -f 1)

printf "\nCalibrating %s with extended sweep\n" "${graph}"
printf "Graph file   : %s\n" "${file}"
printf "Ground truth : %s\n\n" "${ground_truth}"

log_dir="${base_dir}/logs/calibration"
mkdir -p "${log_dir}"
summary_json="${log_dir}/${graph}_extended_summary.json"

cd "${base_dir}" || exit 1

time python calibrate_cbne.py \
  --path "${file}" \
  --target "${ground_truth}" \
  --epsilon 0.05 0.1 0.2 \
  --deg-limit -1 10 12 14 16 \
  --iter-start 5000 \
  --iter-max 120000 \
  --iter-factor 1.15 \
  --seeds 123 234 345 456 567 678 789 987 \
  --tolerance 3e-4 \
  --min-improvement 5e-5 \
  --max-stalled 5 \
  --device cpu \
  --summary-json "${summary_json}" \
  --log-dir "${log_dir}/${graph}"

printf "\nCalibration complete. Summary: %s\n" "${summary_json}"
