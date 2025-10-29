#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

run_step() {
  local name="$1"
  shift
  local status
  local start_ts end_ts elapsed
  start_ts=$(date +%s)
  log "Starting ${name}"
  set +e
  "$@"
  status=$?
  set -e
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  if [[ $status -eq 0 ]]; then
    log "Completed ${name} (elapsed: ${elapsed}s)"
  else
    log "FAILED ${name} (elapsed: ${elapsed}s)" >&2
  fi
  return $status
}

if run_step "1_fine_tune_part1" bash "${SCRIPT_DIR}/1_fine_tune_after_tune_and_run_part1.sh"; then
  run_step "2_fine_tune_part2" bash "${SCRIPT_DIR}/2_fine_tune_after_tune_and_run_part2.sh" || exit 1
else
  log "Skipping 2_fine_tune_part2 because 1_fine_tune_part1 failed." >&2
  exit 1
fi
