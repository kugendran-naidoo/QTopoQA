#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS_DIR=""
MODE="all"
ALLOW_MISSING=0

usage() {
  cat <<'EOF'
Usage: ./run_results_summary.sh --results-dir PATH [--mode MODE]

Aggregates per-target summary metrics into consolidated CSV files.

Required arguments:
  --results-dir PATH   Directory containing per-target result subdirectories.

Optional arguments:
  --mode MODE          One of: dockq, hit-rate, all (default: all).
  --allow-missing      Do not fail when targets are missing summary files.
  -h, --help           Show this help message and exit.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --results-dir requires a path." >&2
        usage
        exit 1
      fi
      RESULTS_DIR="$2"
      shift 2
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Error: --mode requires a value (dockq|hit-rate|all)." >&2
        usage
        exit 1
      fi
      MODE="$2"
      shift 2
      ;;
    --allow-missing)
      ALLOW_MISSING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${RESULTS_DIR}" ]]; then
  echo "Error: --results-dir is required." >&2
  usage
  exit 1
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Error: results directory not found: ${RESULTS_DIR}" >&2
  exit 1
fi

RESULTS_DIR_ABS="$(cd "${RESULTS_DIR}" && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python -m qtdaqa.new_dynamic_features.model_inference.results_summary \
  --results-dir "${RESULTS_DIR_ABS}" \
  --mode "${MODE}" \
  $( [[ "${ALLOW_MISSING}" -eq 1 ]] && printf -- "--allow-missing" )
