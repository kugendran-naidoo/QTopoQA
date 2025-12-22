set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUTPUT="output/topoqa_10A_small_ARM"
FEATURE_FILE="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/feature_configs/feature-config.yaml.topoqa.10A.2"

DATASET_DIR="${REPO_ROOT}/datasets/training/adjusted/smaller_pilot_batch_Dockground_MAF2"
WORK_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/work"
GRAPH_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/graph_data"
LOG_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/logs_${OUTPUT}"

# mk dirs
mkdir -p "${WORK_DIR}"
mkdir -p "${GRAPH_DIR}"

cd "${SCRIPT_DIR}"

time ./run_graph_builder2.sh --pdb-warnings --no-sort-artifacts \
  --dataset-dir "${DATASET_DIR}" \
  --work-dir "${WORK_DIR}" \
  --graph-dir "${GRAPH_DIR}" \
  --log-dir "${LOG_DIR}" \
  --jobs 16 \
  --feature-config "${FEATURE_FILE}"
