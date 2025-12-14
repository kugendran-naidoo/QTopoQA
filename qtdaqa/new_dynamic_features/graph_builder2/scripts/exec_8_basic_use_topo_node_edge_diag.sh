set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUTPUT="output/8_topoqa_basic_topo_node_edge_diag_ARM"
FEATURE_FILE="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/feature_configs/feature-config.yaml.topoqa.10A"

DATASET_DIR="${REPO_ROOT}/datasets/training/adjusted/pilot_batch_Dockground_MAF2"
WORK_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/work"
GRAPH_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/graph_data"
LOG_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/logs_${OUTPUT}"

# mk dirs
mkdir -p "${WORK_DIR}"
mkdir -p "${GRAPH_DIR}"

cd "${SCRIPT_DIR}"

export QTOPO_TOPO_TRACE=1
export QTOPO_TOPO_TRACE_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/trace/topo"

export QTOPO_NODE_TRACE=1
export QTOPO_NODE_TRACE_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/trace/node"

export QTOPO_EDGE_TRACE=1
export QTOPO_EDGE_TRACE_DIR="${REPO_ROOT}/qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/trace/edge"

time ./run_graph_builder2.sh --pdb-warnings --no-sort-artifacts \
  --dataset-dir "${DATASET_DIR}" \
  --work-dir "${WORK_DIR}" \
  --graph-dir "${GRAPH_DIR}" \
  --log-dir "${LOG_DIR}" \
  --jobs 1 \
  --feature-config "${FEATURE_FILE}"
