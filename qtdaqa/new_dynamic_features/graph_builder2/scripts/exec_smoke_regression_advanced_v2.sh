#!/usr/bin/env bash
set -euo pipefail

# Smoke regression for topology/persistence_k_partite_advanced/v2 (fixed-width schema).
# Runs minimal/lean/heavy/heavy_stratified on a tiny dataset and asserts metadata/schema invariants:
# - topology_columns.json exists and has unique columns
# - topology_feature_dim matches topology_columns length-1 and _topology_schema.dim
# - topology_module id is correct
#
# Usage: ./exec_smoke_regression_advanced_v2.sh [dataset_dir] [jobs]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DATASET_DIR="${1:-${REPO_ROOT}/datasets/training/adjusted/smaller_pilot_batch_Dockground_MAF2}"
JOBS="${2:-4}"

run_smoke() {
  local preset_name="$1"
  local cfg_path="$2"
  local output_dir="${SCRIPT_DIR}/output/smoke_advanced_v2_${preset_name}"
  local work_dir="${output_dir}/work"
  local graph_dir="${output_dir}/graph_data"
  local log_dir="${output_dir}/logs"

  rm -rf "${output_dir}"
  mkdir -p "${work_dir}" "${graph_dir}"

  echo "== Running advanced/v2 preset ${preset_name} with config ${cfg_path} ==" >&2
  time "${SCRIPT_DIR}/../run_graph_builder2.sh" --pdb-warnings --no-sort-artifacts \
    --dataset-dir "${DATASET_DIR}" \
    --work-dir "${work_dir}" \
    --graph-dir "${graph_dir}" \
    --log-dir "${log_dir}" \
    --jobs "${JOBS}" \
    --feature-config "${cfg_path}"

python - "${graph_dir}" "${preset_name}" "${work_dir}" <<'PY'
import json
from pathlib import Path
import sys

graph_dir = Path(sys.argv[1])
preset = sys.argv[2]
work_dir = Path(sys.argv[3])

module_id = "topology/persistence_k_partite_advanced/v2"

meta_path = graph_dir / "graph_metadata.json"
if not meta_path.exists():
    raise SystemExit(f"{preset}: graph_metadata.json missing at {meta_path}")
meta = json.loads(meta_path.read_text())

topo_cols_path = graph_dir / "topology_columns.json"
if not topo_cols_path.exists():
    raise SystemExit(f"{preset}: topology_columns.json missing at {topo_cols_path}")
cols = json.loads(topo_cols_path.read_text())
if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
    raise SystemExit(f"{preset}: topology_columns.json is not a list of strings")
if len(cols) != len(set(cols)):
    dupes = [c for c in cols if cols.count(c) > 1][:20]
    raise SystemExit(f"{preset}: duplicate topology columns detected (sample): {dupes}")

topo_dim = meta.get("topology_feature_dim")
schema = meta.get("_topology_schema") or {}
schema_dim = schema.get("dim")
if topo_dim != len(cols) - 1:
    raise SystemExit(f"{preset}: topology_feature_dim={topo_dim} != len(topology_columns)-1 ({len(cols)-1})")
if schema_dim != topo_dim:
    raise SystemExit(f"{preset}: _topology_schema.dim={schema_dim} != topology_feature_dim={topo_dim}")

topo_module = meta.get("topology_module")
if topo_module != module_id:
    raise SystemExit(f"{preset}: topology_module={topo_module!r} != {module_id!r}")

# Ensure the module emitted its schema spec artifact (module-only debug file).
spec_path = work_dir / "topology" / "topology_schema_spec.json"
if not spec_path.exists():
    raise SystemExit(f"{preset}: topology_schema_spec.json missing at {spec_path}")

print(f"{preset}: OK topo_dim={topo_dim} cols={len(cols)}")
PY
}

CFG_BASE="${SCRIPT_DIR}/../feature_configs/advanced_k_partite_v2"
run_smoke "minimal" "${CFG_BASE}/feature-config.yaml.advanced_v2_minimal.10A"
run_smoke "lean" "${CFG_BASE}/feature-config.yaml.advanced_v2_lean.10A"
run_smoke "heavy" "${CFG_BASE}/feature-config.yaml.advanced_v2_heavy.10A"
run_smoke "heavy_stratified" "${CFG_BASE}/feature-config.yaml.advanced_v2_heavy_stratified.10A"

echo "Advanced/v2 smoke regression passed."

