python - <<'PY'
import json, pathlib
root = pathlib.Path("output/full_heavy_10A_smaller/graph_data")
meta = json.loads((root/"graph_metadata.json").read_text())
print("topology_feature_dim:", meta.get("topology_feature_dim"))
print("node_feature_dim:", meta.get("node_feature_dim"))
print("topology_columns exists:", (root/"topology_columns.json").exists())
print("topology_module:", meta.get("topology_module"))
PY
