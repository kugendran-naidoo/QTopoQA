python - <<'PY'
import torch
from pathlib import Path

base = Path("qtdaqa/new_dynamic_features/graph_builder2/output/lapl_edge_pool_heavy_10A/graph_data")
pt_path = next((base / "2bnq").glob("*.pt"), None)
if not pt_path:
    raise SystemExit("No .pt found under graph_data/2bnq")
data = torch.load(pt_path)
print("pt file:", pt_path.name)
print("edge_attr dim:", data.edge_attr.shape[1])    # expected 2081 for heavy with topo_dim=172
print("node dim:", data.x.shape[1])                 # expected 204 (32 DSSP + 172 topo)
print("edge_feature_dim meta:", data.metadata.get("edge_feature_dim"))
print("edge_module:", data.metadata.get("edge_module"))
PY
