import json
import sys
import types
from pathlib import Path

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.tensor = lambda *args, **kwargs: None
    torch_stub.load = lambda *args, **kwargs: None
    torch_stub.save = lambda *args, **kwargs: None
    torch_stub.utils = types.SimpleNamespace()
    torch_stub.utils.data = types.SimpleNamespace(Dataset=object)
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_stub.utils
    sys.modules["torch.utils.data"] = torch_stub.utils.data

if "torch_geometric" not in sys.modules:
    tg_module = types.ModuleType("torch_geometric")
    tg_data_module = types.ModuleType("torch_geometric.data")
    tg_data_module.Data = object
    tg_data_module.Batch = object
    tg_loader_module = types.ModuleType("torch_geometric.loader")
    tg_loader_module.DataLoader = object
    tg_module.data = tg_data_module
    tg_module.loader = tg_loader_module
    sys.modules["torch_geometric"] = tg_module
    sys.modules["torch_geometric.data"] = tg_data_module
    sys.modules["torch_geometric.loader"] = tg_loader_module

BASE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = BASE_DIR / "common"
if COMMON_DIR.exists() and str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from feature_metadata import GraphFeatureMetadata  # type: ignore

from qtdaqa.new_dynamic_features.model_inference import inference_topoqa_cpu as infer


def test_inference_schema_summary_written(tmp_path: Path) -> None:
    cfg = infer.InferenceConfig(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        checkpoint_path=tmp_path / "legacy.ckpt",
        output_file=tmp_path / "out.csv",
        label_file=None,
        batch_size=1,
        num_workers=0,
        builder=infer.BuilderConfig(),
    )
    cfg.work_dir.mkdir(parents=True)
    final_schema = {
        "edge_schema": {"dim": 11, "module": "edge/test"},
        "topology_schema": {"dim": 2},
    }
    checkpoint_meta = {
        "edge_schema": {"dim": 11},
        "topology_schema": {"dim": 2},
    }
    graph_metadata = GraphFeatureMetadata(
        edge_schema={"dim": 11, "module": "edge/test"},
        node_schema={"dim": 5},
        module_registry={"edge": {"id": "edge/test"}},
        metadata_path="graph_metadata.json",
    )
    path = infer._write_inference_schema_summary(cfg, graph_metadata, final_schema, checkpoint_meta)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["final_schema"]["edge_schema"]["dim"] == 11
    assert payload["checkpoint"] == str(cfg.checkpoint_path)
    assert payload["graph_metadata"]["edge_schema"]["dim"] == 11
