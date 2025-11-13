from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

# Stub torch before importing the tool.
torch_stub = types.ModuleType("torch")


def _torch_save(obj, path, *args, **kwargs):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _torch_load(path, map_location=None, *args, **kwargs):
    with open(path, "rb") as handle:
        return pickle.load(handle)


torch_stub.save = _torch_save
torch_stub.load = _torch_load
sys.modules["torch"] = torch_stub

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_inference.override_tool import regenerate_graph_metadata as tool  # noqa: E402


class DummyTensor:
    def __init__(self, shape):
        self._shape = tuple(shape)

    def dim(self) -> int:
        return len(self._shape)

    @property
    def shape(self):
        return self._shape


class DummyGraph:
    def __init__(self, edge_dim: int, node_dim: int):
        self.edge_attr = DummyTensor((5, edge_dim))
        self.x = DummyTensor((7, node_dim))
        self.metadata = {"edge_info": {"feature_dim": edge_dim}}


def _write_pt(path: Path, edge_dim: int, node_dim: int) -> None:
    data = DummyGraph(edge_dim=edge_dim, node_dim=node_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_stub.save(data, path)


def test_regenerate_graph_metadata(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    _write_pt(graph_dir / "sample_a.pt", edge_dim=11, node_dim=4)
    _write_pt(graph_dir / "nested" / "sample_b.pt", edge_dim=11, node_dim=4)

    feature_config = tmp_path / "features.yaml"
    feature_config.write_text(
        "interface:\n  module: interface/polar_cutoff/v1\nnode:\n  module: node/dssp_topo_merge/v1\nedge:\n  module: edge/legacy_band/v11\n  params:\n    distance_max: 10.0\n",
        encoding="utf-8",
    )

    output = tmp_path / "graph_metadata.json"
    path = tool.regenerate_graph_metadata(
        graph_dir=graph_dir,
        feature_config=feature_config,
        output_path=output,
        builder_id="graph_builder2",
        builder_version="2.0.0",
        builder_module="gb2",
        builder_schema_version=1,
        include_feature_text=True,
        edge_dumps_enabled=True,
        edge_dump_dir=tmp_path / "edges",
        edge_dump_configured_dir=tmp_path / "edges_conf",
        sample_limit=None,
    )

    assert path == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "_builder" in payload
    builder_block = payload["_builder"]
    assert builder_block["feature_config"]["text"] is not None
    assert builder_block["edge_dumps"]["enabled"] is True

    entries = {key: value for key, value in payload.items() if not key.startswith("_")}
    assert "sample_a" in entries and "nested/sample_b" in entries
    sample_entry = entries["sample_a"]
    assert sample_entry["edge_module"] == "edge/legacy_band/v11"
    assert sample_entry["edge_params"]["distance_max"] == 10.0
    assert sample_entry["edge_metadata"]["feature_dim"] == 11
    assert len(sample_entry["node_feature_columns"]) == 4
    entry_builder = sample_entry["builder"]
    assert entry_builder["feature_config"]["text"] is None
