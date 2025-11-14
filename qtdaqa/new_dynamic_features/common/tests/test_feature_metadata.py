from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict

import pytest

# Provide a lightweight torch stub before importing the shared module.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.load = lambda *args, **kwargs: None  # type: ignore
    sys.modules["torch"] = torch_stub

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.common import feature_metadata as fm


class _DummyTensor:
    def __init__(self, shape):
        self._shape = tuple(shape)

    def dim(self) -> int:
        return len(self._shape)

    @property
    def shape(self):
        return self._shape


class _DummyGraph:
    def __init__(self, metadata: Dict[str, object]):
        self.edge_attr = _DummyTensor((2, 11))
        self.x = _DummyTensor((3, 5))
        self.metadata = metadata


def _write_graph_metadata(tmp_path: Path, payload: Dict[str, object]) -> Path:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = graph_dir / "graph_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    # Touch a .pt file so loader inspects tensors.
    (graph_dir / "sample.pt").write_text("pt", encoding="utf-8")
    return graph_dir


def test_load_graph_feature_metadata_parses_builder_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "_builder": {
            "id": "graph_builder2",
            "version": "2.0.0",
            "feature_config": {"path": "/tmp/features.yaml", "sha256": "abc123", "text": "edge: {}"},
            "edge_dumps": {"enabled": True, "output_directory": "/tmp/edges"},
        },
        "sample": {
            "edge_metadata": {"feature_dim": 11},
            "node_feature_columns": ["a", "b"],
        },
    }
    graph_dir = _write_graph_metadata(tmp_path, payload)

    monkeypatch.setattr(fm.torch, "load", lambda *args, **kwargs: _DummyGraph(metadata={}))  # type: ignore[attr-defined]

    metadata = fm.load_graph_feature_metadata(graph_dir, max_pt_samples=1)
    assert metadata.builder_id() == "graph_builder2"
    assert metadata.builder_version() == "2.0.0"
    assert metadata.builder_feature_config_digest() == "abc123"
    edge_dump = metadata.builder_edge_dumps()
    assert edge_dump and edge_dump.get("enabled") is True
    serialized = metadata.to_dict()
    assert serialized["builder"]["id"] == "graph_builder2"


def test_load_graph_feature_metadata_uses_per_graph_builder_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "sample": {
            "edge_metadata": {"feature_dim": 11},
            "node_feature_columns": ["a", "b"],
        },
    }
    graph_dir = _write_graph_metadata(tmp_path, payload)

    per_graph_builder = {
        "id": "graph_builder",
        "version": "1.0.0",
        "feature_config": {"path": "/tmp/legacy.yaml", "sha256": "legacy"},
    }
    monkeypatch.setattr(
        fm.torch,
        "load",
        lambda *args, **kwargs: _DummyGraph(metadata={"builder": per_graph_builder}),  # type: ignore[attr-defined]
    )

    metadata = fm.load_graph_feature_metadata(graph_dir, max_pt_samples=1)
    assert metadata.builder_id() == "graph_builder"
    assert metadata.builder_feature_config_path() == "/tmp/legacy.yaml"


def test_discover_summary_path_supports_prefixed_logs(tmp_path: Path) -> None:
    graph_dir = tmp_path / "gb2" / "output" / "graph_data"
    graph_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = tmp_path / "gb2" / "logs_11d_topoqa"
    summary = logs_dir / "graph_builder_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    payload = {"edge": {"output_dir": str(graph_dir)}}
    summary.write_text(json.dumps(payload), encoding="utf-8")

    discovered = fm._discover_summary_path(graph_dir, explicit=None)
    assert discovered == summary.resolve()
