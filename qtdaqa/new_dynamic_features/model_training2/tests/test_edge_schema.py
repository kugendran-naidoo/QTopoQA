from __future__ import annotations

import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[2]
GRAPH_ROOT = BASE_DIR / "model_training2" / "tests" / "fixtures" / "graph_data"
COMMON_DIR = BASE_DIR / "common"
if COMMON_DIR.exists() and str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from feature_metadata import load_graph_feature_metadata  # type: ignore  # noqa: E402


def _load_graph(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        raise AssertionError(f"{path} missing edge_attr tensor")
    return edge_attr


def _require_graph_path() -> Path:
    graph_path = GRAPH_ROOT / "1a2k" / "1a2k_1.pt"
    assert graph_path.exists(), f"Fixture graph not found at {graph_path}"
    return graph_path


def test_feature_metadata_detects_edge_schema():
    graph_path = _require_graph_path()
    edge_attr = _load_graph(graph_path)
    metadata = load_graph_feature_metadata(GRAPH_ROOT, sample_models=["1a2k/1a2k_1"])
    assert metadata.edge_schema["dim"] == edge_attr.shape[1]
    assert metadata.edge_schema.get("module")
    assert metadata.edge_schema.get("variant")


def test_feature_metadata_fallback_without_json(tmp_path: Path):
    graph_path = _require_graph_path()
    temp_graph_dir = tmp_path / "graph_data"
    temp_graph_dir.mkdir(parents=True, exist_ok=True)
    copied = temp_graph_dir / "1a2k" / "1a2k_1.pt"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_bytes(graph_path.read_bytes())

    metadata = load_graph_feature_metadata(temp_graph_dir)
    assert metadata.edge_schema["dim"] == _load_graph(copied).shape[1]
    assert metadata.edge_schema.get("module")
