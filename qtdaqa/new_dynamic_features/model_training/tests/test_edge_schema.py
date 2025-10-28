from __future__ import annotations

import torch
from pathlib import Path


GRAPH_ROOT = Path(__file__).resolve().parents[3] / "new_graph_builder" / "output" / "pilot_batch" / "graph_data"


def _load_graph(name: str):
    path = GRAPH_ROOT / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path)


def test_edge_dim_matches_config():
    data = _load_graph("1nbf_1")
    assert data.edge_attr.shape[1] == 24
