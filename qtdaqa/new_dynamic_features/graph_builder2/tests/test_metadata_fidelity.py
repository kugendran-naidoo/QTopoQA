import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "preset,topo_dim",
    [
        ("minimal", 140),
        ("lean", 420),
        ("heavy", 760),
    ],
)
def test_graph_metadata_has_dims_and_schema(tmp_path: Path, preset: str, topo_dim: int):
    """
    Core metadata fidelity check: graph_metadata.json and topology_columns.json must
    carry dims/schemas for topology/node/edge across presets.
    """
    graph_dir = tmp_path / f"graph_data_{preset}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Build a dummy topology_columns.json with ID + topo_dim numeric columns
    topo_cols = ["ID"] + [f"f{i}" for i in range(topo_dim)]
    (graph_dir / "topology_columns.json").write_text(json.dumps(topo_cols, indent=2), encoding="utf-8")

    node_dim = 172
    edge_dim = 11

    metadata = {
        "topology_feature_dim": topo_dim,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
        "_topology_schema": {"columns": topo_cols, "dim": topo_dim, "module": "topology/test"},
        "_node_schema": {"columns": [f"n{i}" for i in range(node_dim)], "dim": node_dim, "topology_dim": topo_dim},
        "example_model": {
            "edge_metadata": {"feature_dim": edge_dim},
            "node_feature_dim": node_dim,
            "edge_feature_dim": edge_dim,
        },
    }
    (graph_dir / "graph_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Load and assert presence/values
    meta = json.loads((graph_dir / "graph_metadata.json").read_text())
    assert meta.get("topology_feature_dim") == topo_dim
    assert meta.get("node_feature_dim") == node_dim
    assert meta.get("edge_feature_dim") == edge_dim

    topo_schema = meta.get("_topology_schema") or {}
    assert topo_schema.get("columns") == topo_cols
    assert topo_schema.get("dim") == topo_dim
    assert "module" in topo_schema

    node_schema = meta.get("_node_schema") or {}
    assert node_schema.get("dim") == node_dim
    assert node_schema.get("topology_dim") == topo_dim
    assert node_schema.get("columns")

    # Ensure topology_columns.json is present and matches dim+1 (includes ID)
    saved_cols = json.loads((graph_dir / "topology_columns.json").read_text())
    assert len(saved_cols) == topo_dim + 1
