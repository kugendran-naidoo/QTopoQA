from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from qtdaqa.new_dynamic_features.graph_builder2.lib import edge_runner
from qtdaqa.new_dynamic_features.graph_builder2.modules.base import EdgeFeatureModule, EdgeBuildResult, build_metadata


class DummyEdge(EdgeFeatureModule):
    module_id = "edge/dummy/v1"
    module_kind = "edge"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="dummy edge",
        description="dummy",
    )

    def build_edges(self, **kwargs):
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 1), dtype=np.float32)
        return EdgeBuildResult(edge_index=edge_index, edge_attr=edge_attr, metadata={"feature_dim": 1})


def test_topology_columns_written(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    interface_dir = tmp_path / "iface"
    topo_dir = tmp_path / "topo"
    node_dir = tmp_path / "node"
    graph_dir = tmp_path / "graph"
    log_dir = tmp_path / "logs"
    for d in (dataset_dir, interface_dir, topo_dir, node_dir, graph_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy pdb and matching interface/topology/node files
    pdb_path = dataset_dir / "model.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")
    (interface_dir / "model.interface.txt").write_text("c<A>r<1>R<GLY> 0 0 0\n", encoding="utf-8")
    (topo_dir / "model.topology.csv").write_text("ID,a,b\nc<A>r<1>R<GLY>,1,2\n", encoding="utf-8")
    (node_dir / "model.node_fea.csv").write_text("ID,x\na,1\n", encoding="utf-8")

    # Stub out parsers/structures to avoid heavy deps
    monkeypatch.setattr(edge_runner, "_parse_interface_file", lambda path: [edge_runner.InterfaceResidue("c<A>r<1>R<GLY>", "A", 1, " ", "GLY", np.array([0.0,0.0,0.0]))])
    class DummyStructureCache:
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr(edge_runner, "StructureCache", DummyStructureCache)

    res = edge_runner.run_edge_stage(
        dataset_dir=dataset_dir,
        interface_dir=interface_dir,
        topology_dir=topo_dir,
        node_dir=node_dir,
        output_dir=graph_dir,
        log_dir=log_dir,
        edge_module=DummyEdge(params={}),
        jobs=1,
        edge_dump_dir=None,
        builder_info=None,
        sort_artifacts=False,
        module_registry={"topology": {"id": "topology/persistence_partial_k_partite/v1"}},
    )
    assert (graph_dir / "topology_columns.json").exists()
    # run_edge_stage returns success count; ensure no failures
    assert res["failures"] == []


def test_topology_columns_missing_raises(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    interface_dir = tmp_path / "iface"
    topo_dir = tmp_path / "topo"
    node_dir = tmp_path / "node"
    graph_dir = tmp_path / "graph"
    log_dir = tmp_path / "logs"
    for d in (dataset_dir, interface_dir, topo_dir, node_dir, graph_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Only a node file; no topology CSVs -> should raise
    (dataset_dir / "model.pdb").write_text("ATOM\n", encoding="utf-8")
    (interface_dir / "model.interface.txt").write_text("c<A>r<1>R<GLY> 0 0 0\n", encoding="utf-8")
    (node_dir / "model.node_fea.csv").write_text("ID,x\na,1\n", encoding="utf-8")

    monkeypatch.setattr(edge_runner, "_parse_interface_file", lambda path: [])
    class DummyStructureCache:
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr(edge_runner, "StructureCache", DummyStructureCache)

    with pytest.raises(RuntimeError):
        edge_runner.run_edge_stage(
            dataset_dir=dataset_dir,
            interface_dir=interface_dir,
            topology_dir=topo_dir,
            node_dir=node_dir,
            output_dir=graph_dir,
            log_dir=log_dir,
            edge_module=DummyEdge(params={}),
            jobs=1,
            edge_dump_dir=None,
            builder_info=None,
            sort_artifacts=False,
            module_registry={},
        )
