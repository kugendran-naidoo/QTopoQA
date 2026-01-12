from pathlib import Path
import json

import numpy as np

from qtdaqa.new_dynamic_features.graph_builder2.lib import edge_runner
from qtdaqa.new_dynamic_features.graph_builder2.modules.base import EdgeBuildResult, EdgeFeatureModule, build_metadata


class DummyEdge(EdgeFeatureModule):
    module_id = "edge/dummy_schema/v1"
    module_kind = "edge"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="dummy edge schema",
        description="dummy",
    )

    def build_edges(self, **kwargs):
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 3), dtype=np.float32)
        return EdgeBuildResult(edge_index=edge_index, edge_attr=edge_attr, metadata={"feature_dim": 3})


def test_edge_schema_columns_emitted(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    interface_dir = tmp_path / "iface"
    topo_dir = tmp_path / "topo"
    node_dir = tmp_path / "node"
    graph_dir = tmp_path / "graph"
    log_dir = tmp_path / "logs"
    for d in (dataset_dir, interface_dir, topo_dir, node_dir, graph_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "model.pdb").write_text("ATOM\n", encoding="utf-8")
    (interface_dir / "model.interface.txt").write_text("c<A>r<1>R<GLY> 0 0 0\n", encoding="utf-8")
    (topo_dir / "model.topology.csv").write_text("ID,a\nc<A>r<1>R<GLY>,1\n", encoding="utf-8")
    (node_dir / "model.node_fea.csv").write_text("ID,x\nc<A>r<1>R<GLY>,1\n", encoding="utf-8")

    monkeypatch.setattr(
        edge_runner,
        "_parse_interface_file",
        lambda path: [edge_runner.InterfaceResidue("c<A>r<1>R<GLY>", "A", 1, " ", "GLY", np.array([0.0, 0.0, 0.0]))],
    )

    class DummyStructureCache:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(edge_runner, "StructureCache", DummyStructureCache)

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
        module_registry={"topology": {"id": "topology/persistence_null/v1"}},
    )

    meta = json.loads((graph_dir / "graph_metadata.json").read_text(encoding="utf-8"))
    edge_schema = meta.get("_edge_schema") or {}
    assert edge_schema.get("columns") is not None
    assert len(edge_schema["columns"]) == edge_schema.get("dim")
    assert (graph_dir / "edge_columns.json").exists()

    node_schema = meta.get("_node_schema") or {}
    assert node_schema.get("columns") is not None
    assert len(node_schema["columns"]) == node_schema.get("dim")
    assert (graph_dir / "node_columns.json").exists()
