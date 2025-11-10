import json
import os
import sys
import types
from pathlib import Path

import pytest
import importlib.util

USE_REAL_DEPS = os.environ.get("QTOPO_TEST_USE_REAL_DEPS") == "1"

if not USE_REAL_DEPS:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda *args, **kwargs: None
    numpy_stub.zeros = lambda *args, **kwargs: None
    numpy_stub.empty = lambda *args, **kwargs: None
    numpy_stub.ndarray = object
    numpy_stub.int64 = "int64"
    numpy_stub.float32 = "float32"
    sys.modules["numpy"] = numpy_stub

if not USE_REAL_DEPS:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    pandas_stub.read_csv = lambda *args, **kwargs: None
    sys.modules["pandas"] = pandas_stub

if not USE_REAL_DEPS:
    bio_module = types.ModuleType("Bio")
    bio_pdb_module = types.ModuleType("Bio.PDB")
    bio_pdb_atom_module = types.ModuleType("Bio.PDB.Atom")

    class DummyParser:
        def __init__(self, *args, **kwargs):
            pass

        def get_structure(self, *args, **kwargs):
            return {}

    class DummyNeighborSearch:
        def __init__(self, *args, **kwargs):
            pass

    class DummyAtom:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyResidue:
        pass

    class DummyChain:
        pass

    class DummyModel:
        pass

    bio_pdb_module.PDBParser = DummyParser
    bio_pdb_module.NeighborSearch = DummyNeighborSearch
    bio_pdb_atom_module.Atom = DummyAtom
    bio_pdb_residue_module = types.ModuleType("Bio.PDB.Residue")
    bio_pdb_residue_module.Residue = DummyResidue
    bio_pdb_chain_module = types.ModuleType("Bio.PDB.Chain")
    bio_pdb_chain_module.Chain = DummyChain
    bio_pdb_model_module = types.ModuleType("Bio.PDB.Model")
    bio_pdb_model_module.Model = DummyModel
    bio_pdb_structure_module = types.ModuleType("Bio.PDB.Structure")
    bio_pdb_structure_module.Structure = DummyModel
    bio_module.PDB = bio_pdb_module
    sys.modules.setdefault("Bio", bio_module)
    sys.modules.setdefault("Bio.PDB", bio_pdb_module)
    sys.modules.setdefault("Bio.PDB.Atom", bio_pdb_atom_module)
    sys.modules.setdefault("Bio.PDB.Residue", bio_pdb_residue_module)
    sys.modules.setdefault("Bio.PDB.Chain", bio_pdb_chain_module)
    sys.modules.setdefault("Bio.PDB.Model", bio_pdb_model_module)
    sys.modules.setdefault("Bio.PDB.Structure", bio_pdb_structure_module)

if not USE_REAL_DEPS:
    torch_stub = types.ModuleType("torch")
    torch_stub.tensor = lambda *args, **kwargs: None
    torch_stub.save = lambda *args, **kwargs: None
    torch_stub.load = lambda *args, **kwargs: None
    torch_stub.long = "long"
    sys.modules["torch"] = torch_stub

if not USE_REAL_DEPS:
    tg_module = types.ModuleType("torch_geometric")
    tg_data_module = types.ModuleType("torch_geometric.data")
    tg_data_module.Data = object
    tg_data_module.Batch = object
    tg_module.data = tg_data_module
    sys.modules["torch_geometric"] = tg_module
    sys.modules["torch_geometric.data"] = tg_data_module

from qtdaqa.new_dynamic_features.graph_builder2.lib import schema_summary


class DummyMetadata:
    def __init__(self, edge_dim: int, node_dim: int) -> None:
        self.edge_schema = {"module": "edge/test_module", "dim": edge_dim}
        self.node_schema = {"dim": node_dim}
        self.module_registry = {"edge": {"id": "edge/test_module"}}
        self.metadata_path = "graph_metadata.json"
        self.summary_path = None
        self.notes = []

    def to_dict(self):
        return {
            "edge_schema": self.edge_schema,
            "node_schema": self.node_schema,
            "module_registry": self.module_registry,
            "metadata_path": self.metadata_path,
            "summary_path": self.summary_path,
            "notes": self.notes,
        }


@pytest.mark.parametrize("edge_dim,node_dim", [(3, 2)])
def test_schema_summary_written(tmp_path: Path, monkeypatch, edge_dim: int, node_dim: int) -> None:
    graph_dir = tmp_path / "graph_data"
    graph_dir.mkdir()

    def fake_loader(path: Path):
        assert path == graph_dir
        return DummyMetadata(edge_dim=edge_dim, node_dim=node_dim)

    monkeypatch.setattr(schema_summary, "load_graph_feature_metadata", fake_loader)

    schema_summary.write_schema_summary(graph_dir)

    summary_path = graph_dir / "schema_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["edge_schema"]["module"] == "edge/test_module"
    assert summary["edge_schema"]["dim"] == edge_dim
    assert summary["node_schema"]["dim"] == node_dim
