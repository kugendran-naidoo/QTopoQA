from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder.lib.edge_common import InterfaceResidue
from qtdaqa.new_dynamic_features.graph_builder.modules.edge_legacy_v11 import LegacyEdgeModuleV11
from qtdaqa.new_dynamic_features.graph_builder.modules.edge_multiscale_v24 import MultiscaleEdgeModuleV24


class _FakeResidue:
    def __init__(self, coords):
        self._coords = [np.asarray(coord, dtype=float) for coord in coords]

    def get_atoms(self):
        for coord in self._coords:
            yield _FakeAtom(coord)


class _FakeAtom:
    def __init__(self, coord):
        self._coord = coord

    def get_coord(self):
        return self._coord


class _DummyStructure:
    def __init__(self):
        self._store = {
            ("A", 1, " "): _FakeResidue([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            ("B", 2, " "): _FakeResidue([[0.0, 3.0, 0.0], [0.0, 3.5, 0.0]]),
        }

    def get_residue(self, chain_id: str, residue_seq: int, insertion_code: str):
        return self._store.get((chain_id, residue_seq, insertion_code or " "))


def _build_test_residues():
    residues = [
        InterfaceResidue(
            descriptor="c<A>r<1>R<ALA>",
            chain_id="A",
            residue_seq=1,
            insertion_code=" ",
            residue_name="ALA",
            coord=np.array([0.0, 0.0, 0.0]),
        ),
        InterfaceResidue(
            descriptor="c<B>r<2>R<ARG>",
            chain_id="B",
            residue_seq=2,
            insertion_code=" ",
            residue_name="ARG",
            coord=np.array([0.0, 3.5, 0.0]),
        ),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    return residues, id_to_index


def test_multiscale_edge_module_produces_expected_features():
    residues, id_to_index = _build_test_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0]})

    module = MultiscaleEdgeModuleV24()
    result = module.build_edges(
        model_key="test_model",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=node_df,
        interface_path=Path("iface"),
        topology_path=Path("topo"),
        node_path=Path("node"),
        pdb_path=Path("pdb"),
        dump_path=None,
    )

    assert result.edge_index.shape[0] == 4  # two directed edges
    assert result.edge_attr.shape[0] == 4
    distance = result.edge_attr[0, 0]
    assert np.isclose(distance, 3.5, atol=1e-3)
    inverse_distance = result.edge_attr[0, 1]
    assert np.isclose(inverse_distance, 1.0 / 3.5, atol=1e-3)
    assert result.metadata["edge_feature_variant"] == "multi_scale_v24"


def test_legacy_edge_module_matches_11d_shape(tmp_path):
    residues, id_to_index = _build_test_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0]})

    module = LegacyEdgeModuleV11(distance_max=12.0)
    dump_path = tmp_path / "edges.csv"
    result = module.build_edges(
        model_key="legacy_model",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=node_df,
        interface_path=Path("iface"),
        topology_path=Path("topo"),
        node_path=Path("node"),
        pdb_path=Path("pdb"),
        dump_path=dump_path,
    )

    assert result.edge_attr.shape[1] == 11  # distance + 10 histogram bins
    assert result.edge_index.shape[0] == result.edge_attr.shape[0]
    assert dump_path.exists()
    assert result.metadata["edge_feature_variant"] == "legacy_v11"
