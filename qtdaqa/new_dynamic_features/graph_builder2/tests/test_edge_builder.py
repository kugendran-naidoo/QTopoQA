from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common import InterfaceResidue
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.legacy_v11 import LegacyEdgeModuleV11
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.legacy_plus_topo_pair import (
    LegacyPlusTopoPairEdgeModule,
)
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.multi_scale_v24 import MultiscaleEdgeModuleV24
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.neo_v24 import NeoEdgeModuleV24


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
            ("A", 2, " "): _FakeResidue([[0.0, 1.5, 0.0], [0.0, 2.0, 0.0]]),
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


def _build_same_chain_residues():
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
            descriptor="c<A>r<2>R<ARG>",
            chain_id="A",
            residue_seq=2,
            insertion_code=" ",
            residue_name="ARG",
            coord=np.array([0.0, 2.0, 0.0]),
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
    metadata = result.metadata
    assert metadata["edge_feature_variant"] == "multi_scale_v24"
    assert metadata["histogram_bins"] == module.builder.hist_bins.tolist()
    assert metadata["contact_threshold"] == module.builder.contact_threshold
    assert metadata["include_inverse_distance"] is True
    assert metadata["include_unit_vector"] is True
    assert metadata["unit_vector_epsilon"] == module.builder.unit_vector_epsilon


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


def test_multiscale_edge_module_zero_edges_has_fixed_width():
    residues, id_to_index = _build_same_chain_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0]})

    module = MultiscaleEdgeModuleV24()
    result = module.build_edges(
        model_key="no_edges_model",
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

    expected_dim = module.builder.feature_dim
    assert result.edge_index.shape == (0, 2)
    assert result.edge_attr.shape == (0, expected_dim)
    assert result.metadata["edge_count"] == 0
    assert result.metadata["feature_dim"] == expected_dim


def test_legacy_edge_module_zero_edges_preserves_metadata():
    residues, id_to_index = _build_same_chain_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0]})

    module = LegacyEdgeModuleV11()
    result = module.build_edges(
        model_key="legacy_no_edges",
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

    assert result.edge_index.shape == (0, 2)
    assert result.edge_attr.shape == (0, 11)
    assert result.metadata["edge_count"] == 0
    assert result.metadata["feature_dim"] == 11


def test_legacy_plus_topo_pair_module_combines_features():
    residues, id_to_index = _build_test_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0]})

    module = LegacyPlusTopoPairEdgeModule(distance_max=12.0, include_neighbors=False, neighbor_distance=0.0)
    result = module.build_edges(
        model_key="legacy_topo_model",
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

    assert result.edge_attr.shape[1] == 31  # 11 legacy + 20 topology stats
    assert result.edge_index.shape[0] == result.edge_attr.shape[0]
    metadata = result.metadata
    assert metadata["edge_feature_variant"] == "legacy_plus_topo_pair"
    assert metadata["topology_feature_dim"] == 20
    assert metadata["legacy_feature_dim"] == 11
    assert metadata["feature_dim"] == 31


def test_neo_edge_module_hybrid_features(tmp_path):
    residues, id_to_index = _build_test_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0]})

    module = NeoEdgeModuleV24(
        histogram_bins=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        contact_thresholds=[5.0, 10.0],
        histogram_mode="density_times_contact",
    )
    dump_path = tmp_path / "neo_edges.csv"
    result = module.build_edges(
        model_key="neo_model",
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

    assert result.edge_attr.shape[1] == module.builder.feature_dim
    assert dump_path.exists()
    metadata = result.metadata
    assert metadata["edge_feature_variant"] == "neo_v24"
    assert metadata["histogram_bins"] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert metadata["long_band_mask"] is True
    assert metadata["contact_thresholds"] == [5.0, 10.0]
    assert metadata["include_inverse_distance"] is True
    assert metadata["short_contact_max"] == module.builder.short_contact_max
    assert metadata["contact_normalizer"] == module.builder.contact_normalizer
