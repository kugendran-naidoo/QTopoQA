from __future__ import annotations

import numpy as np

from qtdaqa.new_graph_builder.lib.config import load_config
from qtdaqa.new_graph_builder.lib.pt_writer import EdgeFeatureBuilder, InterfaceResidue


class _FakeAtom:
    def __init__(self, coord):
        self._coord = np.asarray(coord, dtype=float)

    def get_coord(self):
        return self._coord


class _FakeResidue:
    def __init__(self, coords):
        self._atoms = [_FakeAtom(c) for c in coords]

    def get_atoms(self):
        return iter(self._atoms)


class _DummyStructure:
    def __init__(self):
        self._map = {
            ("A", 1, " "): _FakeResidue([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            ("B", 2, " "): _FakeResidue([[0.0, 3.0, 0.0], [0.0, 3.5, 0.0]]),
        }

    def get_residue(self, chain_id: str, residue_seq: int, insertion_code: str):
        return self._map.get((chain_id, residue_seq, insertion_code or " "))


def test_edge_feature_builder_produces_multiscale_edges():
    config = load_config(None)
    builder = EdgeFeatureBuilder(config.edge)

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
    structure = _DummyStructure()
    edges, features = builder.build_edges(residues, id_to_index, structure, edge_dump_path=None)

    assert edges.shape[0] == 4  # two directed edges per pair
    assert features.shape[0] == 4
    # distance should equal approx 3.5
    assert np.isclose(features[0, 0], 3.5, atol=1e-3)
    # inverse distance present
    if config.edge.include_inverse_distance:
        assert np.isclose(features[0, 1], 1.0 / 3.5, atol=1e-3)
