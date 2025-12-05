from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common import InterfaceResidue
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.edge_plus_lightweight_mol_v1 import (
    EdgePlusLightweightMoLModule,
)


class DummyAtom:
    def __init__(self, coord):
        self._coord = coord

    def get_coord(self):
        return self._coord


class DummyResidue:
    def __init__(self, coords):
        self._coords = coords

    def get_atoms(self):
        return [DummyAtom(c) for c in self._coords]


class DummyStructure:
    def __init__(self, residues):
        self._residues = residues

    def get_residue(self, chain_id: str, residue_seq: int, insertion_code: str):
        return self._residues.get((chain_id, residue_seq, insertion_code), None)


def _interface_residue(descriptor: str, chain: str, resseq: int, coords) -> InterfaceResidue:
    return InterfaceResidue(
        descriptor=descriptor,
        chain_id=chain,
        residue_seq=resseq,
        insertion_code=" ",
        residue_name="ALA",
        coord=np.asarray(coords, dtype=float),
    )


def test_edge_lightweight_mol_build_edges_deterministic(tmp_path: Path) -> None:
    residues: List[InterfaceResidue] = [
        _interface_residue("A_001_A", "A", 1, [0.0, 0.0, 0.0]),
        _interface_residue("B_002_B", "B", 2, [0.0, 0.0, 5.0]),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}

    dummy_residues = {
        ("A", 1, " "): DummyResidue([[0.0, 0.0, 0.0]]),
        ("B", 2, " "): DummyResidue([[0.0, 0.0, 5.0]]),
    }
    structure = DummyStructure(dummy_residues)

    module = EdgePlusLightweightMoLModule(
        distance_min=0.0,
        distance_max=10.0,
        lap_estimator="exact",
        lap_size_threshold=10,
        lap_max_neighbors=4,
    )

    node_df = pd.DataFrame()  # unused
    dump_path = tmp_path / "edges.csv"
    result = module.build_edges(
        model_key="dummy",
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

    edge_index = result.edge_index
    edge_attr = result.edge_attr
    # four directed edges because the module emits both directions per (i,j)
    assert edge_index.shape[0] == 4
    pairs = {(edge_index[k, 0], edge_index[k, 1]) for k in range(edge_index.shape[0])}
    assert pairs == {(0, 1), (1, 0)}
    # feature_dim should match 11 hist + 5 moments
    assert edge_attr.shape[1] == 16
    assert result.metadata.get("feature_dim") == 16
    # CSV dump exists and has header + 4 rows
    assert dump_path.exists()
    lines = dump_path.read_text().strip().splitlines()
    assert len(lines) == 5


def test_edge_lightweight_mol_metadata_and_profile(tmp_path: Path) -> None:
    residues: List[InterfaceResidue] = [
        _interface_residue("A_001_A", "A", 1, [0.0, 0.0, 0.0]),
        _interface_residue("B_002_B", "B", 2, [0.0, 0.0, 5.0]),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    dummy_residues = {
        ("A", 1, " "): DummyResidue([[0.0, 0.0, 0.0]]),
        ("B", 2, " "): DummyResidue([[0.0, 0.0, 5.0]]),
    }
    structure = DummyStructure(dummy_residues)
    module = EdgePlusLightweightMoLModule(
        distance_min=0.0,
        distance_max=10.0,
        lap_profile=True,
        lap_max_neighbors=3,
    )
    node_df = pd.DataFrame()
    result = module.build_edges(
        model_key="dummy",
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
    meta = result.metadata
    assert meta["feature_dim"] == 16
    assert meta["edge_feature_variant"] == "edge_plus_lightweight_MoL/lean"
    assert meta["lap_max_neighbors"] == 3
    assert meta.get("lap_profile") is True
    assert isinstance(meta.get("lap_time_sec"), float)


def test_edge_lightweight_mol_validate_params_rejects_bad_estimator():
    with pytest.raises(ValueError):
        EdgePlusLightweightMoLModule.validate_params({"lap_estimator": "bogus"})
