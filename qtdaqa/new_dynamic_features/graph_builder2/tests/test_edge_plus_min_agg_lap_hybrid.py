from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.edge_plus_min_agg_lap_hybrid import (
    EdgePlusMinAggLapHybridModule,
)
from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common import InterfaceResidue


class _Atom:
    def __init__(self, coord):
        self._coord = np.asarray(coord, dtype=float)

    def get_coord(self):
        return self._coord


class _Residue:
    def __init__(self, atoms: List[_Atom]):
        self._atoms = atoms

    def get_atoms(self):
        return self._atoms


class _Structure:
    def __init__(self, residues):
        self.residues = residues

    def get_residue(self, chain_id, resnum, insertion_code):
        return self.residues.get((chain_id, resnum, insertion_code))


def _res(descriptor: str, chain: str, resnum: int, coord) -> InterfaceResidue:
    return InterfaceResidue(
        descriptor=descriptor,
        chain_id=chain,
        residue_seq=resnum,
        insertion_code=" ",
        residue_name="GLY",
        coord=np.asarray(coord, dtype=float),
    )


def test_config_template_has_dim_hint_and_comments():
    tmpl = EdgePlusMinAggLapHybridModule.config_template()
    notes = tmpl.get("notes", {})
    assert notes.get("expected_topology_dim") == 172
    formulas = notes.get("feature_dim_formula", {})
    assert "lean" in formulas and "heavy" in formulas
    comments = tmpl.get("param_comments", {})
    for key in ["distance_min", "distance_max", "scale_histogram", "include_norms", "include_cosine", "variant", "include_minmax", "jobs"]:
        assert key in comments


def test_build_edges_lean_dim_and_order(tmp_path: Path):
    module = EdgePlusMinAggLapHybridModule()
    topo_dim = 4  # small for test; expected agg = 3*4 + norms(2) + cosine(1) = 15
    residues = [
        _res("c<A>r<1>R<GLY>", "A", 1, [0.0, 0.0, 0.0]),
        _res("c<B>r<2>R<GLY>", "B", 2, [1.0, 0.0, 0.0]),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    structure = _Structure(
        {
            ("A", 1, " "): _Residue([_Atom([0.0, 0.0, 0.0])]),
            ("B", 2, " "): _Residue([_Atom([1.0, 0.0, 0.0])]),
        }
    )
    topo_df = pd.DataFrame(
        [
            ["c<A>r<1>R<GLY>", *[1.0, 2.0, 3.0, 4.0]],
            ["c<B>r<2>R<GLY>", *[2.0, 3.0, 4.0, 5.0]],
        ],
        columns=["ID", "f1", "f2", "f3", "f4"],
    )
    topo_path = tmp_path / "topology.csv"
    topo_df.to_csv(topo_path, index=False)

    result = module.build_edges(
        model_key="dummy",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=pd.DataFrame(),
        interface_path=tmp_path / "iface.txt",
        topology_path=topo_path,
        node_path=tmp_path / "node.csv",
        pdb_path=tmp_path / "pdb.pdb",
        dump_path=None,
    )

    expected_agg = 3 * topo_dim + 3  # norms(2) + cosine(1)
    expected_dim = 11 + expected_agg
    assert result.edge_attr.shape[1] == expected_dim
    assert result.metadata["feature_dim"] == expected_dim
    assert result.metadata["variant"] == "lean"
    assert result.edge_index.shape[0] == 4  # two residues -> 4 directed edges


def test_build_edges_heavy_dim_and_metadata(tmp_path: Path):
    module = EdgePlusMinAggLapHybridModule(include_minmax=True, variant="heavy")
    topo_dim = 3
    residues = [
        _res("c<A>r<1>R<GLY>", "A", 1, [0.0, 0.0, 0.0]),
        _res("c<B>r<2>R<GLY>", "B", 2, [1.0, 0.0, 0.0]),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    structure = _Structure(
        {
            ("A", 1, " "): _Residue([_Atom([0.0, 0.0, 0.0])]),
            ("B", 2, " "): _Residue([_Atom([1.0, 0.0, 0.0])]),
        }
    )
    topo_df = pd.DataFrame(
        [
            ["c<A>r<1>R<GLY>", *[1.0, 2.0, 3.0]],
            ["c<B>r<2>R<GLY>", *[2.0, 3.0, 4.0]],
        ],
        columns=["ID", "f1", "f2", "f3"],
    )
    topo_path = tmp_path / "topology.csv"
    topo_df.to_csv(topo_path, index=False)

    result = module.build_edges(
        model_key="dummy",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=pd.DataFrame(),
        interface_path=tmp_path / "iface.txt",
        topology_path=topo_path,
        node_path=tmp_path / "node.csv",
        pdb_path=tmp_path / "pdb.pdb",
        dump_path=None,
    )

    expected_agg = 3 * topo_dim + 2 * topo_dim + 3  # lean agg + min/max
    expected_dim = 11 + expected_agg
    assert result.edge_attr.shape[1] == expected_dim
    assert result.metadata["variant"] == "heavy"
    assert result.metadata["include_minmax"] is True
    # ordering is deterministic; four directed edges
    assert result.edge_index.shape[0] == 4
