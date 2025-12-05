from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.edge_plus_min_agg_topo import (
    EdgePlusMinAggTopoModule,
    HIST_DIM,
    DEFAULT_TOPO_DIM_HINT,
)
from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common import InterfaceResidue


class StubAtom:
    def __init__(self, coord):
        self._coord = np.asarray(coord, dtype=float)

    def get_coord(self):
        return self._coord


class StubResidue:
    def __init__(self, atoms: List[StubAtom]):
        self._atoms = atoms

    def get_atoms(self):
        return self._atoms


class StubStructure:
    def __init__(self, residues):
        self.residues = residues

    def get_residue(self, chain_id, resnum, insertion_code):
        return self.residues.get((chain_id, resnum, insertion_code))


def _make_residue(descriptor: str, chain: str, resnum: int, coord) -> InterfaceResidue:
    return InterfaceResidue(
        descriptor=descriptor,
        chain_id=chain,
        residue_seq=resnum,
        insertion_code=" ",
        residue_name="GLY",
        coord=np.asarray(coord, dtype=float),
    )


def test_config_template_has_dim_hint_and_comments() -> None:
    tmpl = EdgePlusMinAggTopoModule.config_template()
    assert tmpl["module"] == EdgePlusMinAggTopoModule.module_id
    notes = tmpl.get("notes", {})
    assert notes.get("expected_topology_dim") == DEFAULT_TOPO_DIM_HINT
    assert "feature_dim_formula" in notes
    comments = tmpl.get("param_comments", {})
    assert "distance_min" in comments and "variant" in comments


def test_validate_params_rejects_bad_variant():
    params = {"variant": "bad"}
    try:
        EdgePlusMinAggTopoModule.validate_params(params)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid variant")


def test_build_edges_leans_variant_feature_dim_and_order(tmp_path: Path):
    module = EdgePlusMinAggTopoModule()
    # small topology dim to make assertions simple
    topo_dim = 4
    residues = [
        _make_residue("c<A>r<1>R<GLY>", "A", 1, [0.0, 0.0, 0.0]),
        _make_residue("c<B>r<2>R<GLY>", "B", 2, [1.0, 0.0, 0.0]),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    atoms_a = [StubAtom([0.0, 0.0, 0.0])]
    atoms_b = [StubAtom([1.0, 0.0, 0.0])]
    structure = StubStructure(
        {
            ("A", 1, " "): StubResidue(atoms_a),
            ("B", 2, " "): StubResidue(atoms_b),
        }
    )
    # build topo CSV
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
    # feature dim = hist 11 + agg (3*topo_dim + norms 2 + cosine 1) = 11 + (12+3)=26
    assert result.edge_attr.shape[1] == HIST_DIM + (3 * topo_dim + 3)
    assert result.metadata["feature_dim"] == result.edge_attr.shape[1]
    # deterministic ordering: two residues => 4 directed edges; distances identical
    assert result.edge_index.shape[0] == 4
    assert np.all(result.edge_attr[:, 0] == result.edge_attr[0, 0])


def test_build_edges_heavy_includes_minmax(tmp_path: Path):
    module = EdgePlusMinAggTopoModule(include_minmax=True, variant="heavy")
    topo_dim = 2
    residues = [
        _make_residue("c<A>r<1>R<GLY>", "A", 1, [0.0, 0.0, 0.0]),
        _make_residue("c<B>r<2>R<GLY>", "B", 2, [1.0, 0.0, 0.0]),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    atoms_a = [StubAtom([0.0, 0.0, 0.0])]
    atoms_b = [StubAtom([1.0, 0.0, 0.0])]
    structure = StubStructure(
        {
            ("A", 1, " "): StubResidue(atoms_a),
            ("B", 2, " "): StubResidue(atoms_b),
        }
    )
    topo_df = pd.DataFrame(
        [
            ["c<A>r<1>R<GLY>", *[1.0, 2.0]],
            ["c<B>r<2>R<GLY>", *[2.0, 3.0]],
        ],
        columns=["ID", "f1", "f2"],
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
    # Heavy: agg = 3*topo_dim + 2*topo_dim (minmax) + norms 2 + cosine 1
    expected_agg = 3 * topo_dim + 2 * topo_dim + 3
    assert result.edge_attr.shape[1] == HIST_DIM + expected_agg
    assert result.metadata["variant"] == "heavy"
    assert result.metadata["include_minmax"] is True
