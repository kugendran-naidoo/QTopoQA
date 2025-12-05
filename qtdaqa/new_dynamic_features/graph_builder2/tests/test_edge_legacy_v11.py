from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.legacy_v11 import (
    LegacyEdgeModuleV11,
    FEATURE_DIM,
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


def test_config_template_has_dim_and_comments() -> None:
    tmpl = LegacyEdgeModuleV11.config_template()
    assert tmpl["module"] == LegacyEdgeModuleV11.module_id
    assert tmpl.get("notes", {}).get("feature_dim") == FEATURE_DIM
    comments = tmpl.get("param_comments", {})
    assert "distance_min" in comments
    assert "distance_max" in comments


def test_validate_params_rejects_bad_window():
    params = {"distance_min": 5.0, "distance_max": 1.0}
    try:
        LegacyEdgeModuleV11.validate_params(params)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid distance window")


def test_build_edges_deterministic_and_feature_dim(tmp_path: Path):
    module = LegacyEdgeModuleV11()
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
    result = module.build_edges(
        model_key="dummy",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=pd.DataFrame(),
        interface_path=tmp_path / "iface.txt",
        topology_path=tmp_path / "topo.csv",
        node_path=tmp_path / "node.csv",
        pdb_path=tmp_path / "pdb.pdb",
        dump_path=None,
    )
    # Expect 4 edges (bidirectional for each loop iteration) sorted by src,dst,distance
    assert result.edge_attr.shape[1] == FEATURE_DIM
    assert result.edge_index.shape[0] == 4
    # Distances should be identical, ordering deterministic
    assert np.all(result.edge_attr[:, 0] == result.edge_attr[0, 0])
    # Metadata includes feature_dim and variant
    assert result.metadata["feature_dim"] == FEATURE_DIM
    assert result.metadata["edge_feature_variant"] == "legacy_v11"
    assert result.metadata["distance_window"] == [0.0, 10.0]
