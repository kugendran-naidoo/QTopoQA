from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common import InterfaceResidue
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.edge_plus_pool_agg_topo import (
    EdgePlusPoolAggTopoModule,
)


class _FakeAtom:
    def __init__(self, coord):
        self._coord = coord

    def get_coord(self):
        return self._coord


class _FakeResidue:
    def __init__(self, coords):
        self._coords = [np.asarray(coord, dtype=float) for coord in coords]

    def get_atoms(self):
        for coord in self._coords:
            yield _FakeAtom(coord)


class _DummyStructure:
    def __init__(self):
        self._store = {
            ("A", 1, " "): _FakeResidue([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            ("B", 2, " "): _FakeResidue([[0.0, 3.0, 0.0], [0.0, 3.5, 0.0]]),
            ("B", 3, " "): _FakeResidue([[0.0, 5.0, 0.0], [0.0, 5.5, 0.0]]),
        }

    def get_residue(self, chain_id: str, residue_seq: int, insertion_code: str):
        return self._store.get((chain_id, residue_seq, insertion_code or " "))


def _build_residues() -> Tuple[List[InterfaceResidue], dict]:
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
        InterfaceResidue(
            descriptor="c<B>r<3>R<GLY>",
            chain_id="B",
            residue_seq=3,
            insertion_code=" ",
            residue_name="GLY",
            coord=np.array([0.0, 5.0, 0.0]),
        ),
    ]
    id_to_index = {res.descriptor: idx for idx, res in enumerate(residues)}
    return residues, id_to_index


def _write_topology(tmp_path: Path, ids: List[str], vectors: List[List[float]]) -> Path:
    feature_cols = [f"f{i}" for i in range(len(vectors[0]))]
    path = tmp_path / "topology.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID"] + feature_cols)
        for identifier, vec in zip(ids, vectors):
            writer.writerow([identifier] + list(vec))
    return path


def test_edge_plus_pool_agg_topo_builds_expected_shapes(tmp_path):
    residues, id_to_index = _build_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0, 3.0]})
    topo_vectors = [[1.0, 0.5, 0.25], [0.2, 0.3, 0.4], [0.3, 0.1, 0.2]]
    topology_path = _write_topology(tmp_path, node_df["ID"].tolist(), topo_vectors)

    module = EdgePlusPoolAggTopoModule(scale_histogram=False, pool_k=1)  # keep histogram block unscaled
    result = module.build_edges(
        model_key="test_model",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=node_df,
        interface_path=Path("iface"),
        topology_path=topology_path,
        node_path=Path("node"),
        pdb_path=Path("pdb"),
        dump_path=None,
    )

    topo_dim = len(topo_vectors[0])
    endpoint_agg_dim = topo_dim * 4 + 2 + 1  # concat + mean + abs diff + norms + cosine
    pooled_agg_dim = topo_dim * 4 + 2 + 1    # same structure on pooled means
    agg_dim = endpoint_agg_dim + pooled_agg_dim
    expected_dim = module._HIST_DIM + agg_dim

    assert result.edge_index.shape[0] == result.edge_attr.shape[0]
    assert result.edge_attr.shape[1] == expected_dim
    assert result.metadata["edge_feature_variant"] == "edge_plus_pool_agg_topo/lean"
    assert result.metadata["variant"] == "lean"
    assert result.metadata["pool_k"] == 1
    assert result.metadata["feature_dim"] == expected_dim
    assert result.metadata["topology_feature_dim"] == topo_dim
    assert result.metadata["include_norms"] is True
    assert result.metadata["include_cosine"] is True


def test_edge_plus_pool_agg_topo_rejects_heavy_variant():
    with pytest.raises(ValueError):
        EdgePlusPoolAggTopoModule.validate_params({"variant": "bogus"})


def test_edge_plus_pool_agg_topo_heavy_includes_minmax(tmp_path):
    residues, id_to_index = _build_residues()
    structure = _DummyStructure()
    node_df = pd.DataFrame({"ID": [res.descriptor for res in residues], "feat": [1.0, 2.0, 3.0]})
    topo_vectors = [[1.0, 0.5, 0.25], [0.2, 0.3, 0.4], [0.3, 0.1, 0.2]]
    topology_path = _write_topology(tmp_path, node_df["ID"].tolist(), topo_vectors)

    module = EdgePlusPoolAggTopoModule(scale_histogram=False, pool_k=1, variant="heavy", include_minmax=True)
    result = module.build_edges(
        model_key="test_heavy",
        residues=residues,
        id_to_index=id_to_index,
        structure=structure,
        node_df=node_df,
        interface_path=Path("iface"),
        topology_path=topology_path,
        node_path=Path("node"),
        pdb_path=Path("pdb"),
        dump_path=None,
    )

    topo_dim = len(topo_vectors[0])
    endpoint_agg_dim = topo_dim * 6 + 2 + 1  # concat + mean + abs diff + min + max + norms + cosine
    pooled_agg_dim = topo_dim * 6 + 2 + 1    # pooled block mirrors endpoint with min/max
    agg_dim = endpoint_agg_dim + pooled_agg_dim
    expected_dim = module._HIST_DIM + agg_dim

    assert result.edge_attr.shape[1] == expected_dim
    assert result.metadata["variant"] == "heavy"
    assert result.metadata["include_minmax"] is True
    assert result.metadata["edge_feature_variant"] == "edge_plus_pool_agg_topo/heavy"
    # Sorted by (src, dst, distance)
    assert result.edge_index.tolist() == [[0, 1], [0, 2], [1, 0], [2, 0]]


def test_edge_plus_pool_agg_topo_template_has_dim_hint_and_comments():
    template = EdgePlusPoolAggTopoModule.config_template()
    comments = template["param_comments"]
    required_comments = [
        "distance_min",
        "distance_max",
        "scale_histogram",
        "pool_k",
        "include_norms",
        "include_cosine",
        "variant",
        "include_minmax",
        "jobs",
    ]
    for key in required_comments:
        assert key in comments
    assert "dim_hint" in template
    # heavy alternate should carry dim hint and variant flag
    alternates = template.get("alternates", [])
    assert alternates, "expected heavy alternate in template"
    assert "dim_hint" in alternates[0]
    assert alternates[0]["params"]["variant"] == "heavy"
