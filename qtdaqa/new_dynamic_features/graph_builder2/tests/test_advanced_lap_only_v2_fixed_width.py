from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_res(chain: str, num: int, name: str, coord):
    edge_common = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common"
    )
    return edge_common.InterfaceResidue(
        descriptor=f"c<{chain}>r<{num}>R<{name}>",
        chain_id=chain,
        residue_seq=num,
        insertion_code=" ",
        residue_name=name,
        coord=np.asarray(coord, dtype=float),
    )


class DummyStructure:
    def __init__(self, n_atoms: int):
        self._atoms = [object()] * n_atoms

    def get_atoms(self):
        return iter(self._atoms)


def test_fixed_width_columns_constant_across_chain_counts(monkeypatch, tmp_path: Path):
    mod = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_k_partite_advanced_laplacian_only_v2"
    )
    new_topo = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.lib.new_topological_features"
    )

    residues_ab = [_make_res("A", 1, "GLY", (0.0, 0.0, 0.0)), _make_res("B", 2, "ALA", (1.0, 0.0, 0.0))]
    residues_a = [_make_res("A", 1, "GLY", (0.0, 0.0, 0.0))]

    def fake_loader(path: Path):
        if path.name.endswith("ab.interface.txt"):
            return residues_ab
        return residues_a

    lap_cols = mod._build_lap_columns(4, (1, 2), (0.1,))

    monkeypatch.setattr(mod, "_parse_interface_file", fake_loader)
    monkeypatch.setattr(new_topo, "_load_structure", lambda path: DummyStructure(10))
    monkeypatch.setattr(mod, "_laplacian_features", lambda *args, **kwargs: [1.0] * len(lap_cols))

    df_ab, _, _ = mod._run_lap_only_v2_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "ab.interface.txt",
        preset="lean",
        k_max=2,
        secondary_partition="none",
        secondary_k_max=4,
        lap_graph_mode="cross_chain",
        lap_graph_mode_primary="all",
        lap_distance_cutoff=8.0,
        lap_k_neighbors=None,
        lap_max_neighbors=128,
        lap_edge_weight="gaussian",
        lap_sigma=4.0,
        lap_eigs_count=4,
        lap_moment_orders=(1, 2),
        lap_heat_times=(0.1,),
        lap_include_entropy=True,
        lap_normalize="sym",
        max_atoms=25000,
        max_block_seconds=10.0,
        slow_threshold=0.0,
        dssp_timeout_seconds=0.0,
        dssp_slow_threshold=0.0,
    )

    df_a, _, _ = mod._run_lap_only_v2_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "a.interface.txt",
        preset="lean",
        k_max=2,
        secondary_partition="none",
        secondary_k_max=4,
        lap_graph_mode="cross_chain",
        lap_graph_mode_primary="all",
        lap_distance_cutoff=8.0,
        lap_k_neighbors=None,
        lap_max_neighbors=128,
        lap_edge_weight="gaussian",
        lap_sigma=4.0,
        lap_eigs_count=4,
        lap_moment_orders=(1, 2),
        lap_heat_times=(0.1,),
        lap_include_entropy=True,
        lap_normalize="sym",
        max_atoms=25000,
        max_block_seconds=10.0,
        slow_threshold=0.0,
        dssp_timeout_seconds=0.0,
        dssp_slow_threshold=0.0,
    )

    assert list(df_ab.columns) == list(df_a.columns)
    assert len(set(df_ab.columns)) == len(df_ab.columns)
    assert "primary.s1.present" in df_a.columns
    assert df_a["primary.s1.present"].eq(0).all()


def test_deterministic_slot_columns_present(monkeypatch, tmp_path: Path):
    mod = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_k_partite_advanced_laplacian_only_v2"
    )
    new_topo = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.lib.new_topological_features"
    )

    residues = [_make_res("B", 2, "ALA", (1.0, 0.0, 0.0)), _make_res("A", 1, "GLY", (0.0, 0.0, 0.0))]
    lap_cols = mod._build_lap_columns(4, (1, 2), (0.1,))

    monkeypatch.setattr(mod, "_parse_interface_file", lambda path: residues)
    monkeypatch.setattr(new_topo, "_load_structure", lambda path: DummyStructure(10))
    monkeypatch.setattr(mod, "_laplacian_features", lambda *args, **kwargs: [1.0] * len(lap_cols))

    df, _, slot_map = mod._run_lap_only_v2_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "iface.interface.txt",
        preset="lean",
        k_max=2,
        secondary_partition="none",
        secondary_k_max=4,
        lap_graph_mode="cross_chain",
        lap_graph_mode_primary="all",
        lap_distance_cutoff=8.0,
        lap_k_neighbors=None,
        lap_max_neighbors=128,
        lap_edge_weight="gaussian",
        lap_sigma=4.0,
        lap_eigs_count=4,
        lap_moment_orders=(1, 2),
        lap_heat_times=(0.1,),
        lap_include_entropy=True,
        lap_normalize="sym",
        max_atoms=25000,
        max_block_seconds=10.0,
        slow_threshold=0.0,
        dssp_timeout_seconds=0.0,
        dssp_slow_threshold=0.0,
    )

    assert slot_map["primary_slots"][0]["label"] == "A"
    assert slot_map["primary_slots"][1]["label"] == "B"
    assert any(col.startswith("primary.s0.") for col in df.columns)
    assert any(col.startswith("primary.s1.") for col in df.columns)
    assert any(col.startswith("pair.s0_s1.") for col in df.columns)


def test_validate_params_rejects_bad_graph_mode():
    mod = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_k_partite_advanced_laplacian_only_v2"
    )
    with pytest.raises(ValueError):
        mod.PersistenceKPartiteAdvancedLaplacianOnlyV2Module.validate_params({"lap_graph_mode": "bad"})
