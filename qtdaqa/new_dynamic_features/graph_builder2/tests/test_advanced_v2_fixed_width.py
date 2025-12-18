from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd


def _make_res(chain: str, num: int, name: str):
    mod = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.lib.new_topological_features"
    )
    return mod.ResidueDescriptor(
        chain_id=chain,
        residue_number=num,
        residue_name=name,
        insertion_code=" ",
        raw_descriptor=f"c<{chain}>r<{num}>R<{name}>",
    )


class DummyStructure:
    def __init__(self, n_atoms: int):
        self._atoms = [object()] * n_atoms

    def get_atoms(self):
        return iter(self._atoms)


def test_fixed_width_columns_constant_across_chain_counts(monkeypatch, tmp_path: Path):
    mod = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_k_partite_advanced_v2"
    )
    new_topo = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.lib.new_topological_features"
    )

    residues_ab = [_make_res("A", 1, "GLY"), _make_res("B", 2, "ALA")]
    residues_a = [_make_res("A", 1, "GLY")]

    def fake_loader(path: Path):
        if path.name.endswith("ab.interface.txt"):
            return residues_ab, None
        return residues_a, None

    monkeypatch.setattr(mod, "_load_interface_descriptors", fake_loader)
    monkeypatch.setattr(new_topo, "_load_structure", lambda path: DummyStructure(10))

    def fake_compute(pdb_path, res_list, config, **kwargs):
        ids = [r.raw_descriptor for r in res_list]
        return pd.DataFrame({"ID": ids, "x": range(len(ids)), "y": range(len(ids))})

    monkeypatch.setattr(new_topo, "compute_features_for_residues", fake_compute)

    cfg = new_topo.TopologicalConfig(
        neighbor_distance=8.0,
        filtration_cutoff=8.0,
        min_persistence=0.01,
        element_filters=[("C",)],
        dedup_sort=False,
        workers=None,
    )

    df_ab, notes_ab, _ = mod._run_adv_v2_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "ab.interface.txt",
        base_config=cfg,
        strat_config=cfg,
        preset="lean",
        secondary_partition="none",
        k_max=2,
        secondary_k_max=4,
        enable_cross_bias=False,
        penalty_value=4.0,
        max_atoms=25000,
        max_block_seconds=10.0,
        enable_polar_block=False,
        polar_hbond_weight=False,
        polar_hbond_weight_factor=0.5,
        polar_hbond_energy_cutoff=-0.5,
        polar_hbond_inter_only=False,
        enable_typed_block=False,
        slow_threshold=0.0,
        dssp_timeout_seconds=0.0,
        dssp_slow_threshold=0.0,
    )

    df_a, notes_a, _ = mod._run_adv_v2_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "a.interface.txt",
        base_config=cfg,
        strat_config=cfg,
        preset="lean",
        secondary_partition="none",
        k_max=2,
        secondary_k_max=4,
        enable_cross_bias=False,
        penalty_value=4.0,
        max_atoms=25000,
        max_block_seconds=10.0,
        enable_polar_block=False,
        polar_hbond_weight=False,
        polar_hbond_weight_factor=0.5,
        polar_hbond_energy_cutoff=-0.5,
        polar_hbond_inter_only=False,
        enable_typed_block=False,
        slow_threshold=0.0,
        dssp_timeout_seconds=0.0,
        dssp_slow_threshold=0.0,
    )

    assert list(df_ab.columns) == list(df_a.columns)
    assert len(set(df_ab.columns)) == len(df_ab.columns)

    # Chain B is missing in the second input, so slot primary.s1 should be padded.
    assert "primary.s1.present" in df_a.columns
    assert df_a["primary.s1.present"].eq(0).all()


def test_deterministic_slot_columns_present(monkeypatch, tmp_path: Path):
    mod = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_k_partite_advanced_v2"
    )
    new_topo = importlib.import_module(
        "qtdaqa.new_dynamic_features.graph_builder2.lib.new_topological_features"
    )

    residues = [_make_res("B", 2, "ALA"), _make_res("A", 1, "GLY")]

    monkeypatch.setattr(mod, "_load_interface_descriptors", lambda path: (residues, None))
    monkeypatch.setattr(new_topo, "_load_structure", lambda path: DummyStructure(10))
    monkeypatch.setattr(
        new_topo,
        "compute_features_for_residues",
        lambda pdb_path, res_list, config, **kwargs: pd.DataFrame(
            {"ID": [r.raw_descriptor for r in res_list], "x": range(len(res_list)), "y": range(len(res_list))}
        ),
    )

    cfg = new_topo.TopologicalConfig(
        neighbor_distance=8.0,
        filtration_cutoff=8.0,
        min_persistence=0.01,
        element_filters=[("C",)],
        dedup_sort=False,
        workers=None,
    )

    df, _, slot_map = mod._run_adv_v2_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "iface.interface.txt",
        base_config=cfg,
        strat_config=cfg,
        preset="lean",
        secondary_partition="none",
        k_max=2,
        secondary_k_max=4,
        enable_cross_bias=False,
        penalty_value=4.0,
        max_atoms=25000,
        max_block_seconds=10.0,
        enable_polar_block=False,
        polar_hbond_weight=False,
        polar_hbond_weight_factor=0.5,
        polar_hbond_energy_cutoff=-0.5,
        polar_hbond_inter_only=False,
        enable_typed_block=False,
        slow_threshold=0.0,
        dssp_timeout_seconds=0.0,
        dssp_slow_threshold=0.0,
    )

    # Sorted chain IDs: A then B => primary.s0 should map to A.
    assert slot_map["primary_slots"][0]["label"] == "A"
    assert slot_map["primary_slots"][1]["label"] == "B"

    # Fixed slots are present in the schema regardless of per-structure chain order.
    assert any(col.startswith("primary.s0.") for col in df.columns)
    assert any(col.startswith("primary.s1.") for col in df.columns)
    assert any(col.startswith("pair.s0_s1.") for col in df.columns)
