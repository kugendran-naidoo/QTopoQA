import pandas as pd
import pytest

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology import (
    persistence_partial_k_partite_v1 as mod,
)
from qtdaqa.new_dynamic_features.graph_builder2.lib import new_topological_features


class DummyStructure:
    def __init__(self, atoms):
        self._atoms = atoms

    def get_atoms(self):
        return iter(self._atoms)


def _make_res(chain_id: str, resnum: int, name: str) -> new_topological_features.ResidueDescriptor:
    return new_topological_features.ResidueDescriptor(
        chain_id=chain_id,
        residue_number=resnum,
        residue_name=name,
        insertion_code=" ",
        raw_descriptor=f"c<{chain_id}>r<{resnum}>R<{name}>",
    )


def test_run_partial_single_aligns_blocks(monkeypatch, tmp_path):
    # Two chains with 3 residues total to exercise base/cross/per-chain/dssp/polar.
    residues = [_make_res("A", 1, "GLY"), _make_res("A", 2, "ALA"), _make_res("B", 3, "LYS")]

    # Patch interface loader
    monkeypatch.setattr(mod, "_load_interface_descriptors", lambda path: (residues, None))
    # Patch structure loader (atom count gate)
    monkeypatch.setattr(new_topological_features, "_load_structure", lambda path: DummyStructure([1, 2, 3]))

    def fake_compute(pdb_path, res_list, config, **kwargs):
        # Return predictable IDs and two numeric cols sized to res_list
        ids = [r.to_string() for r in res_list]
        return pd.DataFrame({"ID": ids, "x": range(len(ids)), "y": range(len(ids))})

    monkeypatch.setattr(new_topological_features, "compute_features_for_residues", fake_compute)

    base_cfg = new_topological_features.TopologicalConfig(
        neighbor_distance=8.0,
        filtration_cutoff=8.0,
        min_persistence=0.01,
        element_filters=[["C"]],
        dedup_sort=False,
        workers=None,
    )

    df, notes = mod._run_partial_single(
        pdb_path=tmp_path / "dummy.pdb",
        interface_path=tmp_path / "dummy.interface.txt",
        base_config=base_cfg,
        enable_cross_bias=True,
        enable_cross_only=True,
        enable_per_chain=True,
        enable_dssp_block=True,
        enable_polar_block=True,
        penalty_value=4.0,
        intra_penalty_lambda=0.0,
        max_chains=2,
        max_atoms=25000,
        max_block_seconds=10.0,
        polar_hbond_weight=False,
        slow_threshold=0.0,
    )

    # Expect rows equal to all residues and columns for each block (base + cross + cross-only + per-chain A + per-chain B + dssp + polar)
    assert len(df) == len(residues)
    # Each block contributes 2 cols; 7 blocks total -> 14 numeric + ID
    assert df.shape[1] == 1 + 14
    # Per-chain blocks should be aligned to full residue set (zeros for non-chain residues).
    # Block order by position after inserting ID at col 0:
    # base(2), cross(2), cross_only(2), per_chain_A(2), per_chain_B(2), dssp(2), polar(2)
    ids = df["ID"].tolist()
    chain_a_idxs = [i for i, rid in enumerate(ids) if rid.startswith("c<A>")]
    chain_b_idxs = [i for i, rid in enumerate(ids) if rid.startswith("c<B>")]
    base_start = 1  # numeric columns start after ID
    per_chain_a_start = base_start + 3 * 2  # skip base, cross, cross_only
    per_chain_b_start = per_chain_a_start + 2
    for idx in chain_b_idxs:
        assert (df.iloc[idx, per_chain_a_start : per_chain_a_start + 2] == 0).all()
    for idx in chain_a_idxs:
        assert (df.iloc[idx, per_chain_b_start : per_chain_b_start + 2] == 0).all()
    # No errors recorded
    assert notes == []


def test_validate_params_rejects_bad_preset():
    with pytest.raises(ValueError):
        mod.PersistencePartialKPartiteModule.validate_params({"preset": "unknown"})
