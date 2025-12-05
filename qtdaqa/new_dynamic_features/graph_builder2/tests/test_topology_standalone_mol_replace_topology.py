import numpy as np
import pytest

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.standalone_MoL_replace_topology_v1 import (
    StandaloneMoLReplaceTopologyModule,
)
from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_common import InterfaceResidue


def _fake_residues():
    return [
        InterfaceResidue("c<A>r<1>R<GLY>", "A", 1, " ", "GLY", np.array([0.0, 0.0, 0.0])),
        InterfaceResidue("c<B>r<2>R<ALA>", "B", 2, " ", "ALA", np.array([1.0, 0.0, 0.0])),
        InterfaceResidue("c<B>r<3>R<ALA>", "B", 3, " ", "ALA", np.array([0.0, 1.0, 0.0])),
    ]


def test_validate_params_sets_sigma_for_gaussian():
    params = {"lap_weight": "gaussian", "neighbor_distance": 8.0}
    StandaloneMoLReplaceTopologyModule.validate_params(params)
    assert params["lap_sigma"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        StandaloneMoLReplaceTopologyModule.validate_params({"lap_weight": "invalid"})


def test_feature_dim_single_and_multi_scale():
    mod_single = StandaloneMoLReplaceTopologyModule(lap_multi_radii=None)
    assert mod_single._feature_dim == 55
    mod_multi = StandaloneMoLReplaceTopologyModule()
    assert mod_multi._feature_dim == 55 * 3
    # ensure columns align with dim
    assert len(mod_multi._columns) == mod_multi._feature_dim


def test_compute_features_lengths_match_dim():
    residues = _fake_residues()
    mod = StandaloneMoLReplaceTopologyModule()
    feats, lap_time = mod._compute_features(residues)
    assert set(feats.keys()) == {r.descriptor for r in residues}
    for values in feats.values():
        assert len(values) == mod._feature_dim
        assert not np.isnan(values).any()
    assert lap_time is None


def test_slq_fallback_path():
    # Force degenerate neighborhood (max_neighbors=1) and slq estimator to hit SLQ path
    residues = _fake_residues()
    mod = StandaloneMoLReplaceTopologyModule(
        lap_multi_radii=None,
        lap_estimator="slq",
        lap_max_neighbors=1,
        lap_eigs_count=4,
    )
    feats, _ = mod._compute_features(residues)
    assert set(feats.keys()) == {r.descriptor for r in residues}
    for values in feats.values():
        assert len(values) == mod._feature_dim
        assert not np.isnan(values).any()


def test_weight_modes_and_ordering_deterministic():
    residues = _fake_residues()
    modes = ["unweighted", "gaussian", "inverse"]
    feature_maps = []
    for mode in modes:
        mod = StandaloneMoLReplaceTopologyModule(lap_multi_radii=None, lap_weight=mode)
        feats, _ = mod._compute_features(residues)
        feature_maps.append(feats)
    # Ensure deterministic ordering and stable outputs across repeated calls for same mode
    mod_repeat = StandaloneMoLReplaceTopologyModule(lap_multi_radii=None, lap_weight="unweighted")
    feats_a, _ = mod_repeat._compute_features(residues)
    feats_b, _ = mod_repeat._compute_features(residues)
    assert feats_a == feats_b


def test_lap_profile_returns_time():
    residues = _fake_residues()
    mod = StandaloneMoLReplaceTopologyModule(lap_profile=True, lap_multi_radii=None)
    _, lap_time = mod._compute_features(residues)
    assert lap_time is not None
    assert lap_time >= 0.0
