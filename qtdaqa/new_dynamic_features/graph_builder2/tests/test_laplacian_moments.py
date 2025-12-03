from __future__ import annotations

import numpy as np

from qtdaqa.new_dynamic_features.graph_builder2.lib.laplacian_moments import (
    LaplacianMomentConfig,
    compute_laplacian_moments,
)
from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.lightweight_mol_v1 import (
    TopologyLightweightMoLModule,
)
from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.edge_plus_lightweight_mol_v1 import (
    EdgePlusLightweightMoLModule,
)


def test_laplacian_moments_exact_two_node_line():
    # Two nodes connected: eigenvalues {0, 2}
    adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    raw, centered = compute_laplacian_moments(
        adj, moment_orders=(1, 2, 3, 4), config=LaplacianMomentConfig(size_threshold=10)
    )
    # raw moments are means of powers of eigenvalues
    assert np.isclose(raw[0], 1.0)
    assert np.isclose(raw[1], 2.0)
    assert np.isclose(raw[2], 4.0)
    assert np.isclose(raw[3], 8.0)
    # centered moments (variance, skew, kurt proxy)
    assert np.isclose(centered[0], 1.0)  # variance
    assert np.isclose(centered[1], 0.0)  # skew
    assert np.isclose(centered[2], 1.0)  # kurt (central 4th moment)


def test_config_templates_have_alias_and_desc():
    topo_template = TopologyLightweightMoLModule.config_template()
    assert "alias" in topo_template and "Lap" in topo_template["alias"]
    assert "summary" in topo_template and "Laplacian" in topo_template["summary"]
    assert "description" in topo_template and "normalized-Laplacian" in topo_template["description"]

    edge_template = EdgePlusLightweightMoLModule.config_template()
    assert "alias" in edge_template and "Edge 16D" in edge_template["alias"]
    assert "summary" in edge_template and "Laplacian" in edge_template["summary"]
    assert "description" in edge_template and "0-10 A" in edge_template["description"]


def test_laplacian_moments_slq_path_with_seed():
    adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    config = LaplacianMomentConfig(size_threshold=0, estimator="slq", slq_probes=16, seed=123)
    raw, centered = compute_laplacian_moments(adj, moment_orders=(1, 2, 3), config=config)
    # SLQ estimates should be finite and roughly near exact values (1,2,4 for this graph)
    assert all(np.isfinite(raw))
    assert raw[0] > 0.5 and raw[0] < 1.5
    assert raw[1] > 1.0 and raw[1] < 3.5
    assert raw[2] > 2.0 and raw[2] < 7.0
    # centered moments should be finite
    assert all(np.isfinite(centered))
