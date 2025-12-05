from pathlib import Path
import json

from qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_basic_v1 import (
    PersistenceTopologyModule,
    FEATURE_DIM,
)


def test_config_template_contains_dim_hint_and_params() -> None:
    tmpl = PersistenceTopologyModule.config_template()
    assert tmpl["module"] == PersistenceTopologyModule.module_id
    params = tmpl["params"]
    assert params["neighbor_distance"] == 8.0
    assert params["filtration_cutoff"] == 8.0
    assert tmpl.get("notes", {}).get("feature_dim") == FEATURE_DIM
    # Param comments should exist for key params
    comments = tmpl.get("param_comments", {})
    assert "neighbor_distance" in comments
    assert "filtration_cutoff" in comments


def test_validate_params_accepts_defaults(tmp_path: Path) -> None:
    params = dict(PersistenceTopologyModule._metadata.defaults)
    # Should not raise
    PersistenceTopologyModule.validate_params(params)


def test_validate_params_rejects_bad_values() -> None:
    params = dict(PersistenceTopologyModule._metadata.defaults)
    params["neighbor_distance"] = -1
    try:
        PersistenceTopologyModule.validate_params(params)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for negative neighbor_distance"


def test_feature_dim_hint_serialises() -> None:
    """Ensure config_template notes are JSON serialisable (used in --create-feature-config)."""
    tmpl = PersistenceTopologyModule.config_template()
    json.dumps(tmpl)
