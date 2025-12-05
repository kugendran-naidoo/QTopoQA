import pytest

from qtdaqa.new_dynamic_features.graph_builder2.modules.node.dssp_topo_merge_v1 import (
    DSSPTopologyNodeModule,
    FEATURE_DIM_HINT,
    TOPO_DIM_HINT,
    NODE_BASE_DIM,
)


def test_config_template_has_dim_hint_and_comments() -> None:
    tmpl = DSSPTopologyNodeModule.config_template()
    assert tmpl["module"] == DSSPTopologyNodeModule.module_id
    notes = tmpl.get("notes", {})
    assert "feature_dim_hint" in notes
    assert str(TOPO_DIM_HINT) in notes["feature_dim_hint"]
    assert str(FEATURE_DIM_HINT) in notes["feature_dim_hint"]
    comments = tmpl.get("param_comments", {})
    assert "drop_na" in comments
    assert "jobs" in comments


def test_validate_params_accepts_defaults() -> None:
    params = dict(DSSPTopologyNodeModule._metadata.defaults)
    DSSPTopologyNodeModule.validate_params(params)


def test_validate_params_rejects_bad_jobs() -> None:
    params = dict(DSSPTopologyNodeModule._metadata.defaults)
    params["jobs"] = -1
    with pytest.raises(ValueError):
        DSSPTopologyNodeModule.validate_params(params)


def test_validate_params_normalises_drop_na_str() -> None:
    params = dict(DSSPTopologyNodeModule._metadata.defaults)
    params["drop_na"] = "yes"
    DSSPTopologyNodeModule.validate_params(params)
    assert params["drop_na"] is True
