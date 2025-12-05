import pytest
import pandas as pd
from pathlib import Path

from qtdaqa.new_dynamic_features.graph_builder2.modules.node.dssp_topo_merge_passthrough_v1 import (
    DSSPTopologyMergePassthrough,
    NODE_BASE_DIM,
    TOPO_DIM_HINT,
)


def test_config_template_has_dim_hint_and_comments() -> None:
    tmpl = DSSPTopologyMergePassthrough.config_template()
    assert tmpl["module"] == DSSPTopologyMergePassthrough.module_id
    notes = tmpl.get("notes", {})
    assert "feature_dim_hint" in notes
    assert str(TOPO_DIM_HINT) in notes["feature_dim_hint"]
    assert str(NODE_BASE_DIM) in notes["feature_dim_hint"]
    comments = tmpl.get("param_comments", {})
    assert "drop_na" in comments
    assert "jobs" in comments


def test_validate_params_accepts_defaults() -> None:
    params = dict(DSSPTopologyMergePassthrough._metadata.defaults)
    DSSPTopologyMergePassthrough.validate_params(params)


def test_validate_params_rejects_bad_jobs() -> None:
    params = dict(DSSPTopologyMergePassthrough._metadata.defaults)
    params["jobs"] = 0
    with pytest.raises(ValueError):
        DSSPTopologyMergePassthrough.validate_params(params)


def test_validate_params_normalises_drop_na_str() -> None:
    params = dict(DSSPTopologyMergePassthrough._metadata.defaults)
    params["drop_na"] = "no"
    DSSPTopologyMergePassthrough.validate_params(params)
    assert params["drop_na"] is False
