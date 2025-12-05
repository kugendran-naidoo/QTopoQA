from pathlib import Path

import pytest

from qtdaqa.new_dynamic_features.graph_builder2.modules.interface.polar_cutoff_v1 import (
    DefaultInterfaceModule,
)


def test_config_template_has_comments_and_notes() -> None:
    tmpl = DefaultInterfaceModule.config_template()
    assert tmpl["module"] == DefaultInterfaceModule.module_id
    comments = tmpl.get("param_comments", {})
    assert "cutoff" in comments
    assert "jobs" in comments
    assert "coordinate_decimals" in comments
    notes = tmpl.get("notes", {})
    assert "determinism" in notes
    assert "jobs_precedence" in notes


def test_validate_params_accepts_defaults() -> None:
    params = dict(DefaultInterfaceModule._metadata.defaults)
    DefaultInterfaceModule.validate_params(params)


def test_validate_params_rejects_negative_cutoff() -> None:
    params = dict(DefaultInterfaceModule._metadata.defaults)
    params["cutoff"] = -1
    with pytest.raises(ValueError):
        DefaultInterfaceModule.validate_params(params)


def test_metadata_contains_id_and_kind() -> None:
    meta = DefaultInterfaceModule.metadata()
    assert meta.module_id == "interface/polar_cutoff/v1"
    assert meta.module_kind == "interface"
