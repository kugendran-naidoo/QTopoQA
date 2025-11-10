from __future__ import annotations

from qtdaqa.new_dynamic_features.graph_builder2.tests import utils_stub_modules  # noqa: F401

from pathlib import Path
from typing import Dict, List

import pytest

from qtdaqa.new_dynamic_features.graph_builder2 import graph_builder2 as graph_builder


def _module(module_id: str, *, alias: str | None = None, params: Dict[str, object] | None = None, summary: str = ""):
    return {
        "module_id": module_id,
        "alias": alias,
        "params": params or {},
        "summary": summary or f"Summary for {module_id}",
    }


def test_write_feature_config_uses_live_module_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_map: Dict[str, List[Dict[str, object]]] = {
        "interface": [
            _module("interface/custom_alt/v1", summary="Alt interface"),
            _module("interface/polar_cutoff/v1", alias="TopoQA default", params={"cutoff": 10.0, "coordinate_decimals": 3}),
        ],
        "topology": [
            _module("topology/persistence_basic/v1", params={"neighbor_distance": 8.0}),
        ],
        "node": [
            _module("node/custom/v2"),
            _module("node/dssp_topo_merge/v1", params={"drop_na": False}),
        ],
        "edge": [
            _module("edge/legacy_band/v11", params={"distance_max": 10.0}),
            _module("edge/multi_scale/v24", params={"histogram_bins": [0.0, 2.0], "contact_threshold": 5.0}),
        ],
        "mol": [
            _module("mol/custom_stage/v1", summary="Mol builder"),
        ],
    }
    monkeypatch.setattr(graph_builder, "_collect_module_templates", lambda: module_map)

    output_path = tmp_path / "example.feature-config.yaml"
    graph_builder.write_feature_config(output_path)
    text = output_path.read_text()

    assert "interface:\n  module: interface/polar_cutoff/v1" in text
    assert "cutoff: 10.0" in text
    assert "#   - interface/custom_alt/v1" in text  # alternate listing
    assert "node:\n  module: node/dssp_topo_merge/v1" in text  # preferred default selected even when not first
    assert "edge:\n  module: edge/multi_scale/v24" in text
    assert "# Alternate edge modules:" in text
    assert "OPTIONAL stage: mol" in text
    assert "mol:\n  module: mol/custom_stage/v1" in text


def test_render_template_warns_when_stage_missing() -> None:
    module_map = {
        "interface": [_module("interface/polar_cutoff/v1")],
        "node": [_module("node/dssp_topo_merge/v1")],
        "edge": [_module("edge/multi_scale/v24")],
    }
    text = graph_builder._render_feature_config_template(module_map)
    assert "# OPTIONAL stage: topology" in text
    assert "No registered topology modules were found" in text
