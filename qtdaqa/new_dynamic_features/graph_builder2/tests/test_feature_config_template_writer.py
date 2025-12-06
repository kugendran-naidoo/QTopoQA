from __future__ import annotations

from qtdaqa.new_dynamic_features.graph_builder2.tests import utils_stub_modules  # noqa: F401

from pathlib import Path
from typing import Dict, List

import pytest

from qtdaqa.new_dynamic_features.graph_builder2 import graph_builder2 as graph_builder


def _module(
    module_id: str,
    *,
    alias: str | None = None,
    params: Dict[str, object] | None = None,
    summary: str = "",
    description: str = "",
    param_comments: Dict[str, str] | None = None,
):
    return {
        "module_id": module_id,
        "alias": alias,
        "params": params or {},
        "param_comments": param_comments or {},
        "summary": summary or f"Summary for {module_id}",
        "description": description or f"Description for {module_id}",
    }


def test_write_feature_config_uses_live_module_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_map: Dict[str, List[Dict[str, object]]] = {
        "interface": [
            _module("interface/custom_alt/v1", summary="Alt interface"),
            _module(
                "interface/polar_cutoff/v1",
                alias="n x 3D TopoQA default 10A cutoff",
                description="Interface description",
                params={"cutoff": 10.0, "coordinate_decimals": -1, "jobs": 16},
                param_comments={"coordinate_decimals": "skip rounding to keep raw coords"},
            ),
        ],
        "topology": [
            _module(
                "topology/persistence_basic/v1",
                alias="140D summarizes interface features TopoQA default",
                summary="Topology summary",
                description="Topology description",
                params={
                    "neighbor_distance": 8.0,
                    "element_filters": [["C"], ["C", "N"]],
                    "jobs": 16,
                },
            ),
        ],
        "node": [
            _module("node/custom/v2"),
            _module(
                "node/dssp_topo_merge/v1",
                alias="172D (140D Topological consumes Interface + 32D) TopoQA default",
                summary="Node summary",
                description="Node description",
                params={"drop_na": True, "jobs": 16},
                param_comments={
                    "drop_na": (
                        "matches fea_df_clean = fea_df.dropna() in both inference_model.py "
                        "and k_mac_inference_pca_tsne4.py"
                    )
                },
            ),
        ],
        "edge": [
            _module(
                "edge/legacy_band/v11",
                alias="Edge 11D legacy",
                summary="Legacy summary",
                description="Legacy description",
                params={"distance_min": 0.0, "distance_max": 10.0, "scale_features": True, "jobs": 16},
            ),
            _module(
                "edge/multi_scale/v24",
                alias="24D Scalars",
                summary="Edge summary",
                description="Edge description",
                params={"histogram_bins": [0.0, 2.0, 4.0], "contact_threshold": 5.0, "jobs": 16},
            ),
            _module(
                "edge/neo/v24",
                alias="Neo hybrid multi-scale",
                summary="Neo summary",
                description="Neo description",
                params={"contact_thresholds": [4.0, 8.0], "jobs": 16},
            ),
        ],
        "mol": [
            _module("mol/custom_stage/v1", summary="Mol builder"),
        ],
    }
    monkeypatch.setattr(graph_builder, "_collect_module_templates", lambda: module_map)

    output_path = tmp_path / "example.feature-config.yaml"
    graph_builder.write_feature_config(output_path)
    text = output_path.read_text()

    assert "options:" in text
    assert "topology_round_decimals" in text

    assert "interface:\n  module: interface/polar_cutoff/v1" in text
    assert "cutoff: 10.0" in text
    assert "coordinate_decimals: -1  # skip rounding to keep raw coords" in text
    assert '  alias: "n x 3D TopoQA default 10A cutoff"' in text
    assert '  summary: "Summary for interface/polar_cutoff/v1"' in text
    assert '  description: "Interface description"' in text
    assert "jobs: 16" in text
    assert "#   - interface/custom_alt/v1" in text  # alternate listing
    assert 'topology:\n  module: topology/persistence_basic/v1' in text
    assert '  summary: "Topology summary"' in text
    assert '  description: "Topology description"' in text
    assert "node:\n  module: node/dssp_topo_merge/v1" in text  # preferred default selected even when not first
    assert (
        "drop_na: true  # matches fea_df_clean = fea_df.dropna() in both inference_model.py "
        "and k_mac_inference_pca_tsne4.py"
    ) in text
    assert '  summary: "Node summary"' in text
    assert '  description: "Node description"' in text
    assert "edge:\n  module: edge/legacy_band/v11" in text
    assert '  alias: "Edge 11D legacy"' in text
    assert "  # dim: 11" in text
    assert '  summary: "Legacy summary"' in text
    assert '  description: "Legacy description"' in text
    assert "element_filters:" in text and "- [C, N]" in text
    assert "distance_max: 10.0" in text
    assert "scale_features: true" in text
    legacy_block = text.split("edge:\n  module: edge/legacy_band/v11", 1)[1]
    legacy_block = legacy_block.split("# Alternate edge modules")[0]
    assert "jobs: 16" in legacy_block
    assert "# Alternate edge modules:" in text
    assert "OPTIONAL stage: mol" in text
    assert "mol:\n  module: mol/custom_stage/v1" in text


def test_write_feature_config_can_include_alternates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_map: Dict[str, List[Dict[str, object]]] = {
        "edge": [
            _module(
                "edge/legacy_band/v11",
                alias="Edge 11D legacy",
                params={"distance_min": 0.0, "distance_max": 10.0, "scale_features": True, "jobs": 16},
            ),
            _module(
                "edge/multi_scale/v24",
                alias="24D Scalars",
                summary="Edge summary",
                description="Edge description",
                params={"histogram_bins": [0.0, 2.0], "jobs": 16},
            ),
            _module(
                "edge/neo/v24",
                alias="Neo hybrid multi-scale",
                summary="Neo summary",
                description="Neo description",
                params={"contact_thresholds": [4.0, 8.0], "jobs": 16},
            ),
        ],
    }
    monkeypatch.setattr(graph_builder, "_collect_module_templates", lambda: module_map)

    output_path = tmp_path / "example.feature-config.yaml"
    graph_builder.write_feature_config(output_path, include_alternates=True)
    text = output_path.read_text()

    assert "edge:\n  module: edge/legacy_band/v11" in text
    assert "  # dim: 11" in text
    assert "# Alternate edge modules (uncomment to use):" in text
    assert "# edge:" in text
    assert "#   module: edge/multi_scale/v24  # alias: 24D Scalars" in text
    assert '#   alias: "24D Scalars"' in text
    assert "#   # dim: 24" in text
    assert '#   summary: "Edge summary"' in text
    assert "#     jobs: 16" in text
    assert "#   module: edge/neo/v24  # alias: Neo hybrid multi-scale" in text
    assert '#   alias: "Neo hybrid multi-scale"' in text
    assert '#   summary: "Neo summary"' in text
    assert "#     jobs: 16" in text


def test_render_template_warns_when_stage_missing() -> None:
    module_map = {
        "interface": [_module("interface/polar_cutoff/v1")],
        "node": [_module("node/dssp_topo_merge/v1")],
        "edge": [_module("edge/multi_scale/v24")],
    }
    text = graph_builder._render_feature_config_template(module_map)
    assert "# OPTIONAL stage: topology" in text
    assert "No registered topology modules were found" in text


def test_format_module_listing_includes_dim_hint() -> None:
    from qtdaqa.new_dynamic_features.graph_builder2.modules.registry import FeatureModuleMetadata

    class DummyModule:
        module_id = "edge/dim_test/v1"
        default_alias = "Edge 42D test"

        @classmethod
        def metadata(cls):
            return FeatureModuleMetadata(
                module_id=cls.module_id,
                module_kind="edge",
                summary="dummy",
                description="dummy",
            )

        @classmethod
        def list_params(cls):
            return {}

    meta = DummyModule.metadata()
    lines = graph_builder._format_module_listing(meta, DummyModule)
    assert any("dim" in line and "42" in line for line in lines)
