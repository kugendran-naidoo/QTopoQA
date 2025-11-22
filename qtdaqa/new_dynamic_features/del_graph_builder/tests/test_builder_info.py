from __future__ import annotations

from pathlib import Path

from qtdaqa.new_dynamic_features.graph_builder import builder_info


def test_builder_info_snapshot_and_sanitise(tmp_path: Path) -> None:
    config_path = tmp_path / "feature-config.yaml"
    config_path.write_text("interface: {module: interface/polar_cutoff/v1}\nnode: {module: node/dssp_topo_merge/v1}\nedge: {module: edge/multi_scale/v24}\n", encoding="utf-8")

    info = builder_info.build_builder_info(
        feature_config_path=config_path,
        edge_dump_enabled=True,
        edge_dump_dir=tmp_path / "edges",
        configured_edge_dump_dir=tmp_path / "edges",
        selection_options={"defaults": {"jobs": 4}},
    )

    assert info["id"] == builder_info.BUILDER_ID
    assert info["feature_config"]["sha256"]
    assert info["feature_config"]["text"]

    sanitised = builder_info.sanitize_builder_info(info, include_feature_text=False)
    assert sanitised["feature_config"]["text"] is None
    assert info["feature_config"]["text"] is not None
