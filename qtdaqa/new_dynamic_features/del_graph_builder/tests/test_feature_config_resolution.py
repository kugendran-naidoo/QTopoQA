from __future__ import annotations

from argparse import Namespace
import logging
from pathlib import Path

import pytest

from qtdaqa.new_dynamic_features.graph_builder import graph_builder
from qtdaqa.new_dynamic_features.graph_builder.lib.features_config import load_feature_config


def _make_args(*, feature_config: str, work_dir: Path) -> Namespace:
    return Namespace(
        feature_config=feature_config,
        work_dir=str(work_dir),
    )


def test_resolve_feature_config_errors_when_user_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    args = _make_args(feature_config=str(missing), work_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        graph_builder._resolve_feature_config(args)  # type: ignore[attr-defined]


def _write_minimal_config(path: Path, *, include_extra: bool = False) -> None:
    text = """
interface:
  module: interface/polar_cutoff/v1
  params: {cutoff: 10.0}

node:
  module: node/dssp_topo_merge/v1
  params: {drop_na: false}

edge:
  module: edge/legacy_band/v11
  params: {distance_max: 8.0}
"""
    if include_extra:
        text += """
mol:
  module: custom/mol_stage/v1
  params: {}
"""
    path.write_text(text, encoding="utf-8")


def test_resolve_feature_config_reads_user_file(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path)
    args = _make_args(feature_config=str(config_path), work_dir=tmp_path)

    selection = graph_builder._resolve_feature_config(args)  # type: ignore[attr-defined]

    assert selection.edge["module"] == "edge/legacy_band/v11"


def test_load_feature_config_validates_required_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("edge:\n  module: edge/multi_scale/v24\n  params: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_feature_config(config_path)


def test_optional_stage_allowed(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path, include_extra=True)
    selection = load_feature_config(config_path)
    graph_builder._validate_feature_selection(selection)  # type: ignore[attr-defined]
    assert selection.edge["module"] == "edge/legacy_band/v11"


def test_unknown_stage_emits_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path, include_extra=True)
    caplog.set_level(logging.WARNING)
    load_feature_config(config_path)
    assert any("custom stage 'mol'" in message for message in caplog.messages)


def test_invalid_element_filters_fail_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path)
    selection = load_feature_config(config_path)
    # Simulate tuple literal string
    selection.topology["params"]["element_filters"] = "(('C',), ('N',))"
    with pytest.raises(ValueError):
        graph_builder._validate_feature_selection(selection)  # type: ignore[attr-defined]


def test_interface_cutoff_must_be_positive(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path)
    selection = load_feature_config(config_path)
    selection.interface["params"]["cutoff"] = -2
    with pytest.raises(ValueError):
        graph_builder._validate_feature_selection(selection)  # type: ignore[attr-defined]


def test_edge_histogram_bins_validated(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path)
    selection = load_feature_config(config_path)
    selection.edge["params"]["histogram_bins"] = [2, 1, 3]
    with pytest.raises(ValueError):
        graph_builder._validate_feature_selection(selection)  # type: ignore[attr-defined]

def test_parse_args_requires_all_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path)
    with pytest.raises(SystemExit):
        graph_builder.parse_args(  # type: ignore[attr-defined]
            [
                "--dataset-dir",
                str(tmp_path),
                "--work-dir",
                str(tmp_path),
                "--graph-dir",
                str(tmp_path),
                "--feature-config",
                str(config_path),
            ]
        )
    err = capsys.readouterr().err
    assert "--log-dir" in err


def test_resolve_edge_dump_prefers_cli() -> None:
    assert graph_builder._resolve_edge_dump(True, False) is True  # type: ignore[attr-defined]
    assert graph_builder._resolve_edge_dump(False, True) is False  # type: ignore[attr-defined]


def test_resolve_edge_dump_falls_back_to_config() -> None:
    assert graph_builder._resolve_edge_dump(None, False) is False  # type: ignore[attr-defined]
    assert graph_builder._resolve_edge_dump(None, True) is True  # type: ignore[attr-defined]


def test_resolve_edge_dump_defaults_true() -> None:
    assert graph_builder._resolve_edge_dump(None, None) is True  # type: ignore[attr-defined]


def test_resolve_edge_dump_dir_prefers_cli(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cli_dir = tmp_path / "cli"
    resolved = graph_builder._resolve_edge_dump_dir(work_dir, cli_dir, "/ignored")  # type: ignore[attr-defined]
    assert resolved == cli_dir.resolve()


def test_resolve_edge_dump_dir_uses_config(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    config_dir = tmp_path / "config"
    resolved = graph_builder._resolve_edge_dump_dir(work_dir, None, str(config_dir))  # type: ignore[attr-defined]
    assert resolved == config_dir.resolve()


def test_resolve_edge_dump_dir_defaults_to_work(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    resolved = graph_builder._resolve_edge_dump_dir(work_dir, None, None)  # type: ignore[attr-defined]
    assert resolved == (work_dir / "edge_features").resolve()
