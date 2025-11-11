from __future__ import annotations


from qtdaqa.new_dynamic_features.graph_builder2.tests import utils_stub_modules  # noqa: F401

from argparse import Namespace
import logging
from pathlib import Path

import pytest

from qtdaqa.new_dynamic_features.graph_builder.graph_builder import _validate_feature_selection  # type: ignore
from qtdaqa.new_dynamic_features.graph_builder2 import graph_builder2 as graph_builder
from types import SimpleNamespace
from qtdaqa.new_dynamic_features.graph_builder2.lib.features_config import load_feature_config


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
    assert "mol" in selection.extras
    assert selection.extras["mol"]["module"] == "custom/mol_stage/v1"
    graph_builder._validate_feature_selection(selection)  # type: ignore[attr-defined]
    assert selection.edge["module"] == "edge/legacy_band/v11"


def test_unknown_stage_emits_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "features.yaml"
    _write_minimal_config(config_path, include_extra=True)
    caplog.set_level(logging.INFO)
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


def test_determine_effective_jobs_prefers_cli() -> None:
    jobs, source = graph_builder._determine_effective_jobs(cli_jobs=12, config_jobs=5)
    assert jobs == 12
    assert source == "cli"


def test_determine_effective_jobs_falls_back_to_config() -> None:
    jobs, source = graph_builder._determine_effective_jobs(cli_jobs=None, config_jobs=6)
    assert jobs == 6
    assert source == "config"


def test_determine_effective_jobs_defaults_to_modules() -> None:
    jobs, source = graph_builder._determine_effective_jobs(cli_jobs=None, config_jobs=None)
    assert jobs is None
    assert source == "module"


def test_apply_job_defaults_updates_modules_when_jobs_missing() -> None:
    modules = {
        "edge": SimpleNamespace(params={"jobs": None}),
        "node": SimpleNamespace(params={"jobs": "auto"}),
    }
    graph_builder._apply_job_defaults(modules, jobs=10)
    assert modules["edge"].params["jobs"] == 10
    assert modules["node"].params["jobs"] == 10


def test_apply_job_defaults_leaves_modules_when_no_fallback() -> None:
    modules = {
        "edge": SimpleNamespace(params={"jobs": None}),
    }
    graph_builder._apply_job_defaults(modules, jobs=None)
    assert modules["edge"].params["jobs"] is None


def test_resolve_edge_dump_prefers_cli() -> None:
    assert graph_builder._resolve_edge_dump(True, False) is True
    assert graph_builder._resolve_edge_dump(False, True) is False


def test_resolve_edge_dump_falls_back_to_config() -> None:
    assert graph_builder._resolve_edge_dump(None, False) is False
    assert graph_builder._resolve_edge_dump(None, True) is True


def test_resolve_edge_dump_defaults_to_true() -> None:
    assert graph_builder._resolve_edge_dump(None, None) is True


def test_resolve_edge_dump_dir_prefers_cli(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cli_dir = tmp_path / "cli-target"
    resolved = graph_builder._resolve_edge_dump_dir(work_dir, cli_dir, "/ignored")
    assert resolved == cli_dir.resolve()


def test_resolve_edge_dump_dir_uses_config_when_cli_missing(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    config_dir = tmp_path / "config-target"
    resolved = graph_builder._resolve_edge_dump_dir(work_dir, None, str(config_dir))
    assert resolved == config_dir.resolve()


def test_resolve_edge_dump_dir_defaults_to_work_subdir(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    resolved = graph_builder._resolve_edge_dump_dir(work_dir, None, None)
    assert resolved == (work_dir / "edge_features").resolve()
