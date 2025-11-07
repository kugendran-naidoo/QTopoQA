import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

# Minimal torch stub so metadata utilities avoid importing the real library.
torch_stub = ModuleType("torch")
torch_stub.tensor = lambda *args, **kwargs: SimpleNamespace()
torch_stub.load = lambda *args, **kwargs: SimpleNamespace()
torch_stub.is_tensor = lambda *_args, **_kwargs: False
sys.modules["torch"] = torch_stub

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_inference import builder_runner  # noqa: E402
from qtdaqa.new_dynamic_features.model_inference.builder_runner import BuilderConfig  # noqa: E402


def _make_config(tmp_path: Path, reuse: bool = True):
    data_dir = tmp_path / "data"
    work_dir = tmp_path / "work"
    output_file = tmp_path / "preds.csv"
    checkpoint = tmp_path / "model.ckpt"
    for directory in (data_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("ckpt", encoding="utf-8")
    return SimpleNamespace(
        data_dir=data_dir,
        work_dir=work_dir,
        checkpoint_path=checkpoint,
        output_file=output_file,
        label_file=None,
        batch_size=4,
        num_workers=0,
        builder=BuilderConfig(jobs=2),
        reuse_existing_graphs=reuse,
    )


def test_graph_metadata_matches_success(tmp_path: Path) -> None:
    final_schema = {"edge_schema": {"module": "edge/multi_scale/v24", "dim": 24}}
    fake_metadata = SimpleNamespace(edge_schema={"module": "edge/multi_scale/v24", "dim": 24})
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        return_value=fake_metadata,
    ):
        result, reason = builder_runner._graph_metadata_matches(tmp_path, final_schema)
    assert result is True
    assert reason == ""


def test_graph_metadata_matches_failure(tmp_path: Path) -> None:
    final_schema = {"edge_schema": {"module": "edge/multi_scale/v24"}}
    fake_metadata = SimpleNamespace(edge_schema={"module": "edge/legacy_band/v11"})
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        return_value=fake_metadata,
    ):
        result, reason = builder_runner._graph_metadata_matches(tmp_path, final_schema)
    assert result is False
    assert "edge schema key 'module' mismatch" in reason


def test_ensure_graph_dir_reuses_when_metadata_matches(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, reuse=True)
    graph_dir = Path(cfg.work_dir) / "graph_data"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "dummy.pt").write_text("pt", encoding="utf-8")
    final_schema = {"edge_schema": {"module": "edge/multi_scale/v24"}}

    with mock.patch.object(
        builder_runner, "_graph_metadata_matches", return_value=(True, "")
    ) as matches_mock, mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.run_graph_builder"
    ) as run_mock:
        chosen_dir = builder_runner.ensure_graph_dir(cfg, final_schema)

    assert chosen_dir == graph_dir
    matches_mock.assert_called_once()
    run_mock.assert_not_called()


def test_ensure_graph_dir_rebuilds_when_metadata_mismatch(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, reuse=True)
    graph_dir = Path(cfg.work_dir) / "graph_data"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "dummy.pt").write_text("pt", encoding="utf-8")
    final_schema = {"edge_schema": {"module": "edge/multi_scale/v24"}}
    rebuilt_dir = Path(cfg.work_dir) / "rebuilt_graphs"

    with mock.patch.object(
        builder_runner, "_graph_metadata_matches", return_value=(False, "edge mismatch")
    ) as matches_mock, mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.run_graph_builder",
        return_value=rebuilt_dir,
    ) as run_mock:
        chosen_dir = builder_runner.ensure_graph_dir(cfg, final_schema)

    assert chosen_dir == rebuilt_dir
    matches_mock.assert_called_once()
    run_mock.assert_called_once_with(cfg)


def test_ensure_graph_dir_builds_when_no_cached_graphs(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, reuse=True)
    final_schema = {"edge_schema": {"module": "edge/multi_scale/v24"}}
    rebuilt_dir = Path(cfg.work_dir) / "rebuilt_graphs"

    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.run_graph_builder",
        return_value=rebuilt_dir,
    ) as run_mock:
        chosen_dir = builder_runner.ensure_graph_dir(cfg, final_schema)

    assert chosen_dir == rebuilt_dir
    run_mock.assert_called_once_with(cfg)
