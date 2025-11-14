import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict

import pytest

# Stub torch early to avoid importing heavyweight dependencies.
torch_stub = ModuleType("torch")
torch_stub.tensor = lambda *args, **kwargs: SimpleNamespace()
torch_stub.load = lambda *args, **kwargs: SimpleNamespace()
torch_stub.is_tensor = lambda *_args, **_kwargs: False

class _FakeTensor:
    pass

torch_stub.Tensor = _FakeTensor
sys.modules["torch"] = torch_stub

torch_utils_stub = ModuleType("torch.utils")
torch_utils_data_stub = ModuleType("torch.utils.data")


class _Dataset:
    pass


torch_utils_data_stub.Dataset = _Dataset
torch_utils_stub.data = torch_utils_data_stub
sys.modules["torch.utils"] = torch_utils_stub
sys.modules["torch.utils.data"] = torch_utils_data_stub

torch_geo_stub = ModuleType("torch_geometric")
torch_geo_data_stub = ModuleType("torch_geometric.data")


class _Data:
    pass


class _Batch:
    @staticmethod
    def from_data_list(_items):
        return SimpleNamespace()


torch_geo_data_stub.Data = _Data
torch_geo_data_stub.Batch = _Batch
torch_geo_loader_stub = ModuleType("torch_geometric.loader")


class _DataLoader:
    def __init__(*args, **kwargs):
        pass


torch_geo_loader_stub.DataLoader = _DataLoader
sys.modules["torch_geometric"] = torch_geo_stub
sys.modules["torch_geometric.data"] = torch_geo_data_stub
sys.modules["torch_geometric.loader"] = torch_geo_loader_stub

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_inference import inference_topoqa_cpu  # noqa: E402


def _write_cfg(tmp_path: Path, content: str) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path


def test_load_config_structured(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
        paths:
          data_dir: ./data
          label_file: null
          work_dir: ./work
          output_file: ./out.csv
          checkpoint: ./model.ckpt

        batch_size: 64
        num_workers: 2

        builder:
          jobs: 3

        options:
          reuse_existing_graphs: true
        """,
    )
    cfg = inference_topoqa_cpu.load_config(cfg_path)
    assert cfg.data_dir == cfg_path.parent / "data"
    assert cfg.label_file is None
    assert cfg.output_file == cfg_path.parent / "out.csv"
    assert cfg.builder.jobs == 3
    assert cfg.reuse_existing_graphs is True
    assert cfg.batch_size == 64
    assert cfg.config_name == cfg_path.name


def test_load_config_rejects_legacy_keys(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
        data_dir: ./data
        work_dir: ./work
        output_file: ./out.csv
        checkpoint_path: ./model.ckpt
        """,
    )
    with pytest.raises(ValueError, match="structured layout"):
        inference_topoqa_cpu.load_config(cfg_path)


def test_load_config_missing_required_path(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
        paths:
          data_dir: ./data
          work_dir: ./work
          checkpoint: ./model.ckpt

        builder:
          jobs: 1
        """,
    )
    cfg = inference_topoqa_cpu.load_config(cfg_path)
    assert cfg.output_file is None


def test_load_config_accepts_schema_override(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
        paths:
          data_dir: ./data
          work_dir: ./work
          output_file: ./out.csv
          checkpoint: ./model.ckpt

        builder:
          jobs: 2

        options:
          reuse_existing_graphs: true
          use_checkpoint_schema: false

        interface_schema:
          module: interface/polar_cutoff/v1
          jobs: 4
        edge_schema:
          module: edge/legacy_band/v11
          dim: 11
        topology_schema:
          summary: {}
          jobs: 5
        node_schema:
          module: node/dssp_topo_merge/v1
          jobs: 6
        """,
    )
    cfg = inference_topoqa_cpu.load_config(cfg_path)
    assert cfg.interface_schema["module"] == "interface/polar_cutoff/v1"
    assert cfg.interface_schema["jobs"] == 4
    assert cfg.edge_schema["module"] == "edge/legacy_band/v11"
    assert cfg.edge_schema["dim"] == 11
    assert cfg.topology_schema["summary"] == {}
    assert cfg.topology_schema["jobs"] == 5
    assert cfg.node_schema["module"] == "node/dssp_topo_merge/v1"
    assert cfg.node_schema["jobs"] == 6
    assert cfg.use_checkpoint_schema is False


def test_load_config_auto_selects_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training_root = tmp_path / "training_runs"
    run_one = training_root / "run_one"
    run_two = training_root / "run_two"
    for run_dir in (run_one, run_two):
        (run_dir / "model_checkpoints").mkdir(parents=True, exist_ok=True)
    ckpt_one = run_one / "model_checkpoints" / "best.ckpt"
    ckpt_two = run_two / "model_checkpoints" / "best.ckpt"
    ckpt_one.write_text("one", encoding="utf-8")
    ckpt_two.write_text("two", encoding="utf-8")

    summaries = {
        run_one.resolve(): {
            "best_selection_metric": 0.25,
            "best_val_loss": 0.12,
            "best_checkpoint": str(ckpt_one),
            "run_name": "run_one",
        },
        run_two.resolve(): {
            "best_selection_metric": 0.15,
            "best_val_loss": 0.10,
            "best_checkpoint": str(ckpt_two),
            "run_name": "run_two",
        },
    }

    def fake_summarise(run_dir: Path) -> Dict[str, object]:
        return summaries[run_dir.resolve()]

    monkeypatch.setattr(inference_topoqa_cpu.train_cli, "_summarise_run", fake_summarise)

    cfg_path = _write_cfg(
        tmp_path,
        f"""
        paths:
          data_dir: ./data
          work_dir: ./work
          output_file: ./out.csv
          training_root: {training_root}

        builder:
          jobs: 2
        """,
    )
    cfg = inference_topoqa_cpu.load_config(cfg_path)
    assert cfg.checkpoint_path == ckpt_two.resolve()
    assert cfg.training_root == training_root.resolve()


def test_load_config_auto_select_checkpoint_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training_root = tmp_path / "training_runs"
    run_dir = training_root / "run_zero"
    run_dir.mkdir(parents=True, exist_ok=True)

    def fake_summarise(_run_dir: Path) -> Dict[str, object]:
        return {}

    monkeypatch.setattr(inference_topoqa_cpu.train_cli, "_summarise_run", fake_summarise)

    cfg_path = _write_cfg(
        tmp_path,
        f"""
        paths:
          data_dir: ./data
          work_dir: ./work
          output_file: ./out.csv
          training_root: {training_root}

        builder:
          jobs: 2
        """,
    )
    with pytest.raises(RuntimeError, match="No eligible checkpoints"):
        inference_topoqa_cpu.load_config(cfg_path)


def test_load_config_requires_training_root_when_autoselect(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
        paths:
          data_dir: ./data
          work_dir: ./work
          output_file: ./out.csv
          training_root: ./missing_runs
        builder:
          jobs: 1
        """,
    )
    with pytest.raises(FileNotFoundError, match="Training root does not exist"):
        inference_topoqa_cpu.load_config(cfg_path)


def test_guard_schema_overrides(tmp_path: Path) -> None:
    cfg = inference_topoqa_cpu.InferenceConfig(
        data_dir=tmp_path,
        work_dir=tmp_path,
        checkpoint_path=tmp_path / "ckpt.chkpt",
        output_file=tmp_path / "out.csv",
        edge_schema={"module": "edge/foo"},
        training_root=tmp_path,
    )
    checkpoint_meta = {"edge_schema": {"module": "edge/bar"}}
    with pytest.raises(RuntimeError, match="edge_schema.module mismatch"):
        inference_topoqa_cpu._guard_schema_overrides(cfg, checkpoint_meta)

    cfg.edge_schema = {"module": "edge/foo"}
    checkpoint_meta = {"edge_schema": {"module": "edge/foo"}}
    # Should not raise
    inference_topoqa_cpu._guard_schema_overrides(cfg, checkpoint_meta)


def test_log_checkpoint_banner_formats(caplog, tmp_path: Path) -> None:
    cfg = inference_topoqa_cpu.InferenceConfig(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        checkpoint_path=tmp_path / "ckpt.ckpt",
        output_file=tmp_path / "out.csv",
        config_name="config.yaml.test",
    )
    caplog.set_level("INFO")
    inference_topoqa_cpu._log_checkpoint_banner(cfg, surround_blank=True)
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "",
        f"Checkpoint file: {cfg.checkpoint_path}",
        f"Inference config: {cfg.config_name}",
        "",
    ]
    caplog.clear()
    inference_topoqa_cpu._log_checkpoint_banner(cfg, surround_blank=False)
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"Checkpoint file: {cfg.checkpoint_path}",
        f"Inference config: {cfg.config_name}",
    ]


def test_cli_overrides_preserve_builder_identifier(tmp_path: Path) -> None:
    feature_cfg = tmp_path / "features.yaml"
    feature_cfg.write_text(
        """
        interface:
          module: interface/default
        node:
          module: node/default
        edge:
          module: edge/default
        """,
        encoding="utf-8",
    )
    cfg_path = _write_cfg(
        tmp_path,
        """
        paths:
          data_dir: ./data
          work_dir: ./work
          output_file: ./out.csv
          checkpoint: ./model.ckpt

        builder:
          id: graph_builder2
          feature_config: ./features.yaml
          jobs: 2
        """,
    )
    cfg = inference_topoqa_cpu.load_config(cfg_path)
    args = SimpleNamespace(
        data_dir=None,
        work_dir=str(tmp_path / "cli_work"),
        checkpoint_path=None,
        output_file=str(tmp_path / "cli_out.csv"),
        label_file=None,
        batch_size=None,
        num_workers=None,
        builder_jobs=7,
        reuse_existing_graphs=False,
    )
    merged = inference_topoqa_cpu._merge_cli_overrides(cfg, args)
    assert merged.builder.builder_name == "graph_builder2"
    assert merged.builder.feature_config == cfg.builder.feature_config
    assert merged.builder.jobs == 7
