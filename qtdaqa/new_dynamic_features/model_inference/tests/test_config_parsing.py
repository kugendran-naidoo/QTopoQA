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
    with pytest.raises(KeyError, match="paths.output_file"):
        inference_topoqa_cpu.load_config(cfg_path)


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
