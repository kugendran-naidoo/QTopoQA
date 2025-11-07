import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import yaml

# Stub torch (and dependents) before importing builder_runner.
torch_stub = ModuleType("torch")
torch_stub.tensor = lambda *args, **kwargs: SimpleNamespace()
torch_stub.load = lambda *args, **kwargs: SimpleNamespace()
torch_stub.is_tensor = lambda *_args, **_kwargs: False
sys.modules["torch"] = torch_stub

torch_utils_stub = ModuleType("torch.utils")
torch_utils_data_stub = ModuleType("torch.utils.data")


class _Dataset:
    pass


torch_utils_data_stub.Dataset = _Dataset
torch_utils_stub.data = torch_utils_data_stub
sys.modules["torch.utils"] = torch_utils_stub
sys.modules["torch.utils.data"] = torch_utils_data_stub

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_inference.builder_runner import (  # noqa: E402
    BuilderConfig,
    parse_builder_config,
    run_graph_builder,
    _write_metadata_feature_config,
)


class _DummyConfig:
    def __init__(self, data_dir: Path, work_dir: Path, builder: BuilderConfig):
        self.data_dir = data_dir
        self.work_dir = work_dir
        self.builder = builder


def _make_dummy_config(tmp_path: Path, builder: BuilderConfig) -> _DummyConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    work_dir = tmp_path / "work"
    return _DummyConfig(data_dir=data_dir, work_dir=work_dir, builder=builder)


def test_prepare_feature_config_applies_overrides(tmp_path: Path) -> None:
    builder = BuilderConfig(dump_edges=False, topology_dedup_sort=True)
    config_path = builder.prepare_feature_config(tmp_path)
    assert config_path is not None
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert payload["options"]["edge_dump"] is False
    assert payload["topology"]["params"]["dedup_sort"] is True
    for key in ("interface", "topology", "node", "edge"):
        assert "module" in payload[key]


def test_prepare_feature_config_uses_inline_features_without_mutation(tmp_path: Path) -> None:
    features = {
        "interface": {"module": "interface/polar_cutoff/v1", "params": {"cutoff": 15.0}},
        "topology": {"module": "topology/persistence_basic/v1", "params": {"dedup_sort": False}},
        "node": {"module": "node/dssp_topo_merge/v1", "params": {}},
        "edge": {"module": "edge/legacy_band/v11", "params": {"distance_max": 12.0}},
        "options": {"edge_dump": True},
    }
    original = json.loads(json.dumps(features))
    builder = BuilderConfig(features=features)
    path = builder.prepare_feature_config(tmp_path)
    assert path is not None
    generated = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert generated == original
    assert builder.features == features


def test_parse_builder_config_accepts_jobs_only(tmp_path: Path) -> None:
    config = parse_builder_config({"jobs": 6}, tmp_path)
    assert config.jobs == 6
    assert config.dump_edges is None
    assert config.topology_dedup_sort is None
    assert config.feature_config is None
    assert config.features == {}


def test_parse_builder_config_rejects_legacy_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_builder_config({"dump_edges": False}, tmp_path)
    with pytest.raises(ValueError):
        parse_builder_config({"topology_dedup_sort": True}, tmp_path)
    with pytest.raises(ValueError):
        parse_builder_config({"features": {"edge": {}}}, tmp_path)
    with pytest.raises(ValueError):
        parse_builder_config({"feature_config": "features.yaml"}, tmp_path)


def test_run_graph_builder_invokes_cli_with_feature_config(tmp_path: Path) -> None:
    builder = BuilderConfig(dump_edges=False)
    cfg = _make_dummy_config(tmp_path, builder)

    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.subprocess.run"
    ) as mocked_run:
        mocked_run.return_value = SimpleNamespace(returncode=0)
        result_dir = run_graph_builder(cfg)

    assert result_dir == cfg.work_dir / "graph_data"
    generated_features = cfg.work_dir / "builder_features" / "features.generated.yaml"
    assert generated_features.exists()
    invoked_cmd = mocked_run.call_args[0][0]
    assert invoked_cmd[0] == sys.executable
    assert invoked_cmd[1:3] == ["-m", "qtdaqa.new_dynamic_features.graph_builder.graph_builder"]
    assert "--feature-config" in invoked_cmd


def test_run_graph_builder_prefers_metadata_feature_config(tmp_path: Path) -> None:
    builder = BuilderConfig()
    cfg = _make_dummy_config(tmp_path, builder)
    metadata_path = tmp_path / "metadata_features.yaml"
    metadata_path.write_text("options: {}", encoding="utf-8")

    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.subprocess.run"
    ) as mocked_run:
        mocked_run.return_value = SimpleNamespace(returncode=0)
        run_graph_builder(cfg, metadata_feature_config=metadata_path)

    invoked_cmd = mocked_run.call_args[0][0]
    feature_flag_index = invoked_cmd.index("--feature-config")
    assert invoked_cmd[feature_flag_index + 1] == str(metadata_path)
    generated_features = cfg.work_dir / "builder_features" / "features.generated.yaml"
    assert not generated_features.exists(), "metadata path should bypass builder overrides"


def test_write_metadata_feature_config_produces_expected_yaml(tmp_path: Path) -> None:
    feature_metadata = {
        "module_registry": {
            "interface": {"id": "interface/polar_cutoff/v1", "defaults": {"cutoff": 13.5}},
            "topology": {"id": "topology/persistence_basic/v1", "defaults": {"neighbor_distance": 7.0}},
            "node": {"id": "node/dssp_topo_merge/v1", "defaults": {"drop_na": True}},
            "edge": {"id": "edge/multi_scale/v24", "defaults": {"contact_threshold": 5.0}},
        },
        "edge_schema": {"module_params": {"contact_threshold": 6.0}},
    }

    path = _write_metadata_feature_config(feature_metadata, tmp_path)
    assert path is not None
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert payload["edge"]["params"]["contact_threshold"] == 6.0
    assert payload["interface"]["module"] == "interface/polar_cutoff/v1"
    assert "options" in payload
