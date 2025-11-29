import json
import sys
from pathlib import Path
import subprocess
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

from qtdaqa.new_dynamic_features.common.feature_metadata import GraphFeatureMetadata  # noqa: E402
from qtdaqa.new_dynamic_features.model_inference.builder_runner import (  # noqa: E402
    BuilderConfig,
    parse_builder_config,
    run_graph_builder,
    _write_metadata_feature_config,
    _build_schema_feature_payload,
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
    with pytest.raises(FileNotFoundError):
        parse_builder_config({"feature_config": "features.dummy"}, tmp_path)


def test_parse_builder_config_accepts_explicit_feature_config(tmp_path: Path) -> None:
    features = tmp_path / "features.yaml"
    features.write_text("edge: {}", encoding="utf-8")
    config = parse_builder_config({"feature_config": str(features)}, tmp_path)
    assert config.feature_config == features.resolve()
    rel_config = parse_builder_config({"feature_config": features.name}, tmp_path)
    assert rel_config.feature_config == features.resolve()


def test_parse_builder_config_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc:
        parse_builder_config({"jobs": 2, "sort_artefacts": False}, tmp_path)
    assert "unrecognized keys" in str(exc.value)


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


def test_run_graph_builder_classifies_subprocess_error(tmp_path: Path) -> None:
    builder = BuilderConfig()
    cfg = _make_dummy_config(tmp_path, builder)

    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=2, cmd=["python", "-m", "builder"]),
    ):
        with pytest.raises(RuntimeError) as exc:
            run_graph_builder(cfg)
    assert "Graph builder failed" in str(exc.value)
    assert "return code 2" in str(exc.value)


def test_run_graph_builder_respects_custom_module(tmp_path: Path) -> None:
    builder = BuilderConfig()
    cfg = _make_dummy_config(tmp_path, builder)

    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.subprocess.run"
    ) as mocked_run:
        mocked_run.return_value = SimpleNamespace(returncode=0)
        run_graph_builder(cfg, builder_module="custom.module")

    invoked_cmd = mocked_run.call_args[0][0]
    assert invoked_cmd[1:3] == ["-m", "custom.module"]


def test_prepare_feature_config_writes_defaults_when_no_overrides(tmp_path: Path) -> None:
    builder = BuilderConfig()
    path = builder.prepare_feature_config(tmp_path)
    assert path is not None
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert set(payload.keys()) >= {"interface", "topology", "node", "edge", "options"}


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


def test_write_metadata_feature_config_accepts_metadata_object(tmp_path: Path) -> None:
    feature_metadata = GraphFeatureMetadata(
        module_registry={
            "interface": {"id": "interface/polar_cutoff/v1", "params": {"cutoff": 12.0}},
            "topology": {"id": "topology/persistence_basic/v1", "params": {"neighbor_distance": 9.0}},
            "node": {"id": "node/dssp_topo_merge/v1", "params": {"drop_na": False}},
            "edge": {"id": "edge/multi_scale/v24", "defaults": {"contact_threshold": 5.5}},
        },
        edge_schema={"module_params": {"contact_threshold": 4.5}},
    )

    path = _write_metadata_feature_config(feature_metadata, tmp_path)
    assert path is not None
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert payload["edge"]["params"]["contact_threshold"] == 4.5
    assert payload["topology"]["params"]["neighbor_distance"] == 9.0


def test_write_metadata_feature_config_falls_back_to_metadata_file(tmp_path: Path) -> None:
    metadata_path = tmp_path / "graph_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "module_registry": {
                    "interface": {"id": "interface/polar_cutoff/v1"},
                    "topology": {"id": "topology/persistence_basic/v1"},
                    "node": {"id": "node/dssp_topo_merge/v1"},
                    "edge": {"id": "edge/multi_scale/v24"},
                },
                "edge_schema": {"module_params": {"contact_threshold": 7.5}},
            }
        ),
        encoding="utf-8",
    )

    path = _write_metadata_feature_config(None, tmp_path, fallback_metadata_path=metadata_path)
    assert path is not None
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert payload["edge"]["params"]["contact_threshold"] == 7.5


def test_write_metadata_feature_config_uses_builder_snapshot_text(tmp_path: Path) -> None:
    feature_metadata = {
        "builder": {
            "feature_config": {
                "text": "interface:\n  module: interface/polar_cutoff/v1\noptions: {}\n",
            }
        }
    }

    path = _write_metadata_feature_config(feature_metadata, tmp_path)
    assert path is not None
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert payload["interface"]["module"] == "interface/polar_cutoff/v1"


def test_write_metadata_feature_config_uses_builder_snapshot_path(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.yaml"
    snapshot_path.write_text("options: {}\ninterface: {module: interface/polar_cutoff/v1}\n", encoding="utf-8")
    feature_metadata = {
        "builder": {
            "feature_config": {
                "path": str(snapshot_path),
            }
        }
    }

    path = _write_metadata_feature_config(feature_metadata, tmp_path)
    assert path is not None
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert payload["interface"]["module"] == "interface/polar_cutoff/v1"


def test_build_schema_feature_payload_overrides_edge_params() -> None:
    payload = _build_schema_feature_payload(
        {
            "edge_schema": {
                "module": "edge/legacy_band/v11",
                "module_params": {"distance_max": 9.0},
            }
        }
    )
    assert payload is not None
    assert payload["edge"]["module"] == "edge/legacy_band/v11"
    assert payload["edge"]["params"]["distance_max"] == 9.0


def test_build_schema_feature_payload_copies_stage_jobs() -> None:
    payload = _build_schema_feature_payload(
        {
            "interface_schema": {"module": "interface/custom", "jobs": 5},
            "topology_schema": {"module": "topology/persistence_basic/v1", "jobs": 6},
            "node_schema": {"module": "node/dssp_topo_merge/v1", "jobs": 7},
            "edge_schema": {
                "module": "edge/legacy_band/v11",
                "module_params": {"distance_max": 9.0},
                "jobs": 8,
            },
        }
    )
    assert payload is not None
    assert payload["interface"]["jobs"] == 5
    assert payload["topology"]["jobs"] == 6
    assert payload["node"]["jobs"] == 7
    assert payload["edge"]["jobs"] == 8
