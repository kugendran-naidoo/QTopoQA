import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

# Minimal torch stub so metadata utilities don't load the real library in tests.
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

from qtdaqa.new_dynamic_features.model_inference import builder_runner  # noqa: E402
from qtdaqa.new_dynamic_features.model_inference.inference_topoqa_cpu import (  # noqa: E402
    InferenceConfig,
    _run_schema_check,
)


def test_load_and_validate_metadata_success(tmp_path: Path) -> None:
    metadata_obj = SimpleNamespace(
        edge_schema={"module": "edge/multi_scale/v24", "dim": 24, "variant": "multi_scale_v24"},
        node_schema={"dim": 140},
        metadata_path=str(tmp_path / "graph_metadata.json"),
    )
    final_schema = {
        "edge_schema": {"module": "edge/multi_scale/v24", "dim": 24, "variant": "multi_scale_v24"},
        "topology_schema": {"dim": 140},
    }
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        return_value=metadata_obj,
    ):
        result = builder_runner.validate_graph_metadata(tmp_path, final_schema)
    assert result is metadata_obj


def test_load_and_validate_metadata_edge_mismatch(tmp_path: Path) -> None:
    metadata_obj = SimpleNamespace(
        edge_schema={"module": "edge/legacy_band/v11", "dim": 24},
        node_schema={"dim": 140},
        metadata_path=str(tmp_path / "graph_metadata.json"),
    )
    final_schema = {
        "edge_schema": {"module": "edge/multi_scale/v24", "dim": 24},
        "topology_schema": {"dim": 140},
    }
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        return_value=metadata_obj,
    ):
        with pytest.raises(RuntimeError) as exc:
            builder_runner.validate_graph_metadata(tmp_path, final_schema)
    assert "edge_schema.module" in str(exc.value)


def test_load_and_validate_metadata_topology_mismatch(tmp_path: Path) -> None:
    metadata_obj = SimpleNamespace(
        edge_schema={"module": "edge/multi_scale/v24", "dim": 24},
        node_schema={"dim": 128},
        metadata_path=str(tmp_path / "graph_metadata.json"),
    )
    final_schema = {
        "edge_schema": {"module": "edge/multi_scale/v24", "dim": 24},
        "topology_schema": {"dim": 140},
    }
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        return_value=metadata_obj,
    ):
        with pytest.raises(RuntimeError) as exc:
            builder_runner.validate_graph_metadata(tmp_path, final_schema)
    assert "topology/node dim" in str(exc.value)


def test_load_and_validate_metadata_load_error(tmp_path: Path) -> None:
    final_schema = {"edge_schema": {"module": "edge/multi_scale/v24"}}
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        side_effect=OSError("missing metadata"),
    ):
        with pytest.raises(RuntimeError) as exc:
            builder_runner.validate_graph_metadata(tmp_path, final_schema)
    assert "Unable to load graph metadata" in str(exc.value)


def test_validate_metadata_allows_defaulted_params(tmp_path: Path) -> None:
    # Metadata omits a param that is present in the checkpoint schema, but the module_registry
    # records it as a default. Validation should treat them as equivalent.
    metadata_obj = SimpleNamespace(
        edge_schema={"module": "edge/edge_plus_pool_agg_topo/v1", "dim": 854, "module_params": {}},
        node_schema={"dim": 140},
        module_registry={
            "edge": {
                "id": "edge/edge_plus_pool_agg_topo/v1",
                "defaults": {"include_norms": True, "distance_max": 10.0},
            }
        },
        metadata_path=str(tmp_path / "graph_metadata.json"),
    )
    final_schema = {
        "edge_schema": {
            "module": "edge/edge_plus_pool_agg_topo/v1",
            "dim": 854,
            "module_params": {"include_norms": True},
        },
        "topology_schema": {"dim": 140},
    }
    with mock.patch(
        "qtdaqa.new_dynamic_features.model_inference.builder_runner.load_graph_feature_metadata",
        return_value=metadata_obj,
    ):
        result = builder_runner.validate_graph_metadata(tmp_path, final_schema)
    assert result is metadata_obj


def test_preflight_registry_missing_module(monkeypatch) -> None:
    final_schema = {"edge_schema": {"module": "edge/does_not_exist/v1", "module_params": {}}}

    def _raise_missing(module_id: str, **params):
        raise KeyError(module_id)

    monkeypatch.setattr(builder_runner, "instantiate_module", _raise_missing)
    with pytest.raises(RuntimeError) as exc:
        builder_runner._preflight_registry_support(final_schema)
    assert "not available" in str(exc.value)


def test_preflight_registry_rejects_params(monkeypatch) -> None:
    final_schema = {"edge_schema": {"module": "edge/edge_plus_pool_agg_topo/v1", "module_params": {"variant": "mega"}}}

    def _reject(module_id: str, **params):
        raise ValueError("unsupported variant")

    monkeypatch.setattr(builder_runner, "instantiate_module", _reject)
    with pytest.raises(RuntimeError) as exc:
        builder_runner._preflight_registry_support(final_schema)
    assert "rejected parameters" in str(exc.value)


def test_schema_check_uses_existing_graphs(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    graph_dir = work_dir / "graph_data"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "dummy.pt").touch()

    cfg = InferenceConfig(
        data_dir=tmp_path,
        work_dir=work_dir,
        checkpoint_path=tmp_path / "ckpt",
        results_dir=tmp_path / "results",
        output_file=tmp_path / "out.csv",
        check_schema=True,
    )

    calls: list[str] = []
    monkeypatch.setattr(builder_runner, "_preflight_registry_support", lambda fs: calls.append("preflight"))
    monkeypatch.setattr(builder_runner, "validate_graph_metadata", lambda gd, fs: calls.append("validate"))

    _run_schema_check(cfg, {"edge_schema": {"module": "edge/multi_scale/v24"}})
    assert calls == ["preflight", "validate"]
