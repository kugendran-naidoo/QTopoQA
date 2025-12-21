from __future__ import annotations

import json
import logging
from pathlib import Path

# Allow importing shared feature_metadata module without installing package
import sys
import types

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.load = lambda *args, **kwargs: None
    torch_stub.tensor = lambda *args, **kwargs: None
    sys.modules["torch"] = torch_stub

BASE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = BASE_DIR / "common"
if COMMON_DIR.exists() and str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from feature_metadata import GraphFeatureMetadata  # type: ignore

from qtdaqa.new_dynamic_features.model_training2.common.metadata_artifacts import (
    write_feature_metadata_artifacts,
)


def test_training_schema_summary_written(tmp_path: Path) -> None:
    metadata = GraphFeatureMetadata(
        edge_schema={"dim": 24, "module": "edge/test_module"},
        node_schema={"dim": 12},
        module_registry={"edge": {"id": "edge/test_module"}},
        topology_schema_spec={"schema_version": "vX", "dim": 1234},
        feature_config={"path": "features.yaml", "sha256": "deadbeef", "text": "edge:\n  module: test"},
    )
    metadata_dict = metadata.to_dict()
    metadata_dict["topology_schema"] = {"module": "topology/test_module"}
    metadata_dict["topology_schema_spec"] = metadata.topology_schema_spec
    metadata_dict["feature_config"] = metadata.feature_config

    logger = logging.getLogger("test_training_schema_summary_written")
    logger.addHandler(logging.NullHandler())

    metadata_path, summary_path = write_feature_metadata_artifacts(tmp_path, metadata, metadata_dict, logger)

    assert metadata_path.exists()
    assert summary_path is not None and summary_path.exists()

    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["edge_schema"]["dim"] == 24
    assert summary_payload["topology_schema"]["module"] == "topology/test_module"
    assert summary_payload["topology_schema_spec"]["schema_version"] == "vX"
    assert summary_payload["topology_schema_spec"]["dim"] == 1234
    assert summary_payload["feature_config"]["sha256"] == "deadbeef"

    metadata_payload = json.loads(metadata_path.read_text())
    assert metadata_payload["summary_path"] == str(summary_path)
    assert metadata_payload["topology_schema_spec"]["schema_version"] == "vX"
    assert metadata_payload["feature_config"]["sha256"] == "deadbeef"
