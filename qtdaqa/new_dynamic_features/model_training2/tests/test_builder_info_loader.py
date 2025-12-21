from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_training2 import builder_metadata


def test_load_builder_info_prefers_global_entry(tmp_path: Path) -> None:
    payload = {
        "_builder": {"id": "graph_builder2", "version": "2.0.0"},
        "sample_model": {"edge_module": "edge/legacy_band/v11"},
    }
    metadata_path = tmp_path / "graph_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    builder_info = builder_metadata.load_builder_info_from_metadata(str(metadata_path))
    assert builder_info is not None
    assert builder_info["id"] == "graph_builder2"


def test_load_builder_info_falls_back_to_model_entries(tmp_path: Path) -> None:
    payload = {
        "sample_model": {
            "edge_module": "edge/multi_scale/v24",
            "builder": {"id": "graph_builder", "version": "1.0.0"},
        }
    }
    metadata_path = tmp_path / "graph_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    builder_info = builder_metadata.load_builder_info_from_metadata(str(metadata_path))
    assert builder_info is not None
    assert builder_info["id"] == "graph_builder"


def test_log_builder_provenance_emits_digest(caplog) -> None:
    info = {
        "id": "graph_builder2",
        "version": "2.0.0",
        "feature_config": {"path": "/tmp/features.yaml", "sha256": "abc123"},
        "edge_dumps": {"enabled": True, "output_directory": "/tmp/edges"},
    }
    logger = logging.getLogger("test_log_builder_provenance")
    with caplog.at_level(logging.INFO):
        builder_metadata.log_builder_provenance(logger, info)
    assert "graph_builder2" in caplog.text
    assert "sha256=abc123" in caplog.text
    assert "Edge dumps: enabled" in caplog.text
