from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BUILDER_ID = "graph_builder2"
BUILDER_VERSION = "2.0.0"
BUILDER_SCHEMA_VERSION = 1
BUILDER_ENTRYPOINT = "qtdaqa.new_dynamic_features.graph_builder2.graph_builder2"


def _compute_feature_config_snapshot(path: Path) -> Dict[str, object]:
    resolved = path.resolve()
    text = resolved.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "path": str(resolved),
        "sha256": digest,
        "text": text,
    }


def build_builder_info(
    *,
    feature_config_path: Path,
    edge_dump_enabled: bool,
    edge_dump_dir: Optional[Path],
    configured_edge_dump_dir: Optional[Path],
    selection_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Capture the builder identity and configuration snapshot for this run."""
    snapshot = _compute_feature_config_snapshot(feature_config_path)
    options = copy.deepcopy(selection_options) if selection_options else {}
    return {
        "id": BUILDER_ID,
        "version": BUILDER_VERSION,
        "schema_version": BUILDER_SCHEMA_VERSION,
        "module": BUILDER_ENTRYPOINT,
        "command": list(sys.argv),
        "feature_config": snapshot,
        "edge_dumps": {
            "enabled": bool(edge_dump_enabled),
            "output_directory": str(edge_dump_dir) if edge_dump_dir else None,
            "configured_directory": str(configured_edge_dump_dir) if configured_edge_dump_dir else None,
        },
        "options": options,
    }


def sanitize_builder_info(builder_info: Dict[str, object], *, include_feature_text: bool) -> Dict[str, object]:
    payload = copy.deepcopy(builder_info)
    if not include_feature_text:
        feature_config = payload.get("feature_config")
        if isinstance(feature_config, dict) and "text" in feature_config:
            feature_config["text"] = None
    return payload

