from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional


def load_builder_info_from_metadata(metadata_path: Optional[str]) -> Optional[Dict[str, object]]:
    """Load the builder info block from graph_metadata.json if present."""
    if not metadata_path:
        return None
    path = Path(metadata_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    builder_entry = payload.get("_builder")
    if isinstance(builder_entry, dict):
        return builder_entry
    for entry in payload.values():
        if isinstance(entry, dict):
            candidate = entry.get("builder")
            if isinstance(candidate, dict):
                return candidate
    return None


def log_builder_provenance(logger: logging.Logger, builder_info: Dict[str, object]) -> None:
    """Emit informative log lines describing the graph builder provenance."""
    builder_id = builder_info.get("id") or builder_info.get("module") or "unknown builder"
    version = builder_info.get("version")
    if version:
        logger.info("Graph builder: %s (version %s)", builder_id, version)
    else:
        logger.info("Graph builder: %s", builder_id)

    feature_config = builder_info.get("feature_config")
    if isinstance(feature_config, dict):
        path = feature_config.get("path")
        digest = feature_config.get("sha256")
        details: Dict[str, str] = {}
        if path:
            details["path"] = str(path)
        if digest:
            details["sha256"] = str(digest)
        if details:
            detail_text = ", ".join(f"{key}={value}" for key, value in details.items())
            logger.info("Feature config snapshot: %s", detail_text)

    edge_dumps = builder_info.get("edge_dumps")
    if isinstance(edge_dumps, dict):
        enabled = edge_dumps.get("enabled")
        state = "enabled" if enabled else "disabled"
        output_dir = edge_dumps.get("output_directory") or edge_dumps.get("configured_directory")
        if output_dir:
            logger.info("Edge dumps: %s (output: %s)", state, output_dir)
        else:
            logger.info("Edge dumps: %s", state)
