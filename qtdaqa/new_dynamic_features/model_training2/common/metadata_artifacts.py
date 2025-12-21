from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

try:  # package import when executing via `python -m`
    from qtdaqa.new_dynamic_features.common.feature_metadata import GraphFeatureMetadata
except ImportError:  # pragma: no cover - support direct script execution
    try:
        from ...common.feature_metadata import GraphFeatureMetadata  # type: ignore
    except ImportError:  # pragma: no cover
        from feature_metadata import GraphFeatureMetadata  # type: ignore


def write_feature_metadata_artifacts(
    output_dir: Path,
    metadata_obj: GraphFeatureMetadata,
    metadata_dict: Dict[str, object],
    logger: logging.Logger,
) -> Tuple[Path, Optional[Path]]:
    metadata_path = output_dir / "feature_metadata.json"
    summary_path = output_dir / "schema_summary.json"

    summary_payload = metadata_obj.to_dict()
    topology_schema = metadata_dict.get("topology_schema")
    if topology_schema is not None:
        summary_payload["topology_schema"] = topology_schema
    topo_spec = metadata_dict.get("topology_schema_spec")
    if topo_spec is not None:
        summary_payload["topology_schema_spec"] = topo_spec

    actual_summary_path: Optional[Path] = summary_path
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        metadata_obj.summary_path = str(summary_path)
        metadata_dict["summary_path"] = str(summary_path)
        logger.info("Schema summary written to %s", summary_path)
    except OSError as exc:
        logger.warning("Unable to write schema summary file: %s", exc)
        actual_summary_path = None

    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_dict, handle, indent=2)
        logger.info("Feature metadata written to %s", metadata_path)
    except OSError as exc:
        logger.warning("Unable to write feature metadata file: %s", exc)

    return metadata_path, actual_summary_path
