from __future__ import annotations

import json
import logging
from pathlib import Path

try:  # support package execution and direct script invocation
    from ...common.feature_metadata import load_graph_feature_metadata
except ImportError:  # pragma: no cover
    from common.feature_metadata import load_graph_feature_metadata  # type: ignore

LOG = logging.getLogger(__name__)


def write_schema_summary(graph_dir: Path) -> Path | None:
    """Generate a human-readable schema summary next to graph_metadata.json."""
    summary_path = graph_dir / "schema_summary.json"
    try:
        metadata = load_graph_feature_metadata(graph_dir)
        payload = metadata.to_dict()
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("Schema summary written to %s", summary_path)
        return summary_path
    except Exception as exc:  # pragma: no cover - best effort
        LOG.warning("Unable to write schema summary for %s: %s", graph_dir, exc)
        return None
