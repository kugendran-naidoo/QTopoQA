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
        # Fallback: if module_registry is empty, try to load co-located graph_builder_summary.json
        if not metadata.module_registry:
            builder_summary = graph_dir / "graph_builder_summary.json"
            if builder_summary.exists():
                try:
                    payload = json.loads(builder_summary.read_text(encoding="utf-8"))
                    modules = payload.get("modules")
                    if isinstance(modules, dict) and modules:
                        metadata.module_registry = modules
                        metadata.summary_path = str(builder_summary)
                except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - best effort
                    LOG.warning("Unable to read builder summary at %s: %s", builder_summary, exc)

        # For readability, emit module_registry in the same order as the feature config stages.
        if metadata.module_registry:
            ordered: dict[str, object] = {}
            for stage in ("interface", "topology", "node", "edge"):
                if stage in metadata.module_registry:
                    ordered[stage] = metadata.module_registry[stage]
            for key, value in metadata.module_registry.items():
                if key not in ordered:
                    ordered[key] = value
            metadata.module_registry = ordered

        payload = metadata.to_dict()
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("Schema summary written to %s", summary_path)
        return summary_path
    except Exception as exc:  # pragma: no cover - best effort
        LOG.warning("Unable to write schema summary for %s: %s", graph_dir, exc)
        return None
