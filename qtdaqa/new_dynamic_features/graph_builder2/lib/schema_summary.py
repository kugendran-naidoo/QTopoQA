from __future__ import annotations

import json
import logging
from pathlib import Path
import traceback
import os

try:  # support package execution and direct script invocation
    from ...common.feature_metadata import load_graph_feature_metadata
    from ...common.feature_metadata import GraphFeatureMetadata
except ImportError:  # pragma: no cover
    from common.feature_metadata import load_graph_feature_metadata, GraphFeatureMetadata  # type: ignore

LOG = logging.getLogger(__name__)


def _load_topology_columns(graph_dir: Path) -> list[str] | None:
    """
    Attempt to load topology column names for schema summary enrichment.

    Source path can be overridden with QTOPO_TOPOLOGY_COLUMNS_PATH; defaults to
    <graph_dir>/topology_columns.json. Missing or invalid files emit a warning.
    """
    override = os.environ.get("QTOPO_TOPOLOGY_COLUMNS_PATH")
    candidate = Path(override).expanduser() if override else graph_dir / "topology_columns.json"
    if not candidate.exists():
        LOG.warning("Topology columns file not found for schema summary: %s", candidate)
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            return data
        LOG.warning("Topology columns file %s is not a list of strings; skipping.", candidate)
    except Exception as exc:  # pragma: no cover - best effort
        LOG.warning("Unable to load topology columns from %s: %s", candidate, exc)
    return None


def write_schema_summary(graph_dir: Path) -> Path | None:
    """Generate a human-readable schema summary next to graph_metadata.json."""
    summary_path = graph_dir / "schema_summary.json"
    try:
        try:
            # Skip PT sampling to keep this lightweight and avoid failures when torch is unavailable.
            metadata = load_graph_feature_metadata(graph_dir, max_pt_samples=0)
        except Exception as exc:
            try:
                # Backward-compatible fallback for loaders without max_pt_samples (e.g., tests).
                metadata = load_graph_feature_metadata(graph_dir)  # type: ignore[arg-type]
            except Exception as exc2:
                LOG.warning("Falling back to minimal schema summary for %s: %s", graph_dir, exc2)
                metadata = GraphFeatureMetadata()
                metadata.notes.append(f"schema_summary fallback: {exc2}")
                # Keep a breadcrumb to the graph metadata path for debugging.
                metadata.metadata_path = str((graph_dir / "graph_metadata.json").resolve())
                LOG.debug("schema_summary fallback traceback:\n%s", traceback.format_exc())
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
        # Explicitly mirror node feature columns for convenience
        if hasattr(metadata, "node_feature_columns") and metadata.node_feature_columns:
            payload["node_feature_columns"] = list(metadata.node_feature_columns)
        topo_cols = _load_topology_columns(graph_dir)
        if topo_cols is not None:
            payload["topology_columns"] = topo_cols
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("Schema summary written to %s", summary_path)
        return summary_path
    except Exception as exc:  # pragma: no cover - best effort
        LOG.warning("Unable to write schema summary for %s: %s", graph_dir, exc)
        return None
