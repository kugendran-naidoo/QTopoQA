from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class GraphFeatureMetadata:
    edge_schema: Dict[str, object] = field(default_factory=dict)
    node_schema: Dict[str, object] = field(default_factory=dict)
    module_registry: Dict[str, object] = field(default_factory=dict)
    metadata_path: Optional[str] = None
    summary_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "edge_schema": self.edge_schema,
            "node_schema": self.node_schema,
            "module_registry": self.module_registry,
            "metadata_path": self.metadata_path,
            "summary_path": self.summary_path,
            "notes": self.notes,
        }


def _ensure_list(value: Optional[Sequence[str]]) -> List[str]:
    if not value:
        return []
    seen = set()
    ordered: List[str] = []
    for item in value:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _get_nested(entry: Dict[str, object], path: Sequence[str]):
    current: object = entry
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _collect_consistent(
    entries: Sequence[Dict[str, object]],
    path: Sequence[str],
    *,
    note_label: str,
    metadata: GraphFeatureMetadata,
):
    if not entries:
        return None
    first = _get_nested(entries[0], path)
    if first is None:
        return None
    for entry in entries[1:]:
        other = _get_nested(entry, path)
        if other != first:
            metadata.notes.append(
                f"Inconsistent {note_label}: encountered {other!r} but expected {first!r}"
            )
            break
    return first


def _model_to_graph_path(graph_dir: Path, model: str) -> Path:
    return (graph_dir / f"{model}.pt").resolve()


def _iter_sample_paths(graph_dir: Path, models: Sequence[str], limit: int) -> List[Path]:
    paths: List[Path] = []
    for model in models:
        candidate = _model_to_graph_path(graph_dir, model)
        if candidate.exists():
            paths.append(candidate)
        else:
            # Caller is responsible for logging notes about missing files.
            continue
        if len(paths) >= limit:
            break
    if not paths:
        # fall back to discovering any graphs under the directory
        for candidate in sorted(graph_dir.rglob("*.pt")):
            paths.append(candidate.resolve())
            if len(paths) >= limit:
                break
    return paths


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_summary_path(graph_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        path = explicit.resolve()
        return path if path.exists() else None

    target_dir = graph_dir.resolve()
    for parent in graph_dir.parents:
        logs_dir = parent / "logs"
        if not logs_dir.is_dir():
            continue
        for summary in sorted(logs_dir.rglob("graph_builder_summary.json"), reverse=True):
            try:
                data = _load_json(summary)
            except (OSError, json.JSONDecodeError):
                continue
            edge_info = data.get("edge", {})
            output_dir = edge_info.get("output_dir")
            if not output_dir:
                continue
            try:
                output_path = Path(output_dir).resolve()
            except OSError:
                continue
            if output_path == target_dir:
                return summary.resolve()
    return None


def load_graph_feature_metadata(
    graph_dir: Path,
    *,
    sample_models: Optional[Sequence[str]] = None,
    metadata_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    max_pt_samples: int = 32,
) -> GraphFeatureMetadata:
    graph_dir = Path(graph_dir).resolve()
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_dir}")

    metadata = GraphFeatureMetadata()

    selected_models = _ensure_list(sample_models)
    metadata_file = metadata_path.resolve() if metadata_path else (graph_dir / "graph_metadata.json")
    if metadata_file.exists():
        try:
            raw_entries = _load_json(metadata_file)
        except (OSError, json.JSONDecodeError) as exc:
            metadata.notes.append(f"Failed to parse graph metadata at {metadata_file}: {exc}")
            raw_entries = {}
        metadata.metadata_path = str(metadata_file)
    else:
        raw_entries = {}
        metadata.notes.append(f"graph_metadata.json not found at {metadata_file}")

    entries: List[Dict[str, object]] = []
    missing_models: List[str] = []
    if raw_entries:
        if selected_models:
            for model in selected_models:
                entry = raw_entries.get(model)
                if entry:
                    entries.append(entry)
                else:
                    missing_models.append(model)
        else:
            entries = list(raw_entries.values())

    if missing_models:
        metadata.notes.append(
            f"{len(missing_models)} model entries missing from graph_metadata.json (examples: "
            f"{', '.join(missing_models[:5])})"
        )

    edge_dim = _collect_consistent(
        entries,
        ("edge_metadata", "feature_dim"),
        note_label="edge feature_dim",
        metadata=metadata,
    )
    edge_variant = _collect_consistent(
        entries,
        ("edge_metadata", "edge_feature_variant"),
        note_label="edge feature_variant",
        metadata=metadata,
    )
    edge_module = _collect_consistent(
        entries,
        ("edge_module",),
        note_label="edge module id",
        metadata=metadata,
    )
    edge_params = _collect_consistent(
        entries,
        ("edge_params",),
        note_label="edge parameters",
        metadata=metadata,
    )
    edge_bands = _collect_consistent(
        entries,
        ("edge_metadata", "bands"),
        note_label="edge bands",
        metadata=metadata,
    )

    node_columns = _collect_consistent(
        entries,
        ("node_feature_columns",),
        note_label="node feature columns",
        metadata=metadata,
    )

    if edge_dim is not None:
        try:
            edge_dim = int(edge_dim)
        except (TypeError, ValueError):
            metadata.notes.append(f"Unable to coerce edge dimension {edge_dim!r} to int.")
            edge_dim = None

    node_dim = None
    if isinstance(node_columns, list):
        node_dim = len(node_columns)

    metadata.edge_schema.update(
        {
            key: value
            for key, value in (
                ("dim", edge_dim),
                ("module", edge_module),
                ("variant", edge_variant),
                ("module_params", edge_params),
                ("bands", edge_bands),
            )
            if value is not None
        }
    )
    if metadata.metadata_path:
        metadata.edge_schema.setdefault("source", metadata.metadata_path)

    if node_columns is not None:
        metadata.node_schema["columns"] = node_columns
    if node_dim is not None:
        metadata.node_schema["dim"] = node_dim
    if metadata.metadata_path:
        metadata.node_schema.setdefault("source", metadata.metadata_path)

    summary_file = _discover_summary_path(graph_dir, summary_path)
    if summary_file:
        metadata.summary_path = str(summary_file)
        try:
            summary_payload = _load_json(summary_file)
        except (OSError, json.JSONDecodeError) as exc:
            metadata.notes.append(f"Failed to parse graph builder summary at {summary_file}: {exc}")
            summary_payload = {}
        modules = summary_payload.get("modules")
        if isinstance(modules, dict):
            metadata.module_registry = modules
            edge_info = modules.get("edge")
            if isinstance(edge_info, dict):
                alias = edge_info.get("alias")
                if alias:
                    metadata.edge_schema.setdefault("alias", alias)
                jobs = edge_info.get("jobs")
                if jobs is not None:
                    metadata.edge_schema.setdefault("jobs", jobs)
                summary_text = edge_info.get("summary")
                if summary_text:
                    metadata.edge_schema.setdefault("summary", summary_text)

    graph_paths = _iter_sample_paths(graph_dir, selected_models, max_pt_samples)
    edge_dims_from_graphs: List[int] = []
    node_dims_from_graphs: List[int] = []
    metadata_dicts: List[Dict[str, object]] = []

    for graph_path in graph_paths:
        try:
            data = torch.load(graph_path, map_location="cpu")
        except (OSError, RuntimeError) as exc:
            metadata.notes.append(f"Failed to load {graph_path}: {exc}")
            continue

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and hasattr(edge_attr, "dim"):
            if edge_attr.dim() == 1:
                edge_dims_from_graphs.append(int(edge_attr.shape[0]))
            elif edge_attr.dim() >= 2:
                edge_dims_from_graphs.append(int(edge_attr.shape[-1]))

        node_attr = getattr(data, "x", None)
        if node_attr is not None and hasattr(node_attr, "dim"):
            if node_attr.dim() == 1:
                node_dims_from_graphs.append(int(node_attr.shape[0]))
            elif node_attr.dim() >= 2:
                node_dims_from_graphs.append(int(node_attr.shape[-1]))

        graph_meta = getattr(data, "metadata", None)
        if isinstance(graph_meta, dict):
            metadata_dicts.append(graph_meta)

    def _resolve_dim(primary: Optional[int], derived: Sequence[int], label: str) -> Optional[int]:
        unique_dims = sorted({dim for dim in derived if dim is not None})
        if not unique_dims:
            return primary
        if len(unique_dims) > 1:
            metadata.notes.append(
                f"Inconsistent {label} derived from graphs: {unique_dims}"
            )
        derived_dim = unique_dims[-1]
        if primary is None:
            return derived_dim
        if primary != derived_dim:
            metadata.notes.append(
                f"{label} mismatch between metadata ({primary}) and graphs ({derived_dim}); "
                f"using {derived_dim}"
            )
            return derived_dim
        return primary

    resolved_edge_dim = _resolve_dim(metadata.edge_schema.get("dim"), edge_dims_from_graphs, "edge feature dimension")
    if resolved_edge_dim is not None:
        metadata.edge_schema["dim"] = resolved_edge_dim

    resolved_node_dim = _resolve_dim(metadata.node_schema.get("dim"), node_dims_from_graphs, "node feature dimension")
    if resolved_node_dim is not None:
        metadata.node_schema["dim"] = resolved_node_dim

    if metadata_dicts:
        edge_modules = {
            entry.get("edge_module")
            for entry in metadata_dicts
            if isinstance(entry.get("edge_module"), str)
        }
        if edge_modules:
            if len(edge_modules) == 1:
                metadata.edge_schema.setdefault("module", edge_modules.pop())
            else:
                metadata.notes.append(f"Multiple edge modules found in graphs: {sorted(edge_modules)}")
        edge_infos = [
            entry.get("edge_info") for entry in metadata_dicts if isinstance(entry.get("edge_info"), dict)
        ]
        if edge_infos:
            info = edge_infos[0]
            variant = info.get("edge_feature_variant")
            if variant:
                metadata.edge_schema.setdefault("variant", variant)
            bands = info.get("bands")
            if bands:
                metadata.edge_schema.setdefault("bands", bands)
            counted_dim = info.get("feature_dim")
            if counted_dim and "dim" not in metadata.edge_schema:
                try:
                    metadata.edge_schema["dim"] = int(counted_dim)
                except (TypeError, ValueError):
                    metadata.notes.append(f"Unable to coerce feature_dim {counted_dim!r} from graph metadata.")

    if "dim" not in metadata.edge_schema or metadata.edge_schema["dim"] is None:
        raise RuntimeError(
            "Unable to determine edge feature dimension from metadata; "
            "ensure graph_metadata.json or .pt files include edge_attr tensors."
        )

    return metadata
