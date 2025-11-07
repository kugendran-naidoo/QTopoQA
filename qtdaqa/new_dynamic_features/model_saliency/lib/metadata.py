"""
Utilities for loading graph feature metadata and feature group taxonomy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

from ...common.feature_metadata import load_graph_feature_metadata, GraphFeatureMetadata


@dataclass
class FeatureGroup:
    name: str
    columns: Sequence[str]
    indices: Sequence[int]
    description: Optional[str] = None


@dataclass
class GraphFeatureContext:
    metadata: GraphFeatureMetadata
    node_groups: List[FeatureGroup]
    edge_groups: List[FeatureGroup]


def load_metadata(
    graph_dir: Path,
    *,
    sample_models: Optional[Sequence[str]] = None,
    metadata_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> GraphFeatureMetadata:
    return load_graph_feature_metadata(
        graph_dir,
        sample_models=sample_models,
        metadata_path=metadata_path,
        summary_path=summary_path,
    )


def _group_columns_by_prefix(columns: Sequence[str], delimiter: str = "_") -> Mapping[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for col in columns:
        prefix = col.split(delimiter, 1)[0] if delimiter in col else "misc"
        groups.setdefault(prefix, []).append(col)
    return groups


def derive_feature_groups(metadata: GraphFeatureMetadata) -> GraphFeatureContext:
    node_columns = metadata.node_schema.get("columns")
    if not isinstance(node_columns, list):
        node_columns = []

    module_registry = metadata.module_registry if isinstance(metadata.module_registry, dict) else {}
    node_info = module_registry.get("node", {}) if isinstance(module_registry.get("node"), dict) else {}
    edge_info = module_registry.get("edge", {}) if isinstance(module_registry.get("edge"), dict) else {}

    node_groups: List[FeatureGroup] = []
    column_index = {col: idx for idx, col in enumerate(node_columns)}
    if node_columns:
        grouped = _group_columns_by_prefix(node_columns)
        for name, cols in grouped.items():
            desc = None
            if name == "dssp" and node_info.get("summary"):
                desc = f"DSSP-derived features ({node_info['summary']})"
            elif name.startswith("ph"):
                desc = "Persistent homology statistics"
            indices = [column_index[c] for c in cols if c in column_index]
            node_groups.append(FeatureGroup(name=name, columns=cols, indices=indices, description=desc))

    edge_groups = [
        FeatureGroup(
            name=edge_info.get("alias") or edge_info.get("id") or "edge_features",
            columns=[],
            indices=(),
            description=edge_info.get("summary"),
        )
    ]

    return GraphFeatureContext(metadata=metadata, node_groups=node_groups, edge_groups=edge_groups)


def load_graph(path: Path) -> Data:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)!r}")
    return data


def resolve_graph_paths(graph_dir: Path, models: Iterable[str]) -> List[Tuple[str, Path]]:
    resolved: List[Tuple[str, Path]] = []
    for model in models:
        candidate = graph_dir / f"{model}.pt"
        if candidate.exists():
            resolved.append((model, candidate))
        else:
            alt = candidate.with_suffix("")  # fallback without .pt
            if alt.exists():
                resolved.append((model, alt))
    return resolved


def export_metadata(context: GraphFeatureContext, output_path: Path) -> None:
    payload = {
        "edge_schema": context.metadata.edge_schema,
        "node_schema": context.metadata.node_schema,
        "module_registry": context.metadata.module_registry,
        "notes": context.metadata.notes,
        "node_groups": [
            {
                "name": group.name,
                "columns": list(group.columns),
                "indices": list(group.indices),
                "description": group.description,
            }
            for group in context.node_groups
        ],
        "edge_groups": [
            {
                "name": group.name,
                "columns": list(group.columns),
                "indices": list(group.indices),
                "description": group.description,
            }
            for group in context.edge_groups
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
