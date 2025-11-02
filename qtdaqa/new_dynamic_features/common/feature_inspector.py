#!/usr/bin/env python3
"""
Feature inspector for dynamic TopoQA graph datasets.

This CLI interrogates a directory of PyTorch Geometric `.pt` files, infers the
edge/node schema, and surfaces diagnostic information such as inconsistent
dimensions or missing metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

try:  # Allow running as package or standalone script.
    from .feature_metadata import GraphFeatureMetadata, load_graph_feature_metadata
except ImportError:  # pragma: no cover - fallback when executed directly
    from feature_metadata import GraphFeatureMetadata, load_graph_feature_metadata


DEFAULT_MAX_GRAPHS = 25


@dataclass
class GraphDiagnostics:
    path: Path
    edge_dim: Optional[int]
    node_dim: Optional[int]
    has_metadata: bool
    edge_module: Optional[str]
    notes: List[str]


def format_override_block(metadata: GraphFeatureMetadata) -> str:
    """
    Render a commented YAML block that mirrors the metadata-derived override schemas.
    """
    edge = dict(getattr(metadata, "edge_schema", {}) or {})
    topo = dict(getattr(metadata, "topology_schema", {}) or {})
    if not topo:
        topology_entry = metadata.module_registry.get("topology") if isinstance(metadata.module_registry, dict) else None
        if isinstance(topology_entry, dict):
            for key in ("id", "alias", "summary", "jobs"):
                value = topology_entry.get(key)
                if value is not None:
                    topo.setdefault(key, value)
            defaults = topology_entry.get("defaults")
            if isinstance(defaults, dict) and defaults:
                topo.setdefault("defaults", defaults)
            parameters = topology_entry.get("parameters")
            if isinstance(parameters, dict) and parameters:
                topo.setdefault("parameters", parameters)

    lines = [
        "# Suggested override block (paste into a config and uncomment if needed).",
        "# edge/topology overrides are optional: the loader already reads every feature stage from graph_metadata.json",
        "# (see feature_metadata.module_registry). Only define this block when you deliberately want to override the",
        "# recorded metadata for a specific stage.",
        "# edge_schema:  # edge metadata",
    ]
    preferred_edge_keys = (
        "module",
        "variant",
        "dim",
        "bands",
        "use_layer_norm",
        "module_params",
        "jobs",
        "summary",
    )
    added_any = False
    for key in preferred_edge_keys:
        if key in edge:
            value = edge[key]
            if isinstance(value, (dict, list)):
                lines.append(f"#   {key}: {json.dumps(value, ensure_ascii=True)}")
            else:
                lines.append(f"#   {key}: {value!r}")
            added_any = True

    if not added_any and edge:
        for key, value in edge.items():
            if isinstance(value, (dict, list)):
                lines.append(f"#   {key}: {json.dumps(value, ensure_ascii=True)}")
            else:
                lines.append(f"#   {key}: {value!r}")

    lines.append("# topology_schema:  # topology metadata")
    if topo:
        for key, value in topo.items():
            if isinstance(value, (dict, list)):
                lines.append(f"#   {key}: {json.dumps(value, ensure_ascii=True)}")
            else:
                lines.append(f"#   {key}: {value!r}")
    else:
        lines.append("#   {}  # (metadata is empty)")
    return "\n".join(lines)


def _summarise_metadata(metadata: GraphFeatureMetadata) -> Dict[str, object]:
    summary = {
        "edge_schema": dict(metadata.edge_schema),
        "node_schema": dict(metadata.node_schema),
        "metadata_path": metadata.metadata_path,
        "summary_path": metadata.summary_path,
        "notes": list(metadata.notes),
    }
    if metadata.module_registry:
        summary["module_registry"] = {
            kind: {
                key: info.get(key)
                for key in ("id", "alias", "jobs", "summary")
                if isinstance(info, dict)
            }
            for kind, info in metadata.module_registry.items()
            if isinstance(info, dict)
        }
    return summary


def _diagnose_graph(path: Path) -> GraphDiagnostics:
    notes: List[str] = []
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - handled in CLI
        return GraphDiagnostics(path=path, edge_dim=None, node_dim=None, has_metadata=False, edge_module=None, notes=[f"load_error: {exc}"])

    edge_attr = getattr(data, "edge_attr", None)
    node_attr = getattr(data, "x", None)
    edge_dim = None
    node_dim = None
    if edge_attr is not None and hasattr(edge_attr, "dim"):
        if edge_attr.dim() == 1:
            edge_dim = int(edge_attr.shape[0])
        elif edge_attr.dim() >= 2:
            edge_dim = int(edge_attr.shape[-1])
        else:
            notes.append(f"edge_attr has unexpected rank {edge_attr.dim()}")
    else:
        notes.append("edge_attr missing or empty")

    if node_attr is not None and hasattr(node_attr, "dim"):
        if node_attr.dim() == 1:
            node_dim = int(node_attr.shape[0])
        elif node_attr.dim() >= 2:
            node_dim = int(node_attr.shape[-1])
        else:
            notes.append(f"x has unexpected rank {node_attr.dim()}")
    else:
        notes.append("node feature tensor missing or empty")

    metadata = getattr(data, "metadata", None)
    has_metadata = isinstance(metadata, dict)
    edge_module = metadata.get("edge_module") if has_metadata else None

    return GraphDiagnostics(
        path=path,
        edge_dim=edge_dim,
        node_dim=node_dim,
        has_metadata=has_metadata,
        edge_module=edge_module if isinstance(edge_module, str) else None,
        notes=notes,
    )


def _iter_graphs(graph_dir: Path, limit: Optional[int], recursive: bool) -> Iterable[Path]:
    glob_pattern = "**/*.pt" if recursive else "*.pt"
    paths = [path for path in sorted(graph_dir.glob(glob_pattern)) if path.is_file()]
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        yield path


def inspect_graphs(
    graph_dir: Path,
    *,
    metadata_path: Optional[Path],
    summary_path: Optional[Path],
    max_graphs: int,
    enumerate_all: bool,
    json_output: Optional[Path],
    emit_override_block: bool,
) -> None:
    metadata = load_graph_feature_metadata(
        graph_dir=graph_dir,
        metadata_path=metadata_path,
        summary_path=summary_path,
        max_pt_samples=max_graphs,
    )

    if emit_override_block:
        print(format_override_block(metadata))
        return

    per_graph: List[GraphDiagnostics] = []
    for graph_path in _iter_graphs(graph_dir, None if enumerate_all else max_graphs, recursive=True):
        per_graph.append(_diagnose_graph(graph_path))
        if not enumerate_all and len(per_graph) >= max_graphs:
            break

    summary = _summarise_metadata(metadata)
    summary["graph_count_scanned"] = len(per_graph)

    def _format_diag(diag: GraphDiagnostics) -> Dict[str, object]:
        return {
            "path": str(diag.path.relative_to(graph_dir)),
            "edge_dim": diag.edge_dim,
            "node_dim": diag.node_dim,
            "has_metadata": diag.has_metadata,
            "edge_module": diag.edge_module,
            "notes": diag.notes,
        }

    report = {
        "graph_dir": str(graph_dir),
        "summary": summary,
        "graphs": [_format_diag(diag) for diag in per_graph],
    }

    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect TopoQA graph feature schemas stored in PyG .pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "graph_dir",
        type=Path,
        nargs="?",
        help="Directory containing .pt graph files or nested subdirectories.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional path to graph_metadata.json (defaults to <graph_dir>/graph_metadata.json).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path to graph_builder_summary.json for richer context.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=DEFAULT_MAX_GRAPHS,
        help="Maximum number of .pt files to load for diagnostics (ignored with --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan every .pt file under graph_dir (may be slow on large datasets).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional file path to write the inspection report as JSON.",
    )
    parser.add_argument(
        "--emit-override-block",
        action="store_true",
        help="Print a commented YAML block with edge/topology override hints derived from metadata.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.graph_dir is None:
        parser.print_help(sys.stderr)
        return 1

    graph_dir = args.graph_dir.resolve()
    if not graph_dir.exists():
        parser.error(f"Graph directory does not exist: {graph_dir}")

    try:
        inspect_graphs(
            graph_dir=graph_dir,
            metadata_path=args.metadata_path,
            summary_path=args.summary_path,
            max_graphs=max(1, args.max_graphs),
            enumerate_all=bool(args.all),
            json_output=args.json_output.resolve() if args.json_output else None,
            emit_override_block=bool(args.emit_override_block),
        )
    except Exception as exc:  # pragma: no cover - CLI surfaces errors directly
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
