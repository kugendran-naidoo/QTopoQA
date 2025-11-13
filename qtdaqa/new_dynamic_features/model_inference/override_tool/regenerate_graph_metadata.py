#!/usr/bin/env python3
"""Regenerate graph_metadata.json using a feature-config and sample .pt tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

try:  # pragma: no cover - allow running as script or module
    import torch
except ImportError:  # pragma: no cover
    raise SystemExit("torch is required to run this tool.")

from qtdaqa.new_dynamic_features.graph_builder2.lib.features_config import (
    FeatureSelection,
    load_feature_config,
)


DEFAULT_BUILDER_ID = "graph_builder2"
DEFAULT_BUILDER_VERSION = "unknown"
DEFAULT_BUILDER_MODULE = "qtdaqa.new_dynamic_features.graph_builder2.graph_builder2"


@dataclass
class GraphSample:
    key: str
    path: Path
    edge_dim: Optional[int]
    node_dim: Optional[int]
    metadata: Optional[Dict[str, object]]


def _compute_dim(tensor) -> Optional[int]:
    if tensor is None or not hasattr(tensor, "dim"):
        return None
    dim_fn = getattr(tensor, "dim")
    try:
        rank = int(dim_fn())
    except Exception:
        return None
    try:
        shape = tensor.shape
    except Exception:
        return None
    if rank == 1 and len(shape) >= 1:
        return int(shape[0])
    if rank >= 2 and len(shape) >= 1:
        return int(shape[-1])
    return None


def _iter_graphs(graph_dir: Path, limit: Optional[int]) -> Iterable[Path]:
    count = 0
    for path in sorted(graph_dir.rglob("*.pt")):
        if not path.is_file():
            continue
        yield path
        count += 1
        if limit and count >= limit:
            break


def _relative_model_key(graph_dir: Path, path: Path) -> str:
    try:
        relative = path.relative_to(graph_dir)
    except ValueError:
        relative = path
    if relative.suffix == ".pt":
        relative = relative.with_suffix("")
    return relative.as_posix()


def _load_samples(graph_dir: Path, limit: Optional[int]) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for path in _iter_graphs(graph_dir, limit):
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Unable to load {path}: {exc}") from exc
        edge_dim = _compute_dim(getattr(data, "edge_attr", None))
        node_dim = _compute_dim(getattr(data, "x", None))
        metadata = getattr(data, "metadata", None)
        key = _relative_model_key(graph_dir, path)
        samples.append(GraphSample(key=key, path=path, edge_dim=edge_dim, node_dim=node_dim, metadata=metadata))
    if not samples:
        raise RuntimeError(f"No .pt files found under {graph_dir}")
    return samples


def _compute_feature_snapshot(config_path: Path, include_text: bool) -> Dict[str, object]:
    text = config_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    snapshot = {
        "path": str(config_path.resolve()),
        "sha256": digest,
        "text": text if include_text else None,
    }
    return snapshot


def _sanitize_builder(builder_info: Dict[str, object]) -> Dict[str, object]:
    sanitized = json.loads(json.dumps(builder_info))
    feature_config = sanitized.get("feature_config")
    if isinstance(feature_config, dict):
        feature_config["text"] = None
    return sanitized


def _builder_edge_dump_payload(enabled: bool, output_dir: Optional[Path], configured_dir: Optional[Path]) -> Dict[str, object]:
    return {
        "enabled": enabled,
        "output_directory": str(output_dir) if output_dir else None,
        "configured_directory": str(configured_dir) if configured_dir else None,
    }


def _generate_node_columns(sample: GraphSample) -> List[str]:
    if sample.metadata and isinstance(sample.metadata, dict):
        existing = sample.metadata.get("node_feature_columns")
        if isinstance(existing, list) and all(isinstance(item, str) for item in existing):
            return existing
    if sample.node_dim:
        return [f"feature_{idx:03d}" for idx in range(sample.node_dim)]
    return []


def _build_entry(
    sample: GraphSample,
    selection: FeatureSelection,
    builder_stub: Optional[Dict[str, object]],
) -> Dict[str, object]:
    metadata = {}
    if sample.metadata and isinstance(sample.metadata, dict):
        edge_info = sample.metadata.get("edge_info")
        if isinstance(edge_info, dict):
            metadata.update(edge_info)
    if sample.edge_dim is not None:
        metadata.setdefault("feature_dim", sample.edge_dim)

    entry = {
        "edge_module": selection.edge.get("module"),
        "edge_params": selection.edge.get("params", {}),
        "edge_metadata": metadata,
        "node_feature_columns": _generate_node_columns(sample),
    }
    if builder_stub:
        entry["builder"] = builder_stub
    return entry


def regenerate_graph_metadata(
    graph_dir: Path,
    feature_config: Path,
    output_path: Path,
    *,
    builder_id: str,
    builder_version: str,
    builder_module: str,
    builder_schema_version: int,
    include_feature_text: bool,
    edge_dumps_enabled: bool,
    edge_dump_dir: Optional[Path],
    edge_dump_configured_dir: Optional[Path],
    sample_limit: Optional[int],
) -> Path:
    graph_dir = graph_dir.resolve()
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    feature_config = feature_config.resolve()
    if not feature_config.exists():
        raise FileNotFoundError(f"Feature config not found: {feature_config}")

    selection = load_feature_config(feature_config)
    samples = _load_samples(graph_dir, sample_limit)

    builder_info = {
        "id": builder_id,
        "version": builder_version,
        "schema_version": builder_schema_version,
        "module": builder_module,
        "feature_config": _compute_feature_snapshot(feature_config, include_feature_text),
        "edge_dumps": _builder_edge_dump_payload(edge_dumps_enabled, edge_dump_dir, edge_dump_configured_dir),
        "command": os.environ.get("REGENERATE_GRAPH_METADATA_CMD"),
    }
    builder_stub = _sanitize_builder(builder_info)

    payload: Dict[str, object] = {"_builder": builder_info}
    for sample in samples:
        payload[sample.key] = _build_entry(sample, selection, builder_stub)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate graph_metadata.json using a feature-config and existing .pt graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory containing *.pt graph files.")
    parser.add_argument("--feature-config", type=Path, required=True, help="Path to the feature-config YAML.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output graph_metadata.json path (defaults to <graph-dir>/graph_metadata.regenerated.json).",
    )
    parser.add_argument("--builder-id", default=DEFAULT_BUILDER_ID, help="Builder identifier to record.")
    parser.add_argument("--builder-version", default=DEFAULT_BUILDER_VERSION, help="Builder version string.")
    parser.add_argument(
        "--builder-module",
        default=DEFAULT_BUILDER_MODULE,
        help="Builder module/entrypoint to record in metadata.",
    )
    parser.add_argument(
        "--builder-schema-version",
        type=int,
        default=1,
        help="Schema version for the builder metadata block.",
    )
    parser.add_argument(
        "--include-feature-text",
        action="store_true",
        help="Embed the full feature-config text inside the builder snapshot (_builder.feature_config.text).",
    )
    parser.add_argument(
        "--edge-dumps-enabled",
        action="store_true",
        help="Record that edge CSV dumps are available for this dataset.",
    )
    parser.add_argument(
        "--edge-dump-dir",
        type=Path,
        default=None,
        help="Path to the directory storing edge CSV dumps (if available).",
    )
    parser.add_argument(
        "--edge-dump-configured-dir",
        type=Path,
        default=None,
        help="Configured edge dump directory recorded in builder metadata.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of .pt files inspected when building metadata.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Path:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_path = args.output or (args.graph_dir / "graph_metadata.regenerated.json")
    os.environ.setdefault("REGENERATE_GRAPH_METADATA_CMD", " ".join(parser.prog.split()))
    return regenerate_graph_metadata(
        args.graph_dir,
        args.feature_config,
        output_path,
        builder_id=args.builder_id,
        builder_version=args.builder_version,
        builder_module=args.builder_module,
        builder_schema_version=int(args.builder_schema_version),
        include_feature_text=bool(args.include_feature_text),
        edge_dumps_enabled=bool(args.edge_dumps_enabled),
        edge_dump_dir=args.edge_dump_dir,
        edge_dump_configured_dir=args.edge_dump_configured_dir,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":  # pragma: no cover
    path = main()
    print(f"graph_metadata written to {path}")
