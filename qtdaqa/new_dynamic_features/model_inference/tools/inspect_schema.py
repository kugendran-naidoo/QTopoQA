#!/usr/bin/env python3
"""Utility to inspect edge/node schema dims from a checkpoint and/or graph_dir."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import torch

try:
    from ..builder_runner import load_graph_feature_metadata  # type: ignore
except Exception:  # pragma: no cover
    from qtdaqa.new_dynamic_features.common.feature_metadata import load_graph_feature_metadata  # type: ignore


def _print_checkpoint(path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("feature_metadata") or {}
    edge = meta.get("edge_schema") or {}
    node = meta.get("node_schema") or {}
    print(f"[checkpoint] {path}")
    print(json.dumps({"edge_schema": edge, "node_schema": node}, indent=2))


def _print_graph_dir(graph_dir: Path) -> None:
    meta = load_graph_feature_metadata(graph_dir, max_pt_samples=0)
    print(f"[graph_dir] {graph_dir}")
    print(
        json.dumps(
            {
                "edge_schema": meta.edge_schema,
                "node_schema": meta.node_schema,
                "module_registry": meta.module_registry,
                "metadata_path": meta.metadata_path,
                "summary_path": meta.summary_path,
            },
            indent=2,
        )
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint", type=Path, help="Path to checkpoint (.chkpt)")
    parser.add_argument("-g", "--graph-dir", type=Path, help="Path to graph_dir containing graph_metadata.json")
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.graph_dir:
        parser.error("Provide at least one of --checkpoint or --graph-dir")

    if args.checkpoint:
        _print_checkpoint(args.checkpoint.resolve())
    if args.graph_dir:
        _print_graph_dir(args.graph_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
