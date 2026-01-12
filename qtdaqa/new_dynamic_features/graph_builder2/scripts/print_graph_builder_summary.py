#!/usr/bin/env python3
"""Print feature dims + metadata map from graph_builder_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _resolve_summary_path(target: Path) -> Optional[Path]:
    if target.is_file():
        return target
    if target.is_dir():
        direct = target / "graph_builder_summary.json"
        if direct.exists():
            return direct
        nested = target / "graph_data" / "graph_builder_summary.json"
        if nested.exists():
            return nested
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print feature dims + metadata map from graph_builder_summary.json.",
    )
    parser.add_argument(
        "output_dir",
        help="Path to a graph_data dir, output run dir, or graph_builder_summary.json.",
    )
    args = parser.parse_args()

    summary_path = _resolve_summary_path(Path(args.output_dir).expanduser())
    if summary_path is None:
        print(f"Unable to locate graph_builder_summary.json under {args.output_dir}")
        return 2

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    dims = data.get("feature_dims") or {}
    print("[Feature Dimensions]")
    if dims:
        for key in ("topology_feature_dim", "node_feature_dim", "edge_feature_dim"):
            if key in dims:
                print(f"  {key}: {dims.get(key)}")
    else:
        print("  (missing)")

    meta_map = data.get("metadata_map") or {}
    print("\n[Metadata Map]")
    if meta_map:
        preferred_order = [
            "graph_metadata.json",
            "topology_columns.json",
            "node_columns.json",
            "edge_columns.json",
            "graph_manifest.json",
            "graph_builder_summary.json",
            "graph_builder_summary.log",
        ]
        seen = set()
        ordered_items = []
        for name in preferred_order:
            if name in meta_map:
                ordered_items.append((name, meta_map.get(name)))
                seen.add(name)
            else:
                fallback_path = summary_path.parent / name
                ordered_items.append(
                    (
                        name,
                        {
                            "description": "Not listed in metadata map (expected file).",
                            "path": str(fallback_path),
                        },
                    )
                )
                seen.add(name)
        for name, info in meta_map.items():
            if name not in seen:
                ordered_items.append((name, info))
        for name, info in ordered_items:
            description = ""
            path_value = ""
            if isinstance(info, dict):
                description = info.get("description", "")
                path_value = info.get("path", "")
            print(f"  {name}: {description}")
            if path_value:
                exists = Path(path_value).exists()
                status = "OK" if exists else "MISSING"
                print(f"    path: {path_value} [{status}]")
    else:
        print("  (missing)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
