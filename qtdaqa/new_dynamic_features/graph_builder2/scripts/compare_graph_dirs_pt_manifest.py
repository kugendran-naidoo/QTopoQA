#!/usr/bin/env python3
"""
Compare two graph_builder2 graph_data directories by their .pt inventory.

This is a lightweight helper for ablation studies (e.g., baseline PH vs null-topology)
to ensure both runs train on the same set of graphs.

It does not load .pt files (no torch dependency). It only compares filenames.

Usage:
  python compare_graph_dirs_pt_manifest.py <graph_dir_a> <graph_dir_b> [--write <out.json>]

Output:
  - counts for A, B, intersection, only-in-A, only-in-B
  - optional JSON with lists of relative paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _list_pt_files(graph_dir: Path) -> list[str]:
    if not graph_dir.exists():
        raise SystemExit(f"Graph dir does not exist: {graph_dir}")
    paths = []
    for path in graph_dir.rglob("*.pt"):
        try:
            rel = path.relative_to(graph_dir)
        except ValueError:
            rel = Path(path.name)
        paths.append(str(rel))
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_dir_a", type=Path)
    parser.add_argument("graph_dir_b", type=Path)
    parser.add_argument("--write", type=Path, default=None, help="Write JSON report to this path.")
    args = parser.parse_args()

    a = args.graph_dir_a
    b = args.graph_dir_b
    list_a = _list_pt_files(a)
    list_b = _list_pt_files(b)
    set_a = set(list_a)
    set_b = set(list_b)

    inter = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    print(f"A: {a} -> {len(list_a)} graphs")
    print(f"B: {b} -> {len(list_b)} graphs")
    print(f"Intersection: {len(inter)}")
    print(f"Only in A: {len(only_a)}")
    print(f"Only in B: {len(only_b)}")

    if args.write:
        payload = {
            "graph_dir_a": str(a),
            "graph_dir_b": str(b),
            "count_a": len(list_a),
            "count_b": len(list_b),
            "intersection": inter,
            "only_a": only_a,
            "only_b": only_b,
        }
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote report: {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

