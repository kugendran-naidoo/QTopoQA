#!/usr/bin/env python3
"""
Estimate the in-memory footprint of a PyG batch given a directory of .pt graphs.

Usage:
  python estimate_batch_memory.py --path-to-graph_data /path/to/graph_data \
    --estimate-batch-size 24 --graph-sample 10

Arguments:
  --path-to-graph_data   Path to the graph_data directory containing .pt files (required).
  --estimate-batch-size  Batch size to estimate (e.g., 16, 24, 32). Default: 16.
  --graph-sample         Number of .pt files to sample. Default: 5.

Notes:
  - This sums the sizes of edge_attr, edge_index, and x (node features) per graph.
  - It reports the average graph size and the estimated batch size in MB.
  - It excludes runtime overhead (dataloader, intermediate tensors); add ~30–50% headroom.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def graph_size_bytes(data: object) -> int:
    total = 0
    if hasattr(data, "edge_attr") and getattr(data, "edge_attr") is not None:
        edge_attr = getattr(data, "edge_attr")
        total += edge_attr.numel() * edge_attr.element_size()
    if hasattr(data, "edge_index") and getattr(data, "edge_index") is not None:
        edge_index = getattr(data, "edge_index")
        total += edge_index.numel() * edge_index.element_size()
    if hasattr(data, "x") and getattr(data, "x") is not None:
        x = getattr(data, "x")
        total += x.numel() * x.element_size()
    return total


def estimate_batch(dir_path: Path, sample_n: int, batch_size: int) -> None:
    paths = sorted(dir_path.rglob("*.pt"))
    if not paths:
        print(f"No .pt files found under {dir_path}")
        sys.exit(1)
    sample_paths = paths[:sample_n]
    sizes = []
    for path in sample_paths:
        data = torch.load(path, map_location="cpu")
        sizes.append(graph_size_bytes(data))
    avg = sum(sizes) / len(sizes)
    est_batch = avg * batch_size
    est_with_overhead = est_batch * 1.5  # ~50% overhead allowance
    print(f"Sampled {len(sizes)} graphs from {dir_path}")
    print(f"Avg graph size: {avg/1e6:.2f} MB")
    print(f"Estimated batch size {batch_size}: {est_batch/1e6:.2f} MB (excluding overhead)")
    print(f"Estimated batch size {batch_size} with ~50% overhead: {est_with_overhead/1e6:.2f} MB")
    print("Note: overhead covers loader/intermediate tensors; adjust if your workload differs.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate PyG batch memory from graph_data .pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-to-graph_data",
        required=True,
        type=Path,
        help="Path to graph_data directory containing .pt files",
    )
    parser.add_argument(
        "--estimate-batch-size",
        type=int,
        default=16,
        help="Batch size to estimate (e.g., 16, 24, 32)",
    )
    parser.add_argument(
        "--graph-sample",
        type=int,
        default=5,
        help="Number of .pt files to sample",
    )
    args = parser.parse_args(argv)
    if args.graph_sample <= 0 or args.estimate_batch_size <= 0:
        parser.error("Batch size and graph sample must be positive integers.")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    estimate_batch(
        dir_path=args.path_to_graph_data,
        sample_n=args.graph_sample,
        batch_size=args.estimate_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
