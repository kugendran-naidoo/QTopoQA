#!/usr/bin/env python3
"""Inspect batch vectors in saved TopoQA graph files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch_geometric.data import Data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PyG batch vectors in saved .pt graphs")
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing graph .pt files (e.g. graph_data)",
    )
    parser.add_argument(
        "--train-label-file",
        required=True,
        help="CSV listing training models (must contain a MODEL column)",
    )
    parser.add_argument(
        "--val-label-file",
        required=True,
        help="CSV listing validation models (must contain a MODEL column)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="How many stale examples to show",
    )
    return parser.parse_args()


def load_model_list(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "MODEL" not in df.columns:
        raise ValueError(f"{path} does not contain a MODEL column")
    return set(df["MODEL"].astype(str))


def inspect_graph(data_path: Path) -> tuple[bool, dict[str, object]]:
    data: Data = torch.load(data_path)

    batch = getattr(data, "batch", None)
    num_nodes = int(data.num_nodes) if data.num_nodes is not None else 0

    # Determine staleness
    stale = False
    reason = None
    if batch is None:
        stale = True
        reason = "missing"
    else:
        if batch.numel() != num_nodes:
            stale = True
            reason = f"length {batch.numel()} != num_nodes {num_nodes}"
        elif batch.dtype != torch.long:
            stale = True
            reason = f"dtype {batch.dtype}"

    summary: dict[str, object] = {
        "path": data_path,
        "num_nodes": num_nodes,
        "stale": stale,
        "reason": reason,
        "unique": None,
        "counts": None,
    }

    if batch is not None and not stale:
        unique, counts = torch.unique(batch.cpu(), return_counts=True)
        summary["unique"] = unique.tolist()
        summary["counts"] = counts.tolist()
    return stale, summary


def main() -> None:
    args = parse_args()

    graph_dir = Path(args.graph_dir).resolve()
    if not graph_dir.is_dir():
        raise SystemExit(f"Graph directory not found: {graph_dir}")

    train_models = load_model_list(Path(args.train_label_file).resolve())
    val_models = load_model_list(Path(args.val_label_file).resolve())
    wanted = train_models | val_models

    available = {Path(p).stem for p in os.listdir(graph_dir)}
    models = sorted(wanted & available)
    if not models:
        raise SystemExit("No overlapping models between labels and graph directory")

    stale_examples: list[dict[str, object]] = []
    typical_examples: list[dict[str, object]] = []
    stale_count = 0

    for model in models:
        pt_path = graph_dir / f"{model}.pt"
        stale, summary = inspect_graph(pt_path)
        if stale:
            stale_count += 1
            if len(stale_examples) < args.num_examples:
                stale_examples.append(summary)
        else:
            if len(typical_examples) < args.num_examples:
                typical_examples.append(summary)

    total = len(models)
    ok_count = total - stale_count

    print("=== Batch Vector Inspection ===")
    print(f"Graph directory : {graph_dir}")
    print(f"Total graphs    : {total}")
    print(f"Stale batches   : {stale_count}")
    print(f"Healthy batches : {ok_count}")
    print()

    if stale_examples:
        print("-- Examples of stale batch vectors --")
        for entry in stale_examples:
            print(f"{entry['path'].name}: {entry['reason']}")
        print()
    else:
        print("No stale batch vectors detected.\n")

    if typical_examples:
        print("-- Typical batch vector summary (first few) --")
        for entry in typical_examples:
            unique = entry.get("unique") or []
            counts = entry.get("counts") or []
            print(
                f"{entry['path'].name}: num_nodes={entry['num_nodes']} "
                f"unique={unique} counts={counts}"
            )


if __name__ == "__main__":
    main()
