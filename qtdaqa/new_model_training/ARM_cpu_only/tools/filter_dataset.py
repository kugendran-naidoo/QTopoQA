#!/usr/bin/env python3
"""Auto-filter train/val CSVs to entries with available graph files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter TopoQA train/val CSVs to the subset that has matching .pt graphs. "
            "Use this when you generate a new batch of graphs (e.g., pilot subset) and need "
            "train/val files that only contain models present in that graph directory. "
            "The script never runs automatically—call it manually before training if you "
            "want refreshed CSVs."
        )
    )
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory containing .pt graphs")
    parser.add_argument("--train", type=Path, required=True, help="Input train.csv")
    parser.add_argument("--val", type=Path, required=True, help="Input val.csv")
    parser.add_argument("--train-out", type=Path, required=True, help="Output train CSV")
    parser.add_argument("--val-out", type=Path, required=True, help="Output val CSV")
    parser.add_argument("--val-split", type=float, default=None, help="Optional fraction of filtered train to reserve for validation if val.csv has no overlap")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for optional re-split")
    return parser.parse_args()


def filter_df(df: pd.DataFrame, available: set[str]) -> pd.DataFrame:
    return df[df["MODEL"].isin(available)].copy()


def main() -> int:
    if len(sys.argv) == 1:
        print(
            "filter_dataset.py\n"
            "------------------\n"
            "Purpose: Manually regenerate train/val CSVs *after* you build a new set of graphs. "
            "Run it whenever your graph directory changes so that training only sees rows with "
            "available `.pt` files.\n\n"
            "Example:\n"
            "  python tools/filter_dataset.py \\\n"
            "      --graph-dir ../../new_graph_builder/output/pilot_batch/graph_data \\\n"
            "      --train ./train.csv --val ./val.csv \\\n"
            "      --train-out ./train_pilot.csv --val-out ./val_pilot.csv \\\n"
            "      --val-split 0.15 --seed 42\n"
        )
        return 0

    args = parse_args()
    available = {path.stem for path in args.graph_dir.glob("*.pt")}
    if not available:
        raise SystemExit(f"No .pt files found under {args.graph_dir}")

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    train_filtered = filter_df(train_df, available)
    val_filtered = filter_df(val_df, available)

    if args.val_split is not None:
        combined = pd.concat([train_filtered, val_filtered], axis=0, ignore_index=True)
        if combined.empty:
            raise SystemExit("No overlapping entries to split")
        val_size = max(1, int(len(combined) * args.val_split))
        val_filtered = combined.sample(n=val_size, random_state=args.seed)
        train_filtered = combined.drop(val_filtered.index)

    train_filtered.to_csv(args.train_out, index=False)
    val_filtered.to_csv(args.val_out, index=False)
    print(f"train rows: {len(train_filtered)} | val rows: {len(val_filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
