#!/usr/bin/env python3
"""Print feature metadata embedded in a training checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show feature metadata stored in a TopoQA checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to the .ckpt/.chkpt file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint.get("feature_metadata")
    if metadata is None:
        print("No feature_metadata section found in checkpoint")
        return 1
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
