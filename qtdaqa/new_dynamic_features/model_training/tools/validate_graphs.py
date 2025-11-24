#!/usr/bin/env python3
"""Order-agnostic validator for graph_dir (.pt) files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def _node_permutation(x: torch.Tensor) -> torch.Tensor:
    x_cpu = x.detach().cpu()
    n = x_cpu.size(0)
    if n == 0:
        return torch.arange(0, device=x.device, dtype=torch.long)
    keys = [np.arange(n)]
    x_np = x_cpu.numpy()
    if x_np.ndim == 1:
        keys.append(x_np)
    else:
        for col in x_np.T[::-1]:
            keys.append(col)
    order = np.lexsort(tuple(keys))
    return torch.from_numpy(order).to(x.device, dtype=torch.long)


def _edge_permutation(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.arange(0, device=edge_index.device, dtype=torch.long)
    edge_cpu = edge_index.detach().cpu()
    src = edge_cpu[0].numpy()
    dst = edge_cpu[1].numpy()
    e = src.shape[0]
    keys = [np.arange(e)]
    if edge_attr is not None and torch.is_tensor(edge_attr) and edge_attr.numel() > 0:
        attr_np = edge_attr.detach().cpu().numpy()
        if attr_np.ndim == 1:
            keys.append(attr_np)
        else:
            for col in attr_np.T[::-1]:
                keys.append(col)
    keys.append(dst)
    keys.append(src)
    order = np.lexsort(tuple(keys))
    return torch.from_numpy(order).to(edge_index.device, dtype=torch.long)


def canonicalize_graph(data) -> Dict[str, object]:
    """Return a canonical representation (sorted) for hashing."""
    if hasattr(data, "keys"):
        fields = {k: data[k] for k in data.keys()}
    elif isinstance(data, dict):
        fields = dict(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    x = fields.get("x")
    edge_index = fields.get("edge_index")
    edge_attr = fields.get("edge_attr")

    if torch.is_tensor(x) and x.numel() > 0:
        perm = _node_permutation(x)
        fields["x"] = x[perm]
        batch = fields.get("batch")
        if torch.is_tensor(batch) and batch.numel() == perm.numel():
            fields["batch"] = batch[perm]
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
        if torch.is_tensor(edge_index) and edge_index.numel() > 0:
            remapped = perm_inv[edge_index]
            order = _edge_permutation(remapped, edge_attr if torch.is_tensor(edge_attr) else None)
            fields["edge_index"] = remapped[:, order]
            if torch.is_tensor(edge_attr) and edge_attr.size(0) == order.numel():
                fields["edge_attr"] = edge_attr[order]
    else:
        if torch.is_tensor(edge_index) and edge_index.numel() > 0:
            order = _edge_permutation(edge_index, edge_attr if torch.is_tensor(edge_attr) else None)
            fields["edge_index"] = edge_index[:, order]
            if torch.is_tensor(edge_attr) and edge_attr.size(0) == order.numel():
                fields["edge_attr"] = edge_attr[order]

    return fields


def _hash_tensor(t: torch.Tensor) -> str:
    buf = bytearray()
    buf.extend(str(t.dtype).encode())
    buf.extend(str(tuple(t.shape)).encode())
    buf.extend(t.detach().cpu().numpy().tobytes())
    return hashlib.sha256(buf).hexdigest()


def hash_canonical_graph(fields: Dict[str, object], *, ignore_metadata: bool = False) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    keys = ["x", "edge_index", "edge_attr", "edge_weight", "batch", "metadata"]
    if ignore_metadata:
        keys = [k for k in keys if k != "metadata"]
    for key in keys:
        val = fields.get(key)
        if torch.is_tensor(val):
            hashes[key] = _hash_tensor(val)
        elif isinstance(val, dict):
            hashes[key] = hashlib.sha256(json.dumps(val, sort_keys=True).encode()).hexdigest()
    return hashes


def walk_graphs(graph_dir: Path) -> Iterable[Path]:
    for path in sorted(graph_dir.rglob("*.pt")):
        if path.is_file():
            yield path


def validate(
    graph_dir: Path,
    manifest: Optional[Path],
    create_manifest: bool,
    sample: Optional[int],
    ignore_metadata: bool,
) -> int:
    graph_dir = graph_dir.resolve()
    if not graph_dir.is_dir():
        print(f"[validate_graphs] ERROR: graph_dir not found: {graph_dir}", file=sys.stderr)
        return 2

    manifest_data: Dict[str, Dict[str, str]] = {}
    if manifest and manifest.exists():
        try:
            manifest_data = json.loads(manifest.read_text())
        except Exception as exc:
            print(f"[validate_graphs] ERROR: failed to read manifest: {exc}", file=sys.stderr)
            return 2

    missing = []
    mismatched = []
    calculated: Dict[str, Dict[str, str]] = {}

    for idx, pt_path in enumerate(walk_graphs(graph_dir)):
        if sample and idx >= sample:
            break
        rel = str(pt_path.relative_to(graph_dir))
        try:
            data = torch.load(pt_path, map_location="cpu")
            canonical = canonicalize_graph(data)
            hashes = hash_canonical_graph(canonical, ignore_metadata=ignore_metadata)
            calculated[rel] = hashes
        except Exception as exc:
            mismatched.append((rel, f"load error: {exc}"))
            continue
        if manifest_data:
            expected = manifest_data.get(rel)
            if expected is not None and ignore_metadata and "metadata" in expected:
                # Drop metadata from expected when ignoring metadata in hashing.
                expected = {k: v for k, v in expected.items() if k != "metadata"}
            if expected is None:
                missing.append(rel)
            elif expected != hashes:
                mismatched.append((rel, "hash mismatch"))

    if manifest and create_manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(calculated, indent=2, sort_keys=True))
        print(f"[validate_graphs] Manifest written: {manifest}")
        # If we just created it, treat as success.
        manifest_data = calculated
        mismatched = []
        missing = []

    if mismatched or (manifest_data and missing):
        if mismatched:
            print(f"[validate_graphs] MISMATCHED: {len(mismatched)} (showing first 10)")
            for rel, reason in mismatched[:10]:
                print(f"  {rel}: {reason}")
        if manifest_data and missing:
            print(f"[validate_graphs] MISSING in manifest: {len(missing)} (showing first 10)")
            for rel in missing[:10]:
                print(f"  {rel}")
        return 1

    print(f"[validate_graphs] OK: validated {len(calculated)} graph(s) in {graph_dir}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", required=True, type=Path, help="Path to graph_dir containing .pt files")
    parser.add_argument("--manifest", type=Path, help="Path to manifest of canonical hashes for comparison")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write (or overwrite) manifest with current canonical hashes instead of comparing",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Validate only the first N graphs (for quick spot checks). Omit to validate all.",
    )
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Ignore the 'metadata' field when hashing (compare tensors only).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    return validate(
        args.graph_dir,
        args.manifest,
        args.write_manifest,
        args.sample,
        args.ignore_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
