#!/usr/bin/env python3
"""Compare PyG graph pickles listed in two manifests."""
from __future__ import annotations

import argparse
import math
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import numpy as np


def _read_manifest(path: Path) -> List[Path]:
    out: List[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            candidate = Path(raw)
            if candidate.suffix.lower() != ".pt":
                continue
            out.append(candidate)
    return out


def _target_from_path(path: Path) -> str:
    parent = path.parent
    if parent.name.lower() == "graph" and parent.parent is not None:
        return parent.parent.name
    return parent.name


def _group_by_filename(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in paths:
        grouped.setdefault(path.name, []).append(path)
    return grouped


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number))


def _compare_scalars(a: Any, b: Any) -> bool:
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False
    return math.isclose(af, bf, rel_tol=1e-6, abs_tol=1e-6)


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.is_sparse:
        return t.coalesce()
    return t


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, str]:
    if a.dtype != b.dtype:
        return False, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return False, f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    a_norm = _normalize_tensor(a)
    b_norm = _normalize_tensor(b)
    if a_norm.is_sparse or b_norm.is_sparse:
        if not (a_norm.is_sparse and b_norm.is_sparse):
            return False, "sparse tensor type mismatch"
        same_idx = torch.equal(a_norm.indices(), b_norm.indices())
        same_vals = torch.equal(a_norm.values(), b_norm.values())
        if not (same_idx and same_vals):
            return False, "sparse tensor values differ"
        return True, ""
    if not torch.equal(a_norm, b_norm):
        return False, "tensor values differ"
    return True, ""


def _iter_data_items(obj: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            yield str(key), obj[key]
        return
    keys = getattr(obj, "keys", None)
    if callable(keys):
        key_list = list(keys())  # type: ignore[arg-type]
    elif isinstance(keys, (list, tuple)):
        key_list = list(keys)
    else:
        key_list = []
    for key in key_list:
        yield str(key), getattr(obj, key)


def _summarize_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[len={len(value)}]"
    if isinstance(value, dict):
        return f"dict[len={len(value)}]"
    if isinstance(value, (int, float, str, bool)):
        return repr(value)
    return type(value).__name__


def _summarize_object(obj: Any) -> str:
    try:
        items = list(_iter_data_items(obj))
    except Exception:
        items = []
    if not items:
        return "no attributes"
    pieces = []
    for key, val in items:
        try:
            summary = _summarize_value(val)
        except Exception:
            summary = "<uninspectable>"
        pieces.append(f"{key}={summary}")
    return ", ".join(pieces)


def _compare_values(a: Any, b: Any, path: str = "root", details: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    details = details if details is not None else []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        ok, detail = _compare_tensors(a, b)
        if not ok:
            details.append(f"{path}: {detail}")
        return ok, details
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            details.append(f"{path}: ndarray shape mismatch {a.shape} vs {b.shape}")
            return False, details
        if not np.array_equal(a, b):
            details.append(f"{path}: ndarray values differ")
            return False, details
        return True, details
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            details.append(f"{path}: sequence length mismatch {len(a)} vs {len(b)}")
            return False, details
        for idx, (va, vb) in enumerate(zip(a, b)):
            ok, details = _compare_values(va, vb, f"{path}[{idx}]", details)
            if not ok:
                continue
        if type(a) != type(b):
            details.append(f"{path}: sequence type mismatch {type(a).__name__} vs {type(b).__name__}")
        return (False, details) if details else (True, details)
    if isinstance(a, dict) and isinstance(b, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            missing = a_keys - b_keys
            extra = b_keys - a_keys
            components: List[str] = []
            if missing:
                components.append(f"missing keys {sorted(missing)}")
            if extra:
                components.append(f"extra keys {sorted(extra)}")
            details.append(f"{path}: {'; '.join(components)}")
            return False, details
        for key in sorted(a_keys):
            ok, details = _compare_values(a[key], b[key], f"{path}.{key}", details)
        return (False, details) if details else (True, details)
    if hasattr(a, "keys") and hasattr(b, "keys"):
        a_items = {k: v for k, v in _iter_data_items(a)}
        b_items = {k: v for k, v in _iter_data_items(b)}
        return _compare_values(a_items, b_items, path, details)
    if (_is_number(a) and _is_number(b)) or isinstance(a, bool) and isinstance(b, bool):
        if _compare_scalars(a, b):
            return True, details
        details.append(f"{path}: scalar mismatch {a} vs {b}")
        return False, details
    if a == b:
        return True, details
    details.append(f"{path}: value mismatch {a!r} vs {b!r}")
    return False, details


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare graphs listed in manifests.")
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--original-list", type=Path, default=default_dir / "original_graphs.list")
    parser.add_argument("--new-list", type=Path, default=default_dir / "new_graphs.list")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=default_dir / "compare_graph",
        help="Base path for log files; timestamped suffixes and extensions are added automatically.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    original_paths = _read_manifest(args.original_list)
    new_paths = _read_manifest(args.new_list)

    new_by_name = _group_by_filename(new_paths)
    original_names = {p.name for p in original_paths}
    new_names = {p.name for p in new_paths}

    successes: List[str] = []
    failures: List[str] = []
    missing: List[str] = []
    new_only: List[str] = sorted(new_names - original_names)

    for orig_path in sorted(original_paths):
        target = _target_from_path(orig_path)
        filename = orig_path.name
        entry_name = f"{target}/{filename}" if target else filename

        candidates = new_by_name.get(filename, [])
        if not candidates:
            missing.append(f"[MISSING] {entry_name} -> no entry in new manifest")
            continue
        if len(candidates) > 1:
            matched = [p for p in candidates if _target_from_path(p).lower() == target.lower()]
            if len(matched) == 1:
                new_path = matched[0]
            elif len(matched) == 0:
                failures.append(f"[FAIL] {entry_name} -> multiple candidates in new manifest; none match target")
                continue
            else:
                failures.append(f"[FAIL] {entry_name} -> multiple candidates match target in new manifest")
                continue
        else:
            new_path = candidates[0]
        try:
            orig_obj = torch.load(orig_path, map_location="cpu")
        except Exception as exc:
            failures.append(f"[FAIL] {entry_name} -> unable to load original: {exc}")
            continue
        try:
            new_obj = torch.load(new_path, map_location="cpu")
        except Exception as exc:
            failures.append(f"[FAIL] {entry_name} -> unable to load new: {exc}")
            continue

        equal, detail = _compare_values(orig_obj, new_obj, path=entry_name)
        if equal:
            summary = _summarize_object(orig_obj)
            successes.append(f"[SUCCESS] {entry_name} :: {summary}")
        else:
            failures.append(f"[FAIL] {entry_name} -> {detail}")

    timestamp = time.strftime("%Y-%m-%d_%H:%M")
    base_path = args.log_file
    if base_path.is_dir() or not base_path.suffix:
        base_dir = base_path if base_path.is_dir() else base_path.parent
        base_name = base_path.stem if base_path.stem else "compare_graph"
    else:
        base_dir = base_path.parent
        base_name = base_path.stem
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history_dir = log_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    for entry in list(log_dir.iterdir()):
        if entry == history_dir:
            continue
        target = history_dir / entry.name
        if entry.is_file():
            if target.exists():
                target.unlink()
            entry.replace(target)
        elif entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(entry), str(target))
    base_name = base_name or "compare_graph"
    timestamped_base = f"{base_name}.{timestamp}"
    success_path = log_dir / f"{timestamped_base}.successes.log"
    failure_path = log_dir / f"{timestamped_base}.failures.log"
    general_path = log_dir / f"{timestamped_base}.general.log"

    for path, lines in (
        (success_path, successes),
        (failure_path, failures),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")

    general_lines: List[str] = [
        f"[SUMMARY] compared {len(original_paths)} graphs from original_graphs.list",
        f"[SUMMARY] success={len(successes)} failure={len(failures)} missing={len(missing)}",
        f"[SUMMARY] extra files in new_graphs.list not in original_graphs.list = {len(new_only)}",
        f"[INFO] successes log: {success_path}",
        f"[INFO] failures log: {failure_path}",
    ]
    if missing:
        general_lines.append(f"[INFO] listing missing entries ({len(missing)}):")
        general_lines.extend(missing)
    if new_only:
        general_lines.append(f"[INFO] files only in new manifest ({len(new_only)}):")
        for name in new_only:
            candidates = new_by_name.get(name, [])
            for candidate in candidates:
                general_lines.append(f"  {candidate}")

    with general_path.open("w", encoding="utf-8") as handle:
        for line in general_lines:
            handle.write(f"{line}\n")

    rel_success = success_path.relative_to(base_dir)
    rel_failure = failure_path.relative_to(base_dir)
    rel_general = general_path.relative_to(base_dir)

    print(f"Compared {len(original_paths)} graphs. Success={len(successes)} Fail={len(failures)} Missing={len(missing)}")
    print(f"Extra files in new_graphs.list not in original_graphs.list = {len(new_only)}")
    print(
        "Logs written to:\n"
        f"  successes -> {rel_success}\n"
        f"  failures  -> {rel_failure}\n"
        f"  general   -> {rel_general}"
    )
    return 0 if not failures and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
