from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping


def trim_suffix(stem: str, suffixes: Iterable[str]) -> str:
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem = stem.rstrip("_- .")
            lower = stem.lower()
    return stem


def normalise_interface_name(name: str) -> str:
    return trim_suffix(Path(name).stem, (".interface", "interface", "iface"))


def normalise_topology_name(name: str) -> str:
    return trim_suffix(Path(name).stem, (".topology", "topology"))


def normalise_node_name(name: str) -> str:
    return trim_suffix(Path(name).stem, (".node_fea", "node_fea", "node"))


def structure_model_key(dataset_dir: Path, structure_path: Path) -> str:
    try:
        relative = structure_path.relative_to(dataset_dir)
    except ValueError:
        return structure_path.stem
    parent_parts = [part for part in relative.parent.parts if part not in ("", ".")]
    if parent_parts:
        return str(PurePosixPath(*parent_parts, structure_path.stem))
    return structure_path.stem


def gather_files(root: Path, patterns: Iterable[str], normalise) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                normalised = normalise(path.name)
                key = relative_key(root, path, normalised)
                mapping.setdefault(key, path)
    return mapping


def relative_key(root: Path, path: Path, name: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return name
    parent_parts = [part for part in relative.parent.parts if part not in ("", ".")]
    if parent_parts:
        return str(PurePosixPath(*parent_parts, name))
    return name


def index_structures(dataset_dir: Path, suffixes: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for ext in suffixes:
        for path in dataset_dir.rglob(f"*{ext}"):
            if path.is_file():
                key = structure_model_key(dataset_dir, path)
                mapping.setdefault(key, path)
    return mapping
