from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .registry import FeatureModule, FeatureModuleMetadata, metadata_from_dict


def _require_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def require_positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    numeric = _require_float(value, label)
    if allow_zero:
        if numeric < 0.0:
            raise ValueError(f"{label} must be >= 0.")
    else:
        if numeric <= 0.0:
            raise ValueError(f"{label} must be > 0.")
    return numeric


def require_float(value: Any, label: str) -> float:
    return _require_float(value, label)


def require_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def require_positive_int(value: Any, label: str) -> int:
    integer = require_int(value, label)
    if integer <= 0:
        raise ValueError(f"{label} must be > 0.")
    return integer


def require_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"{label} must be a boolean.")


def ensure_sorted_float_sequence(
    values: Sequence[Any],
    label: str,
    *,
    min_length: int = 2,
    allow_equal: bool = False,
) -> Sequence[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a sequence of numbers.")
    if len(values) < min_length:
        raise ValueError(f"{label} must contain at least {min_length} entries.")
    floats = [require_float(entry, label) for entry in values]
    comparator = (lambda prev, curr: curr >= prev) if allow_equal else (lambda prev, curr: curr > prev)
    for idx in range(1, len(floats)):
        if not comparator(floats[idx - 1], floats[idx]):
            raise ValueError(f"{label} must be sorted ascending.")
    return floats


class ConfigurableFeatureModule(FeatureModule):
    """Convenience base class that merges provided params with defaults."""

    _metadata: FeatureModuleMetadata

    @classmethod
    def metadata(cls) -> FeatureModuleMetadata:
        return cls._metadata

    def __init__(self, **params: Any) -> None:
        defaults = self.metadata().defaults
        merged = dict(defaults)
        merged.update(params)
        super().__init__(**merged)  # type: ignore[arg-type]
        self.params = merged

    @classmethod
    def describe(cls) -> Mapping[str, Any]:
        meta = cls.metadata()
        return {
            "id": meta.module_id,
            "kind": meta.module_kind,
            "summary": meta.summary,
            "description": meta.description,
            "inputs": list(meta.inputs),
            "outputs": list(meta.outputs),
            "parameters": dict(meta.parameters),
            "defaults": dict(meta.defaults),
            "notes": dict(meta.notes),
        }


def build_metadata(**data: Any) -> FeatureModuleMetadata:
    return metadata_from_dict(data)


class InterfaceFeatureModule(ConfigurableFeatureModule):
    def extract_interfaces(self, *args, **kwargs):
        raise NotImplementedError


class TopologyFeatureModule(ConfigurableFeatureModule):
    def generate_topology(self, *args, **kwargs):
        raise NotImplementedError


class NodeFeatureModule(ConfigurableFeatureModule):
    def generate_nodes(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class EdgeBuildResult:
    edge_index: Any
    edge_attr: Any
    metadata: Mapping[str, Any]


class EdgeFeatureModule(ConfigurableFeatureModule):
    def build_graphs(self, *args, **kwargs):
        raise NotImplementedError

    def build_edges(self, *args, **kwargs) -> EdgeBuildResult:
        raise NotImplementedError
