from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


@dataclass
class EdgeBand:
    label: str
    min_distance: float
    max_distance: float


@dataclass
class EdgeConfig:
    bands: List[EdgeBand] = field(
        default_factory=lambda: [
            EdgeBand(label="short", min_distance=0.0, max_distance=6.0),
            EdgeBand(label="medium", min_distance=6.0, max_distance=10.0),
            EdgeBand(label="long", min_distance=10.0, max_distance=14.0),
        ]
    )
    histogram_bins: Sequence[float] = tuple(float(x) for x in range(0, 22, 2))
    contact_threshold: float = 5.0
    include_inverse_distance: bool = True
    include_unit_vector: bool = True
    unit_vector_epsilon: float = 1e-8
    normalize: str = "none"  # none|minmax|zscore


@dataclass
class InterfaceConfig:
    cutoff: float = 10.0
    coordinate_decimals: int = -1


@dataclass
class TopologyConfigData:
    neighbor_distance: float = 8.0
    filtration_cutoff: float = 8.0
    min_persistence: float = 0.01
    dedup_sort: bool = False


@dataclass
class BuilderConfig:
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    topology: TopologyConfigData = field(default_factory=TopologyConfigData)
    node_drop_na: bool = False
    edge: EdgeConfig = field(default_factory=EdgeConfig)


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(default)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataclass_from_dict(cls, data: Dict[str, Any]):
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: Optional[Path]) -> BuilderConfig:
    default_dict: Dict[str, Any] = {
        "interface": {
            "cutoff": 10.0,
            "coordinate_decimals": -1,
        },
        "topology": {
            "neighbor_distance": 8.0,
            "filtration_cutoff": 8.0,
            "min_persistence": 0.01,
            "dedup_sort": False,
        },
        "node_drop_na": False,
        "edge": {
            "bands": [
                {"label": "short", "min_distance": 0.0, "max_distance": 6.0},
                {"label": "medium", "min_distance": 6.0, "max_distance": 10.0},
                {"label": "long", "min_distance": 10.0, "max_distance": 14.0},
            ],
            "histogram_bins": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "contact_threshold": 5.0,
            "include_inverse_distance": True,
            "include_unit_vector": True,
            "unit_vector_epsilon": 1e-8,
            "normalize": "none",
        },
    }

    user_dict: Dict[str, Any] = {}
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config at {config_path} must be a mapping.")
            user_dict = loaded

    merged = _merge_dict(default_dict, user_dict)
    edge_bands = [_dataclass_from_dict(EdgeBand, band) for band in merged["edge"]["bands"]]
    edge_cfg = EdgeConfig(
        bands=edge_bands,
        histogram_bins=tuple(float(x) for x in merged["edge"].get("histogram_bins", [])),
        contact_threshold=float(merged["edge"].get("contact_threshold", 5.0)),
        include_inverse_distance=bool(merged["edge"].get("include_inverse_distance", True)),
        include_unit_vector=bool(merged["edge"].get("include_unit_vector", True)),
        unit_vector_epsilon=float(merged["edge"].get("unit_vector_epsilon", 1e-8)),
        normalize=str(merged["edge"].get("normalize", "none")).lower(),
    )

    cfg = BuilderConfig(
        interface=_dataclass_from_dict(InterfaceConfig, merged["interface"]),
        topology=_dataclass_from_dict(TopologyConfigData, merged["topology"]),
        node_drop_na=bool(merged.get("node_drop_na", False)),
        edge=edge_cfg,
    )
    return cfg


def config_as_dict(config: BuilderConfig) -> Dict[str, Any]:
    def _band(b: EdgeBand) -> Dict[str, Any]:
        return {
            "label": b.label,
            "min_distance": b.min_distance,
            "max_distance": b.max_distance,
        }

    return {
        "interface": {
            "cutoff": config.interface.cutoff,
            "coordinate_decimals": config.interface.coordinate_decimals,
        },
        "topology": {
            "neighbor_distance": config.topology.neighbor_distance,
            "filtration_cutoff": config.topology.filtration_cutoff,
            "min_persistence": config.topology.min_persistence,
            "dedup_sort": config.topology.dedup_sort,
        },
        "node_drop_na": config.node_drop_na,
        "edge": {
            "bands": [_band(band) for band in config.edge.bands],
            "histogram_bins": list(config.edge.histogram_bins),
            "contact_threshold": config.edge.contact_threshold,
            "include_inverse_distance": config.edge.include_inverse_distance,
            "include_unit_vector": config.edge.include_unit_vector,
            "unit_vector_epsilon": config.edge.unit_vector_epsilon,
            "normalize": config.edge.normalize,
        },
    }
