from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import FeatureModule, FeatureModuleMetadata, metadata_from_dict


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
