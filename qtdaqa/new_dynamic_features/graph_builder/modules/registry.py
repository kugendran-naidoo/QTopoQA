from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type


@dataclass(frozen=True)
class FeatureModuleMetadata:
    module_id: str
    module_kind: str
    summary: str
    description: str
    inputs: Tuple[str, ...] = field(default_factory=tuple)
    outputs: Tuple[str, ...] = field(default_factory=tuple)
    parameters: Mapping[str, str] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)


class FeatureModule:
    """Base class for all feature modules."""

    module_id: str = ""
    module_kind: str = ""

    @classmethod
    def metadata(cls) -> FeatureModuleMetadata:
        raise NotImplementedError

    def __init__(self, **params: Any) -> None:
        self.params = {**self.metadata().defaults, **params}


FeatureModuleFactory = Type[FeatureModule]

_REGISTRY: Dict[str, FeatureModuleFactory] = {}


def register_feature_module(module_cls: FeatureModuleFactory) -> FeatureModuleFactory:
    module_id = module_cls.module_id
    if not module_id:
        raise ValueError(f"Feature module {module_cls} does not define module_id.")
    if module_id in _REGISTRY:
        raise ValueError(f"Feature module '{module_id}' already registered.")
    _REGISTRY[module_id] = module_cls
    return module_cls


def get_module_class(module_id: str) -> FeatureModuleFactory:
    try:
        return _REGISTRY[module_id]
    except KeyError as exc:
        raise KeyError(f"Feature module '{module_id}' is not registered.") from exc


def list_modules(kind: Optional[str] = None) -> Iterable[FeatureModuleMetadata]:
    for module_cls in _REGISTRY.values():
        meta = module_cls.metadata()
        if kind and meta.module_kind != kind:
            continue
        yield meta


def instantiate_module(module_id: str, **params: Any) -> FeatureModule:
    module_cls = get_module_class(module_id)
    return module_cls(**params)


def metadata_from_dict(data: Mapping[str, Any]) -> FeatureModuleMetadata:
    return FeatureModuleMetadata(
        module_id=data["module_id"],
        module_kind=data["module_kind"],
        summary=data.get("summary", ""),
        description=data.get("description", data.get("summary", "")),
        inputs=tuple(data.get("inputs", ())),
        outputs=tuple(data.get("outputs", ())),
        parameters=dict(data.get("parameters", {})),
        defaults=dict(data.get("defaults", {})),
        notes=dict(data.get("notes", {})),
    )
