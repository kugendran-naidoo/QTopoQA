"""Feature module registry for the dynamic graph builder.

Importing this package normally registers all built-in modules. Tests can skip
the eager imports by setting ``QTOPO_SKIP_MODULE_REGISTRY=1`` in the
environment, which allows importing lightweight helpers (e.g., base classes)
without pulling heavy scientific dependencies.
"""

import importlib
import os
import pkgutil
from typing import Iterable

from .registry import (
    FeatureModule,
    FeatureModuleFactory,
    FeatureModuleMetadata,
    register_feature_module,
    get_module_class,
    list_modules,
    instantiate_module,
)

_DISCOVERY_EXCLUDES = {"base", "registry"}


def _iter_module_names() -> Iterable[str]:
    """Yield fully-qualified module names for all stage packages."""
    prefix = __name__ + "."
    for module_info in pkgutil.walk_packages(__path__, prefix=prefix):
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name in _DISCOVERY_EXCLUDES:
            continue
        yield module_info.name


def _auto_import_modules() -> None:
    for module_name in _iter_module_names():
        importlib.import_module(module_name)


if os.environ.get("QTOPO_SKIP_MODULE_REGISTRY") != "1":
    _auto_import_modules()

__all__ = [
    "FeatureModule",
    "FeatureModuleFactory",
    "FeatureModuleMetadata",
    "register_feature_module",
    "get_module_class",
    "list_modules",
    "instantiate_module",
]
