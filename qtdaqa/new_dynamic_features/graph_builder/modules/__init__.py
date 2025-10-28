"""
Feature module registry for the dynamic graph builder.

Importing this package registers the built-in interface, topology, node, and
edge feature implementations. New feature modules can register themselves by
calling ``register_feature_module`` during import.
"""

from .registry import (
    FeatureModule,
    FeatureModuleFactory,
    FeatureModuleMetadata,
    register_feature_module,
    get_module_class,
    list_modules,
    instantiate_module,
)

# Import built-in modules so they register on package import.
from . import interface_default  # noqa: F401
from . import topology_persistence_v1  # noqa: F401
from . import node_dssp_topo_v1  # noqa: F401
from . import edge_multiscale_v24  # noqa: F401
from . import edge_legacy_v11  # noqa: F401

__all__ = [
    "FeatureModule",
    "FeatureModuleFactory",
    "FeatureModuleMetadata",
    "register_feature_module",
    "get_module_class",
    "list_modules",
    "instantiate_module",
]
