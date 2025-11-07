"""
Model saliency toolkit for dynamic TopoQA graphs.

This package now stores its implementation under :mod:`model_saliency.lib`.
The top-level module exposes key helpers directly and provides compatibility
aliases so legacy imports such as ``model_saliency.runner`` continue to work.
"""
from __future__ import annotations

import importlib
import sys
import types
from importlib import metadata as _metadata

from .lib import (  # re-export common entry points
    CheckpointConfig,
    GraphSelection,
    SaliencyRequest,
    default_output_dir,
    run_saliency,
)

__all__ = [
    "__version__",
    "CheckpointConfig",
    "GraphSelection",
    "SaliencyRequest",
    "default_output_dir",
    "run_saliency",
]

_SUBMODULE_ALIASES = {
    "captum_utils": ".lib.captum_utils",
    "config": ".lib.config",
    "data_loader": ".lib.data_loader",
    "metadata": ".lib.metadata",
    "model_loader": ".lib.model_loader",
    "pyg_explainer": ".lib.pyg_explainer",
    "runner": ".lib.runner",
}


class _LazyModule(types.ModuleType):
    def __init__(self, name: str, target: str):
        super().__init__(name)
        self.__dict__["_target"] = target
        self.__dict__["_loaded"] = None

    def _load(self):
        module = self.__dict__["_loaded"]
        if module is None:
            module = importlib.import_module(self.__dict__["_target"], __name__)
            sys.modules[self.__name__] = module
            self.__dict__["_loaded"] = module
        return module

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __dir__(self):
        return dir(self._load())


def _install_aliases() -> None:
    for alias, target in _SUBMODULE_ALIASES.items():
        name = f"{__name__}.{alias}"
        if name not in sys.modules:
            sys.modules[name] = _LazyModule(name, target)


_install_aliases()


def __getattr__(name: str):
    if name == "__version__":
        try:
            return _metadata.version("model_saliency")
        except _metadata.PackageNotFoundError:  # pragma: no cover - local dev
            return "0.0.dev0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
