from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Dict

from qtdaqa.new_dynamic_features.graph_builder2 import graph_builder2 as graph_builder
from qtdaqa.new_dynamic_features.graph_builder2.modules.registry import FeatureModuleMetadata


class _DummyModule:
    default_alias = "DummyAlias"

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return {"alpha": "Alpha param", "beta": "Beta param"}


class _LegacyModule:
    default_alias = None


def _metadata(module_id: str, defaults: Dict[str, object]) -> FeatureModuleMetadata:
    return FeatureModuleMetadata(
        module_id=module_id,
        module_kind="edge",
        summary="Summary text",
        description="Description text",
        defaults=defaults,
        parameters={"alpha": "legacy alpha desc"},
    )


def test_format_module_listing_prefers_module_list_params() -> None:
    meta = _metadata("edge/dummy/v1", {"alpha": 1.0, "beta": 2.0})
    lines = graph_builder._format_module_listing(meta, _DummyModule)
    assert "edge/dummy/v1 (alias: DummyAlias)" in lines[0]
    assert "alpha" in "\n".join(lines)
    assert "Alpha param" in "\n".join(lines)  # pulled from module list_params override
    assert "default=1.0" in "\n".join(lines)


def test_format_module_listing_falls_back_to_metadata_parameters() -> None:
    meta = _metadata("edge/legacy/v0", {"alpha": 9})
    lines = graph_builder._format_module_listing(meta, _LegacyModule)
    body = "\n".join(lines)
    assert "edge/legacy/v0" in body
    assert "legacy alpha desc" in body  # fallback came from metadata.parameters


def test_list_modules_markdown_output() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        graph_builder._list_registered_modules(output_format="markdown")
    text = buffer.getvalue()
    assert "# Registered Feature Modules" in text
    assert "### interface modules" in text
    assert "- `interface/polar_cutoff/v1`" in text
