from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
import io

from qtdaqa.new_dynamic_features.graph_builder2 import graph_builder2 as graph_builder
from qtdaqa.new_dynamic_features.graph_builder2 import modules as module_registry


def test_feature_config_includes_lightweight_mol_modules(tmp_path: Path) -> None:
    module_registry._auto_import_modules()
    output_path = tmp_path / "example.feature-config.yaml"
    graph_builder.write_feature_config(output_path, include_alternates=True)
    text = output_path.read_text()
    assert "topology/lightweight_MoL/v1" in text
    assert "#   module: topology/lightweight_MoL/v1" in text
    assert "edge/edge_plus_lightweight_MoL/v1" in text
    assert "#   module: edge/edge_plus_lightweight_MoL/v1" in text


def test_module_listing_includes_lightweight_mol() -> None:
    module_registry._auto_import_modules()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        graph_builder._list_registered_modules(output_format="markdown")
    text = buffer.getvalue()
    assert "topology/lightweight_MoL/v1" in text
    assert "edge/edge_plus_lightweight_MoL/v1" in text
import os
import pytest

if os.environ.get("QTOPO_TEST_USE_REAL_DEPS") != "1":
    pytest.skip("Requires real deps for full module registry import", allow_module_level=True)
