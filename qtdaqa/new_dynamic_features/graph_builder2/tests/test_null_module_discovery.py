from pathlib import Path

from qtdaqa.new_dynamic_features.graph_builder2.graph_builder2 import write_feature_config
from qtdaqa.new_dynamic_features.graph_builder2.modules import list_modules


def test_null_module_is_registered() -> None:
    # Some test environments set QTOPO_SKIP_MODULE_REGISTRY=1, so modules may not be eagerly imported.
    # Importing the module explicitly ensures it registers itself for discovery/listing.
    import qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_null_v1  # noqa: F401
    module_ids = {meta.module_id for meta in list_modules(kind="topology")}
    assert "topology/persistence_null/v1" in module_ids


def test_create_feature_config_includes_null_alternate(tmp_path: Path) -> None:
    import qtdaqa.new_dynamic_features.graph_builder2.modules.topology.persistence_null_v1  # noqa: F401
    out = tmp_path / "example.feature-config.yaml"
    write_feature_config(out, include_alternates=True)
    text = out.read_text(encoding="utf-8")
    assert "topology/persistence_null/v1" in text
