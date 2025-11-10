from qtdaqa.new_dynamic_features.graph_builder2.modules.base import (  # noqa: E402
    ConfigurableFeatureModule,
    build_metadata,
)


class _DummyModule(ConfigurableFeatureModule):
    module_id = "edge/dummy/v0"
    module_kind = "edge"
    default_alias = "Dummy"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="dummy summary",
        description="dummy description",
        inputs=("foo",),
        outputs=("bar",),
        parameters={"alpha": "Alpha value", "beta": "Beta value"},
        defaults={"alpha": 1.0, "beta": 2.0},
    )


def test_config_template_uses_metadata_defaults() -> None:
    template = _DummyModule.config_template()
    assert template["module"] == _DummyModule.module_id
    assert template["alias"] == "Dummy"
    assert template["params"] == {"alpha": 1.0, "beta": 2.0}

    # Ensure copies are returned so callers cannot mutate class defaults.
    template["params"]["alpha"] = 999
    assert _DummyModule.metadata().defaults["alpha"] == 1.0


def test_list_params_reflects_metadata_parameters() -> None:
    params = _DummyModule.list_params()
    assert params["alpha"] == "Alpha value"
    assert params["beta"] == "Beta value"
