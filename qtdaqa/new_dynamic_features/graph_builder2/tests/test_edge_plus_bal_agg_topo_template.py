from qtdaqa.new_dynamic_features.graph_builder2.modules.edge.edge_plus_bal_agg_topo import (
    EdgePlusBalAggTopoModule,
    DEFAULT_TOPO_DIM_HINT,
    HIST_DIM,
)


def test_config_template_has_dim_hint_and_comments() -> None:
    tmpl = EdgePlusBalAggTopoModule.config_template()
    assert tmpl["module"] == EdgePlusBalAggTopoModule.module_id
    notes = tmpl.get("notes", {})
    assert notes.get("expected_topology_dim") == DEFAULT_TOPO_DIM_HINT
    assert "feature_dim_formula" in notes
    comments = tmpl.get("param_comments", {})
    for key in ("distance_min", "distance_max", "scale_histogram", "variant", "jobs"):
        assert key in comments


def test_variant_defaults_in_template() -> None:
    tmpl = EdgePlusBalAggTopoModule.config_template()
    base_params = tmpl["params"]
    assert base_params["variant"] == "lean"
    for alt in tmpl.get("alternates", []):
        if alt.get("params", {}).get("variant") == "heavy":
            assert alt["params"].get("include_minmax") is True
