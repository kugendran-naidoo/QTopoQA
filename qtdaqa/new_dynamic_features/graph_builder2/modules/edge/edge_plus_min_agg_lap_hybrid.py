from __future__ import annotations

from .edge_plus_min_agg_topo import EdgePlusMinAggTopoModule
from ..base import build_metadata
from ..registry import register_feature_module


@register_feature_module
class EdgePlusMinAggLapHybridModule(EdgePlusMinAggTopoModule):
    """
    Hybrid aggregation edge module for PH + Laplacian topology vectors.

    Semantics and behaviour mirror edge_plus_min_agg_topo/v1; this variant is
    documented for use alongside topology/persistence_laplacian_hybrid/v1
    (172D per-residue vectors). Aggregates per-edge as:
      legacy 11D histogram + concat(u_topo, v_topo, |u-v|) [+ norms/cosine/minmax per params]
    Deterministic ordering preserved (src_idx, dst_idx, distance).
    """

    module_id = "edge/edge_plus_min_agg_lap_hybrid/v1"
    module_kind = "edge"
    default_alias = "11D Legacy + hybrid topo (PH+Lap 172D)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Legacy 11D histogram + hybrid topo concat/abs-diff (PH+Lap 172D).",
        description=(
            "Same aggregation as edge_plus_min_agg_topo/v1, intended for use with "
            "topology/persistence_laplacian_hybrid/v1 (per-residue PH + Laplacian vectors, "
            "expected topo_dim≈172). Concatenates (u_topo, v_topo, |u-v|) and optional norms/cosine; "
            "heavy variant adds min/max. Prepends the legacy 11D distance histogram and preserves "
            "deterministic edge ordering (src_idx, dst_idx, distance)."
        ),
        inputs=("interface_residues", "pdb_structure", "topology_csv"),
        outputs=("edge_index", "edge_attr"),
        parameters=dict(EdgePlusMinAggTopoModule._metadata.parameters),
        defaults=dict(EdgePlusMinAggTopoModule._metadata.defaults),
        notes={"expected_topology_dim": 172},
    )

    @classmethod
    def config_template(cls) -> dict:
        template = super().config_template()
        param_comments = dict(template.get("param_comments", {}))
        param_comments.setdefault("note", "Use with topology/persistence_laplacian_hybrid/v1 (172D PH+Lap).")
        template["param_comments"] = param_comments
        template["alias"] = cls.default_alias
        return template
