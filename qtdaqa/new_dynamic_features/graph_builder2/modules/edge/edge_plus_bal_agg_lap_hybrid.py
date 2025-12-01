from __future__ import annotations

from .edge_plus_bal_agg_topo import EdgePlusBalAggTopoModule
from ..base import build_metadata
from ..registry import register_feature_module


@register_feature_module
class EdgePlusBalAggLapHybridModule(EdgePlusBalAggTopoModule):
    """
    Hybrid aggregation edge module for PH + Laplacian topology vectors.

    Behaviour matches edge_plus_bal_agg_topo/v1; this variant is documented
    for pairing with topology/persistence_laplacian_hybrid/v1 (172D PH+Lap
    per-residue vectors). Aggregation: legacy 11D histogram + balanced topo
    concat(u, v, mean, |u-v|) with optional norms/cosine; heavy variant adds
    min/max. Deterministic edge ordering (src_idx, dst_idx, distance) preserved.
    """

    module_id = "edge/edge_plus_bal_agg_lap_hybrid/v1"
    module_kind = "edge"
    default_alias = "Legacy 11D Edge + 691D {(hist + agg + norms + cosine) from 172D laplacian_hybrid} = Edge 702D (Lean) | Legacy 11D Edge + 1035D {(hist + agg + norms + cosine + minmax) from laplacian_hybrid} = Edge 1046D (Heavy)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Legacy 11D histogram + hybrid balanced topo agg (PH+Lap 172D).",
        description=(
            "Same aggregation as edge_plus_bal_agg_topo/v1, intended for use with "
            "topology/persistence_laplacian_hybrid/v1 (per-residue PH + Laplacian vectors, "
            "expected topo_dim≈172). Prepends the legacy 11D distance histogram, then "
            "concatenates (u_topo, v_topo, mean(u,v), |u-v|) with optional norms/cosine; "
            "heavy variant adds per-dimension min/max. Preserves deterministic edge ordering "
            "(src_idx, dst_idx, distance)."
        ),
        inputs=("interface_residues", "pdb_structure", "topology_csv"),
        outputs=("edge_index", "edge_attr"),
        parameters=dict(EdgePlusBalAggTopoModule._metadata.parameters),
        defaults=dict(EdgePlusBalAggTopoModule._metadata.defaults),
        notes={"expected_topology_dim": 172},
    )

    @classmethod
    def config_template(cls) -> dict:
        params = dict(cls._metadata.defaults)
        param_comments = {
            "note": "Use with topology/persistence_laplacian_hybrid/v1 (172D PH+Lap).",
            "variant": "lean or heavy",
            "include_minmax": "heavy variant only; adds per-dimension min/max blocks",
        }
        heavy_params = dict(params)
        heavy_params.update({"variant": "heavy", "include_minmax": True})
        heavy_alias = (
            "Legacy 11D Edge + 691D {(hist + agg + norms + cosine) from 172D laplacian_hybrid} = Edge 702D (Lean) | "
            "Legacy 11D Edge + 1035D {(hist + agg + norms + cosine + minmax) from laplacian_hybrid} = Edge 1046D (Heavy)"
        )
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "alternates": [
            {
                "module": cls.module_id,
                "alias": heavy_alias,
                "params": heavy_params,
                "param_comments": param_comments,
                "summary": cls._metadata.summary,
                "description": cls._metadata.description,
            }
        ]
        }
