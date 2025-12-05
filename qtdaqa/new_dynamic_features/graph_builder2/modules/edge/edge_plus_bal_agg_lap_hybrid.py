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
            "distance_min": "Cα distance window start; edges with distance <= min are skipped",
            "distance_max": "Cα distance window end; edges with distance >= max are skipped",
            "scale_histogram": "Scale only the legacy 11D (distance + 10-bin) block across each graph",
            "include_norms": "Add L2 norms of endpoint topology vectors (PH+Lap 172D)",
            "include_cosine": "Add cosine similarity between endpoint topology vectors",
            "variant": "lean or heavy; heavy adds per-dimension min/max blocks",
            "include_minmax": "heavy variant only; adds per-dimension min/max blocks",
            "jobs": "Honors CLI --jobs > config default_jobs > module jobs; deterministic ordering; dumps resorted by edge_runner",
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
            "dim_hint": "# dim (lean): 11 + 4*topo_dim + norms + cosine; heavy adds +2*topo_dim (min/max). E.g., topo_dim=172 -> lean 702, heavy 1046.",
            "notes": {
                "expected_topology_dim": 172,
                "feature_dim_formula": {
                    "lean": "11 + (4*topo_dim) + norms(2) + cosine(1) => 702 when topo_dim=172",
                    "heavy": "lean + 2*topo_dim (min/max) => 1046 when topo_dim=172",
                },
            },
            "alternates": [
            {
                "module": cls.module_id,
                "alias": heavy_alias,
                "params": heavy_params,
                "param_comments": param_comments,
                "dim_hint": "# dim (lean): 11 + 4*topo_dim + norms + cosine; heavy adds +2*topo_dim (min/max). E.g., topo_dim=172 -> lean 702, heavy 1046.",
                "summary": cls._metadata.summary,
                "description": cls._metadata.description,
                "notes": {
                    "expected_topology_dim": 172,
                    "feature_dim_formula": {
                        "lean": "11 + (4*topo_dim) + norms(2) + cosine(1) => 702 when topo_dim=172",
                        "heavy": "lean + 2*topo_dim (min/max) => 1046 when topo_dim=172",
                    },
                },
            }
        ]
        }

    def build_edges(self, *args, **kwargs):
        result = super().build_edges(*args, **kwargs)
        variant = self.params.get("variant", "lean")
        result.metadata["edge_feature_variant"] = f"edge_plus_bal_agg_lap_hybrid/{variant}"
        return result
