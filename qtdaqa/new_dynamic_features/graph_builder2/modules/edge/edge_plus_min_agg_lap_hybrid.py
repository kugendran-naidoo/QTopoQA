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
    default_alias = "Legacy 11D Edge + 519D {(hist + agg + norms + cosine) from 172D laplacian_hybrid} = Edge 530D (Lean) | Legacy 11D Edge + 863D {(hist + agg + norms + cosine + minmax) from laplacian_hybrid} = Edge 874D (Heavy)"
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
            "Legacy 11D Edge + 519D {(hist + agg + norms + cosine) from 172D laplacian_hybrid} = Edge 530D (Lean) | "
            "Legacy 11D Edge + 863D {(hist + agg + norms + cosine + minmax) from laplacian_hybrid} = Edge 874D (Heavy)"
        )
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "notes": {
                "expected_topology_dim": 172,
                "feature_dim_formula": {
                    "lean": "11 + (3*topo_dim) + norms(2) + cosine(1) => 530 when topo_dim=172",
                    "heavy": "lean + 2*topo_dim (min/max) => 874 when topo_dim=172",
                },
            },
            "alternates": [
                {
                    "module": cls.module_id,
                    "alias": heavy_alias,
                    "params": heavy_params,
                    "param_comments": param_comments,
                    "summary": cls._metadata.summary,
                    "description": cls._metadata.description,
                    "notes": {
                        "expected_topology_dim": 172,
                        "feature_dim_formula": {
                            "lean": "11 + (3*topo_dim) + norms(2) + cosine(1) => 530 when topo_dim=172",
                            "heavy": "lean + 2*topo_dim (min/max) => 874 when topo_dim=172",
                        },
                    },
                }
            ],
        }

    def build_edges(self, *args, **kwargs):
        result = super().build_edges(*args, **kwargs)
        # Adjust variant label to lap_hybrid namespace for clarity
        variant = self.params.get("variant", "lean")
        result.metadata["edge_feature_variant"] = f"edge_plus_min_agg_lap_hybrid/{variant}"
        return result
