"""
High-level saliency runner coordinating metadata, attributions, and exports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from .captum_utils import (
    AttributionResult,
    aggregate_feature_groups,
    compute_edge_attributions,
    compute_node_attributions,
)
from .config import SaliencyRequest
from .data_loader import GraphSample, ensure_batch, load_graph_samples
from .metadata import derive_feature_groups, export_metadata, load_metadata
from .model_loader import model_context
from .pyg_explainer import run_gnn_explainer


def _export_attributions(results: List[AttributionResult], output_dir: Path) -> None:
    records: List[Dict[str, object]] = []
    for result in results:
        node_attr = result.node_attr.detach().cpu().numpy().tolist() if result.node_attr is not None else None
        edge_attr = result.edge_attr.detach().cpu().numpy().tolist() if result.edge_attr is not None else None
        records.append(
            {
                "graph": result.name,
                "prediction": result.prediction,
                "target": result.target,
                "node_attributions": node_attr,
                "edge_attributions": edge_attr,
                "feature_groups": result.feature_groups,
            }
        )
    (output_dir / "attributions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_saliency(request: SaliencyRequest) -> None:
    request.ensure_output_dir()
    request.graphs.validate()
    metadata = load_metadata(
        request.graphs.graph_dir,
        sample_models=request.graphs.models,
        metadata_path=request.graph_metadata_path,
        summary_path=request.graph_summary_path,
    )
    context = derive_feature_groups(metadata)
    export_metadata(context, request.output_dir / "metadata.json")

    graph_samples: List[GraphSample] = load_graph_samples(
        request.graphs.graph_dir,
        list(request.graphs.iter_models()),
    )
    if not graph_samples:
        raise RuntimeError("No graph samples resolved for saliency analysis.")
    results: List[AttributionResult] = []

    with model_context(request.checkpoint) as model:
        for sample in graph_samples:
            data = ensure_batch(sample.data)
            prediction = float(model([data]).detach().cpu().item())
            target = float(data.y.cpu().item()) if getattr(data, "y", None) is not None else None

            node_attr = compute_node_attributions(
                model,
                data,
                method=request.gradient_method,
                integration_steps=request.integration_steps,
                noise_tunnel_samples=request.noise_tunnel_samples,
                feature_groups=context.node_groups,
            )
            edge_attr = compute_edge_attributions(
                model,
                data,
                integration_steps=request.integration_steps,
            )
            feature_group_scores = aggregate_feature_groups(node_attr, context.node_groups)

            if request.pyg_explainer:
                explanation = run_gnn_explainer(model, data)
                if explanation is not None:
                    torch.save(explanation, request.output_dir / f"{sample.name}_explainer.pt")

            results.append(
                AttributionResult(
                    name=sample.name,
                    node_attr=node_attr,
                    edge_attr=edge_attr,
                    prediction=prediction,
                    target=target,
                    feature_groups=feature_group_scores,
                )
            )

    _export_attributions(results, request.output_dir)
