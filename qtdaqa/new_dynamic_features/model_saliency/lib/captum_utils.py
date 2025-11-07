"""
Captum-based attribution utilities for TopoQA GNN modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Data

from .data_loader import ensure_batch
from .metadata import FeatureGroup


@dataclass
class AttributionResult:
    name: str
    node_attr: Optional[Tensor]
    edge_attr: Optional[Tensor]
    prediction: float
    target: Optional[float]
    feature_groups: Mapping[str, float]


def _forward_wrapper(model: LightningModule) -> Callable[[Data], Tensor]:
    def _forward(data: Data) -> Tensor:
        model.eval()
        data = ensure_batch(data)
        return model([data])

    return _forward


def _prepare_input(data: Data) -> Tuple[Tensor, Optional[Tensor]]:
    x = data.x.clone().detach().requires_grad_(True)
    edge_attr = None
    if data.edge_attr is not None:
        edge_attr = data.edge_attr.clone().detach().requires_grad_(True)
    return x, edge_attr


def _apply_noise_tunnel(
    ig: IntegratedGradients,
    samples: int,
    stdev: float = 0.01,
) -> IntegratedGradients:
    if samples <= 0:
        return ig
    return NoiseTunnel(ig, stdev=stdev, nt_type="smoothgrad_sq", nt_samples=samples)


def aggregate_feature_groups(attributions: Tensor, groups: Sequence[FeatureGroup]) -> Dict[str, float]:
    if attributions is None:
        return {}
    scores: Dict[str, float] = {}
    for group in groups:
        if not group.indices:
            continue
        group_score = float(attributions[:, group.indices].abs().sum().item())
        scores[group.name] = group_score
    return scores


def compute_node_attributions(
    model: LightningModule,
    data: Data,
    *,
    method: str = "integrated_gradients",
    integration_steps: int = 64,
    noise_tunnel_samples: int = 0,
    feature_groups: Sequence[FeatureGroup] = (),
) -> Tensor:
    forward_fn = _forward_wrapper(model)
    x, _ = _prepare_input(data)
    if method == "saliency":
        explainer = Saliency(forward_fn)
        attributions = explainer.attribute(x)
    else:
        ig = IntegratedGradients(forward_fn)
        ig = _apply_noise_tunnel(ig, noise_tunnel_samples)
        attributions = ig.attribute(x, n_steps=integration_steps)
    return attributions


def compute_edge_attributions(
    model: LightningModule,
    data: Data,
    *,
    integration_steps: int = 64,
) -> Optional[Tensor]:
    if data.edge_attr is None:
        return None
    data = ensure_batch(data)
    data = data.clone()
    data.x = data.x.detach().requires_grad_(True)
    data.edge_attr = data.edge_attr.detach().requires_grad_(True)
    output = model([data])
    if output.requires_grad:
        output.backward(torch.ones_like(output))
        return data.edge_attr.grad
    return None
