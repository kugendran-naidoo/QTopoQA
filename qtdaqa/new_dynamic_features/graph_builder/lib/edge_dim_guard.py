from __future__ import annotations

from typing import Any, Optional

import numpy as np


def determine_edge_feature_dim(edge_attr: Any) -> int:
    """Return the feature dimension (number of columns) for an edge_attr tensor."""
    if edge_attr is None:
        return 0

    if hasattr(edge_attr, "shape"):
        shape = edge_attr.shape
        if len(shape) == 0:
            return 0
        if len(shape) == 1:
            return 1 if shape[0] > 0 else 0
        return int(shape[1])

    arr = np.asarray(edge_attr)
    if arr.ndim == 0:
        return 0
    if arr.ndim == 1:
        return arr.shape[0]
    return int(arr.shape[1])


def ensure_edge_feature_dim(
    module_id: str,
    edge_attr: Any,
    metadata_dim: Optional[int],
) -> int:
    """
    Guarantee that edge_attr has a positive, consistent feature dimension.

    Returns the validated dimension (also used to backfill metadata) or raises
    ValueError when the tensor is malformed or inconsistent with metadata.
    """
    tensor_dim = determine_edge_feature_dim(edge_attr)
    if tensor_dim <= 0:
        raise ValueError(
            f"{module_id} produced edge_attr with zero feature columns; "
            "this would break downstream batching."
        )

    if metadata_dim is None or int(metadata_dim) <= 0:
        return tensor_dim

    declared = int(metadata_dim)
    if declared != tensor_dim:
        raise ValueError(
            f"{module_id} reported feature_dim={declared} but tensor columns={tensor_dim}."
        )
    return tensor_dim
