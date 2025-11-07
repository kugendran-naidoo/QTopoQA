"""Model subpackage for dynamic TopoQA training."""

from .gat_5_edge1 import GNN_edge1_edgepooling  # noqa: F401
from .gat_with_edge import GATv2ConvWithEdgeEmbedding1  # noqa: F401

__all__ = ["GNN_edge1_edgepooling", "GATv2ConvWithEdgeEmbedding1"]
