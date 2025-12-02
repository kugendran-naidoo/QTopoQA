from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class LaplacianMomentConfig:
    size_threshold: int = 80
    estimator: str = "exact"  # exact|slq
    slq_probes: int = 8
    slq_steps: int = 32  # kept for interface compatibility; not used by simple power estimator
    seed: int | None = None


def _normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Return symmetric normalized Laplacian for an unweighted adjacency matrix."""
    if adj.size == 0:
        return adj
    degrees = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D = np.diag(d_inv_sqrt)
    L = np.eye(adj.shape[0], dtype=float) - D @ adj @ D
    return L


def _power_trace(L: np.ndarray, order: int) -> float:
    """Exact trace of L^order (small matrices)."""
    if order == 1:
        return float(np.trace(L))
    accum = L.copy()
    for _ in range(1, order):
        accum = accum @ L
    return float(np.trace(accum))


def _power_trace_slq(L: np.ndarray, order: int, probes: int, rng: np.random.Generator) -> float:
    """Simple Hutchinson-style estimator for trace(L^order)."""
    if probes <= 0 or order <= 0:
        return 0.0
    n = L.shape[0]
    total = 0.0
    for _ in range(probes):
        z = rng.choice([-1.0, 1.0], size=(n, 1))
        v = z
        for _ in range(order):
            v = L @ v
        total += float(np.sum(z * v))
    return total / float(probes)


def _moments_from_eigvals(eigvals: np.ndarray, moment_orders: Sequence[int]) -> tuple[list[float], list[float]]:
    vals = np.asarray(eigvals, dtype=float)
    if vals.size == 0:
        return [0.0 for _ in moment_orders], [0.0 for _ in moment_orders if _ >= 2]
    raw = []
    for order in moment_orders:
        raw.append(float(np.mean(np.power(vals, order))))
    mean = raw[0] if raw else 0.0
    centered = []
    for order in moment_orders:
        if order < 2:
            continue
        centered.append(float(np.mean(np.power(vals - mean, order))))
    return raw, centered


def compute_laplacian_moments(
    adj: np.ndarray,
    *,
    moment_orders: Sequence[int],
    config: LaplacianMomentConfig,
) -> tuple[list[float], list[float]]:
    """Compute raw and centered Laplacian moments."""
    n = adj.shape[0]
    if n == 0:
        zeros = [0.0 for _ in moment_orders]
        return zeros, [0.0 for _ in moment_orders if _ >= 2]

    L = _normalized_laplacian(adj)
    rng = np.random.default_rng(config.seed)

    # Choose exact or SLQ based on size and requested estimator.
    use_exact = (config.estimator == "exact") and (n <= max(1, int(config.size_threshold)))
    if use_exact:
        eigvals = np.linalg.eigvalsh(L)
    else:
        # Estimate traces of powers, then reconstruct raw/centered moments.
        raw_map = {}
        for order in moment_orders:
            if order <= 0:
                raw_map[order] = 0.0
                continue
            if config.estimator == "slq":
                trace_est = _power_trace_slq(L, order, config.slq_probes, rng)
            else:
                trace_est = _power_trace(L, order)
            raw_map[order] = trace_est / float(n)
        raw = [raw_map.get(order, 0.0) for order in moment_orders]
        mu1 = raw_map.get(1, 0.0)
        mu2 = raw_map.get(2, 0.0)
        mu3 = raw_map.get(3, 0.0)
        mu4 = raw_map.get(4, 0.0)
        centered = []
        for order in moment_orders:
            if order == 2:
                centered.append(mu2 - mu1 * mu1)
            elif order == 3:
                centered.append(mu3 - 3 * mu1 * mu2 + 2 * mu1 * mu1 * mu1)
            elif order == 4:
                centered.append(mu4 - 4 * mu1 * mu3 + 6 * mu1 * mu1 * mu2 - 3 * mu1 ** 4)
            elif order > 4:
                centered.append(0.0)
        return raw, centered

    eigvals = np.clip(eigvals, 0.0, 2.0)
    return _moments_from_eigvals(eigvals, moment_orders)


def build_unweighted_adjacency(node_coords: Sequence[np.ndarray], node_chains: Sequence[str], cutoff: float) -> np.ndarray:
    """Build unweighted adjacency for cross-chain pairs within cutoff."""
    count = len(node_coords)
    if count == 0:
        return np.zeros((0, 0), dtype=float)
    adj = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            if node_chains[i] == node_chains[j]:
                continue
            dist = float(np.linalg.norm(node_coords[i] - node_coords[j]))
            if dist <= cutoff:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj
