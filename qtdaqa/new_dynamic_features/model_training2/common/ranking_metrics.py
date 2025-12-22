from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import warnings

import numpy as np

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    stats = None  # type: ignore
    _HAS_SCIPY = False


def _align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align_pair(a, b)
    if a.size < 2 or b.size < 2:
        return 0.0
    if not _HAS_SCIPY:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coef, _ = stats.spearmanr(a, b, nan_policy="omit")
    if not np.isfinite(coef):
        return 0.0
    return float(coef)


def default_group_key(model_name: str) -> str:
    name = str(model_name) if model_name is not None else ""
    if not name:
        return ""
    return name.split("_", 1)[0]


@dataclass
class RankingSummary:
    mean_regret: float
    mean_spearman: float
    group_count: int
    skipped_groups: int
    mean_group_size: float

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}rank_regret": self.mean_regret,
            f"{prefix}rank_spearman": self.mean_spearman,
            f"{prefix}rank_groups": float(self.group_count),
            f"{prefix}rank_groups_skipped": float(self.skipped_groups),
            f"{prefix}rank_group_size_mean": self.mean_group_size,
        }


def compute_grouped_ranking_metrics(
    names: Sequence[str],
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    group_key_fn=default_group_key,
) -> RankingSummary:
    if not names:
        return RankingSummary(0.0, 0.0, 0, 0, 0.0)

    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for name, pred, tgt in zip(names, predictions, targets):
        group = group_key_fn(str(name))
        grouped.setdefault(group, []).append((float(pred), float(tgt)))

    regrets: List[float] = []
    spearmans: List[float] = []
    sizes: List[int] = []
    skipped = 0

    for _, items in grouped.items():
        if len(items) < 2:
            skipped += 1
            continue
        preds = np.array([p for p, _ in items], dtype=np.float64)
        trues = np.array([t for _, t in items], dtype=np.float64)
        best_true = float(np.max(trues)) if trues.size else 0.0
        top_idx = int(np.argmax(preds)) if preds.size else 0
        top_true = float(trues[top_idx]) if trues.size else 0.0
        regrets.append(best_true - top_true)
        spearmans.append(_safe_spearman(preds, trues))
        sizes.append(len(items))

    if not regrets:
        return RankingSummary(0.0, 0.0, 0, skipped, 0.0)

    return RankingSummary(
        mean_regret=float(np.mean(regrets)),
        mean_spearman=float(np.mean(spearmans)),
        group_count=len(regrets),
        skipped_groups=skipped,
        mean_group_size=float(np.mean(sizes)) if sizes else 0.0,
    )
