#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    stats = None  # type: ignore
    _HAS_SCIPY = False


def _group_key(name: str) -> str:
    return str(name).split("_", 1)[0] if name else ""


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or not _HAS_SCIPY:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coef, _ = stats.spearmanr(a, b, nan_policy="omit")
    if not np.isfinite(coef):
        return 0.0
    return float(coef)


def _compute_group_metrics(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    regrets: List[float] = []
    spearmans: List[float] = []
    for _, group in df.groupby("group"):
        if len(group) < 2:
            continue
        preds = group["pred"].to_numpy(dtype=float)
        trues = group["true"].to_numpy(dtype=float)
        best_true = float(np.max(trues))
        top_true = float(trues[int(np.argmax(preds))])
        regrets.append(best_true - top_true)
        spearmans.append(_safe_spearman(preds, trues))
    return regrets, spearmans


def _bootstrap(values: List[float], n_boot: int, seed: int) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [rng.choice(values) for _ in range(len(values))]
        samples.append(float(np.mean(draw)))
    samples = np.asarray(samples, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap ranking metric stability on a tuning slice.")
    parser.add_argument("--scores", type=Path, required=True, help="CSV with MODEL,true,pred columns.")
    parser.add_argument("--true-col", type=str, default="dockq", help="Column for true score.")
    parser.add_argument("--pred-col", type=str, default="pred", help="Column for predicted score.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    if "MODEL" not in df.columns:
        raise ValueError("CSV must include MODEL column.")
    if args.true_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError("CSV must include true and pred columns.")
    df = df.rename(columns={args.true_col: "true", args.pred_col: "pred"})
    df["group"] = df["MODEL"].astype(str).apply(_group_key)
    df = df.dropna(subset=["true", "pred"])

    regrets, spearmans = _compute_group_metrics(df)
    result = {
        "groups": len(regrets),
        "rank_regret": _bootstrap(regrets, args.n_boot, args.seed),
        "rank_spearman": _bootstrap(spearmans, args.n_boot, args.seed),
    }

    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
