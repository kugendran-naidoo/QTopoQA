#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_BINS = [0.0, 0.23, 0.49, 0.80, 1.0]
DEFAULT_GROUP_COL = "group_key"


def _group_key_from_model(model: object) -> str:
    if model is None:
        return ""
    value = str(model)
    if not value:
        return ""
    return value.split("_", 1)[0]


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "model" not in cols or "dockq" not in cols:
        raise ValueError(f"Label CSV {path} missing MODEL/dockq columns")
    df = df.rename(columns={cols["model"]: "MODEL", cols["dockq"]: "dockq"})
    if "capri" in cols:
        df = df.rename(columns={cols["capri"]: "capri"})
    return df


def _with_features(df: pd.DataFrame, bins: List[float]) -> pd.DataFrame:
    df = df.copy()
    df["dockq"] = pd.to_numeric(df["dockq"], errors="coerce")
    df = df.dropna(subset=["dockq"])
    df["dockq_bin"] = pd.cut(df["dockq"], bins=bins, include_lowest=True)
    df["af2m"] = df["MODEL"].astype(str).str.contains("_af2m", case=False, regex=False)
    if "capri" in df.columns:
        df["capri"] = pd.to_numeric(df["capri"], errors="coerce")
        df["capri_class"] = df["capri"].fillna(-1).astype(int)
    else:
        df["capri_class"] = -1
    df[DEFAULT_GROUP_COL] = df["MODEL"].apply(_group_key_from_model)
    return df


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _key_to_str(key: object) -> str:
    if isinstance(key, tuple):
        return " | ".join(str(item) for item in key)
    return str(key)


def _distribution_series(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    return df.groupby(keys, dropna=False, observed=False).size().sort_values(ascending=False)


def _group_stats(df: pd.DataFrame, group_col: str, min_size: int) -> Dict[str, int]:
    counts = df.groupby(group_col, dropna=False, observed=False).size()
    total_groups = int(counts.size)
    groups_ge2 = int((counts >= min_size).sum())
    singletons = int((counts == 1).sum())
    max_group = int(counts.max()) if total_groups > 0 else 0
    return {
        "groups": total_groups,
        "groups_ge_min": groups_ge2,
        "singletons": singletons,
        "max_group_size": max_group,
    }


def _filter_by_group_size(
    df: pd.DataFrame, *, group_col: str, min_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby(group_col, dropna=False, observed=False).size()
    eligible = counts[counts >= min_size].index
    eligible_df = df[df[group_col].isin(eligible)]
    ineligible_df = df[~df[group_col].isin(eligible)]
    return eligible_df, ineligible_df


def _distribution_dict(df: pd.DataFrame, keys: List[str]) -> Dict[str, int]:
    series = _distribution_series(df, keys)
    return {_key_to_str(key): int(value) for key, value in series.items()}


def _sample_stratified(
    df: pd.DataFrame,
    keys: List[str],
    target_counts: Dict[Tuple[object, ...], int],
    seed: int,
    target_size: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    selections = []
    remaining_idx = set(df.index.tolist())
    grouped = df.groupby(keys, dropna=False, observed=False)
    for key, count in target_counts.items():
        try:
            group = grouped.get_group(key)
        except KeyError:
            continue
        if count <= 0:
            continue
        take = min(count, len(group))
        sampled = group.sample(n=take, random_state=rng.randint(0, 1_000_000))
        selections.append(sampled)
        remaining_idx.difference_update(sampled.index.tolist())

    selected = pd.concat(selections, ignore_index=True) if selections else df.iloc[0:0]
    deficit = target_size - len(selected)
    if deficit > 0 and remaining_idx:
        remaining = df.loc[list(remaining_idx)]
        extra = remaining.sample(n=min(deficit, len(remaining)), random_state=rng.randint(0, 1_000_000))
        selected = pd.concat([selected, extra], ignore_index=True)
    return selected


def _sample_remaining(
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    remaining: int,
    *,
    seed: int,
    replace: bool,
) -> pd.DataFrame:
    if remaining <= 0 or pool.empty:
        return pool.iloc[0:0]
    rng_seed = int(seed) % 2_000_000_000
    if replace:
        return pool.sample(n=remaining, replace=True, random_state=rng_seed)
    remaining_idx = pool.index.difference(selected.index)
    if remaining_idx.empty:
        return pool.iloc[0:0]
    candidate = pool.loc[remaining_idx]
    take = min(remaining, len(candidate))
    return candidate.sample(n=take, replace=False, random_state=rng_seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute dataset stats and build a stratified tuning split.")
    parser.add_argument("--train", type=Path, required=True, help="Training label CSV (MODEL,dockq,capri).")
    parser.add_argument("--val", type=Path, help="Validation label CSV (optional).")
    parser.add_argument("--eval", type=Path, action="append", default=[], help="Eval label CSV(s) to mirror.")
    parser.add_argument("--out", type=Path, required=True, help="Output tuning CSV path.")
    parser.add_argument("--source", choices=("train", "val", "both"), default="train")
    parser.add_argument("--tuning-fraction", type=float, default=0.1, help="Fraction of source to sample.")
    parser.add_argument("--tuning-size", type=int, default=None, help="Absolute number of tuning samples.")
    parser.add_argument("--dockq-bins", type=str, default=",".join(str(x) for x in DEFAULT_BINS))
    parser.add_argument("--stratify", type=str, default="dockq_bin,capri_class,af2m")
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--stats-out", type=Path, default=None, help="Optional JSON stats output.")
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum decoys per target group for tuning.")
    parser.add_argument(
        "--singleton-policy",
        choices=("strict", "replace", "hybrid"),
        default="strict",
        help="How to handle groups with <min-group-size (strict=drop; replace=sample with replacement; hybrid=fill with singletons).",
    )
    args = parser.parse_args()

    bins = [float(x) for x in args.dockq_bins.split(",") if x.strip()]
    keys = [k.strip() for k in args.stratify.split(",") if k.strip()]

    train_df = _with_features(_load_labels(args.train), bins)
    val_df = _with_features(_load_labels(args.val), bins) if args.val else None
    source_df = train_df if args.source == "train" else val_df if args.source == "val" else pd.concat([train_df, val_df])
    if source_df is None:
        raise ValueError("Requested source split is missing.")

    eval_frames = []
    for path in args.eval or []:
        try:
            eval_frames.append(_with_features(_load_labels(path), bins))
        except Exception:
            continue
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else None

    group_stats = _group_stats(source_df, DEFAULT_GROUP_COL, max(1, int(args.min_group_size)))
    stats = {
        "train": _distribution_dict(train_df, keys),
        "val": _distribution_dict(val_df, keys) if val_df is not None else {},
        "eval": _distribution_dict(eval_df, keys) if eval_df is not None else {},
        "group_stats": group_stats,
    }

    if eval_df is not None:
        train_dist = _distribution_series(train_df, keys)
        eval_dist = _distribution_series(eval_df, keys)
        aligned = train_dist.index.union(eval_dist.index)
        stats["js_divergence"] = _js_divergence(
            train_dist.reindex(aligned, fill_value=0).values.astype(float),
            eval_dist.reindex(aligned, fill_value=0).values.astype(float),
        )

    if args.stats_out:
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    else:
        print(json.dumps(stats, indent=2))

    target_size = len(source_df)
    if args.tuning_size is not None:
        target_size = min(target_size, max(1, int(args.tuning_size)))
    elif args.tuning_fraction:
        target_size = max(1, int(round(len(source_df) * float(args.tuning_fraction))))

    if eval_df is not None:
        eval_dist = _distribution_series(eval_df, keys)
        total = eval_dist.sum()
        target_counts = {k: int(round(v / total * target_size)) for k, v in eval_dist.items()}
    else:
        source_dist = _distribution_series(source_df, keys)
        total = source_dist.sum()
        target_counts = {k: int(round(v / total * target_size)) for k, v in source_dist.items()}

    min_group_size = max(1, int(args.min_group_size))
    eligible_df, ineligible_df = _filter_by_group_size(
        source_df, group_col=DEFAULT_GROUP_COL, min_size=min_group_size
    )
    if args.singleton_policy in {"strict", "replace"}:
        pool_df = eligible_df
    else:
        pool_df = eligible_df
    if pool_df.empty:
        raise ValueError(
            "No groups meet the minimum group size; adjust --min-group-size or choose a different policy."
        )

    tuning_df = _sample_stratified(pool_df, keys, target_counts, args.seed, target_size)
    deficit = target_size - len(tuning_df)
    if deficit > 0:
        if args.singleton_policy == "strict":
            extra = _sample_remaining(eligible_df, tuning_df, deficit, seed=args.seed, replace=False)
            tuning_df = pd.concat([tuning_df, extra], ignore_index=True)
        elif args.singleton_policy == "replace":
            extra = _sample_remaining(eligible_df, tuning_df, deficit, seed=args.seed, replace=True)
            tuning_df = pd.concat([tuning_df, extra], ignore_index=True)
        else:  # hybrid
            extra = _sample_remaining(ineligible_df, tuning_df, deficit, seed=args.seed, replace=False)
            tuning_df = pd.concat([tuning_df, extra], ignore_index=True)
    tuning_out = tuning_df[["MODEL", "dockq"]].copy()
    if "capri" in tuning_df.columns:
        tuning_out["capri"] = tuning_df["capri"]
    tuning_out.to_csv(args.out, index=False)

    print(f"Wrote tuning split: {args.out} ({len(tuning_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
