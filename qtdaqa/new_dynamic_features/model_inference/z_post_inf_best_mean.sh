#!/usr/bin/env python3
"""
Summarise inference checkpoints by mean ranking-loss (dockq column) and pick the top K.
Usage: ./vs_code_post_inference_best_mean.sh <run_output_dir> [TOP_K] [dataset1 dataset2 ...]
Example: ./vs_code_post_inference_best_mean.sh output/advanced_heavy_10A_ARM 3
"""

import csv
import sys
from pathlib import Path


def read_mean(path: Path) -> tuple[float, int]:
    total = 0.0
    count = 0
    try:
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    v = float(row.get("dockq", ""))
                except Exception:
                    continue
                total += v
                count += 1
    except FileNotFoundError:
        return 0.0, 0
    return total, count


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: vs_code_post_inference_best_mean.sh <run_output_dir> [TOP_K] [datasets...]\n")
        return 1
    run_dir = Path(sys.argv[1]).resolve()
    top_k = 3
    datasets = ["BM55-AF2", "HAF2", "ABAG-AF3"]
    if len(sys.argv) >= 3:
        try:
            top_k = int(sys.argv[2])
            ds_args = sys.argv[3:]
        except ValueError:
            ds_args = sys.argv[2:]
        if ds_args:
            datasets = ds_args

    results_dir = run_dir / "results"
    if not results_dir.exists():
        sys.stderr.write(f"Results directory not found: {results_dir}\n")
        return 1

    rows = []
    for ckpt_dir in sorted(results_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        ckpt_name = ckpt_dir.name
        ds_means = {}
        total_sum = 0.0
        total_count = 0
        for ds in datasets:
            csv_path = ckpt_dir / ds / "final_results_dockq.csv"
            s, c = read_mean(csv_path)
            if c > 0:
                ds_means[ds] = s / c
                total_sum += s
                total_count += c
            else:
                ds_means[ds] = None
        overall = (total_sum / total_count) if total_count > 0 else None
        rows.append((overall, ckpt_name, ds_means))

    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda x: x[0])

    header = ["checkpoint", "overall_mean"] + [f"{ds}_mean" for ds in datasets]
    print("\t".join(header))
    for overall, ckpt_name, ds_means in rows[:top_k]:
        fields = [ckpt_name, f"{overall:.6f}"]
        for ds in datasets:
            val = ds_means.get(ds)
            fields.append(f"{val:.6f}" if val is not None else "NA")
        print("\t".join(fields))

    return 0


if __name__ == "__main__":
    sys.exit(main())
