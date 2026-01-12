#!/usr/bin/env python3
"""Select FLT baseline run aligned with inference Option B EMA metrics."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _parse_default(line: str, var: str) -> Optional[str]:
    pattern = rf'{var}="\${{{var}:-([^}}]+)}}\"'
    match = re.search(pattern, line)
    if match:
        return match.group(1).strip()
    return None


def _resolve_repo_root(start: Path) -> Optional[Path]:
    current = start
    for _ in range(10):
        if (current / "qtdaqa").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _read_inference_defaults(script_path: Path) -> dict:
    defaults = {}
    for line in script_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        for key in ("TOP_K", "SHORTLIST_METRIC", "TUNING_METRIC"):
            if key in defaults:
                continue
            value = _parse_default(line, key)
            if value is not None:
                defaults[key] = value
    return defaults


def _resolve_metric(value: Optional[str], env_key: str, fallback: str) -> str:
    if value:
        return value
    if env_key in os.environ:
        return os.environ[env_key]
    return fallback


def _run_option_b_select(
    *,
    shortlist_metric: str,
    tuning_metric: str,
    top_k: int,
    training_root: Optional[Path],
    emit_checkpoint: bool,
    repo_root: Optional[Path],
) -> str:
    cmd = [
        sys.executable,
        "-m",
        "qtdaqa.new_dynamic_features.model_training2.tools.option_b_select",
        "--shortlist-metric",
        shortlist_metric,
        "--tuning-metric",
        tuning_metric,
        "--top-k",
        str(top_k),
    ]
    if training_root:
        cmd.extend(["--training-root", str(training_root)])
    if emit_checkpoint:
        cmd.append("--emit-checkpoint-only")
    env = os.environ.copy()
    if repo_root is not None:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.stdout.strip()


def main() -> int:
    script_root = Path(__file__).resolve().parent
    repo_root = _resolve_repo_root(script_root)
    if repo_root and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="Align FLT baseline selection with inference Option B EMA metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python select_flt_baseline_from_inference.py\n"
            "  python select_flt_baseline_from_inference.py "
            "--inference-script /path/to/z_exec_inf_training2_optionB_EMA.sh\n"
            "  python select_flt_baseline_from_inference.py "
            "--training-root /path/to/training_runs2\n"
            "  python select_flt_baseline_from_inference.py --emit-checkpoint\n"
            "\n"
            "Optional overrides (arguments or env vars):\n"
            "  --shortlist-metric / SHORTLIST_METRIC\n"
            "  --tuning-metric / TUNING_METRIC\n"
            "  --top-k / TOP_K\n"
            "\n"
            "Practical guidance (top-k):\n"
            "  - Fewer candidates (top-k=3) biases toward shortlist metric.\n"
            "  - Larger top-k (10+) gives tuning metric more influence.\n"
            "  - If runs are few, keep top-k small (3-5).\n"
            "  - If runs are many and diverse, top-k=10 is reasonable.\n"
        ),
    )
    default_inference = None
    if repo_root:
        default_inference = repo_root / "qtdaqa/new_dynamic_features/model_inference/z_exec_inf_training2_optionB_EMA.sh"
    parser.add_argument(
        "--inference-script",
        type=Path,
        default=default_inference or Path("z_exec_inf_training2_optionB_EMA.sh"),
        help="Path to the inference script to read defaults from.",
    )
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--shortlist-metric", default=None)
    parser.add_argument("--tuning-metric", default=None)
    parser.add_argument("--emit-checkpoint", action="store_true")
    args = parser.parse_args()

    script_path = args.inference_script.expanduser().resolve()
    if not script_path.exists():
        print(f"Error: inference script not found: {script_path}", file=sys.stderr)
        return 2

    defaults = _read_inference_defaults(script_path)
    shortlist_metric = _resolve_metric(
        args.shortlist_metric,
        "SHORTLIST_METRIC",
        defaults.get("SHORTLIST_METRIC", "ema_val_loss"),
    )
    tuning_metric = _resolve_metric(
        args.tuning_metric,
        "TUNING_METRIC",
        defaults.get("TUNING_METRIC", "ema_tuning_hit_rate_023"),
    )
    top_k_default = defaults.get("TOP_K", "10")
    try:
        top_k_value = int(args.top_k or os.environ.get("TOP_K", top_k_default))
    except ValueError:
        top_k_value = 10

    output = _run_option_b_select(
        shortlist_metric=shortlist_metric,
        tuning_metric=tuning_metric,
        top_k=top_k_value,
        training_root=args.training_root,
        emit_checkpoint=args.emit_checkpoint,
        repo_root=repo_root,
    )

    if args.emit_checkpoint:
        print(output)
        return 0

    run_name = None
    shortlist_value = None
    tuning_value = None
    for line in output.splitlines():
        if line.startswith("run="):
            run_name = line.split("=", 1)[1].strip()
        if line.startswith("shortlist_metric_value="):
            shortlist_value = line.split("=", 1)[1].strip()
        if line.startswith(f"{tuning_metric}="):
            tuning_value = line.split("=", 1)[1].strip()
            break

    if not run_name:
        print("Error: failed to parse run name from option_b_select output.", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1

    training_root = args.training_root
    if training_root is None:
        if repo_root:
            training_root = repo_root / "qtdaqa/new_dynamic_features/model_training2/training_runs2"
        else:
            training_root = Path("qtdaqa/new_dynamic_features/model_training2/training_runs2")
    if not training_root.is_absolute() and repo_root is not None:
        training_root = (repo_root / training_root).resolve()
    else:
        training_root = training_root.expanduser().resolve()
    run_dir = training_root / run_name

    print(f"run_name={run_name}")
    print(f"run_dir={run_dir}")
    print(f"shortlist_metric={shortlist_metric}")
    if shortlist_value is not None:
        print(f"shortlist_metric_value={shortlist_value}")
    print(f"tuning_metric={tuning_metric}")
    if tuning_value is not None:
        print(f"tuning_metric_value={tuning_value}")
    print(f"top_k={top_k_value}")
    if not run_dir.exists():
        print("warning=run_dir_missing", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
