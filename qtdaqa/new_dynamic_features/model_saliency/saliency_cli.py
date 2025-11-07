"""
Standalone command-line interface for the saliency toolkit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

if __package__ is None or __package__ == "":
    # Allow execution via ``python saliency_cli.py`` by injecting the repo root.
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from qtdaqa.new_dynamic_features.model_saliency.lib.config import (
        CheckpointConfig,
        GraphSelection,
        SaliencyRequest,
        default_output_dir,
    )
else:
    from .lib.config import (
        CheckpointConfig,
        GraphSelection,
        SaliencyRequest,
        default_output_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run saliency analysis on TopoQA GNN checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Lightning checkpoint (.chkpt)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repository root for module imports",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory containing graph .pt files",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="One or more model identifiers to analyse (defaults to all *.pt in graph-dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on analysed models (applied after --models filtering).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for reports (defaults to outputs/<checkpoint-stem>).",
    )
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=64,
        help="Integrated gradients steps.",
    )
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=0,
        help="NoiseTunnel samples for smoothing (0 disables NoiseTunnel).",
    )
    parser.add_argument(
        "--disable-explainer",
        action="store_true",
        help="Skip PyG GNNExplainer subgraph masks.",
    )
    return parser


def _create_request(args: argparse.Namespace) -> SaliencyRequest:
    checkpoint_cfg = CheckpointConfig(
        checkpoint_path=args.checkpoint.resolve(),
        repo_root=args.repo_root.resolve(),
    )
    graphs = GraphSelection(
        graph_dir=args.graph_dir.resolve(),
        models=list(_normalise_models(args.models)),
        limit=args.limit,
    )
    output_dir = args.output.resolve() if args.output else default_output_dir(args.checkpoint.stem)
    return SaliencyRequest(
        checkpoint=checkpoint_cfg,
        graphs=graphs,
        output_dir=output_dir,
        integration_steps=args.integration_steps,
        noise_tunnel_samples=args.noise_samples,
        pyg_explainer=not args.disable_explainer,
    )


def _normalise_models(models: Iterable[str]) -> Iterable[str]:
    for model in models:
        yield model.strip()


def _resolve_runner() -> Callable[[SaliencyRequest], None]:
    if __package__ is None or __package__ == "":
        from qtdaqa.new_dynamic_features.model_saliency.lib.runner import run_saliency  # type: ignore
    else:
        from .lib.runner import run_saliency  # type: ignore
    return run_saliency


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    request = _create_request(args)
    runner = _resolve_runner()
    runner(request)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
