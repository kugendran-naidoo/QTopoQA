#!/usr/bin/env python3
"""Compare .pt graph files between two directories."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Error: torch is required to compare .pt files.") from exc

RUN_LOG = Path("pt_file_compare_run.log").resolve()
FAIL_LOG = Path("pt_file_compare_failures.log").resolve()
DEFAULT_ABS_TOL = 0.0
DEFAULT_REL_TOL = 0.0


class _TeeStdout:
    """Mirror writes to both stdout and a log file."""

    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def __getattr__(self, attr):
        return getattr(self._primary, attr)


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text before exiting on errors."""

    def error(self, message: str) -> None:  # pragma: no cover
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def _trim_suffix(stem: str, suffixes: tuple[str, ...]) -> str:
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem = stem.rstrip("_- .")
            lower = stem.lower()
    return stem


def _normalise_name(path: Path) -> str:
    return _trim_suffix(path.stem, (".pt", ".pth", ".bin", "graph"))


def _gather_pt_files(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for pattern in ("*.pt", "*.pth", "*.bin"):
        for path in root.rglob(pattern):
            if path.is_file():
                key = _normalise_name(path)
                mapping.setdefault(key, path)
    return mapping


def parse_args(argv: Optional[Iterable[str]] = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = HelpOnErrorArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pt-file-dir1", type=Path, required=True,
                        help="Directory containing baseline .pt files. (required)")
    parser.add_argument("--pt-file-dir2", type=Path, required=True,
                        help="Directory containing candidate .pt files. (required)")
    parser.add_argument("--abs-tolerance", type=float, default=DEFAULT_ABS_TOL,
                        help="Absolute tolerance for numeric comparison (default: %(default)s)")
    parser.add_argument("--rel-tolerance", type=float, default=DEFAULT_REL_TOL,
                        help="Relative tolerance for numeric comparison (default: %(default)s)")
    parser.add_argument("--report", type=Path, default=Path("pt_file_compare_diff_report.txt"),
                        help="Path to write a detailed difference report (default: %(default)s)")
    parser.add_argument("--same-report", type=Path, default=Path("pt_file_compare_same_report.txt"),
                        help="Path to write identical pair report (default: %(default)s)")
    parser.add_argument("--no-flatten-dir1", action="store_true",
                        help="Do not search recursively in pt_file_dir1.")
    parser.add_argument("--no-flatten-dir2", action="store_true",
                        help="Do not search recursively in pt_file_dir2.")
    parser.add_argument("--order-agnostic", action="store_true",
                        help="Treat tensors as order-independent (ignores row/edge ordering).")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args, parser


def _collect_defaults(parser: argparse.ArgumentParser) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for action in parser._actions:
        if action.dest and action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    return defaults


def _log_configuration(log, args: argparse.Namespace, defaults: Dict[str, object]) -> None:
    log("=== compare_pt_files run ===")
    log(f"Start time: {time.strftime('%H:%M:%S')}")
    log("Configuration:")
    for key in sorted(vars(args)):
        log(f"  {key}: {getattr(args, key)}")
    log("Defaults:")
    for key in sorted(defaults):
        log(f"  {key}: {defaults[key]}")


def _to_field_dict(data) -> Dict[str, object]:
    if isinstance(data, dict):
        return {k: data[k] for k in data}
    if hasattr(data, "keys"):
        return {k: data[k] for k in data.keys()}
    raise TypeError(f"Unsupported data type for comparison: {type(data)}")


def _node_permutation(x: torch.Tensor) -> torch.Tensor:
    x_cpu = x.detach().cpu()
    n = x_cpu.size(0)
    if n == 0:
        return torch.arange(0, device=x.device, dtype=torch.long)

    keys = [np.arange(n)]
    x_np = x_cpu.numpy()
    if x_np.ndim == 1:
        keys.append(x_np)
    else:
        for col in x_np.T[::-1]:
            keys.append(col)
    order = np.lexsort(tuple(keys))
    return torch.from_numpy(order).to(x.device, dtype=torch.long)


def _edge_permutation(edge_index: torch.Tensor,
                      edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.arange(0, device=edge_index.device, dtype=torch.long)

    edge_cpu = edge_index.detach().cpu()
    src = edge_cpu[0].numpy()
    dst = edge_cpu[1].numpy()
    e = src.shape[0]

    keys = [np.arange(e)]
    if edge_attr is not None and torch.is_tensor(edge_attr) and edge_attr.numel() > 0:
        attr_np = edge_attr.detach().cpu().numpy()
        if attr_np.ndim == 1:
            keys.append(attr_np)
        else:
            for col in attr_np.T[::-1]:
                keys.append(col)
    keys.append(dst)
    keys.append(src)
    order = np.lexsort(tuple(keys))
    return torch.from_numpy(order).to(edge_index.device, dtype=torch.long)


def _canonicalise_fields(fields: Dict[str, object]) -> Dict[str, object]:
    canon: Dict[str, object] = {}
    for key, value in fields.items():
        if torch.is_tensor(value):
            canon[key] = value.clone()
        else:
            canon[key] = value

    x = canon.get("x")
    if isinstance(x, torch.Tensor) and x.numel() > 0:
        perm = _node_permutation(x)
        canon["x"] = x[perm]

        batch = canon.get("batch")
        if isinstance(batch, torch.Tensor) and batch.numel() == perm.numel():
            canon["batch"] = batch[perm]

        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)

        edge_index = canon.get("edge_index")
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            remapped = perm_inv[edge_index]
            edge_attr = canon.get("edge_attr") if "edge_attr" in canon else None
            if edge_attr is not None and not torch.is_tensor(edge_attr):
                edge_attr = None
            order = _edge_permutation(remapped, edge_attr)
            canon["edge_index"] = remapped[:, order]

            for edge_field in ("edge_attr", "edge_weight"):
                value = canon.get(edge_field)
                if isinstance(value, torch.Tensor) and value.size(0) == order.numel():
                    canon[edge_field] = value[order]

    return canon


def _compare_scalars(a, b, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def _compare_tensors(tensor_a, tensor_b, abs_tol: float, rel_tol: float) -> bool:
    if tensor_a.size() != tensor_b.size():
        return False
    diff = tensor_a - tensor_b
    if diff.numel() == 0:
        return True
    max_abs = float(diff.abs().max())
    if max_abs <= abs_tol:
        return True
    denom = tensor_b.abs().clamp_min(1e-12)
    max_rel = float((diff.abs() / denom).max())
    return max_rel <= rel_tol


def _compare_data_objects(data_a, data_b, abs_tol: float, rel_tol: float) -> List[str]:
    diffs: List[str] = []
    keys_a = sorted(data_a.keys())
    keys_b = sorted(data_b.keys())
    if keys_a != keys_b:
        diffs.append(f"  field mismatch: baseline={keys_a} candidate={keys_b}")
        return diffs
    for key in keys_a:
        value_a = data_a[key]
        value_b = data_b[key]
        if torch.is_tensor(value_a) and torch.is_tensor(value_b):
            if not _compare_tensors(value_a, value_b, abs_tol, rel_tol):
                diffs.append(f"  tensor '{key}' differs")
        else:
            if not _compare_scalars(value_a, value_b, abs_tol, rel_tol):
                diffs.append(f"  field '{key}' differs: baseline={value_a} candidate={value_b}")
    return diffs


def main(argv: Optional[Iterable[str]] = None) -> int:
    args, parser = parse_args(argv)
    defaults = _collect_defaults(parser)

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

    pt_dir1 = args.pt_file_dir1.resolve()
    pt_dir2 = args.pt_file_dir2.resolve()

    report_path = args.report.resolve()
    same_path = args.same_report.resolve()
    if report_path.exists():
        report_path.write_text("", encoding="utf-8")
    if same_path.exists():
        same_path.write_text("", encoding="utf-8")

    start = time.perf_counter()

    with RUN_LOG.open("w", encoding="utf-8") as run_handle, FAIL_LOG.open("w", encoding="utf-8") as fail_handle:
        def log(msg: str) -> None:
            print(msg)
            run_handle.write(msg + "\n")

        _log_configuration(log, args, defaults)

        flatten1 = not args.no_flatten_dir1
        flatten2 = not args.no_flatten_dir2

        if not pt_dir1.is_dir():
            log(f"Error: pt_file_dir1 does not exist: {pt_dir1}")
            fail_handle.write(f"Missing directory: {pt_dir1}\n")
            return 2
        if not pt_dir2.is_dir():
            log(f"Error: pt_file_dir2 does not exist: {pt_dir2}")
            fail_handle.write(f"Missing directory: {pt_dir2}\n")
            return 2

        files1 = _gather_pt_files(pt_dir1 if flatten1 else pt_dir1)
        files2 = _gather_pt_files(pt_dir2 if flatten2 else pt_dir2)

        missing_in_2 = sorted(set(files1) - set(files2))
        missing_in_1 = sorted(set(files2) - set(files1))

        shared = sorted(set(files1) & set(files2))

        identical = 0
        different = 0
        diff_report: List[str] = []
        same_report: List[str] = []

        for key in shared:
            baseline_path = files1[key]
            candidate_path = files2[key]
            try:
                data_a = torch.load(baseline_path, map_location="cpu")
                data_b = torch.load(candidate_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover
                msg = f"DIFFERENT: {key}\n  baseline: {baseline_path}\n  candidate: {candidate_path}\n  load error: {exc}"
                diff_report.append(msg)
                fail_handle.write(msg + "\n")
                different += 1
                continue

            fields_a = _to_field_dict(data_a)
            fields_b = _to_field_dict(data_b)
            if args.order_agnostic:
                fields_a = _canonicalise_fields(fields_a)
                fields_b = _canonicalise_fields(fields_b)

            diffs = _compare_data_objects(
                fields_a,
                fields_b,
                args.abs_tolerance,
                args.rel_tolerance,
            )
            if diffs:
                header = (
                    f"DIFFERENT: {key}\n  baseline: {baseline_path}\n  candidate: {candidate_path}"
                )
                diff_report.append("\n".join([header] + diffs))
                different += 1
            else:
                same_report.append(
                    f"SAME: {key}\n  baseline: {baseline_path}\n  candidate: {candidate_path}"
                )
                identical += 1

        print_missing = []
        for key in missing_in_2:
            path = files1[key]
            msg = f"MISSING in candidate: {key}\n  baseline: {path}"
            diff_report.append(msg)
            fail_handle.write(msg + "\n")
            print_missing.append(msg)
        for key in missing_in_1:
            path = files2[key]
            msg = f"MISSING in baseline: {key}\n  candidate: {path}"
            diff_report.append(msg)
            fail_handle.write(msg + "\n")
            print_missing.append(msg)

        elapsed = time.perf_counter() - start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        log(f"Baseline directory: {pt_dir1}")
        log(f"Candidate directory: {pt_dir2}")
        log(f"Shared files compared: {identical + different}")
        log(f"  Identical files:     {identical}")
        log(f"  Different files:     {different}")
        log(f"Missing in candidate:  {len(missing_in_2)}")
        log(f"Missing in baseline:   {len(missing_in_1)}")
        log(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        if diff_report:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n\n".join(diff_report) + "\n", encoding="utf-8")
            log(f"Detailed differences written to {report_path}")
        else:
            log("No differences detected; no report written.")

        if same_report:
            same_path.parent.mkdir(parents=True, exist_ok=True)
            same_path.write_text("\n".join(same_report) + "\n", encoding="utf-8")
            log(f"Identical file pairs written to {same_path}")
        else:
            log("No identical file pairs recorded; no same-report written.")

        if different or missing_in_1 or missing_in_2:
            log("Differences detected; see logs for details.")
            return 1

        log("All .pt files are identical within tolerances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
