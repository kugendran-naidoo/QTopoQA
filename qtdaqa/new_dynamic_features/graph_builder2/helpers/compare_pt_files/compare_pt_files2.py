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
DEFAULT_STRICT_METADATA = False


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
    parser.add_argument("baseline_dir", type=Path, metavar="baseline_dir",
                        help="Directory containing baseline .pt files. (required)")
    parser.add_argument("candidate_dir", type=Path, metavar="candidate_dir",
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
                        help="Do not search recursively in baseline_dir.")
    parser.add_argument("--no-flatten-dir2", action="store_true",
                        help="Do not search recursively in candidate_dir.")
    parser.add_argument("--order-agnostic", action="store_true",
                        help="Treat tensors as order-independent (ignores row/edge ordering).")
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        default=DEFAULT_STRICT_METADATA,
        help="Require metadata fields to match exactly (relaxed mode drops run-specific builder info).",
    )
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


def _normalise_builder(builder: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    schema_version = builder.get("schema_version")
    if schema_version is not None:
        cleaned["schema_version"] = schema_version
    return cleaned


def _normalise_metadata(metadata: Dict[str, object], *, strict: bool) -> Dict[str, object]:
    if strict:
        return dict(metadata)

    cleaned: Dict[str, object] = {}
    for key, value in metadata.items():
        if key == "builder" and isinstance(value, dict):
            builder_store = _normalise_builder(value)
            if builder_store:
                cleaned[key] = builder_store
            continue
        if key == "feature_config" and not strict:
            continue
        if key == "edge_params" and isinstance(value, dict):
            new_value = {k: v for k, v in value.items() if k not in {"jobs", "requested_jobs", "num_workers"}}
            cleaned[key] = new_value
            continue
        if key == "edge_info" and isinstance(value, dict):
            new_value = {k: v for k, v in value.items() if k not in {"requested_jobs", "jobs"}}
            cleaned[key] = new_value
            continue
        cleaned[key] = value
    return cleaned


def _apply_metadata_normalisation(fields: Dict[str, object], *, strict: bool) -> Dict[str, object]:
    result = dict(fields)
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        result["metadata"] = _normalise_metadata(metadata, strict=strict)
    return result


def _metadata_feature_config_note(meta_a, meta_b) -> Optional[str]:
    def _extract(meta):
        if not isinstance(meta, dict):
            return None, None
        feature_cfg = meta.get("feature_config")
        if not isinstance(feature_cfg, dict):
            return None, None
        sha = feature_cfg.get("sha256") or feature_cfg.get("text_sha256")
        name = feature_cfg.get("name")
        path_value = feature_cfg.get("path")
        if not name and path_value:
            name = Path(path_value).name
        return sha, name

    sha_a, name_a = _extract(meta_a)
    sha_b, name_b = _extract(meta_b)
    if sha_a == sha_b:
        return None
    return (
        "feature_config hash differs: "
        f"baseline=({name_a or 'unknown'}, {sha_a or 'unknown'}) "
        f"candidate=({name_b or 'unknown'}, {sha_b or 'unknown'})"
    )


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


def _canonicalise_fields(fields: Dict[str, object], *, strict_metadata: bool) -> Dict[str, object]:
    canon: Dict[str, object] = {}
    for key, value in fields.items():
        if torch.is_tensor(value):
            canon[key] = value.clone()
        else:
            canon[key] = value

    metadata = canon.get("metadata")
    if isinstance(metadata, dict):
        canon["metadata"] = _normalise_metadata(metadata, strict=strict_metadata)

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

    pt_dir1 = args.baseline_dir.resolve()
    pt_dir2 = args.candidate_dir.resolve()

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
            log(f"Error: baseline_dir does not exist: {pt_dir1}")
            fail_handle.write(f"Missing directory: {pt_dir1}\n")
            return 2
        if not pt_dir2.is_dir():
            log(f"Error: candidate_dir does not exist: {pt_dir2}")
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
        metadata_notes: List[str] = []

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

            raw_fields_a = _to_field_dict(data_a)
            raw_fields_b = _to_field_dict(data_b)

            metadata_note = None
            if not args.strict_metadata:
                metadata_note = _metadata_feature_config_note(
                    raw_fields_a.get("metadata"), raw_fields_b.get("metadata")
                )

            fields_a = _apply_metadata_normalisation(raw_fields_a, strict=args.strict_metadata)
            fields_b = _apply_metadata_normalisation(raw_fields_b, strict=args.strict_metadata)
            if args.order_agnostic:
                fields_a = _canonicalise_fields(fields_a, strict_metadata=args.strict_metadata)
                fields_b = _canonicalise_fields(fields_b, strict_metadata=args.strict_metadata)

            diffs = _compare_data_objects(
                fields_a,
                fields_b,
                args.abs_tolerance,
                args.rel_tolerance,
            )
            if diffs:
                if metadata_note and not args.strict_metadata:
                    diffs.append(f"  NOTE: {metadata_note}")
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
                if metadata_note and not args.strict_metadata:
                    metadata_notes.append(
                        "\n".join(
                            [
                                f"METADATA (ignored): {key}",
                                f"  baseline: {baseline_path}",
                                f"  candidate: {candidate_path}",
                                f"  NOTE: {metadata_note}",
                            ]
                        )
                    )

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

        report_sections: List[str] = []
        if diff_report:
            report_sections.append("=== Tensor differences ===\n" + "\n\n".join(diff_report))
        if metadata_notes:
            report_sections.append(
                "=== Metadata-only differences (ignored for equality) ===\n" + "\n\n".join(metadata_notes)
            )
        if report_sections:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
            if diff_report and metadata_notes:
                log(f"Tensor and metadata-only details written to {report_path.name}")
            elif diff_report:
                log(f"Detailed differences written to {report_path.name}")
            else:
                log(f"Metadata-only differences written to {report_path.name}")
        else:
            log("No differences detected; no report written.")

        if same_report:
            same_path.parent.mkdir(parents=True, exist_ok=True)
            same_path.write_text("\n".join(same_report) + "\n", encoding="utf-8")
            log(f"Identical file pairs written to {same_path.name}")
        else:
            log("No identical file pairs recorded; no same-report written.")

        if metadata_notes and not diff_report:
            log(f"Metadata-only differences detected (ignored for equality): {len(metadata_notes)}")

        if different or missing_in_1 or missing_in_2:
            log("Differences detected; see logs for details.")
            return 1

        log("All .pt files are identical within tolerances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
