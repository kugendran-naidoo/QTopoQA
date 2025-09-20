#!/usr/bin/env python3
"""
qtdaqa.graph_builder

Generates PyG graph files (.pt) ONLY, with configurable feature construction.
It reuses the existing TopoQA feature code but provides JSON-configurable toggles
for node, edge, and topological aspects. Graph files overwrite by default when
configured to do so.

Inputs
- --dataset-dir: folder containing per-target subfolders with .pdb/.cif decoys
- --work-dir:    folder for intermediates (interface, topo, node feature CSVs)
- --out-graphs:  destination for graph .pt files (one per decoy)
- --log-dir:     folder for logs (a per-run timestamped file is created)
- --node-config / --edge-config / --topo-config: JSON files
- --other-config: JSON with general knobs (e.g., overwrite_graphs, jobs)

Compatibility controls (CLI):
- --use-legacy-config / --use-local-json-config: use the legacy (original) graph
  construction for exact byte-for-byte compatibility (default: legacy) or use
  the local JSON-configurable builder for flexible features.
- --verify-original: when using the modular builder, also build the original and
  byte-compare outputs, logging exact_match.

Parallelism (CLI):
- --parallel N: number of decoys to process in parallel across all targets
  (cross-target concurrency). If omitted, falls back to jobs in other.json.

This module separates feature generators so they can be enabled/disabled
independently via configuration while still supporting exact compatibility.
"""
from __future__ import annotations

import argparse
import heapq
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import concurrent.futures as _fut

import numpy as np
import pandas as pd

# Ensure we can import project-local modules/packages
REPO_ROOT = Path(__file__).resolve().parents[1]  # QTopoQA
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so 'qtdaqa' and 'topoqa' packages resolve
TOPOQA_DIR = REPO_ROOT / "topoqa"
if str(TOPOQA_DIR) not in sys.path:
    sys.path.insert(0, str(TOPOQA_DIR))

# Existing feature builders
from src.get_interface import cal_interface  # type: ignore
from src.topo_feature import topo_fea  # type: ignore
from src.node_fea_df import node_fea  # type: ignore

from qtdaqa.lib.config import (
    EdgeConfig,
    NodeConfig,
    OtherConfig,
    TopoConfig,
    load_edge_config,
    load_node_config,
    load_other_config,
    load_topo_config,
)
from qtdaqa.lib.graph_build import build_graph
from qtdaqa.lib.logging_utils import setup_logger
import time
from qtdaqa.lib.original_compat import create_graph_compat
import torch


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_targets(dataset_dir: Path) -> List[Path]:
    return [p for p in dataset_dir.iterdir() if p.is_dir()]


def _list_models(target_dir: Path) -> List[Path]:
    out: List[Path] = []
    for ext in (".pdb", ".cif"):
        out.extend(sorted(target_dir.glob(f"*{ext}")))
    return out


def run_interface_one(pdb_path: Path, out_txt: Path, cutoff: float) -> None:
    # Use existing class with explicit cut
    obj = cal_interface(str(pdb_path), cut=int(cutoff))
    obj.find_and_write(str(out_txt))


def run_topo_one(pdb_path: Path, iface_txt: Path, out_csv: Path, cfg: TopoConfig) -> None:
    # iface file has IDs and coords; read residue IDs to feed to topo_fea
    df = pd.read_csv(iface_txt, sep=" ", names=["ID", "co_1", "co_2", "co_3"])  # noqa
    res_list = list(df["ID"].astype(str).values)
    obj = topo_fea(str(pdb_path), cfg.neighbor_distance, cfg.element_sets, res_list)
    topo_df = obj.cal_fea()
    topo_df.to_csv(out_csv, index=False)


def run_node_fea_one(model_name: str, pdb_dir: Path, iface_dir: Path, topo_dir: Path, out_csv: Path, cfg: NodeConfig) -> None:
    nf = node_fea(model_name, str(pdb_dir), str(iface_dir), str(topo_dir))
    df, _ = nf.calculate_fea()
    # Drop unavailable feature families per config
    drop_cols: List[str] = []
    if not cfg.use_rasa:
        drop_cols += ["rasa"]
    if not cfg.use_phi_psi:
        drop_cols += ["phi", "psi"]
    if not cfg.use_ss8:
        drop_cols += [f"SS8_{i}" for i in range(8)]
    if not cfg.use_aa_onehot:
        drop_cols += [f"AA_{i}" for i in range(21)]
    if not cfg.use_topological:
        drop_cols += [c for c in df.columns if c.startswith("f0_") or c.startswith("f1_")]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.to_csv(out_csv, index=False)


def _model_name_from_path(p: Path) -> str:
    return p.stem


def _iface_path(work_dir: Path, name: str) -> Path:
    return work_dir / "interface_ca" / f"{name}.txt"


def _topo_path(work_dir: Path, name: str) -> Path:
    return work_dir / "node_topo" / f"{name}.csv"


def _node_path(work_dir: Path, name: str) -> Path:
    return work_dir / "node_fea" / f"{name}.csv"


def _graph_path(out_graphs: Path, name: str) -> Path:
    return out_graphs / f"{name}.pt"


def _job_logger_name(run_logger_name: str, target: str, model_stem: str) -> str:
    return f"{run_logger_name}.{target}.{model_stem}"


def _parse_timestamp(line: str) -> Optional[datetime]:
    fragment = line[:19]
    try:
        return datetime.strptime(fragment, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _read_log_blocks(path: Path) -> List[Tuple[datetime, List[str]]]:
    """Return timestamped log blocks preserving multiline records."""
    blocks: List[Tuple[datetime, List[str]]] = []
    last_ts = datetime.min
    with path.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            parsed = _parse_timestamp(line)
            if parsed is None:
                ts = last_ts
            else:
                ts = parsed
                last_ts = parsed
            block = [line]
            while True:
                pos = fh.tell()
                next_line = fh.readline()
                if not next_line:
                    break
                if _parse_timestamp(next_line) is None:
                    block.append(next_line)
                    continue
                fh.seek(pos)
                break
            blocks.append((ts, block))
    return blocks


def _merge_log_files(sources: Sequence[Path], destination: Path) -> None:
    """Merge multiple log files by timestamp into destination."""
    unique_sources = []
    seen: Set[Path] = set()
    for src in sources:
        if src in seen or not src.exists():
            continue
        seen.add(src)
        unique_sources.append(src)

    if not unique_sources:
        return

    counter = 0
    heap: List[Tuple[datetime, int, List[str], Iterator[Tuple[datetime, List[str]]]]] = []
    for src in unique_sources:
        blocks = _read_log_blocks(src)
        if not blocks:
            continue
        iterator = iter(blocks)
        ts, block = next(iterator)
        heapq.heappush(heap, (ts, counter, block, iterator))
        counter += 1

    if not heap:
        return

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with tmp_path.open("w", encoding="utf-8") as out:
        while heap:
            ts, _, block, iterator = heapq.heappop(heap)
            for line in block:
                out.write(line)
            try:
                next_ts, next_block = next(iterator)
            except StopIteration:
                continue
            counter += 1
            heapq.heappush(heap, (next_ts, counter, next_block, iterator))

    tmp_path.replace(destination)


def _teardown_logger(logger: logging.Logger) -> None:
    handler_ids = {id(h) for h in logger.handlers}
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
    base = logging.getLogger("qtdaqa")
    base.handlers = [h for h in base.handlers if id(h) not in handler_ids]


def _worker_entry(item: Tuple[
    Path,  # pdb_path
    Path,  # target_dir
    Path,  # work_dir_target
    Path,  # out_graphs_subdir
    NodeConfig,
    EdgeConfig,
    TopoConfig,
    bool,   # overwrite
    bool,   # exact_compat
    bool,   # verify_with_original
    str,    # job_logger_name
    str,    # job_log_dir
]) -> Tuple[str, str, Optional[str]]:
    (m, tdir, twork, out_sub, node_cfg, edge_cfg, topo_cfg,
     overwrite, exact_compat, verify_with_original, logger_name, log_dir_str) = item
    logger = setup_logger(log_dir_str, name=logger_name)
    try:
        process_one(
            pdb_path=m,
            target_dir=tdir,
            work_dir=twork,
            out_graphs=out_sub,
            node_cfg=node_cfg,
            edge_cfg=edge_cfg,
            topo_cfg=topo_cfg,
            overwrite=overwrite,
            exact_compat=exact_compat,
            verify_with_original=verify_with_original,
            logger=logger,
        )
        return (tdir.name, m.name, None)
    except Exception as e:
        return (tdir.name, m.name, str(e))
    finally:
        _teardown_logger(logger)


def process_one(
    pdb_path: Path,
    target_dir: Path,
    work_dir: Path,
    out_graphs: Path,
    node_cfg: NodeConfig,
    edge_cfg: EdgeConfig,
    topo_cfg: TopoConfig,
    overwrite: bool,
    exact_compat: bool,
    verify_with_original: bool,
    logger: Optional[logging.Logger] = None,
    logger_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> None:
    # Allow initialization inside worker processes
    if logger is None:
        if logger_name and log_dir:
            logger = setup_logger(str(log_dir), name=logger_name)
        else:
            logger = logging.getLogger("qtdaqa")
    name = _model_name_from_path(pdb_path)
    pdb_dir = target_dir
    iface_dir = work_dir / "interface_ca"
    topo_dir = work_dir / "node_topo"
    node_dir = work_dir / "node_fea"
    for d in (iface_dir, topo_dir, node_dir, out_graphs):
        _ensure_dir(d)

    gpath = _graph_path(out_graphs, name)
    if gpath.exists() and not overwrite:
        logger.info(f"skip {name}: graph exists")
        return

    # 1) interface
    iface_txt = _iface_path(work_dir, name)
    t0 = time.perf_counter()
    run_interface_one(pdb_path, iface_txt, cutoff=edge_cfg.cutoff_max)
    dt = time.perf_counter() - t0
    logger.info(f"[interface_extraction] target={target_dir.name} decoy={pdb_path.name} -> {iface_txt.name} [done][time] took {dt:.3f}s")
    # Skip models with no detected interface residues (match original pipeline behavior)
    try:
        if (not iface_txt.exists()) or (iface_txt.stat().st_size == 0):
            logger.info(f"[skip] target={target_dir.name} decoy={pdb_path.name} reason=no_interface")
            return
    except Exception:
        # if any filesystem issue, attempt a conservative read
        try:
            n_lines = sum(1 for _ in open(iface_txt, 'r'))
        except Exception:
            n_lines = 0
        if n_lines == 0:
            logger.info(f"[skip] target={target_dir.name} decoy={pdb_path.name} reason=no_interface")
            return

    # 2) topo (optional depends on node_cfg.use_topological but safe to compute)
    topo_csv = _topo_path(work_dir, name)
    t0 = time.perf_counter()
    run_topo_one(pdb_path, iface_txt, topo_csv, topo_cfg)
    dt = time.perf_counter() - t0
    logger.info(f"[topo_features] target={target_dir.name} decoy={pdb_path.name} -> {topo_csv.name} [done][time] took {dt:.3f}s")

    # 3) node features (filter via node_cfg)
    node_csv = _node_path(work_dir, name)
    t0 = time.perf_counter()
    run_node_fea_one(name, pdb_dir, iface_dir, topo_dir, node_csv, node_cfg)
    dt = time.perf_counter() - t0
    logger.info(f"[node_features] target={target_dir.name} decoy={pdb_path.name} -> {node_csv.name} [done][time] took {dt:.3f}s")

    # 4) build graph (compat mode: use original create_graph)
    if exact_compat:
        t0 = time.perf_counter()
        # local compat API mirrors original signature
        create_graph_compat(name, str(node_dir), str(iface_dir), [f"0-{int(edge_cfg.cutoff_max)}"], str(out_graphs), str(target_dir))
        dt = time.perf_counter() - t0
        logger.info(f"[build_graph(original)] target={target_dir.name} decoy={pdb_path.name} -> {gpath.name} [done][time] took {dt:.3f}s")
    else:
        node_df = pd.read_csv(node_csv)
        iface_df = pd.read_csv(iface_txt, sep=" ", names=["ID", "co_1", "co_2", "co_3"])  # noqa
        t0 = time.perf_counter()
        art = build_graph(name, node_df, iface_df, node_cfg=node_cfg, edge_cfg=edge_cfg, pdb_path=str(pdb_path))
        torch.save(art.to_pyg(), gpath)
        dt = time.perf_counter() - t0
        logger.info(f"[build_graph] target={target_dir.name} decoy={pdb_path.name} -> {gpath.name} [done][time] took {dt:.3f}s")

    # 5) optional verification against original
    if verify_with_original and not exact_compat:
        # generate original into a temporary sibling and compare SEMANTICS (tensors)
        tmp_dir = out_graphs / "_orig_tmp"
        _ensure_dir(tmp_dir)
        t0 = time.perf_counter()
        create_graph_compat(name, str(node_dir), str(iface_dir), [f"0-{int(edge_cfg.cutoff_max)}"], str(tmp_dir), str(target_dir))
        dt = time.perf_counter() - t0
        opath = tmp_dir / f"{name}.pt"
        try:
            a = torch.load(gpath, map_location="cpu")
            b = torch.load(opath, map_location="cpu")

            def _eq_tensor(ta: torch.Tensor, tb: torch.Tensor) -> bool:
                return ta.dtype == tb.dtype and ta.shape == tb.shape and torch.equal(ta, tb)

            def _first_diff(ta: torch.Tensor, tb: torch.Tensor):
                info = {}
                if ta.dtype != tb.dtype or ta.shape != tb.shape:
                    info["shape_a"] = tuple(ta.shape)
                    info["shape_b"] = tuple(tb.shape)
                    info["dtype_a"] = str(ta.dtype)
                    info["dtype_b"] = str(tb.dtype)
                    return info
                if ta.numel() == 0:
                    return info
                # strict equality mask
                diff = ta.ne(tb)
                if diff.any():
                    idx = diff.nonzero(as_tuple=False)[0].tolist()
                    info["first_diff_index"] = idx
                    # capture a few values around
                    try:
                        info["a_val"] = ta.view(-1)[diff.view(-1).nonzero(as_tuple=False)[0]].item()
                        info["b_val"] = tb.view(-1)[diff.view(-1).nonzero(as_tuple=False)[0]].item()
                    except Exception:
                        pass
                return info

            x_ok = _eq_tensor(a.x, b.x)
            ei_ok = _eq_tensor(a.edge_index, b.edge_index)
            ea_ok = _eq_tensor(a.edge_attr, b.edge_attr)

            semantic_equal = x_ok and ei_ok and ea_ok

            details = {}
            if not x_ok:
                details["x"] = _first_diff(a.x, b.x)
            if not ei_ok:
                details["edge_index"] = _first_diff(a.edge_index, b.edge_index)
            if not ea_ok:
                details["edge_attr"] = _first_diff(a.edge_attr, b.edge_attr)

            # Also report on-disk byte equality (informational, not guaranteed for pickle)
            try:
                with open(gpath, 'rb') as fa, open(opath, 'rb') as fb:
                    bytes_equal = (fa.read() == fb.read())
            except Exception:
                bytes_equal = None

            # Canonicalized byte comparison: re-serialize as a stable dict to fixed protocol
            import io, collections
            def _canonical_bytes(obj) -> bytes:
                payload = collections.OrderedDict()
                payload['x'] = obj.x.contiguous()
                payload['edge_index'] = obj.edge_index.to(torch.long).contiguous()
                payload['edge_attr'] = obj.edge_attr.contiguous()
                buf = io.BytesIO()
                torch.save(payload, buf, pickle_protocol=4)
                return buf.getvalue()
            try:
                bytes_equal_canonical = (_canonical_bytes(a) == _canonical_bytes(b))
            except Exception:
                bytes_equal_canonical = None

            logger.info(
                f"[verify] target={target_dir.name} decoy={pdb_path.name} semantic_equal={semantic_equal} "
                f"x_ok={x_ok} edge_index_ok={ei_ok} edge_attr_ok={ea_ok} bytes_equal={bytes_equal} canonical_bytes_equal={bytes_equal_canonical} (orig took {dt:.3f}s)"
            )
            if not semantic_equal and details:
                logger.info(f"[verify] first-difference details: {details}")
        except Exception as e:
            logger.exception(f"[verify] failed compare for {name}: {e}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Configurable graph-only builder for TopoQA")
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--work-dir", type=str, required=True)
    p.add_argument("--out-graphs", type=str, required=True)
    p.add_argument("--log-dir", type=str, required=True)
    # default configs from package
    cfg_base = Path(__file__).parent / "configs"
    p.add_argument("--node-config", type=str, default=str(cfg_base / "node.json"))
    p.add_argument("--edge-config", type=str, default=str(cfg_base / "edge.json"))
    p.add_argument("--topo-config", type=str, default=str(cfg_base / "topo.json"))
    p.add_argument("--other-config", type=str, default=str(cfg_base / "other.json"))
    # CLI options replacing former JSON flags
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-legacy-config", dest="use_legacy_config", action="store_true", help="Use legacy (original) compat builder for exact graphs (default)")
    g.add_argument("--use-local-json-config", dest="use_legacy_config", action="store_false", help="Use local JSON-configurable builder")
    p.set_defaults(use_legacy_config=True)
    p.add_argument("--verify-original", action="store_true", default=False, help="Also build with original and compare bytes when using modular builder")
    p.add_argument("--parallel", type=int, default=None, help="Number of decoys to process in parallel across all targets")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ts = time.strftime("%Y-%m-%d_%H:%M")
    run_logger_name = f"graph_builder.{ts}"
    logger = setup_logger(args.log_dir, name=run_logger_name)
    file_handler = next((h for h in logger.handlers if getattr(h, "baseFilename", None)), None)
    main_log_path = Path(file_handler.baseFilename) if file_handler else None
    worker_log_dir = Path(args.log_dir) / run_logger_name
    _ensure_dir(worker_log_dir)
    worker_log_paths: Set[Path] = set()
    dataset_dir = Path(args.dataset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_graphs = Path(args.out_graphs).resolve()

    node_cfg = load_node_config(args.node_config)
    edge_cfg = load_edge_config(args.edge_config)
    topo_cfg = load_topo_config(args.topo_config)
    other_cfg = load_other_config(args.other_config)

    _ensure_dir(work_dir)
    _ensure_dir(out_graphs)

    targets = _list_targets(dataset_dir)
    if not targets:
        logger.error(f"no targets under {dataset_dir}")
        return 2

    # Collect all work as tasks for cross-target parallelism
    work_items: List[Tuple[Path, Path, Path, Path, NodeConfig, EdgeConfig, TopoConfig, bool, bool, bool, str, str]] = []
    for tdir in targets:
        logger.info(f"[target] {tdir.name}")
        models = _list_models(tdir)
        if not models:
            logger.warning(f"no structures in {tdir}")
            continue
        twork = work_dir / tdir.name
        _ensure_dir(twork)
        for m in models:
            out_sub = out_graphs / tdir.name
            model_stem = _model_name_from_path(m)
            job_logger_name = _job_logger_name(run_logger_name, tdir.name, model_stem)
            worker_log_paths.add(worker_log_dir / f"{job_logger_name}.log")
            work_items.append((
                m, tdir, twork, out_sub,
                node_cfg, edge_cfg, topo_cfg,
                other_cfg.overwrite_graphs,
                args.use_legacy_config,
                args.verify_original,
                job_logger_name,
                str(worker_log_dir),
            ))

    # Determine parallel workers
    workers = args.parallel if args.parallel is not None else int(other_cfg.jobs)
    if not workers or workers <= 1:
        # sequential
        for (m, tdir, twork, out_sub, node_cfg, edge_cfg, topo_cfg,
             overwrite, exact_compat, verify_with_original, _lname, _ldir) in work_items:
            try:
                process_one(
                    pdb_path=m,
                    target_dir=tdir,
                    work_dir=twork,
                    out_graphs=out_sub,
                    node_cfg=node_cfg,
                    edge_cfg=edge_cfg,
                    topo_cfg=topo_cfg,
                    overwrite=overwrite,
                    exact_compat=exact_compat,
                    verify_with_original=verify_with_original,
                    logger=logger,
                )
            except Exception as e:
                logger.exception(f"failed {tdir.name}/{m.name}: {e}")
    else:
        # parallel using processes
        with _fut.ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for target_name, model_name, err in ex.map(_worker_entry, work_items):
                if err:
                    logger.error(f"failed {target_name}/{model_name}: {err}")

    logger.info("[done] graph generation completed")
    existing_worker_logs = sorted([p for p in worker_log_paths if p.exists()])
    if main_log_path and existing_worker_logs:
        logger.info(f"[logs] merging {len(existing_worker_logs)} worker logs")
    _teardown_logger(logger)
    if main_log_path and existing_worker_logs:
        _merge_log_files([main_log_path] + existing_worker_logs, main_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
