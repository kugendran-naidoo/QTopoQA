from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
import time
import traceback

import pandas as pd
import logging
import numpy as np

from ..base import (
    TopologyFeatureModule,
    build_metadata,
    require_bool,
    require_float,
    require_positive_float,
    require_positive_int,
)
from ..registry import register_feature_module
from ...lib import new_topological_features
from ...lib.topology_runner import _load_interface_descriptors

LOG = logging.getLogger("graph_builder")


@register_feature_module
class PersistencePartialKPartiteModule(TopologyFeatureModule):
    """
    Part-aware persistent homology with optional cross-biased/cross-only/per-chain/DSSP/polar blocks.

    Presets:
      - minimal: base PH (140D)
      - lean (default): base + cross-biased + cross-only (~420D)
      - heavy: lean + optional per-chain (cap k=2) + optional DSSP + optional polar mini
    """

    module_id = "topology/persistence_partial_k_partite/v1"
    module_kind = "topology"
    default_alias = "Part-aware PH (minimal/lean/heavy presets; cross-biased + cross-only default) = 420D (lean)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Persistent homology with chain-aware blocks (cross-biased/cross-only) and optional per-chain/DSSP/polar blocks.",
        description=(
            "Extends the basic PH module with part-aware variants: base PH (140D), cross-biased block "
            "(penalize intra-chain distances), cross-only block (inter-chain atoms only), optional per-chain PH "
            "(capped), optional DSSP-stratified block, and optional polar-only mini block. Presets: minimal "
            "(140D), lean (default, ~420D), heavy (adds opt-in blocks)."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "preset": "minimal|lean|heavy (controls which blocks run).",
            "neighbor_distance": "Neighbourhood radius in Å for PH computation.",
            "filtration_cutoff": "Maximum filtration value in Å (applies to β0).",
            "min_persistence": "Minimum persistence threshold for features.",
            "element_filters": "Sequence of element subsets considered for PH statistics.",
            "dedup_sort": "Enable deduplication/sorting of coordinates before PH.",
            "jobs": "Optional override for worker count.",
            # cross-biased
            "enable_cross_bias": "Toggle cross-biased block (intra-chain penalty).",
            "intra_penalty_lambda": "Å penalty added to same-chain pairs in cross-biased block.",
            # cross-only
            "enable_cross_only": "Toggle cross-only block (inter-chain atoms only).",
            # per-chain
            "enable_per_chain": "Toggle per-chain PH blocks.",
            "max_chains": "Cap on per-chain blocks to concatenate.",
            # DSSP
            "enable_dssp_block": "Toggle DSSP-based block (H/E/C classes).",
            # polar
            "enable_polar_block": "Toggle polar-only mini block.",
            "polar_hbond_weight": "Weight polar atoms in DSSP H-bonds lower for polar block.",
            "polar_hbond_weight_factor": "Distance multiplier for H-bonded polar atoms (default=0.5).",
            "polar_hbond_energy_cutoff": "DSSP H-bond energy cutoff (kcal/mol) to consider bonded (default=-0.5).",
            "polar_hbond_inter_only": "If true, only weight inter-chain H-bonded atoms in polar block.",
            "dssp_timeout_seconds": "Per-PDB timeout for DSSP call (skip on exceed).",
            "dssp_slow_threshold": "Log a note if DSSP exceeds this time (seconds).",
            "polar_hbond_weight_factor": "Distance multiplier for H-bonded polar atoms (default=0.5).",
            "polar_hbond_energy_cutoff": "DSSP H-bond energy cutoff (kcal/mol) to consider bonded (default=-0.5).",
            "polar_hbond_inter_only": "If true, only weight inter-chain H-bonded atoms in polar block.",
            "dssp_timeout_seconds": "Per-PDB timeout for DSSP call (skip on exceed).",
            "dssp_slow_threshold": "Log a note if DSSP exceeds this time (seconds).",
            # guardrails
            "max_atoms": "Atom-count cap to skip heavy blocks.",
            "max_block_seconds": "Time cap (s) per heavy block to skip remainder.",
        },
        defaults={
            "preset": "lean",
            "neighbor_distance": 8.0,
            "filtration_cutoff": 8.0,
            "min_persistence": 0.01,
            "element_filters": (
                ("C",),
                ("N",),
                ("O",),
                ("C", "N"),
                ("C", "O"),
                ("N", "O"),
                ("C", "N", "O"),
            ),
            "dedup_sort": False,
            "jobs": 16,
            # cross-biased
            "enable_cross_bias": True,  # lean default
            "intra_penalty_mode": "default",  # default|light|heavy|none|custom
            "intra_penalty_lambda": None,  # used when mode=custom; otherwise derived from neighbor_distance
            # cross-only
            "enable_cross_only": True,  # lean default
            # per-chain
            "enable_per_chain": False,
            "max_chains": 2,
            # DSSP
            "enable_dssp_block": False,
            # polar
            "enable_polar_block": False,
            "polar_hbond_weight": False,
            "polar_hbond_weight_factor": 0.5,
            "polar_hbond_energy_cutoff": -0.5,
            "polar_hbond_inter_only": False,
            "dssp_timeout_seconds": 10.0,
            "dssp_slow_threshold": 2.0,
            "polar_hbond_weight_factor": 0.5,
            "polar_hbond_energy_cutoff": -0.5,
            "polar_hbond_inter_only": False,
            "dssp_timeout_seconds": 10.0,
            "dssp_slow_threshold": 2.0,
            # guardrails
            "max_atoms": 25000,
            "max_block_seconds": 60.0,
        },
    )

    def generate_topology(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        interface_dir: Path,
        work_dir: Path,
        log_dir: Path,
        sort_artifacts: bool = True,
        round_decimals: int | None = None,
        slow_threshold: float = 5.0,
    ):
        pdb_list = list(pdb_paths)
        params = self.params
        preset = (params.get("preset") or "lean").strip().lower()
        neighbor_distance = float(params["neighbor_distance"])
        filtration_cutoff = float(params["filtration_cutoff"])
        min_persistence = float(params["min_persistence"])
        element_filters = params["element_filters"]
        dedup_sort = bool(params.get("dedup_sort", False))
        # Avoid per-residue process pools; keep workers=None for PH
        worker_count = None

        def _flag(value, fallback):
            if value is None:
                return fallback
            return bool(value)

        enable_cross_bias = _flag(params.get("enable_cross_bias"), preset != "minimal")
        enable_cross_only = _flag(params.get("enable_cross_only"), preset != "minimal")
        enable_per_chain = _flag(params.get("enable_per_chain"), preset == "heavy")
        enable_dssp_block = _flag(params.get("enable_dssp_block"), preset == "heavy")
        enable_polar_block = _flag(params.get("enable_polar_block"), preset == "heavy")

        penalty_mode = (params.get("intra_penalty_mode") or "default").strip().lower()
        if penalty_mode not in {"default", "light", "heavy", "none", "custom"}:
            penalty_mode = "default"
        intra_penalty_lambda = params.get("intra_penalty_lambda")
        if penalty_mode == "light":
            penalty_value = neighbor_distance / 4.0
        elif penalty_mode == "heavy":
            penalty_value = neighbor_distance / 1.34
        elif penalty_mode == "none":
            penalty_value = 0.0
        elif penalty_mode == "custom" and intra_penalty_lambda is not None:
            penalty_value = float(intra_penalty_lambda)
        else:
            penalty_value = neighbor_distance / 2.0

        max_chains = int(params.get("max_chains") or 2)
        max_atoms = int(params.get("max_atoms") or 25000)
        max_block_seconds = float(params.get("max_block_seconds") or 60.0)
        polar_hbond_weight = bool(params.get("polar_hbond_weight", False))
        polar_hbond_weight_factor = float(params.get("polar_hbond_weight_factor") or 0.5)
        polar_hbond_energy_cutoff = float(params.get("polar_hbond_energy_cutoff") or -0.5)
        polar_hbond_inter_only = bool(params.get("polar_hbond_inter_only", False))
        dssp_timeout_seconds = float(params.get("dssp_timeout_seconds") or 10.0)
        dssp_slow_threshold = float(params.get("dssp_slow_threshold") or 2.0)

        base_config = new_topological_features.TopologicalConfig(
            neighbor_distance=neighbor_distance,
            filtration_cutoff=filtration_cutoff,
            min_persistence=min_persistence,
            element_filters=element_filters,
            dedup_sort=dedup_sort,
            workers=worker_count,
        )

        dataset_label = dataset_dir.name
        topology_dir = work_dir / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology_log_dir = log_dir / "topology_logs"
        topology_log_dir.mkdir(parents=True, exist_ok=True)

        successes = 0
        failures: List[Tuple[Path, Path, str]] = []
        outputs = []
        start_time = time.perf_counter()
        total = len(pdb_list)
        tasks: List[Tuple[Path, Path, Path, Path]] = []
        for pdb_path in sorted(pdb_list):
            rel = pdb_path.relative_to(dataset_dir) if pdb_path.is_relative_to(dataset_dir) else Path(pdb_path.name)
            iface_path = interface_dir / rel.parent / f"{pdb_path.stem}.interface.txt"
            output_path = topology_dir / rel.parent / f"{pdb_path.stem}.topology.csv"
            log_path = topology_log_dir / rel.parent / f"{pdb_path.stem}.log"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((pdb_path, iface_path, output_path, log_path))

        worker_count = max(1, int(params.get("jobs") or 1))

        def _log_progress():
            elapsed = time.perf_counter() - start_time
            pct = (successes / total) * 100 if total else 100.0
            rate = successes / elapsed if elapsed > 0 else 0.0
            remaining = (total - successes) / rate if rate > 0 else 0.0
            LOG.info(
                "[Topology - %s] %.1f%% complete (%d/%d) ETA %.1f min",
                dataset_label,
                pct,
                successes,
                total,
                remaining / 60.0,
            )

        if worker_count <= 1:
            for pdb_path, iface_path, output_path, log_path in tasks:
                try:
                    self._process_single(
                        pdb_path=pdb_path,
                        interface_path=iface_path,
                        base_config=base_config,
                        enable_cross_bias=enable_cross_bias,
                        enable_cross_only=enable_cross_only,
                        enable_per_chain=enable_per_chain,
                        enable_dssp_block=enable_dssp_block,
                        enable_polar_block=enable_polar_block,
                        penalty_value=penalty_value,
                        intra_penalty_lambda=intra_penalty_lambda,
                        max_chains=max_chains,
                    max_atoms=max_atoms,
                    max_block_seconds=max_block_seconds,
                    polar_hbond_weight=polar_hbond_weight,
                    polar_hbond_weight_factor=polar_hbond_weight_factor,
                    polar_hbond_energy_cutoff=polar_hbond_energy_cutoff,
                    polar_hbond_inter_only=polar_hbond_inter_only,
                    dssp_timeout_seconds=dssp_timeout_seconds,
                    dssp_slow_threshold=dssp_slow_threshold,
                    sort_artifacts=sort_artifacts,
                    round_decimals=round_decimals,
                    slow_threshold=slow_threshold,
                    output_path=output_path,
                    log_path=log_path,
                    )
                    successes += 1
                    if successes % 5 == 0:
                        _log_progress()
                except Exception as exc:  # pragma: no cover
                    failures.append((pdb_path, log_path, str(exc)))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            worker_args = {
                "base_config": base_config,
                "enable_cross_bias": enable_cross_bias,
                "enable_cross_only": enable_cross_only,
                "enable_per_chain": enable_per_chain,
                "enable_dssp_block": enable_dssp_block,
                "enable_polar_block": enable_polar_block,
                "penalty_value": penalty_value,
                "intra_penalty_lambda": intra_penalty_lambda,
                "max_chains": max_chains,
                "max_atoms": max_atoms,
                "max_block_seconds": max_block_seconds,
                "polar_hbond_weight": polar_hbond_weight,
                "polar_hbond_weight_factor": polar_hbond_weight_factor,
                "polar_hbond_energy_cutoff": polar_hbond_energy_cutoff,
                "polar_hbond_inter_only": polar_hbond_inter_only,
                "dssp_timeout_seconds": dssp_timeout_seconds,
                "dssp_slow_threshold": dssp_slow_threshold,
                "sort_artifacts": sort_artifacts,
                "round_decimals": round_decimals,
                "slow_threshold": slow_threshold,
            }

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _process_partial_task,
                        pdb_path,
                        iface_path,
                        output_path,
                        log_path,
                        worker_args,
                    ): (pdb_path, log_path)
                    for pdb_path, iface_path, output_path, log_path in tasks
                }
                for future in as_completed(future_map):
                    pdb_path, log_path = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover
                        failures.append((pdb_path, log_path, str(exc)))
                        continue
                    if result.get("error"):
                        failures.append((pdb_path, log_path, result["error"]))
                    else:
                        successes += 1
                        if successes % 5 == 0:
                            _log_progress()

        elapsed = time.perf_counter() - start_time
        return {
            "output_dir": topology_dir,
            "log_dir": topology_log_dir,
            "success": successes,
            "failures": failures,
            "elapsed": elapsed,
            "processed": len(pdb_list),
        }

    def _process_single(
        self,
        *,
        pdb_path: Path,
        interface_path: Path,
        base_config: new_topological_features.TopologicalConfig,
        enable_cross_bias: bool,
        enable_cross_only: bool,
        enable_per_chain: bool,
        enable_dssp_block: bool,
        enable_polar_block: bool,
        penalty_value: float,
        intra_penalty_lambda: float,
        max_chains: int,
        max_atoms: int,
        max_block_seconds: float,
        polar_hbond_weight: bool,
        polar_hbond_weight_factor: float,
        polar_hbond_energy_cutoff: float,
        polar_hbond_inter_only: bool,
        dssp_timeout_seconds: float,
        dssp_slow_threshold: float,
        sort_artifacts: bool,
        round_decimals: int | None,
        slow_threshold: float,
        output_path: Path,
        log_path: Path,
    ) -> None:
        df, notes = _run_partial_single(
            pdb_path=pdb_path,
            interface_path=interface_path,
            base_config=base_config,
            enable_cross_bias=enable_cross_bias,
            enable_cross_only=enable_cross_only,
            enable_per_chain=enable_per_chain,
            enable_dssp_block=enable_dssp_block,
            enable_polar_block=enable_polar_block,
            penalty_value=penalty_value,
            intra_penalty_lambda=intra_penalty_lambda,
            max_chains=max_chains,
            max_atoms=max_atoms,
            max_block_seconds=max_block_seconds,
            polar_hbond_weight=polar_hbond_weight,
            polar_hbond_weight_factor=polar_hbond_weight_factor,
            polar_hbond_energy_cutoff=polar_hbond_energy_cutoff,
            polar_hbond_inter_only=polar_hbond_inter_only,
            dssp_timeout_seconds=dssp_timeout_seconds,
            dssp_slow_threshold=dssp_slow_threshold,
            slow_threshold=slow_threshold,
        )
        if sort_artifacts and "ID" in df.columns:
            df = df.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
        if round_decimals is not None and round_decimals >= 0:
            num_cols = [c for c in df.columns if c != "ID"]
            df[num_cols] = df[num_cols].round(round_decimals)
        df.to_csv(output_path, index=False)
        log_lines = [f"PDB: {pdb_path}", "Status: SUCCESS", f"Output file: {output_path}", f"Columns: {len(df.columns)-1}"]
        if notes:
            log_lines.append("Notes:")
            log_lines.extend([f"- {note}" for note in notes])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        preset = params.get("preset")
        if preset is not None:
            preset_norm = str(preset).strip().lower()
            if preset_norm not in {"minimal", "lean", "heavy"}:
                raise ValueError("preset must be minimal|lean|heavy.")
            params["preset"] = preset_norm
        for key in ("neighbor_distance", "filtration_cutoff"):
            val = params.get(key)
            if val is not None:
                params[key] = require_positive_float(val, f"topology.params.{key}")
        min_persistence = params.get("min_persistence")
        if min_persistence is not None:
            params["min_persistence"] = require_positive_float(min_persistence, "topology.params.min_persistence", allow_zero=True)
        dedup_sort = params.get("dedup_sort")
        if dedup_sort is not None:
            params["dedup_sort"] = require_bool(dedup_sort, "topology.params.dedup_sort")
        element_filters = params.get("element_filters")
        if element_filters is not None:
            if isinstance(element_filters, str):
                raise ValueError("element_filters must be a list/tuple, not a string.")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")
        for key in ("enable_cross_bias", "enable_cross_only", "enable_per_chain", "enable_dssp_block", "enable_polar_block", "polar_hbond_weight", "polar_hbond_inter_only"):
            val = params.get(key)
            if val is not None:
                params[key] = require_bool(val, f"topology.params.{key}")
        lambda_val = params.get("intra_penalty_lambda")
        if lambda_val is not None:
            params["intra_penalty_lambda"] = require_positive_float(lambda_val, "topology.params.intra_penalty_lambda")
        penalty_mode = params.get("intra_penalty_mode")
        if penalty_mode is not None:
            mode = str(penalty_mode).strip().lower()
            if mode not in {"default", "light", "heavy", "none", "custom"}:
                raise ValueError("topology.params.intra_penalty_mode must be default|light|heavy|none|custom.")
            params["intra_penalty_mode"] = mode
        max_chains = params.get("max_chains")
        if max_chains is not None:
            params["max_chains"] = require_positive_int(max_chains, "topology.params.max_chains")
        max_atoms = params.get("max_atoms")
        if max_atoms is not None:
            params["max_atoms"] = require_positive_int(max_atoms, "topology.params.max_atoms")
        max_block_seconds = params.get("max_block_seconds")
        if max_block_seconds is not None:
            params["max_block_seconds"] = require_positive_float(max_block_seconds, "topology.params.max_block_seconds", allow_zero=True)
        for key in ("polar_hbond_weight_factor", "polar_hbond_energy_cutoff", "dssp_timeout_seconds", "dssp_slow_threshold"):
            val = params.get(key)
            if val is not None:
                params[key] = require_float(val, f"topology.params.{key}")

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        base = super().config_template()
        base["alias"] = cls.default_alias
        params = dict(base.get("params", {}))
        # Make lean defaults explicit for template clarity
        params.update(
            {
                "preset": "lean",
                "enable_cross_bias": True,
                "enable_cross_only": True,
                "enable_per_chain": False,  # default false for lean
                "enable_dssp_block": False,  # default false for lean
                "enable_polar_block": False,  # default false for lean
                "polar_hbond_weight": False,  # default false for lean
            }
        )
        param_comments = {
            "preset": "minimal=140D, lean=base+cross-biased+cross-only (~420D default), heavy=lean+optional per-chain/DSSP/polar",
            "intra_penalty_mode": "default(=nd/2), light(=nd/4), heavy(=nd/1.34), none(=0), custom(use intra_penalty_lambda numeric)",
            "intra_penalty_lambda": "Used only when intra_penalty_mode=custom; otherwise derived from neighbor_distance",
            "max_chains": "Only used when enable_per_chain=true; ignored otherwise",
            "max_atoms": "Guardrail to skip heavy blocks; ignored for minimal/lean",
            "max_block_seconds": "Per-block time cap (s) for heavy blocks; ignored for minimal/lean",
            "polar_hbond_weight": "Toggle H-bond weighting for polar block (heavy only)",
            "polar_hbond_weight_factor": "Distance multiplier for H-bonded polar atoms (default=0.5)",
            "polar_hbond_energy_cutoff": "DSSP H-bond energy cutoff (kcal/mol) to consider bonded (default=-0.5)",
            "polar_hbond_inter_only": "If true, only weight inter-chain H-bonded atoms in polar block",
            "dssp_timeout_seconds": "Per-PDB timeout for DSSP call (skip on exceed)",
            "dssp_slow_threshold": "Log a note if DSSP exceeds this time (seconds)",
        }
        base["params"] = params
        base["param_comments"] = param_comments
        minimal_comments = dict(param_comments)
        minimal_comments["intra_penalty_mode"] = "none for minimal preset"
        minimal_comments.update(
            {
                "polar_hbond_weight": "ignored for minimal preset",
                "polar_hbond_weight_factor": "ignored for minimal preset",
                "polar_hbond_energy_cutoff": "ignored for minimal preset",
                "polar_hbond_inter_only": "ignored for minimal preset",
                "dssp_timeout_seconds": "ignored for minimal preset",
                "dssp_slow_threshold": "ignored for minimal preset",
            }
        )
        alternates = [
            {
                "module": cls.module_id,
                "alias": "Part-aware PH minimal (base only) = 140D",
                "params": {
                    **params,
                    "preset": "minimal",
                    "intra_penalty_mode": "none",  # none for minimal preset
                    "enable_cross_bias": False,
                    "enable_cross_only": False,
                    "enable_per_chain": False,
                    "enable_dssp_block": False,
                    "enable_polar_block": False,
                },
                "param_comments": minimal_comments,
            },
            {
                "module": cls.module_id,
                "alias": "Part-aware PH heavy (lean + optional per-chain/DSSP/polar) ≈ 420D+",
                "params": {
                    **params,
                    "preset": "heavy",
                    "enable_cross_bias": True,
                    "enable_cross_only": True,
                    "enable_per_chain": True,
                    "enable_dssp_block": True,
                    "enable_polar_block": True,
                    "polar_hbond_weight": False,
                },
                "param_comments": param_comments,
            },
        ]
        base["alternates"] = alternates
        return base


def _run_partial_single(
    *,
    pdb_path: Path,
    interface_path: Path,
    base_config: new_topological_features.TopologicalConfig,
    enable_cross_bias: bool,
    enable_cross_only: bool,
    enable_per_chain: bool,
    enable_dssp_block: bool,
    enable_polar_block: bool,
    penalty_value: float,
    intra_penalty_lambda: float,
    max_chains: int,
    max_atoms: int,
    max_block_seconds: float,
    polar_hbond_weight: bool,
    polar_hbond_weight_factor: float,
    polar_hbond_energy_cutoff: float,
    polar_hbond_inter_only: bool,
    dssp_timeout_seconds: float,
    dssp_slow_threshold: float,
    slow_threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    residues, error = _load_interface_descriptors(interface_path)
    if error:
        raise RuntimeError(error)

    structure = new_topological_features._load_structure(pdb_path)  # type: ignore[attr-defined]
    atom_count = sum(1 for _ in structure.get_atoms())
    heavy_allowed = atom_count <= max_atoms
    if not heavy_allowed:
        notes.append(f"Skipped heavy blocks: atom_count {atom_count} > max_atoms {max_atoms}")

    blocks: List[pd.DataFrame] = []
    id_col: List[str] = []

    base_df = new_topological_features.compute_features_for_residues(pdb_path, residues, base_config)
    id_col = base_df["ID"].tolist()
    expected_rows = len(id_col)
    expected_cols = base_df.shape[1] - 1

    id_index = {rid: idx for idx, rid in enumerate(id_col)}

    def _ensure_rows(df: pd.DataFrame, label: str, expected: int) -> pd.DataFrame:
        if len(df) != expected:
            raise RuntimeError(f"{label} row mismatch: expected {expected}, got {len(df)}")
        return df.reset_index(drop=True)

    def _compute_block(label: str, fn, *, expected_rows_override: int | None = None, expected_cols_override: int | str | None = None) -> pd.DataFrame:
        try:
            df = fn()
        except Exception as exc:
            raise RuntimeError(f"{label} block failed: {exc}") from exc
        exp_rows = expected_rows if expected_rows_override is None else expected_rows_override
        if expected_cols_override == "auto":
            exp_cols = df.shape[1] - 1  # drop ID
        else:
            exp_cols = expected_cols if expected_cols_override is None else expected_cols_override
        data = _ensure_rows(df.drop(columns=["ID"]), label, exp_rows)
        if data.shape[1] != exp_cols:
            raise RuntimeError(f"{label} column mismatch: expected {exp_cols}, got {data.shape[1]}")
        return data

    blocks.append(_compute_block("base", lambda: base_df))

    def _timed_block(fn, label: str):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if elapsed > max_block_seconds:
            raise TimeoutError(f"{label} exceeded time cap ({elapsed:.2f}s > {max_block_seconds}s)")
        return result

    if enable_cross_bias:
        def cross_bias_fn():
            cfg = base_config
            return new_topological_features.compute_features_for_residues(
                pdb_path, residues, cfg, bias_mode="intra_penalty", bias_value=penalty_value
            )
        cb_start = time.perf_counter()
        cb_df = _timed_block(lambda: _compute_block("cross_bias", cross_bias_fn), "cross_bias")
        cb_elapsed = time.perf_counter() - cb_start
        if slow_threshold and cb_elapsed > slow_threshold:
            notes.append(f"cross_bias PH slow: {cb_elapsed:.2f}s")
        blocks.append(cb_df)

    if enable_cross_only:
        def cross_only_fn():
            cfg = base_config
            return new_topological_features.compute_features_for_residues(
                pdb_path, residues, cfg, chain_filter="cross_only"
            )
        co_start = time.perf_counter()
        co_df = _timed_block(lambda: _compute_block("cross_only", cross_only_fn), "cross_only")
        co_elapsed = time.perf_counter() - co_start
        if slow_threshold and co_elapsed > slow_threshold:
            notes.append(f"cross_only PH slow: {co_elapsed:.2f}s")
        blocks.append(co_df)

    if enable_per_chain and heavy_allowed:
        chain_map: Dict[str, List[new_topological_features.ResidueDescriptor]] = {}  # type: ignore[attr-defined]
        for res in residues:
            chain_map.setdefault(res.chain_id, []).append(res)
        chain_ids = sorted(chain_map.keys())[:max_chains]
        for chain_id in chain_ids:
            res_list = chain_map[chain_id]
            def chain_fn(res_list=res_list):
                cfg = base_config
                return new_topological_features.compute_features_for_residues(
                    pdb_path, res_list, cfg, chain_filter="within_chain"
                )
            try:
                ch_start = time.perf_counter()
                raw_df = _timed_block(lambda: chain_fn(), f"per_chain_{chain_id}")
                # Align per-chain rows back to full residue ordering
                aligned = np.zeros((expected_rows, expected_cols), dtype=float)
                raw_df = _ensure_rows(raw_df, f"per_chain_{chain_id}", len(res_list))
                feature_vals = raw_df.drop(columns=["ID"]).to_numpy(dtype=float)
                raw_ids = raw_df["ID"].tolist()
                for idx, rid in enumerate(raw_ids):
                    pos = id_index.get(rid)
                    if pos is not None and idx < feature_vals.shape[0]:
                        aligned[pos, :] = feature_vals[idx]
                ch_df = pd.DataFrame(aligned, columns=raw_df.columns.drop("ID"))
                ch_elapsed = time.perf_counter() - ch_start
                if slow_threshold and ch_elapsed > slow_threshold:
                    notes.append(f"per_chain_{chain_id} PH slow: {ch_elapsed:.2f}s")
                blocks.append(ch_df)
            except TimeoutError as exc:
                notes.append(str(exc))
                heavy_allowed = False
                break

    if enable_dssp_block and heavy_allowed:
        def dssp_fn():
            cfg = new_topological_features.TopologicalConfig(
                neighbor_distance=base_config.neighbor_distance,
                filtration_cutoff=base_config.filtration_cutoff,
                min_persistence=base_config.min_persistence,
                element_filters=["all"],
                dedup_sort=base_config.dedup_sort,
                workers=base_config.workers,
            )
            return new_topological_features.compute_features_for_residues(
                pdb_path, residues, cfg, sse_mode=True
                )
        try:
            dssp_start = time.perf_counter()
            dssp_df = _timed_block(lambda: _compute_block("dssp_block", dssp_fn, expected_cols_override="auto"), "dssp_block")
            dssp_elapsed = time.perf_counter() - dssp_start
            if slow_threshold and dssp_elapsed > slow_threshold:
                notes.append(f"dssp_block PH slow: {dssp_elapsed:.2f}s")
            blocks.append(dssp_df)
        except TimeoutError as exc:
            notes.append(str(exc))
            heavy_allowed = False

    if enable_polar_block and heavy_allowed:
        # Optional DSSP-derived H-bond weighting for polar atoms
        hbond_residues: Optional[Set[Tuple[str, int, str]]] = None
        if polar_hbond_weight:
            hbond_residues, dssp_note = _compute_dssp_hbond_residues(
                pdb_path,
                residues,
                energy_cutoff=polar_hbond_energy_cutoff,
                timeout_seconds=dssp_timeout_seconds,
                slow_threshold=dssp_slow_threshold,
            )
            if dssp_note:
                notes.append(dssp_note)
        def polar_fn():
            cfg = new_topological_features.TopologicalConfig(
                neighbor_distance=base_config.neighbor_distance,
                filtration_cutoff=base_config.filtration_cutoff,
                min_persistence=base_config.min_persistence,
                element_filters=[("N",), ("O",)],
                dedup_sort=base_config.dedup_sort,
                workers=base_config.workers,
            )
            return new_topological_features.compute_features_for_residues(
                pdb_path,
                residues,
                cfg,
                polar_mode=True,
                hbond_weight=polar_hbond_weight,
                hbond_residues=hbond_residues,
                hbond_factor=polar_hbond_weight_factor,
                hbond_inter_only=polar_hbond_inter_only,
            )
        try:
            pol_start = time.perf_counter()
            pol_df = _timed_block(lambda: _compute_block("polar_block", polar_fn, expected_cols_override="auto"), "polar_block")
            pol_elapsed = time.perf_counter() - pol_start
            if slow_threshold and pol_elapsed > slow_threshold:
                notes.append(f"polar_block PH slow: {pol_elapsed:.2f}s")
            blocks.append(pol_df)
        except TimeoutError as exc:
            notes.append(str(exc))
            heavy_allowed = False

    try:
        combined = pd.concat(blocks, axis=1)
    except Exception as exc:
        raise RuntimeError(f"Concatenation failed: {exc}") from exc
    combined.insert(0, "ID", id_col)
    return combined, notes


def _compute_dssp_hbond_residues(
    pdb_path: Path,
    residues: List[new_topological_features.ResidueDescriptor],
    *,
    energy_cutoff: float,
    timeout_seconds: float,
    slow_threshold: float,
) -> Tuple[Optional[Set[Tuple[str, int, str]]], Optional[str]]:
    note: Optional[str] = None
    start = time.perf_counter()
    try:
        from Bio.PDB import DSSP, PDBParser  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, f"DSSP skipped: Bio.PDB not available ({exc})"
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", pdb_path)
        model = structure[0]
        dssp = DSSP(model, str(pdb_path))
    except Exception as exc:  # pragma: no cover
        return None, f"DSSP skipped: {exc}"
    elapsed = time.perf_counter() - start
    if elapsed > timeout_seconds:
        return None, f"DSSP skipped: {elapsed:.2f}s exceeded timeout {timeout_seconds:.2f}s"
    if slow_threshold and elapsed > slow_threshold:
        note = f"DSSP slow: {elapsed:.2f}s"

    bonded: Set[Tuple[str, int, str]] = set()
    for res in residues:
        key = (res.chain_id, res.bio_id)
        try:
            entry = dssp[key]
        except KeyError:
            continue
        energies = []
        for idx, val in enumerate(entry):
            if idx >= 5:
                try:
                    energies.append(float(val))
                except Exception:
                    continue
        if any(e <= energy_cutoff for e in energies):
            bonded.add((res.chain_id, res.residue_number, res.insertion_code or " "))
    return bonded if bonded else set(), note


def _process_partial_task(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    log_path: Path,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        df, notes = _run_partial_single(
            pdb_path=pdb_path,
            interface_path=interface_path,
            base_config=options["base_config"],
            enable_cross_bias=options["enable_cross_bias"],
            enable_cross_only=options["enable_cross_only"],
            enable_per_chain=options["enable_per_chain"],
            enable_dssp_block=options["enable_dssp_block"],
            enable_polar_block=options["enable_polar_block"],
            penalty_value=options["penalty_value"],
            intra_penalty_lambda=options["intra_penalty_lambda"],
            max_chains=options["max_chains"],
            max_atoms=options["max_atoms"],
            max_block_seconds=options["max_block_seconds"],
            polar_hbond_weight=options["polar_hbond_weight"],
            polar_hbond_weight_factor=options["polar_hbond_weight_factor"],
            polar_hbond_energy_cutoff=options["polar_hbond_energy_cutoff"],
            polar_hbond_inter_only=options["polar_hbond_inter_only"],
            dssp_timeout_seconds=options["dssp_timeout_seconds"],
            dssp_slow_threshold=options["dssp_slow_threshold"],
            slow_threshold=options["slow_threshold"],
        )
        if options["sort_artifacts"] and "ID" in df.columns:
            df = df.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
        rd = options["round_decimals"]
        if rd is not None and rd >= 0:
            id_col = df["ID"]
            num_df = df.drop(columns=["ID"]).round(rd)
            df = pd.concat([id_col, num_df], axis=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log_lines = [f"PDB: {pdb_path}", "Status: SUCCESS", f"Output file: {output_path}", f"Columns: {len(df.columns)-1}"]
        if notes:
            log_lines.append("Notes:")
            log_lines.extend([f"- {note}" for note in notes])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {"error": None}
    except Exception as exc:  # pragma: no cover
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        log_path.write_text(
            f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\nTraceback:\n{tb}\n",
            encoding="utf-8",
        )
        return {"error": str(exc)}
