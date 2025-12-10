from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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


def _chemotype(res_name: str) -> str:
    r = (res_name or "").upper()
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
    polar = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
    positive = {"LYS", "ARG", "HIS"}
    negative = {"ASP", "GLU"}
    if r in hydrophobic:
        return "HYDROPHOBIC"
    if r in polar:
        return "POLAR"
    if r in positive:
        return "POSITIVE"
    if r in negative:
        return "NEGATIVE"
    return "OTHER"


def _typed_class(res_name: str) -> str:
    """Rudimentary residue typing for typed-contact blocks."""
    t = _chemotype(res_name)
    if t in {"POSITIVE", "NEGATIVE"}:
        return "SALT"
    if t == "POLAR":
        return "HBOND"
    if t == "HYDROPHOBIC":
        return "HYDROPHOBIC"
    return "OTHER"


@register_feature_module
class PersistenceKPartiteAdvancedModule(TopologyFeatureModule):
    """
    Advanced k-partite PH with optional chemistry-aware blocks (typed contacts, weighted filtration toggles, power/landmark stubs).

    Presets:
      - minimal: base PH only (140D)
      - lean: base + cross-only per primary pair + per-primary
      - heavy: lean + cross-bias + polar H-bond + typed-contact
      - heavy_stratified: heavy + secondary stratification (chemotype default; DSSP optional)
      - rich (optional): heavy_stratified + typed-contact + weighted toggle; power/landmark flags available
    """

    module_id = "topology/persistence_k_partite_advanced/v1"
    module_kind = "topology"
    default_alias = "Advanced k-partite PH with typed contacts"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Advanced k-partite persistent homology with optional typed-contact and weighted/polar blocks.",
        description=(
            "Extends the full k-partite module with optional typed-contact blocks and hooks for weighted filtrations, "
            "power/landmark modes. Primary partitions are chains; optional secondary stratification (chemotype/DSSP). "
            "Presets keep backward compatibility while allowing richer chemistry-aware variants."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "preset": "minimal|lean|heavy|heavy_stratified",
            "neighbor_distance": "Neighbourhood radius in Å for PH computation.",
            "filtration_cutoff": "Maximum filtration value in Å (applies to β0).",
            "min_persistence": "Minimum persistence threshold for features.",
            "element_filters": "Sequence of element subsets considered for PH statistics (primary blocks).",
            "element_filters_stratified": "Element filters for stratified blocks (reduced).",
            "dedup_sort": "Enable deduplication/sorting of coordinates before PH.",
            "jobs": "Optional override for worker count.",
            "secondary_partition": "none|dssp|chemotype (mutually exclusive).",
            "k_max": "Cap on number of primary partitions.",
            "secondary_k_max": "Cap on number of secondary partitions.",
            # bias
            "enable_cross_bias": "Toggle cross-biased block (intra-primary penalty).",
            "intra_penalty_mode": "default|light|heavy|none|custom",
            "intra_penalty_lambda": "Å penalty when mode=custom.",
            # guardrails
            "max_atoms": "Atom-count cap to skip heavy blocks.",
            "max_block_seconds": "Time cap (s) per heavy block to skip remainder.",
            # polar mini
            "enable_polar_block": "Toggle polar-only mini block.",
            "polar_hbond_weight": "Weight polar atoms in DSSP H-bonds lower for polar block.",
            "polar_hbond_weight_factor": "Distance multiplier for H-bonded polar atoms (default=0.5).",
            "polar_hbond_energy_cutoff": "DSSP H-bond energy cutoff (kcal/mol) to consider bonded (default=-0.5).",
            "polar_hbond_inter_only": "If true, only weight inter-chain H-bonded atoms in polar block.",
            # typed block
            "enable_typed_block": "Enable typed-contact PH blocks (hydrophobic/salt/H-bond).",
            # dssp stratified
            "dssp_timeout_seconds": "Per-PDB timeout for DSSP call (skip if exceeded).",
            "dssp_slow_threshold": "Log a note if DSSP exceeds this time (seconds).",
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
            "element_filters_stratified": (("C", "N", "O"),),
            "dedup_sort": False,
            "jobs": 16,
            "secondary_partition": "none",
            "k_max": 4,
            "secondary_k_max": 4,
            "enable_cross_bias": False,
            "intra_penalty_mode": "default",
            "intra_penalty_lambda": None,
            "max_atoms": 25000,
            "max_block_seconds": 60.0,
            "enable_polar_block": False,
            "polar_hbond_weight": False,
            "polar_hbond_weight_factor": 0.5,
            "polar_hbond_energy_cutoff": -0.5,
            "polar_hbond_inter_only": False,
            "enable_typed_block": False,
            "dssp_timeout_seconds": 10.0,
            "dssp_slow_threshold": 2.0,
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
        element_filters_strat = params.get("element_filters_stratified") or (("C", "N", "O"),)
        dedup_sort = bool(params.get("dedup_sort", False))
        worker_count = None

        secondary_partition = (params.get("secondary_partition") or "none").strip().lower()
        if secondary_partition not in {"none", "dssp", "chemotype"}:
            secondary_partition = "none"

        k_max = int(params.get("k_max") or 4)
        secondary_k_max = int(params.get("secondary_k_max") or 4)

        def _flag(value, fallback):
            if value is None:
                return fallback
            return bool(value)

        enable_cross_bias = _flag(params.get("enable_cross_bias"), preset in {"heavy", "heavy_stratified"})
        enable_polar_block = _flag(params.get("enable_polar_block"), preset in {"heavy", "heavy_stratified"})
        enable_typed_block = _flag(params.get("enable_typed_block"), preset in {"heavy", "heavy_stratified"})
        enable_weighted_filtration = False
        enable_power_block = False
        enable_landmark_mode = False

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

        strat_config = new_topological_features.TopologicalConfig(
            neighbor_distance=neighbor_distance,
            filtration_cutoff=filtration_cutoff,
            min_persistence=min_persistence,
            element_filters=element_filters_strat,
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
        start_time = time.perf_counter()
        total = len(pdb_list)

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
        if worker_count <= 1:
            for pdb_path, iface_path, output_path, log_path in tasks:
                try:
                    df, notes = self._process_single(
                        pdb_path=pdb_path,
                        interface_path=iface_path,
                        base_config=base_config,
                        strat_config=strat_config,
                        preset=preset,
                        secondary_partition=secondary_partition,
                        k_max=k_max,
                        secondary_k_max=secondary_k_max,
                        enable_cross_bias=enable_cross_bias,
                        penalty_value=penalty_value,
                        max_atoms=max_atoms,
                        max_block_seconds=max_block_seconds,
                        enable_polar_block=enable_polar_block,
                        polar_hbond_weight=polar_hbond_weight,
                        polar_hbond_weight_factor=polar_hbond_weight_factor,
                        polar_hbond_energy_cutoff=polar_hbond_energy_cutoff,
                        polar_hbond_inter_only=polar_hbond_inter_only,
                        enable_typed_block=enable_typed_block,
                        slow_threshold=slow_threshold,
                        dssp_timeout_seconds=dssp_timeout_seconds,
                        dssp_slow_threshold=dssp_slow_threshold,
                    )
                    if sort_artifacts and "ID" in df.columns:
                        df = df.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
                    if round_decimals is not None and round_decimals >= 0:
                        id_col = df["ID"]
                        num_df = df.drop(columns=["ID"]).round(round_decimals)
                        df = pd.concat([id_col, num_df], axis=1)
                    df.to_csv(output_path, index=False)
                    log_lines = [f"PDB: {pdb_path}", "Status: SUCCESS", f"Output file: {output_path}", f"Columns: {len(df.columns)-1}"]
                    if notes:
                        log_lines.append("Notes:")
                        log_lines.extend([f"- {note}" for note in notes])
                    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
                    successes += 1
                    if successes % 5 == 0:
                        _log_progress()
                except Exception as exc:  # pragma: no cover
                    failures.append((pdb_path, log_path, str(exc)))
                    log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            worker_args = {
                "base_config": base_config,
                "strat_config": strat_config,
                "preset": preset,
                "secondary_partition": secondary_partition,
                "k_max": k_max,
                "secondary_k_max": secondary_k_max,
                "enable_cross_bias": enable_cross_bias,
                "penalty_value": penalty_value,
                "max_atoms": max_atoms,
                "max_block_seconds": max_block_seconds,
                "enable_polar_block": enable_polar_block,
                "polar_hbond_weight": polar_hbond_weight,
                "polar_hbond_weight_factor": polar_hbond_weight_factor,
                "polar_hbond_energy_cutoff": polar_hbond_energy_cutoff,
                "polar_hbond_inter_only": polar_hbond_inter_only,
                "enable_typed_block": enable_typed_block,
                "slow_threshold": slow_threshold,
                "dssp_timeout_seconds": dssp_timeout_seconds,
                "dssp_slow_threshold": dssp_slow_threshold,
                "sort_artifacts": sort_artifacts,
                "round_decimals": round_decimals,
            }

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _process_adv_task,
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
        strat_config: new_topological_features.TopologicalConfig,
        preset: str,
        secondary_partition: str,
        k_max: int,
        secondary_k_max: int,
        enable_cross_bias: bool,
        penalty_value: float,
        max_atoms: int,
        max_block_seconds: float,
        enable_polar_block: bool,
        polar_hbond_weight: bool,
    polar_hbond_weight_factor: float,
    polar_hbond_energy_cutoff: float,
    polar_hbond_inter_only: bool,
    enable_typed_block: bool,
    slow_threshold: float,
    dssp_timeout_seconds: float,
    dssp_slow_threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
        return _run_adv_single(
            pdb_path=pdb_path,
            interface_path=interface_path,
            base_config=base_config,
            strat_config=strat_config,
            preset=preset,
            secondary_partition=secondary_partition,
            k_max=k_max,
            secondary_k_max=secondary_k_max,
            enable_cross_bias=enable_cross_bias,
            penalty_value=penalty_value,
            max_atoms=max_atoms,
            max_block_seconds=max_block_seconds,
            enable_polar_block=enable_polar_block,
            polar_hbond_weight=polar_hbond_weight,
            polar_hbond_weight_factor=polar_hbond_weight_factor,
            polar_hbond_energy_cutoff=polar_hbond_energy_cutoff,
            polar_hbond_inter_only=polar_hbond_inter_only,
            enable_typed_block=enable_typed_block,
            enable_weighted_filtration=enable_weighted_filtration,
            enable_power_block=enable_power_block,
            enable_landmark_mode=enable_landmark_mode,
            slow_threshold=slow_threshold,
            dssp_timeout_seconds=dssp_timeout_seconds,
            dssp_slow_threshold=dssp_slow_threshold,
        )

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        preset = params.get("preset")
        if preset is not None:
            preset_norm = str(preset).strip().lower()
            if preset_norm not in {"minimal", "lean", "heavy", "heavy_stratified"}:
                raise ValueError("preset must be minimal|lean|heavy|heavy_stratified.")
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
        if element_filters is not None and isinstance(element_filters, str):
            raise ValueError("element_filters must be a list/tuple, not a string.")
        element_filters_strat = params.get("element_filters_stratified")
        if element_filters_strat is not None and isinstance(element_filters_strat, str):
            raise ValueError("element_filters_stratified must be a list/tuple, not a string.")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")
        for key in (
            "enable_cross_bias",
            "enable_polar_block",
            "polar_hbond_weight",
            "polar_hbond_inter_only",
            "enable_typed_block",
        ):
            val = params.get(key)
            if val is not None:
                params[key] = require_bool(val, f"topology.params.{key}")
        for key in ("k_max", "secondary_k_max", "max_atoms"):
            val = params.get(key)
            if val is not None:
                params[key] = require_positive_int(val, f"topology.params.{key}")
        max_block_seconds = params.get("max_block_seconds")
        if max_block_seconds is not None:
            params["max_block_seconds"] = require_positive_float(max_block_seconds, "topology.params.max_block_seconds", allow_zero=True)
        dssp_timeout = params.get("dssp_timeout_seconds")
        if dssp_timeout is not None:
            params["dssp_timeout_seconds"] = require_positive_float(dssp_timeout, "topology.params.dssp_timeout_seconds", allow_zero=True)
        dssp_slow = params.get("dssp_slow_threshold")
        if dssp_slow is not None:
            params["dssp_slow_threshold"] = require_positive_float(dssp_slow, "topology.params.dssp_slow_threshold", allow_zero=True)
        lambda_val = params.get("intra_penalty_lambda")
        if lambda_val is not None:
            params["intra_penalty_lambda"] = require_positive_float(lambda_val, "topology.params.intra_penalty_lambda")
        for key in ("polar_hbond_weight_factor", "polar_hbond_energy_cutoff"):
            val = params.get(key)
            if val is not None:
                params[key] = require_float(val, f"topology.params.{key}")
        penalty_mode = params.get("intra_penalty_mode")
        if penalty_mode is not None:
            mode = str(penalty_mode).strip().lower()
            if mode not in {"default", "light", "heavy", "none", "custom"}:
                raise ValueError("topology.params.intra_penalty_mode must be default|light|heavy|none|custom.")
            params["intra_penalty_mode"] = mode
        secondary = params.get("secondary_partition")
        if secondary is not None:
            sec = str(secondary).strip().lower()
            if sec not in {"none", "dssp", "chemotype"}:
                raise ValueError("secondary_partition must be none|dssp|chemotype.")
            params["secondary_partition"] = sec

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        base = super().config_template()
        base["alias"] = "Advanced k-partite PH (lean) = 560D+"
        params = dict(base.get("params", {}))
        params.update(
            {
                "preset": "lean",
                "secondary_partition": "none",
                "k_max": 4,
                "secondary_k_max": 4,
                "enable_cross_bias": False,
                "enable_polar_block": False,
                "enable_typed_block": False,
            }
        )
        param_comments = {
            "preset": "minimal=140D, lean=base+cross-pair+per-partition, heavy=~700D (bias+polar+typed, no secondary), heavy_stratified=~900D (adds secondary strat)",
            "secondary_partition": "none|dssp|chemotype (ignored by minimal/lean/heavy; used by heavy_stratified)",
            "k_max": "Cap on number of primary partitions; blocks beyond cap are skipped (lean/heavy/stratified)",
            "secondary_k_max": "Cap on number of secondary partitions (stratified only); ignored otherwise",
            "element_filters_stratified": "Reduced filters for stratified blocks (default C/N/O); used only in heavy_stratified",
            "enable_cross_bias": "Used in heavy/heavy_stratified; ignored by lean/minimal even if true",
            "intra_penalty_mode": "default(=nd/2), light(=nd/4), heavy(=nd/1.34), none(=0), custom(use intra_penalty_lambda numeric); ignored by lean/minimal",
            "intra_penalty_lambda": "Used only when intra_penalty_mode=custom; otherwise derived from neighbor_distance; heavy/heavy_stratified only",
            "max_atoms": "Guardrail to skip heavy blocks; ignored by lean/minimal",
            "max_block_seconds": "Per-block time cap (s) for lean/heavy/stratified; ignored by minimal",
            "enable_polar_block": "Used in heavy/heavy_stratified; ignored by lean/minimal",
            "polar_hbond_weight": "Toggle H-bond weighting for polar block (heavy/heavy_stratified); ignored by lean/minimal",
            "polar_hbond_weight_factor": "Distance multiplier for H-bonded polar atoms (default=0.5); heavy/heavy_stratified",
            "polar_hbond_energy_cutoff": "DSSP H-bond energy cutoff (kcal/mol) (default=-0.5); heavy/heavy_stratified",
            "polar_hbond_inter_only": "If true, only weight inter-chain H-bonded atoms; heavy/heavy_stratified",
            "enable_typed_block": "Enable typed-contact PH (hydrophobic/salt/H-bond); heavy/heavy_stratified default ON",
            "dssp_timeout_seconds": "Per-PDB timeout for DSSP call (skip if exceeded).",
            "dssp_slow_threshold": "Log a note if DSSP exceeds this time (seconds).",
        }
        base["params"] = params
        base["param_comments"] = param_comments
        minimal_comments = dict(param_comments)
        minimal_comments.update(
            {
                "secondary_partition": "ignored for minimal preset",
                "k_max": "ignored for minimal preset",
                "secondary_k_max": "ignored for minimal preset",
                "enable_cross_bias": "ignored for minimal preset",
                "intra_penalty_mode": "ignored for minimal preset",
                "intra_penalty_lambda": "ignored for minimal preset",
                "max_atoms": "ignored for minimal preset",
                "max_block_seconds": "ignored for minimal preset",
                "enable_polar_block": "ignored for minimal preset",
                "polar_hbond_weight": "ignored for minimal preset",
                "polar_hbond_weight_factor": "ignored for minimal preset",
                "polar_hbond_energy_cutoff": "ignored for minimal preset",
                "polar_hbond_inter_only": "ignored for minimal preset",
                "enable_typed_block": "ignored for minimal preset",
                "preset": "standard topoqa = 140D",
            }
        )
        alternates = [
            {
                "module": cls.module_id,
                "alias": "Advanced k-partite PH minimal (base only) = 140D",
                "params": {**params, "preset": "minimal", "enable_cross_bias": False, "enable_polar_block": False, "enable_typed_block": False},
                "param_comments": minimal_comments,
            },
            {
                "module": cls.module_id,
                "alias": "Advanced k-partite PH (heavy) ≈ 700D (bias+polar+typed)",
                "params": {
                    **params,
                    "preset": "heavy",
                    "enable_cross_bias": True,
                    "enable_polar_block": True,
                    "polar_hbond_weight": True,
                    "enable_typed_block": True,
                },
                "param_comments": param_comments,
            },
            {
                "module": cls.module_id,
                "alias": "Advanced k-partite PH (heavy stratified) ≈ 900D (chemotype + polar + typed)",
                "params": {
                    **params,
                    "preset": "heavy_stratified",
                    "secondary_partition": "chemotype",
                    "enable_cross_bias": True,
                    "enable_polar_block": True,
                    "polar_hbond_weight": True,
                    "enable_typed_block": True,
                },
                "param_comments": param_comments,
            },
        ]
        base["alternates"] = alternates
        return base


def _run_adv_single(
    *,
    pdb_path: Path,
    interface_path: Path,
    base_config: new_topological_features.TopologicalConfig,
    strat_config: new_topological_features.TopologicalConfig,
    preset: str,
    secondary_partition: str,
    k_max: int,
    secondary_k_max: int,
    enable_cross_bias: bool,
    penalty_value: float,
    max_atoms: int,
    max_block_seconds: float,
    enable_polar_block: bool,
    polar_hbond_weight: bool,
    polar_hbond_weight_factor: float,
    polar_hbond_energy_cutoff: float,
    polar_hbond_inter_only: bool,
    enable_typed_block: bool,
    slow_threshold: float,
    dssp_timeout_seconds: float,
    dssp_slow_threshold: float,
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

    id_list = [res.raw_descriptor for res in residues]
    expected_rows = len(id_list)
    id_index = {rid: idx for idx, rid in enumerate(id_list)}
    base_df = new_topological_features.compute_features_for_residues(pdb_path, residues, base_config)
    expected_cols = base_df.shape[1] - 1

    blocks: List[pd.DataFrame] = []
    blocks.append(base_df.drop(columns=["ID"]))

    def _timed(fn, label):
        start = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - start
        if elapsed > max_block_seconds:
            raise TimeoutError(f"{label} exceeded time cap ({elapsed:.2f}s > {max_block_seconds}s)")
        if slow_threshold and elapsed > slow_threshold:
            notes.append(f"{label} PH slow: {elapsed:.2f}s")
        return out

    prim_map: Dict[str, List[new_topological_features.ResidueDescriptor]] = {}
    for res in residues:
        prim_map.setdefault(res.chain_id, []).append(res)
    primary_labels = sorted(prim_map.keys())[:k_max]

    def _align_block(raw_df: pd.DataFrame, label: str) -> pd.DataFrame:
        aligned = np.zeros((expected_rows, raw_df.shape[1] - 1), dtype=float)
        raw_ids = raw_df["ID"].tolist()
        features = raw_df.drop(columns=["ID"]).to_numpy(dtype=float)
        for idx, rid in enumerate(raw_ids):
            pos = id_index.get(rid)
            if pos is not None and idx < features.shape[0]:
                aligned[pos, :] = features[idx]
        return pd.DataFrame(aligned, columns=raw_df.columns.drop("ID"))

    if preset in {"lean", "heavy", "heavy_stratified"}:
        for a, b in itertools.combinations(primary_labels, 2):
            pair_res = prim_map.get(a, []) + prim_map.get(b, [])
            if not pair_res:
                continue

            def pair_fn(res_list=pair_res):
                return new_topological_features.compute_features_for_residues(pdb_path, res_list, base_config)

            raw_df = _timed(pair_fn, f"cross_only_{a}_{b}")
            blocks.append(_align_block(raw_df, f"cross_only_{a}_{b}"))

    if preset in {"lean", "heavy", "heavy_stratified"}:
        for a in primary_labels:
            res_list = prim_map.get(a, [])
            if not res_list:
                continue

            def per_fn(res_list=res_list):
                return new_topological_features.compute_features_for_residues(pdb_path, res_list, base_config)

            raw_df = _timed(per_fn, f"per_primary_{a}")
            blocks.append(_align_block(raw_df, f"per_primary_{a}"))

    if preset in {"heavy", "heavy_stratified"} and enable_cross_bias:
        def bias_fn():
            return new_topological_features.compute_features_for_residues(
                pdb_path, residues, base_config, bias_mode="intra_penalty", bias_value=penalty_value
            )

        raw_df = _timed(bias_fn, "cross_bias")
        blocks.append(raw_df.drop(columns=["ID"]))

    if preset == "heavy_stratified":
        if secondary_partition != "none" and heavy_allowed:
            dssp_labels: Dict[Tuple[str, int, str], str] = {}
            if secondary_partition == "dssp":
                dssp_labels = _compute_dssp_labels(
                    pdb_path,
                    residues,
                    timeout_seconds=dssp_timeout_seconds,
                    slow_threshold=dssp_slow_threshold,
                    notes=notes,
                )
                if not dssp_labels:
                    notes.append("DSSP labels unavailable; skipping stratified blocks")
            combined_map: Dict[str, List[new_topological_features.ResidueDescriptor]] = {}
            for res in residues:
                sec_label = None
                if secondary_partition == "chemotype":
                    sec_label = _chemotype(res.residue_name or "")
                elif secondary_partition == "dssp":
                    sec_label = dssp_labels.get((res.chain_id, res.residue_number, res.insertion_code or " "))
                if sec_label is None:
                    continue
                label = f"{res.chain_id}:{sec_label}"
                combined_map.setdefault(label, []).append(res)
            combined_labels = sorted(combined_map.keys())[:secondary_k_max]
            for lbl in combined_labels:
                res_list = combined_map.get(lbl, [])
                if not res_list:
                    continue

                def strat_fn(res_list=res_list):
                    return new_topological_features.compute_features_for_residues(pdb_path, res_list, strat_config)

                raw_df = _timed(strat_fn, f"strat_{lbl}")
                blocks.append(_align_block(raw_df, f"strat_{lbl}"))
            for a, b in itertools.combinations(combined_labels, 2):
                pair_res = combined_map.get(a, []) + combined_map.get(b, [])
                if not pair_res:
                    continue

                def strat_pair_fn(res_list=pair_res):
                    return new_topological_features.compute_features_for_residues(pdb_path, res_list, strat_config)

                raw_df = _timed(strat_pair_fn, f"strat_pair_{a}_{b}")
                blocks.append(_align_block(raw_df, f"strat_pair_{a}_{b}"))

    if preset in {"heavy", "heavy_stratified"} and enable_polar_block and heavy_allowed:
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

        raw_df = _timed(polar_fn, "polar_block")
        blocks.append(raw_df.drop(columns=["ID"]))

    if preset in {"heavy", "heavy_stratified"} and enable_typed_block and heavy_allowed:
        type_map: Dict[str, List[new_topological_features.ResidueDescriptor]] = {}
        for res in residues:
            t = _typed_class(res.residue_name or "")
            type_map.setdefault(t, []).append(res)
        type_labels = [t for t in ["HYDROPHOBIC", "SALT", "HBOND"] if t in type_map]
        for t in type_labels:
            res_list = type_map.get(t, [])
            if not res_list:
                continue

            def typed_fn(res_list=res_list):
                return new_topological_features.compute_features_for_residues(pdb_path, res_list, base_config)

            raw_df = _timed(typed_fn, f"typed_{t}")
            blocks.append(_align_block(raw_df, f"typed_{t}"))
        for a, b in itertools.combinations(type_labels, 2):
            pair_res = type_map.get(a, []) + type_map.get(b, [])
            if not pair_res:
                continue

            def typed_pair_fn(res_list=pair_res):
                return new_topological_features.compute_features_for_residues(pdb_path, res_list, base_config)

            raw_df = _timed(typed_pair_fn, f"typed_pair_{a}_{b}")
            blocks.append(_align_block(raw_df, f"typed_pair_{a}_{b}"))

    combined = pd.concat(blocks, axis=1)
    combined.insert(0, "ID", id_list)
    return combined, notes


def _compute_dssp_labels(
    pdb_path: Path,
    residues: List[new_topological_features.ResidueDescriptor],
    *,
    timeout_seconds: float,
    slow_threshold: float,
    notes: List[str],
) -> Dict[Tuple[str, int, str], str]:
    start = time.perf_counter()
    try:
        try:
            from Bio.PDB import DSSP, PDBParser  # type: ignore
        except Exception as exc:  # pragma: no cover
            notes.append(f"DSSP skipped: Bio.PDB not available ({exc})")
            return {}
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", pdb_path)
        model = structure[0]
        dssp = DSSP(model, str(pdb_path))
    except Exception as exc:  # pragma: no cover
        notes.append(f"DSSP skipped: {exc}")
        return {}
    elapsed = time.perf_counter() - start
    if elapsed > timeout_seconds:
        notes.append(f"DSSP skipped: {elapsed:.2f}s exceeded timeout {timeout_seconds:.2f}s")
        return {}
    if slow_threshold and elapsed > slow_threshold:
        notes.append(f"DSSP slow: {elapsed:.2f}s")

    def _simplify(ss: str) -> str:
        if ss in {"H", "G", "I"}:
            return "H"
        if ss in {"E", "B"}:
            return "E"
        return "C"

    labels: Dict[Tuple[str, int, str], str] = {}
    for res in residues:
        key = (res.chain_id, res.bio_id)
        try:
            entry = dssp[key]
        except KeyError:
            continue
        ss = entry[2] if isinstance(entry, tuple) and len(entry) > 2 else entry.secondary_structure  # type: ignore[attr-defined]
        labels[(res.chain_id, res.residue_number, res.insertion_code or " ")] = _simplify(ss or "C")
    return labels


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

    hbond_res: Set[Tuple[str, int, str]] = set()
    for res in residues:
        key = (res.chain_id, res.bio_id)
        try:
            entry = dssp[key]
        except KeyError:
            continue
        hbonds = []
        if isinstance(entry, tuple):
            hbonds = entry[3:7]
        else:
            hbonds = [
                getattr(entry, "hbonds_1H", None),
                getattr(entry, "hbonds_1O", None),
                getattr(entry, "hbonds_2H", None),
                getattr(entry, "hbonds_2O", None),
            ]
        for hb in hbonds:
            if not hb or not isinstance(hb, tuple) or len(hb) < 2:
                continue
            energy = hb[1]
            if energy is None:
                continue
            try:
                energy_val = float(energy)
            except Exception:
                continue
            if energy_val <= energy_cutoff:
                hbond_res.add((res.chain_id, res.residue_number, res.insertion_code or " "))
                break
    return hbond_res, note


def _process_adv_task(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    log_path: Path,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        df, notes = _run_adv_single(
            pdb_path=pdb_path,
            interface_path=interface_path,
            base_config=options["base_config"],
            strat_config=options["strat_config"],
            preset=options["preset"],
            secondary_partition=options["secondary_partition"],
            k_max=options["k_max"],
            secondary_k_max=options["secondary_k_max"],
            enable_cross_bias=options["enable_cross_bias"],
            penalty_value=options["penalty_value"],
            max_atoms=options["max_atoms"],
            max_block_seconds=options["max_block_seconds"],
            enable_polar_block=options["enable_polar_block"],
            polar_hbond_weight=options["polar_hbond_weight"],
            polar_hbond_weight_factor=options["polar_hbond_weight_factor"],
            polar_hbond_energy_cutoff=options["polar_hbond_energy_cutoff"],
            polar_hbond_inter_only=options["polar_hbond_inter_only"],
            enable_typed_block=options["enable_typed_block"],
            slow_threshold=options["slow_threshold"],
            dssp_timeout_seconds=options["dssp_timeout_seconds"],
            dssp_slow_threshold=options["dssp_slow_threshold"],
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
        log_path.write_text(
            f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n",
            encoding="utf-8",
        )
        return {"error": str(exc)}
