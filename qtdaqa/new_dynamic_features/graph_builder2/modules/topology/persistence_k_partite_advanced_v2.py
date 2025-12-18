from __future__ import annotations

import itertools
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

PH_FEATURES_PER_FILTER = 20
SLOT_MISSINGNESS_DIM = 9  # present + n_res + log1p_n_res + 6 reason one-hots

REASON_FIELDS: Tuple[str, ...] = (
    "reason_present",
    "reason_missing_slot",
    "reason_empty_subset",
    "reason_guardrail_skip",
    "reason_timeout",
    "reason_error",
)


def _chemotype4(res_name: str) -> str:
    r = (res_name or "").upper()
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
    polar = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
    positive = {"LYS", "ARG", "HIS"}
    negative = {"ASP", "GLU"}
    if r in positive:
        return "POS"
    if r in negative:
        return "NEG"
    if r in polar:
        return "POLAR"
    if r in hydrophobic:
        return "HYDRO"
    return "POLAR"


def _typed_class(res_name: str) -> str:
    """Rudimentary residue typing for typed-contact blocks (fixed 3-class set)."""
    t = _chemotype4(res_name)
    if t in {"POS", "NEG"}:
        return "SALT"
    if t == "POLAR":
        return "HBOND"
    return "HYDROPHOBIC"


def _ph_dim_from_filters(element_filters: Sequence[Sequence[str]]) -> int:
    return PH_FEATURES_PER_FILTER * len(element_filters)


@dataclass(frozen=True)
class _SlotSpec:
    slot_id: str
    feature_prefix: str
    feature_cols: Tuple[str, ...]


def _reason_onehot(reason: str) -> Dict[str, float]:
    return {field: float(field == reason) for field in REASON_FIELDS}


def _prefix_columns(prefix: str, cols: Sequence[str]) -> List[str]:
    return [f"{prefix}{col}" for col in cols]


def _make_missingness_frame(
    *,
    slot_prefix: str,
    n_rows: int,
    present: float,
    n_residues: int,
    reason: str,
) -> pd.DataFrame:
    payload: Dict[str, float] = {
        "present": float(present),
        "n_residues": float(n_residues),
        "log1p_n_residues": float(math.log1p(n_residues)),
    }
    payload.update(_reason_onehot(reason))
    cols = _prefix_columns(slot_prefix, payload.keys())
    values = np.tile(np.array(list(payload.values()), dtype=float), (n_rows, 1))
    return pd.DataFrame(values, columns=cols)


def _align_block(
    raw_df: pd.DataFrame,
    *,
    expected_ids: Sequence[str],
    id_to_index: Dict[str, int],
    numeric_cols: Sequence[str],
) -> np.ndarray:
    aligned = np.zeros((len(expected_ids), len(numeric_cols)), dtype=float)
    raw_ids = raw_df["ID"].tolist()
    values = raw_df[list(numeric_cols)].to_numpy(dtype=float)
    for idx, rid in enumerate(raw_ids):
        pos = id_to_index.get(rid)
        if pos is not None and idx < values.shape[0]:
            aligned[pos, :] = values[idx]
    return aligned


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


@register_feature_module
class PersistenceKPartiteAdvancedV2Module(TopologyFeatureModule):
    """
    v2: fixed-width k-partite PH with always-on missingness channels per block slot.

    This module exists to prevent per-structure topology schema drift within a run by:
      - defining a fixed inventory of block slots (base + per-primary + per-primary-pair + optional blocks)
      - padding missing slots with zeros
      - namespacing all columns by block identifier
      - appending slot-level missingness channels to disambiguate zeros from padding
    """

    module_id = "topology/persistence_k_partite_advanced/v2"
    module_kind = "topology"
    default_alias = "Advanced k-partite PH (v2 fixed-width schema; preset-dependent)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Fixed-width k-partite persistent homology with slot padding + missingness channels.",
        description=(
            "Produces a fixed-width topology CSV per structure by padding missing chain/pair/typed/stratified slots "
            "up to configured caps and prefixing every feature column with a block identifier. Always appends "
            "slot-level missingness channels (present/size/reason) so padded zeros are not ambiguous."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "preset": "minimal|lean|heavy|heavy_stratified",
            "neighbor_distance": "Neighbourhood radius in Å for PH computation.",
            "filtration_cutoff": "Maximum filtration value in Å.",
            "min_persistence": "Minimum persistence threshold for features.",
            "element_filters": "Sequence of element subsets considered for PH statistics (primary blocks).",
            "element_filters_stratified": "Element filters for stratified blocks (reduced).",
            "dedup_sort": "Enable deduplication/sorting of coordinates before PH.",
            "jobs": "Optional override for worker count.",
            "secondary_partition": "none|dssp|chemotype (mutually exclusive).",
            "k_max": "Cap on number of primary partitions (slots).",
            "secondary_k_max": "Cap on number of secondary classes used (DSSP:3, chemotype:4).",
            "enable_cross_bias": "Toggle cross-biased block (intra-primary penalty).",
            "intra_penalty_mode": "default|light|heavy|none|custom",
            "intra_penalty_lambda": "Å penalty when mode=custom.",
            "max_atoms": "Atom-count cap to skip heavy blocks.",
            "max_block_seconds": "Time cap (s) per block; slot will be padded on timeout.",
            "enable_polar_block": "Toggle polar-only mini block (N/O).",
            "polar_hbond_weight": "Toggle H-bond weighting in polar block.",
            "polar_hbond_weight_factor": "Distance multiplier for H-bonded polar atoms (default=0.5).",
            "polar_hbond_energy_cutoff": "DSSP H-bond energy cutoff (kcal/mol) (default=-0.5).",
            "polar_hbond_inter_only": "If true, only weight inter-chain H-bonded atoms.",
            "enable_typed_block": "Enable typed-contact PH blocks (hydrophobic/salt/H-bond).",
            "dssp_timeout_seconds": "Per-PDB timeout for DSSP call (skip if exceeded).",
            "dssp_slow_threshold": "Log a note if DSSP exceeds this time (seconds).",
            "write_slot_maps": "Write per-structure slot maps under work/topology_slot_maps/ (debug).",
            "max_topology_dim_allowed": "Warn/fail if expected worst-case topo_dim exceeds this limit (optional).",
            "fail_fast_topology_dim": "If true, raise an error when expected worst-case topo_dim exceeds limit.",
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
            "k_max": 2,
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
            "write_slot_maps": False,
            "max_topology_dim_allowed": None,
            "fail_fast_topology_dim": False,
        },
        notes={"schema_version": "k_partite_advanced_v2_fixed_width"},
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

        secondary_partition = (params.get("secondary_partition") or "none").strip().lower()
        if secondary_partition not in {"none", "dssp", "chemotype"}:
            secondary_partition = "none"

        k_max = int(params.get("k_max") or 2)
        secondary_k_max = int(params.get("secondary_k_max") or 4)

        def _flag(value, fallback):
            if value is None:
                return fallback
            return bool(value)

        enable_cross_bias = _flag(params.get("enable_cross_bias"), preset in {"heavy", "heavy_stratified"})
        enable_polar_block = _flag(params.get("enable_polar_block"), preset in {"heavy", "heavy_stratified"})
        enable_typed_block = _flag(params.get("enable_typed_block"), preset in {"heavy", "heavy_stratified"})

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
        write_slot_maps = bool(params.get("write_slot_maps", False))
        max_topology_dim_allowed = params.get("max_topology_dim_allowed")
        fail_fast_topology_dim = bool(params.get("fail_fast_topology_dim", False))

        worker_count = max(1, int(params.get("jobs") or 1))

        base_config = new_topological_features.TopologicalConfig(
            neighbor_distance=neighbor_distance,
            filtration_cutoff=filtration_cutoff,
            min_persistence=min_persistence,
            element_filters=element_filters,
            dedup_sort=dedup_sort,
            workers=None,
        )

        strat_config = new_topological_features.TopologicalConfig(
            neighbor_distance=neighbor_distance,
            filtration_cutoff=filtration_cutoff,
            min_persistence=min_persistence,
            element_filters=element_filters_strat,
            dedup_sort=dedup_sort,
            workers=None,
        )

        topology_dir = work_dir / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology_log_dir = log_dir / "topology_logs"
        topology_log_dir.mkdir(parents=True, exist_ok=True)
        slot_map_dir = work_dir / "topology_slot_maps"
        if write_slot_maps:
            slot_map_dir.mkdir(parents=True, exist_ok=True)

        # Emit a global schema spec artifact for debugging (module-only; graph_metadata.json will still be canonical).
        schema_spec = _build_schema_spec(
            preset=preset,
            element_filters=element_filters,
            element_filters_stratified=element_filters_strat,
            k_max=k_max,
            secondary_partition=secondary_partition,
            secondary_k_max=secondary_k_max,
            enable_cross_bias=enable_cross_bias,
            enable_polar_block=enable_polar_block,
            enable_typed_block=enable_typed_block,
        )
        expected_worst_case = schema_spec.get("expected_worst_case_topology_dim")
        if isinstance(max_topology_dim_allowed, int) and expected_worst_case is not None:
            if expected_worst_case > max_topology_dim_allowed:
                msg = (
                    f"Expected worst-case topo_dim {expected_worst_case} exceeds max_topology_dim_allowed "
                    f"{max_topology_dim_allowed} (preset={preset}, k_max={k_max})."
                )
                if fail_fast_topology_dim:
                    raise ValueError(msg)
                LOG.warning(msg)
        (topology_dir / "topology_schema_spec.json").write_text(
            json.dumps(schema_spec, indent=2, sort_keys=True), encoding="utf-8"
        )

        dataset_label = dataset_dir.name
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

        tasks: List[Tuple[Path, Path, Path, Path, Optional[Path]]] = []
        for pdb_path in sorted(pdb_list):
            rel = pdb_path.relative_to(dataset_dir) if pdb_path.is_relative_to(dataset_dir) else Path(pdb_path.name)
            iface_path = interface_dir / rel.parent / f"{pdb_path.stem}.interface.txt"
            output_path = topology_dir / rel.parent / f"{pdb_path.stem}.topology.csv"
            log_path = topology_log_dir / rel.parent / f"{pdb_path.stem}.log"
            slot_path = None
            if write_slot_maps:
                slot_path = slot_map_dir / rel.parent / f"{pdb_path.stem}.slot_map.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if slot_path is not None:
                slot_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((pdb_path, iface_path, output_path, log_path, slot_path))

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

        if worker_count <= 1:
            for pdb_path, iface_path, output_path, log_path, slot_path in tasks:
                try:
                    result = _process_adv_v2_task(
                        pdb_path,
                        iface_path,
                        output_path,
                        log_path,
                        slot_path,
                        worker_args,
                    )
                    if result.get("error"):
                        failures.append((pdb_path, log_path, result["error"]))
                    else:
                        successes += 1
                        if successes % 5 == 0:
                            _log_progress()
                except Exception as exc:  # pragma: no cover
                    failures.append((pdb_path, log_path, str(exc)))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _process_adv_v2_task,
                        pdb_path,
                        iface_path,
                        output_path,
                        log_path,
                        slot_path,
                        worker_args,
                    ): (pdb_path, log_path)
                    for pdb_path, iface_path, output_path, log_path, slot_path in tasks
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
            params["min_persistence"] = require_positive_float(
                min_persistence, "topology.params.min_persistence", allow_zero=True
            )
        dedup_sort = params.get("dedup_sort")
        if dedup_sort is not None:
            params["dedup_sort"] = require_bool(dedup_sort, "topology.params.dedup_sort")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")
        for key in ("k_max", "secondary_k_max", "max_atoms"):
            val = params.get(key)
            if val is not None:
                params[key] = require_positive_int(val, f"topology.params.{key}")
        max_block_seconds = params.get("max_block_seconds")
        if max_block_seconds is not None:
            params["max_block_seconds"] = require_positive_float(
                max_block_seconds, "topology.params.max_block_seconds", allow_zero=True
            )
        dssp_timeout = params.get("dssp_timeout_seconds")
        if dssp_timeout is not None:
            params["dssp_timeout_seconds"] = require_positive_float(
                dssp_timeout, "topology.params.dssp_timeout_seconds", allow_zero=True
            )
        dssp_slow = params.get("dssp_slow_threshold")
        if dssp_slow is not None:
            params["dssp_slow_threshold"] = require_positive_float(
                dssp_slow, "topology.params.dssp_slow_threshold", allow_zero=True
            )
        lambda_val = params.get("intra_penalty_lambda")
        if lambda_val is not None:
            params["intra_penalty_lambda"] = require_positive_float(
                lambda_val, "topology.params.intra_penalty_lambda"
            )
        for key in ("polar_hbond_weight_factor", "polar_hbond_energy_cutoff"):
            val = params.get(key)
            if val is not None:
                params[key] = require_float(val, f"topology.params.{key}")
        penalty_mode = params.get("intra_penalty_mode")
        if penalty_mode is not None:
            mode = str(penalty_mode).strip().lower()
            if mode not in {"default", "light", "heavy", "none", "custom"}:
                raise ValueError(
                    "topology.params.intra_penalty_mode must be default|light|heavy|none|custom."
                )
            params["intra_penalty_mode"] = mode
        secondary = params.get("secondary_partition")
        if secondary is not None:
            sec = str(secondary).strip().lower()
            if sec not in {"none", "dssp", "chemotype"}:
                raise ValueError("secondary_partition must be none|dssp|chemotype.")
            params["secondary_partition"] = sec
        for key in (
            "enable_cross_bias",
            "enable_polar_block",
            "polar_hbond_weight",
            "polar_hbond_inter_only",
            "enable_typed_block",
            "write_slot_maps",
            "fail_fast_topology_dim",
        ):
            val = params.get(key)
            if val is not None:
                params[key] = require_bool(val, f"topology.params.{key}")
        max_allowed = params.get("max_topology_dim_allowed")
        if max_allowed is not None:
            params["max_topology_dim_allowed"] = require_positive_int(
                max_allowed, "topology.params.max_topology_dim_allowed"
            )

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        params = dict(cls._metadata.defaults)
        template.update(
            {
                "alias": "Advanced k-partite PH (v2 fixed-width) = preset-dependent",
                "summary": cls._metadata.summary,
                "description": cls._metadata.description,
                "params": params,
            }
        )
        param_comments = {
            "preset": "minimal=base only; lean=base+per-primary+per-pair (fixed slots); heavy=lean+cross-bias+polar+typed; heavy_stratified=heavy+secondary stratification",
            "k_max": "Number of primary slots (chains) in fixed schema; missing slots padded with reason_missing_slot",
            "secondary_partition": "none|dssp|chemotype (heavy_stratified only)",
            "secondary_k_max": "Max secondary classes used (DSSP fixed 3-state H/E/C; chemotype fixed 4-class POS/NEG/POLAR/HYDRO); extra classes truncated deterministically",
            "max_atoms": "Guardrail: if atom_count > max_atoms, heavy blocks are padded with reason_guardrail_skip",
            "max_block_seconds": "Time cap per block; on timeout, slot is padded with reason_timeout",
            "write_slot_maps": "If true, write work/topology_slot_maps/<pdb_id>.slot_map.json for debugging",
            "max_topology_dim_allowed": "Optional safety: warn/fail if expected worst-case topo_dim exceeds this limit",
            "fail_fast_topology_dim": "If true, raise error when expected worst-case topo_dim exceeds limit; otherwise warn only",
        }
        template["param_comments"] = param_comments

        alternates: List[Dict[str, object]] = []
        for preset_name, preset_params in (
            ("minimal", {"preset": "minimal"}),
            (
                "lean",
                {
                    "preset": "lean",
                    "enable_cross_bias": False,
                    "enable_polar_block": False,
                    "enable_typed_block": False,
                    "secondary_partition": "none",
                },
            ),
            (
                "heavy",
                {
                    "preset": "heavy",
                    "enable_cross_bias": True,
                    "enable_polar_block": True,
                    "polar_hbond_weight": True,
                    "enable_typed_block": True,
                    "secondary_partition": "none",
                },
            ),
            (
                "heavy_stratified",
                {
                    "preset": "heavy_stratified",
                    "secondary_partition": "chemotype",
                    "enable_cross_bias": True,
                    "enable_polar_block": True,
                    "polar_hbond_weight": True,
                    "enable_typed_block": True,
                },
            ),
        ):
            alt_params = {**params, **preset_params}
            spec = _build_schema_spec(
                preset=alt_params["preset"],
                element_filters=alt_params["element_filters"],
                element_filters_stratified=alt_params["element_filters_stratified"],
                k_max=int(alt_params.get("k_max") or 2),
                secondary_partition=str(alt_params.get("secondary_partition") or "none"),
                secondary_k_max=int(alt_params.get("secondary_k_max") or 4),
                enable_cross_bias=bool(alt_params.get("enable_cross_bias", False)),
                enable_polar_block=bool(alt_params.get("enable_polar_block", False)),
                enable_typed_block=bool(alt_params.get("enable_typed_block", False)),
            )
            dim = spec.get("expected_worst_case_topology_dim")
            alternates.append(
                {
                    "module": cls.module_id,
                    "alias": f"Advanced k-partite PH (v2 {preset_name}) ≈ topo_dim {dim}",
                    "params": alt_params,
                    "param_comments": param_comments,
                }
            )
        template["alternates"] = alternates
        template.setdefault("notes", {})
        template["notes"].update(
            {
                "schema": "Fixed-width within a run: block slots are padded; columns are namespaced; missingness channels always included.",
                "schema_spec_artifact": "Writes work/topology/topology_schema_spec.json for debugging; canonical dims remain in graph_metadata.json + topology_columns.json.",
            }
        )
        return template


def _build_schema_spec(
    *,
    preset: str,
    element_filters: Sequence[Sequence[str]],
    element_filters_stratified: Sequence[Sequence[str]],
    k_max: int,
    secondary_partition: str,
    secondary_k_max: int,
    enable_cross_bias: bool,
    enable_polar_block: bool,
    enable_typed_block: bool,
) -> Dict[str, object]:
    base_dim = _ph_dim_from_filters(element_filters)
    strat_dim = _ph_dim_from_filters(element_filters_stratified)
    polar_dim = _ph_dim_from_filters([("N",), ("O",)])

    # fixed core slots by preset
    if preset == "minimal":
        core_slots = 1
    else:
        core_slots = 1 + k_max + (k_max * (k_max - 1)) // 2
    total_dim = core_slots * (base_dim + SLOT_MISSINGNESS_DIM)

    if enable_cross_bias and preset in {"heavy", "heavy_stratified"}:
        total_dim += base_dim + SLOT_MISSINGNESS_DIM
    if enable_polar_block and preset in {"heavy", "heavy_stratified"}:
        total_dim += polar_dim + SLOT_MISSINGNESS_DIM
    if enable_typed_block and preset in {"heavy", "heavy_stratified"}:
        typed_classes = 3
        typed_pairs = 3
        total_dim += (typed_classes + typed_pairs) * (base_dim + SLOT_MISSINGNESS_DIM)
    if secondary_partition != "none" and preset == "heavy_stratified":
        if secondary_partition == "dssp":
            sec_classes = min(3, secondary_k_max)
        else:
            sec_classes = min(4, secondary_k_max)
        # per-primary per-class + per-pair per-class, both using strat_dim
        strat_slots = (k_max * sec_classes) + (((k_max * (k_max - 1)) // 2) * sec_classes)
        total_dim += strat_slots * (strat_dim + SLOT_MISSINGNESS_DIM)

    return {
        "module": PersistenceKPartiteAdvancedV2Module.module_id,
        "schema_version": PersistenceKPartiteAdvancedV2Module._metadata.notes.get("schema_version"),
        "preset": preset,
        "k_max": k_max,
        "secondary_partition": secondary_partition,
        "secondary_k_max": secondary_k_max,
        "element_filters": [list(x) for x in element_filters],
        "element_filters_stratified": [list(x) for x in element_filters_stratified],
        "base_dim": base_dim,
        "strat_dim": strat_dim,
        "polar_dim": polar_dim,
        "slot_missingness_dim": SLOT_MISSINGNESS_DIM,
        "expected_worst_case_topology_dim": total_dim,
    }


def _run_adv_v2_single(
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
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
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
    id_to_index = {rid: idx for idx, rid in enumerate(id_list)}
    n_rows = len(id_list)

    # Compute base once (required).
    base_raw = new_topological_features.compute_features_for_residues(pdb_path, residues, base_config)
    base_numeric_cols = [c for c in base_raw.columns if c != "ID"]
    base_aligned = _align_block(
        base_raw,
        expected_ids=id_list,
        id_to_index=id_to_index,
        numeric_cols=base_numeric_cols,
    )

    # Build primary slots (deterministic: lexicographic chain IDs).
    prim_map: Dict[str, List[new_topological_features.ResidueDescriptor]] = {}
    for res in residues:
        prim_map.setdefault(res.chain_id, []).append(res)
    primary_labels = sorted(prim_map.keys())
    slot_chain: List[Optional[str]] = []
    for i in range(k_max):
        slot_chain.append(primary_labels[i] if i < len(primary_labels) else None)

    slot_map: Dict[str, object] = {
        "pdb": str(pdb_path),
        "interface": str(interface_path),
        "k_max": k_max,
        "secondary_partition": secondary_partition,
        "secondary_k_max": secondary_k_max,
        "primary_slots": [{"slot": i, "label": slot_chain[i]} for i in range(k_max)],
        "notes": [],
    }

    def _timed(fn, label: str):
        start = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - start
        if elapsed > max_block_seconds:
            raise TimeoutError(f"{label} exceeded time cap ({elapsed:.2f}s > {max_block_seconds}s)")
        if slow_threshold and elapsed > slow_threshold:
            notes.append(f"{label} slow: {elapsed:.2f}s")
        return out

    blocks: List[pd.DataFrame] = []

    def _add_slot(
        *,
        slot_id: str,
        values: np.ndarray,
        numeric_cols: Sequence[str],
        present: float,
        n_residues: int,
        reason: str,
    ) -> None:
        prefix = f"{slot_id}."
        feature_cols = _prefix_columns(prefix, numeric_cols)
        block_df = pd.DataFrame(values, columns=feature_cols)
        miss_df = _make_missingness_frame(
            slot_prefix=prefix,
            n_rows=n_rows,
            present=present,
            n_residues=n_residues,
            reason=reason,
        )
        blocks.append(pd.concat([block_df, miss_df], axis=1))

    # base slot
    _add_slot(
        slot_id="base",
        values=base_aligned,
        numeric_cols=base_numeric_cols,
        present=1.0,
        n_residues=n_rows,
        reason="reason_present",
    )

    # per-primary slots
    if preset in {"lean", "heavy", "heavy_stratified"}:
        for i in range(k_max):
            chain_id = slot_chain[i]
            slot_id = f"primary.s{i}"
            if chain_id is None:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_missing_slot",
                )
                continue
            res_list = prim_map.get(chain_id, [])
            if not res_list:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_empty_subset",
                )
                continue
            try:
                raw_df = _timed(
                    lambda res_list=res_list: new_topological_features.compute_features_for_residues(
                        pdb_path, res_list, base_config
                    ),
                    slot_id,
                )
                aligned = _align_block(
                    raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=base_numeric_cols
                )
                _add_slot(
                    slot_id=slot_id,
                    values=aligned,
                    numeric_cols=base_numeric_cols,
                    present=1.0,
                    n_residues=len(res_list),
                    reason="reason_present",
                )
            except TimeoutError as exc:
                notes.append(str(exc))
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=len(res_list),
                    reason="reason_timeout",
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{slot_id} error: {exc}")
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=len(res_list),
                    reason="reason_error",
                )

    # per-primary-pair slots (cross-only)
    if preset in {"lean", "heavy", "heavy_stratified"}:
        for i, j in itertools.combinations(range(k_max), 2):
            slot_id = f"pair.s{i}_s{j}"
            a = slot_chain[i]
            b = slot_chain[j]
            if a is None or b is None:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_missing_slot",
                )
                continue
            pair_res = (prim_map.get(a, []) or []) + (prim_map.get(b, []) or [])
            if not pair_res:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_empty_subset",
                )
                continue
            try:
                raw_df = _timed(
                    lambda pair_res=pair_res: new_topological_features.compute_features_for_residues(
                        pdb_path, pair_res, base_config
                    ),
                    slot_id,
                )
                aligned = _align_block(
                    raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=base_numeric_cols
                )
                _add_slot(
                    slot_id=slot_id,
                    values=aligned,
                    numeric_cols=base_numeric_cols,
                    present=1.0,
                    n_residues=len(pair_res),
                    reason="reason_present",
                )
            except TimeoutError as exc:
                notes.append(str(exc))
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=len(pair_res),
                    reason="reason_timeout",
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{slot_id} error: {exc}")
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=len(pair_res),
                    reason="reason_error",
                )

    # optional cross-bias global
    if preset in {"heavy", "heavy_stratified"} and enable_cross_bias:
        slot_id = "cross_bias"
        if not heavy_allowed:
            _add_slot(
                slot_id=slot_id,
                values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                numeric_cols=base_numeric_cols,
                present=0.0,
                n_residues=n_rows,
                reason="reason_guardrail_skip",
            )
        else:
            try:
                raw_df = _timed(
                    lambda: new_topological_features.compute_features_for_residues(
                        pdb_path,
                        residues,
                        base_config,
                        bias_mode="intra_penalty",
                        bias_value=penalty_value,
                    ),
                    slot_id,
                )
                aligned = _align_block(
                    raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=base_numeric_cols
                )
                _add_slot(
                    slot_id=slot_id,
                    values=aligned,
                    numeric_cols=base_numeric_cols,
                    present=1.0,
                    n_residues=n_rows,
                    reason="reason_present",
                )
            except TimeoutError as exc:
                notes.append(str(exc))
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=n_rows,
                    reason="reason_timeout",
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{slot_id} error: {exc}")
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=n_rows,
                    reason="reason_error",
                )

    # optional polar block (N/O)
    if preset in {"heavy", "heavy_stratified"} and enable_polar_block:
        slot_id = "polar"
        polar_cfg = new_topological_features.TopologicalConfig(
            neighbor_distance=base_config.neighbor_distance,
            filtration_cutoff=base_config.filtration_cutoff,
            min_persistence=base_config.min_persistence,
            element_filters=[("N",), ("O",)],
            dedup_sort=base_config.dedup_sort,
            workers=base_config.workers,
        )
        polar_numeric_cols: List[str] = []
        if not heavy_allowed:
            polar_numeric_cols = [f"f{i}" for i in range(_ph_dim_from_filters(polar_cfg.element_filters))]
            _add_slot(
                slot_id=slot_id,
                values=np.zeros((n_rows, len(polar_numeric_cols)), dtype=float),
                numeric_cols=polar_numeric_cols,
                present=0.0,
                n_residues=n_rows,
                reason="reason_guardrail_skip",
            )
        else:
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
            try:
                raw_df = _timed(
                    lambda: new_topological_features.compute_features_for_residues(
                        pdb_path,
                        residues,
                        polar_cfg,
                        polar_mode=True,
                        hbond_weight=polar_hbond_weight,
                        hbond_residues=hbond_residues,
                        hbond_factor=polar_hbond_weight_factor,
                        hbond_inter_only=polar_hbond_inter_only,
                    ),
                    slot_id,
                )
                polar_numeric_cols = [c for c in raw_df.columns if c != "ID"]
                aligned = _align_block(
                    raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=polar_numeric_cols
                )
                _add_slot(
                    slot_id=slot_id,
                    values=aligned,
                    numeric_cols=polar_numeric_cols,
                    present=1.0,
                    n_residues=n_rows,
                    reason="reason_present",
                )
            except TimeoutError as exc:
                notes.append(str(exc))
                polar_numeric_cols = [f"f{i}" for i in range(_ph_dim_from_filters(polar_cfg.element_filters))]
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(polar_numeric_cols)), dtype=float),
                    numeric_cols=polar_numeric_cols,
                    present=0.0,
                    n_residues=n_rows,
                    reason="reason_timeout",
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{slot_id} error: {exc}")
                polar_numeric_cols = [f"f{i}" for i in range(_ph_dim_from_filters(polar_cfg.element_filters))]
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(polar_numeric_cols)), dtype=float),
                    numeric_cols=polar_numeric_cols,
                    present=0.0,
                    n_residues=n_rows,
                    reason="reason_error",
                )

    # optional typed-contact slots (fixed 3-class + 3 pair slots)
    if preset in {"heavy", "heavy_stratified"} and enable_typed_block:
        typed_classes = ("HYDROPHOBIC", "SALT", "HBOND")
        type_map: Dict[str, List[new_topological_features.ResidueDescriptor]] = {t: [] for t in typed_classes}
        for res in residues:
            type_map[_typed_class(res.residue_name or "")].append(res)
        if not heavy_allowed:
            for t in typed_classes:
                slot_id = f"typed.{t}"
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=len(type_map.get(t, [])),
                    reason="reason_guardrail_skip",
                )
            for a, b in itertools.combinations(typed_classes, 2):
                slot_id = f"typed_pair.{a}_{b}"
                pair_res = (type_map.get(a, []) or []) + (type_map.get(b, []) or [])
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                    numeric_cols=base_numeric_cols,
                    present=0.0,
                    n_residues=len(pair_res),
                    reason="reason_guardrail_skip",
                )
        else:
            for t in typed_classes:
                slot_id = f"typed.{t}"
                res_list = type_map.get(t, []) or []
                if not res_list:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                        numeric_cols=base_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_empty_subset",
                    )
                    continue
                try:
                    raw_df = _timed(
                        lambda res_list=res_list: new_topological_features.compute_features_for_residues(
                            pdb_path, res_list, base_config
                        ),
                        slot_id,
                    )
                    aligned = _align_block(
                        raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=base_numeric_cols
                    )
                    _add_slot(
                        slot_id=slot_id,
                        values=aligned,
                        numeric_cols=base_numeric_cols,
                        present=1.0,
                        n_residues=len(res_list),
                        reason="reason_present",
                    )
                except TimeoutError as exc:
                    notes.append(str(exc))
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                        numeric_cols=base_numeric_cols,
                        present=0.0,
                        n_residues=len(res_list),
                        reason="reason_timeout",
                    )
                except Exception as exc:  # pragma: no cover
                    notes.append(f"{slot_id} error: {exc}")
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                        numeric_cols=base_numeric_cols,
                        present=0.0,
                        n_residues=len(res_list),
                        reason="reason_error",
                    )
            for a, b in itertools.combinations(typed_classes, 2):
                slot_id = f"typed_pair.{a}_{b}"
                pair_res = (type_map.get(a, []) or []) + (type_map.get(b, []) or [])
                if not pair_res:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                        numeric_cols=base_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_empty_subset",
                    )
                    continue
                try:
                    raw_df = _timed(
                        lambda pair_res=pair_res: new_topological_features.compute_features_for_residues(
                            pdb_path, pair_res, base_config
                        ),
                        slot_id,
                    )
                    aligned = _align_block(
                        raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=base_numeric_cols
                    )
                    _add_slot(
                        slot_id=slot_id,
                        values=aligned,
                        numeric_cols=base_numeric_cols,
                        present=1.0,
                        n_residues=len(pair_res),
                        reason="reason_present",
                    )
                except TimeoutError as exc:
                    notes.append(str(exc))
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                        numeric_cols=base_numeric_cols,
                        present=0.0,
                        n_residues=len(pair_res),
                        reason="reason_timeout",
                    )
                except Exception as exc:  # pragma: no cover
                    notes.append(f"{slot_id} error: {exc}")
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(base_numeric_cols)), dtype=float),
                        numeric_cols=base_numeric_cols,
                        present=0.0,
                        n_residues=len(pair_res),
                        reason="reason_error",
                    )

    # optional stratified blocks: fixed class sets, deterministic slot ids
    if preset == "heavy_stratified" and secondary_partition != "none":
        strat_numeric_cols: Optional[List[str]] = None
        if secondary_partition == "dssp":
            sec_classes_all = ("H", "E", "C")
        else:
            sec_classes_all = ("POS", "NEG", "POLAR", "HYDRO")
        sec_classes = sec_classes_all[: min(len(sec_classes_all), max(1, secondary_k_max))]

        dssp_labels: Dict[Tuple[str, int, str], str] = {}
        if secondary_partition == "dssp" and heavy_allowed:
            dssp_labels = _compute_dssp_labels(
                pdb_path,
                residues,
                timeout_seconds=dssp_timeout_seconds,
                slow_threshold=dssp_slow_threshold,
                notes=notes,
            )
        if secondary_partition == "dssp" and not dssp_labels:
            notes.append("DSSP labels unavailable; stratified slots will be padded (reason_error).")

        # Build per-residue secondary label
        sec_label_for: Dict[str, str] = {}
        for res in residues:
            if secondary_partition == "chemotype":
                sec_label_for[res.raw_descriptor] = _chemotype4(res.residue_name or "")
            else:
                lbl = dssp_labels.get((res.chain_id, res.residue_number, res.insertion_code or " "))
                if lbl:
                    sec_label_for[res.raw_descriptor] = lbl

        def _strat_subset(chain: Optional[str], sec: str) -> List[new_topological_features.ResidueDescriptor]:
            if chain is None:
                return []
            out: List[new_topological_features.ResidueDescriptor] = []
            for res in prim_map.get(chain, []) or []:
                if sec_label_for.get(res.raw_descriptor) == sec:
                    out.append(res)
            return out

        # Determine strat numeric columns (needed for padding when empty/guardrail).
        try:
            probe_df = new_topological_features.compute_features_for_residues(pdb_path, residues, strat_config)
            strat_numeric_cols = [c for c in probe_df.columns if c != "ID"]
        except Exception:  # pragma: no cover
            strat_numeric_cols = [f"f{i}" for i in range(_ph_dim_from_filters(strat_config.element_filters))]

        for i in range(k_max):
            for sec in sec_classes:
                slot_id = f"strat.primary.s{i}.{sec}"
                chain_id = slot_chain[i]
                if chain_id is None:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_missing_slot",
                    )
                    continue
                if secondary_partition == "dssp" and not dssp_labels:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_error",
                    )
                    continue
                subset = _strat_subset(chain_id, sec)
                if not subset:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_empty_subset",
                    )
                    continue
                if not heavy_allowed:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=len(subset),
                        reason="reason_guardrail_skip",
                    )
                    continue
                try:
                    raw_df = _timed(
                        lambda subset=subset: new_topological_features.compute_features_for_residues(
                            pdb_path, subset, strat_config
                        ),
                        slot_id,
                    )
                    aligned = _align_block(
                        raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=strat_numeric_cols
                    )
                    _add_slot(
                        slot_id=slot_id,
                        values=aligned,
                        numeric_cols=strat_numeric_cols,
                        present=1.0,
                        n_residues=len(subset),
                        reason="reason_present",
                    )
                except TimeoutError as exc:
                    notes.append(str(exc))
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=len(subset),
                        reason="reason_timeout",
                    )
                except Exception as exc:  # pragma: no cover
                    notes.append(f"{slot_id} error: {exc}")
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=len(subset),
                        reason="reason_error",
                    )

        for i, j in itertools.combinations(range(k_max), 2):
            for sec in sec_classes:
                slot_id = f"strat.pair.s{i}_s{j}.{sec}"
                a = slot_chain[i]
                b = slot_chain[j]
                if a is None or b is None:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_missing_slot",
                    )
                    continue
                if secondary_partition == "dssp" and not dssp_labels:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_error",
                    )
                    continue
                subset = _strat_subset(a, sec) + _strat_subset(b, sec)
                if not subset:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_empty_subset",
                    )
                    continue
                if not heavy_allowed:
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=len(subset),
                        reason="reason_guardrail_skip",
                    )
                    continue
                try:
                    raw_df = _timed(
                        lambda subset=subset: new_topological_features.compute_features_for_residues(
                            pdb_path, subset, strat_config
                        ),
                        slot_id,
                    )
                    aligned = _align_block(
                        raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=strat_numeric_cols
                    )
                    _add_slot(
                        slot_id=slot_id,
                        values=aligned,
                        numeric_cols=strat_numeric_cols,
                        present=1.0,
                        n_residues=len(subset),
                        reason="reason_present",
                    )
                except TimeoutError as exc:
                    notes.append(str(exc))
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=len(subset),
                        reason="reason_timeout",
                    )
                except Exception as exc:  # pragma: no cover
                    notes.append(f"{slot_id} error: {exc}")
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(strat_numeric_cols)), dtype=float),
                        numeric_cols=strat_numeric_cols,
                        present=0.0,
                        n_residues=len(subset),
                        reason="reason_error",
                    )

    combined = pd.concat(blocks, axis=1)
    combined.insert(0, "ID", id_list)
    slot_map["notes"] = notes
    return combined, notes, slot_map


def _process_adv_v2_task(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    log_path: Path,
    slot_map_path: Optional[Path],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        df, notes, slot_map = _run_adv_v2_single(
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
        log_lines = [
            f"PDB: {pdb_path}",
            "Status: SUCCESS",
            f"Output file: {output_path}",
            f"Columns: {len(df.columns)-1}",
        ]
        if notes:
            log_lines.append("Notes:")
            log_lines.extend([f"- {note}" for note in notes])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if slot_map_path is not None:
            slot_map_path.write_text(json.dumps(slot_map, indent=2, sort_keys=True), encoding="utf-8")
        return {"error": None}
    except Exception as exc:  # pragma: no cover
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
        return {"error": str(exc)}
