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
from ...lib.edge_common import InterfaceResidue
from ...lib.topology_runner import _INTERFACE_DESCRIPTOR_RE

LOG = logging.getLogger("graph_builder")

SCHEMA_VERSION = "k_partite_advanced_lap_only_v2_fixed_width"
SLOT_MISSINGNESS_DIM = 9  # present + n_res + log1p_n_res + 6 reason one-hots

REASON_FIELDS: Tuple[str, ...] = (
    "reason_present",
    "reason_missing_slot",
    "reason_empty_subset",
    "reason_guardrail_skip",
    "reason_timeout",
    "reason_error",
)

LAP_EIGS_DEFAULT = 16
LAP_MOMENT_DEFAULT = (1, 2, 3, 4)
LAP_HEAT_DEFAULT = (0.1, 1.0, 5.0)


@dataclass(frozen=True)
class _SlotSpec:
    slot_id: str
    feature_prefix: str
    feature_cols: Tuple[str, ...]


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


def _parse_interface_file(path: Path) -> List[InterfaceResidue]:
    residues: List[InterfaceResidue] = []
    pattern = _INTERFACE_DESCRIPTOR_RE
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not parts:
                continue
            descriptor = parts[0]
            match = pattern.match(descriptor)
            if not match:
                continue
            try:
                coord = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
            except (IndexError, ValueError):
                continue
            residues.append(
                InterfaceResidue(
                    descriptor=descriptor,
                    chain_id=match.group("chain"),
                    residue_seq=int(match.group("res")),
                    insertion_code=(match.group("ins") or " ").strip() or " ",
                    residue_name=match.group("resname"),
                    coord=coord,
                )
            )
    return residues


def _build_lap_columns(eigs_count: int, moment_orders: Sequence[Any], heat_times: Sequence[Any]) -> List[str]:
    cols = ["lap_num_nodes"]
    cols.extend([f"lap_eig_{idx:02d}" for idx in range(1, eigs_count + 1)])
    cols.extend(
        [
            "lap_lambda_min",
            "lap_lambda_max",
            "lap_lambda_2",
            "lap_lambda_mean",
            "lap_lambda_var",
            "lap_spectral_radius",
            "lap_condition_guarded",
        ]
    )
    cols.append("lap_entropy")
    cols.extend([f"lap_moment_k{str(order).replace('.', '_')}" for order in moment_orders])
    cols.extend([f"lap_heat_t{str(time).replace('.', '_')}" for time in heat_times])
    return cols


def _lap_feature_dim(eigs_count: int, moment_orders: Sequence[Any], heat_times: Sequence[Any]) -> int:
    return len(_build_lap_columns(eigs_count, moment_orders, heat_times))


LAP_FEATURE_DIM_DEFAULT = _lap_feature_dim(LAP_EIGS_DEFAULT, LAP_MOMENT_DEFAULT, LAP_HEAT_DEFAULT)


def _weight_function(mode: str, sigma: float) -> Any:
    eps = 1e-8
    if mode == "gaussian" and sigma > eps:
        denom = 2.0 * sigma * sigma

        def _gaussian(dist: float) -> float:
            return float(math.exp(-((dist * dist) / denom))) if dist > 0 else 0.0

        return _gaussian

    if mode == "inverse":

        def _inverse(dist: float) -> float:
            return 1.0 / max(dist, eps)

        return _inverse

    def _binary(dist: float) -> float:
        return 1.0 if dist > 0 else 0.0

    return _binary


def _build_neighbors(
    residues: List[InterfaceResidue],
    target_idx: int,
    *,
    graph_mode: str,
    distance_cutoff: float,
    k_neighbors: Optional[int],
    max_neighbors: int,
) -> Tuple[List[int], np.ndarray]:
    target = residues[target_idx]
    candidates: List[Tuple[float, int, str]] = []
    for idx, res in enumerate(residues):
        if idx == target_idx:
            continue
        if graph_mode == "cross_chain" and res.chain_id == target.chain_id:
            continue
        dist = float(np.linalg.norm(res.coord - target.coord))
        if distance_cutoff > 0 and dist > distance_cutoff:
            continue
        candidates.append((dist, idx, res.descriptor))

    candidates.sort(key=lambda item: (item[0], item[2]))
    if k_neighbors is not None and k_neighbors > 0:
        candidates = candidates[:k_neighbors]

    limit = max(1, max_neighbors)
    selected = candidates[: max(0, limit - 1)]
    indices = [target_idx] + [idx for _, idx, _ in selected]
    coords = np.stack([residues[idx].coord for idx in indices], axis=0).astype(float)
    return indices, coords


def _build_laplacian(
    coords: np.ndarray,
    *,
    weight_fn,
    normalize: str,
) -> np.ndarray:
    n = coords.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            w = weight_fn(dist)
            if w <= 0:
                continue
            W[i, j] = w
            W[j, i] = w
    degree = W.sum(axis=1)
    if normalize == "rw":
        D_inv = np.zeros((n, n), dtype=float)
        for i in range(n):
            if degree[i] > 0:
                D_inv[i, i] = 1.0 / degree[i]
        L = np.eye(n, dtype=float) - D_inv @ W
    else:
        D_inv_sqrt = np.zeros((n, n), dtype=float)
        for i in range(n):
            if degree[i] > 0:
                D_inv_sqrt[i, i] = 1.0 / math.sqrt(degree[i])
        L = np.eye(n, dtype=float) - D_inv_sqrt @ W @ D_inv_sqrt
    return L


def _laplacian_features(
    residues: List[InterfaceResidue],
    target_idx: int,
    *,
    graph_mode: str,
    distance_cutoff: float,
    k_neighbors: Optional[int],
    max_neighbors: int,
    edge_weight: str,
    sigma: float,
    eigs_count: int,
    moment_orders: Sequence[int],
    heat_times: Sequence[float],
    normalize: str,
    include_entropy: bool,
) -> List[float]:
    indices, coords = _build_neighbors(
        residues,
        target_idx,
        graph_mode=graph_mode,
        distance_cutoff=distance_cutoff,
        k_neighbors=k_neighbors,
        max_neighbors=max_neighbors,
    )
    node_count = len(indices)
    if node_count == 0:
        lap_eigs = [0.0] * eigs_count
        return [0.0] + lap_eigs + [0.0] * 7 + [0.0] + [0.0] * len(moment_orders) + [0.0] * len(heat_times)

    weight_fn = _weight_function(edge_weight, sigma)
    L = _build_laplacian(coords, weight_fn=weight_fn, normalize=normalize)
    if L.size == 0:
        lap_eigs = [0.0] * eigs_count
        return [float(node_count)] + lap_eigs + [0.0] * 7 + [0.0] * len(moment_orders) + [0.0] * len(heat_times)

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    eigvals_sorted = np.sort(eigvals)
    eps = 1e-9
    zero_count = int(np.sum(eigvals_sorted <= eps))
    positive = eigvals_sorted[eigvals_sorted > eps]

    pos_list = positive.tolist()
    per_eig = pos_list[:eigs_count]
    if len(per_eig) < eigs_count:
        per_eig.extend([0.0] * (eigs_count - len(per_eig)))

    if positive.size == 0:
        lambda_min = lambda_max = lambda_2 = lambda_mean = lambda_var = spectral_radius = 0.0
    else:
        lambda_min = float(positive.min())
        lambda_max = float(positive.max())
        lambda_2 = float(positive.min())
        lambda_mean = float(positive.mean())
        lambda_var = float(positive.var())
        spectral_radius = lambda_max
    condition_guarded = (lambda_max + eps) / max(lambda_min, eps)

    entropy = 0.0
    if include_entropy and positive.size > 0:
        total = float(positive.sum())
        if total > 0:
            p = positive / total
            entropy = float(-np.sum(p * np.log(p + eps)))

    moments: List[float] = []
    for order in moment_orders:
        if order <= 0 or positive.size == 0:
            moments.append(0.0)
        else:
            moments.append(float(np.sum(np.power(positive, order))))

    heats: List[float] = []
    for t in heat_times:
        if t < 0:
            heats.append(0.0)
            continue
        heat_pos = float(np.sum(np.exp(-t * positive))) if positive.size > 0 else 0.0
        heats.append(float(zero_count + heat_pos))

    features: List[float] = [float(node_count)]
    features.extend(per_eig)
    features.extend(
        [
            lambda_min,
            lambda_max,
            lambda_2,
            lambda_mean,
            lambda_var,
            spectral_radius,
            condition_guarded,
        ]
    )
    features.append(entropy)
    features.extend(moments)
    features.extend(heats)
    return features


def _prefix_columns(prefix: str, cols: Iterable[str]) -> List[str]:
    return [f"{prefix}{col}" for col in cols]


def _reason_onehot(reason: str) -> Dict[str, float]:
    return {field: float(field == reason) for field in REASON_FIELDS}


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


def _compute_lap_df(
    residues: List[InterfaceResidue],
    *,
    lap_cols: Sequence[str],
    graph_mode: str,
    distance_cutoff: float,
    k_neighbors: Optional[int],
    max_neighbors: int,
    edge_weight: str,
    sigma: float,
    eigs_count: int,
    moment_orders: Sequence[int],
    heat_times: Sequence[float],
    normalize: str,
    include_entropy: bool,
) -> pd.DataFrame:
    ids = [res.descriptor for res in residues]
    if not residues:
        return pd.DataFrame(columns=["ID", *lap_cols])
    values = np.zeros((len(residues), len(lap_cols)), dtype=float)
    for idx in range(len(residues)):
        values[idx, :] = _laplacian_features(
            residues,
            idx,
            graph_mode=graph_mode,
            distance_cutoff=distance_cutoff,
            k_neighbors=k_neighbors,
            max_neighbors=max_neighbors,
            edge_weight=edge_weight,
            sigma=sigma,
            eigs_count=eigs_count,
            moment_orders=moment_orders,
            heat_times=heat_times,
            normalize=normalize,
            include_entropy=include_entropy,
        )
    df = pd.DataFrame(values, columns=lap_cols)
    df.insert(0, "ID", ids)
    return df


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


def _build_schema_spec(
    *,
    preset: str,
    k_max: int,
    secondary_partition: str,
    secondary_k_max: int,
    lap_dim: int,
) -> Dict[str, Any]:
    slot_dim = lap_dim + SLOT_MISSINGNESS_DIM
    primary_slots = 0
    pair_slots = 0
    if preset in {"lean", "heavy", "heavy_stratified"}:
        primary_slots = k_max
        pair_slots = (k_max * (k_max - 1)) // 2
    total_slots = 1 + primary_slots + pair_slots

    sec_classes = []
    if preset == "heavy_stratified" and secondary_partition != "none":
        if secondary_partition == "dssp":
            sec_classes = ["H", "E", "C"]
        elif secondary_partition == "chemotype":
            sec_classes = ["POS", "NEG", "POLAR", "HYDRO"]
        if secondary_k_max:
            sec_classes = sec_classes[:secondary_k_max]
        total_slots += len(sec_classes) * (primary_slots + pair_slots)

    total_dim = total_slots * slot_dim
    return {
        "module": PersistenceKPartiteAdvancedLaplacianOnlyV2Module.module_id,
        "schema_version": PersistenceKPartiteAdvancedLaplacianOnlyV2Module._metadata.notes.get("schema_version"),
        "preset": preset,
        "k_max": k_max,
        "secondary_partition": secondary_partition,
        "secondary_k_max": secondary_k_max,
        "lap_dim": lap_dim,
        "slot_missingness_dim": SLOT_MISSINGNESS_DIM,
        "expected_worst_case_topology_dim": total_dim,
    }


def _run_lap_only_v2_single(
    *,
    pdb_path: Path,
    interface_path: Path,
    preset: str,
    k_max: int,
    secondary_partition: str,
    secondary_k_max: int,
    lap_graph_mode: str,
    lap_graph_mode_primary: str,
    lap_distance_cutoff: float,
    lap_k_neighbors: Optional[int],
    lap_max_neighbors: int,
    lap_edge_weight: str,
    lap_sigma: float,
    lap_eigs_count: int,
    lap_moment_orders: Sequence[int],
    lap_heat_times: Sequence[float],
    lap_include_entropy: bool,
    lap_normalize: str,
    max_atoms: int,
    max_block_seconds: float,
    slow_threshold: float,
    dssp_timeout_seconds: float,
    dssp_slow_threshold: float,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    notes: List[str] = []
    residues = _parse_interface_file(interface_path)
    if not residues:
        raise RuntimeError(f"No interface residues found in {interface_path}")

    structure = new_topological_features._load_structure(pdb_path)  # type: ignore[attr-defined]
    atom_count = sum(1 for _ in structure.get_atoms())
    heavy_allowed = atom_count <= max_atoms
    if not heavy_allowed:
        notes.append(f"Skipped heavy blocks: atom_count {atom_count} > max_atoms {max_atoms}")

    id_list = [res.descriptor for res in residues]
    id_to_index = {rid: idx for idx, rid in enumerate(id_list)}
    n_rows = len(id_list)

    lap_cols = _build_lap_columns(lap_eigs_count, lap_moment_orders, lap_heat_times)

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

    base_raw = _compute_lap_df(
        residues,
        lap_cols=lap_cols,
        graph_mode=lap_graph_mode,
        distance_cutoff=lap_distance_cutoff,
        k_neighbors=lap_k_neighbors,
        max_neighbors=lap_max_neighbors,
        edge_weight=lap_edge_weight,
        sigma=lap_sigma,
        eigs_count=lap_eigs_count,
        moment_orders=lap_moment_orders,
        heat_times=lap_heat_times,
        normalize=lap_normalize,
        include_entropy=lap_include_entropy,
    )
    base_aligned = _align_block(
        base_raw,
        expected_ids=id_list,
        id_to_index=id_to_index,
        numeric_cols=lap_cols,
    )
    _add_slot(
        slot_id="base",
        values=base_aligned,
        numeric_cols=lap_cols,
        present=1.0,
        n_residues=n_rows,
        reason="reason_present",
    )

    prim_map: Dict[str, List[InterfaceResidue]] = {}
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

    if preset in {"lean", "heavy", "heavy_stratified"}:
        for i in range(k_max):
            chain_id = slot_chain[i]
            slot_id = f"primary.s{i}"
            if chain_id is None:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_missing_slot",
                )
                continue
            res_list = prim_map.get(chain_id, [])
            if not res_list:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_empty_subset",
                )
                continue
            try:
                raw_df = _timed(
                    lambda res_list=res_list: _compute_lap_df(
                        res_list,
                        lap_cols=lap_cols,
                        graph_mode=lap_graph_mode_primary,
                        distance_cutoff=lap_distance_cutoff,
                        k_neighbors=lap_k_neighbors,
                        max_neighbors=lap_max_neighbors,
                        edge_weight=lap_edge_weight,
                        sigma=lap_sigma,
                        eigs_count=lap_eigs_count,
                        moment_orders=lap_moment_orders,
                        heat_times=lap_heat_times,
                        normalize=lap_normalize,
                        include_entropy=lap_include_entropy,
                    ),
                    slot_id,
                )
                aligned = _align_block(
                    raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=lap_cols
                )
                _add_slot(
                    slot_id=slot_id,
                    values=aligned,
                    numeric_cols=lap_cols,
                    present=1.0,
                    n_residues=len(res_list),
                    reason="reason_present",
                )
            except TimeoutError as exc:
                notes.append(str(exc))
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=len(res_list),
                    reason="reason_timeout",
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{slot_id} error: {exc}")
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=len(res_list),
                    reason="reason_error",
                )

    if preset in {"lean", "heavy", "heavy_stratified"}:
        for i, j in itertools.combinations(range(k_max), 2):
            slot_id = f"pair.s{i}_s{j}"
            a = slot_chain[i]
            b = slot_chain[j]
            if a is None or b is None:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_missing_slot",
                )
                continue
            pair_res = (prim_map.get(a, []) or []) + (prim_map.get(b, []) or [])
            if not pair_res:
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=0,
                    reason="reason_empty_subset",
                )
                continue
            try:
                raw_df = _timed(
                    lambda pair_res=pair_res: _compute_lap_df(
                        pair_res,
                        lap_cols=lap_cols,
                        graph_mode=lap_graph_mode,
                        distance_cutoff=lap_distance_cutoff,
                        k_neighbors=lap_k_neighbors,
                        max_neighbors=lap_max_neighbors,
                        edge_weight=lap_edge_weight,
                        sigma=lap_sigma,
                        eigs_count=lap_eigs_count,
                        moment_orders=lap_moment_orders,
                        heat_times=lap_heat_times,
                        normalize=lap_normalize,
                        include_entropy=lap_include_entropy,
                    ),
                    slot_id,
                )
                aligned = _align_block(
                    raw_df, expected_ids=id_list, id_to_index=id_to_index, numeric_cols=lap_cols
                )
                _add_slot(
                    slot_id=slot_id,
                    values=aligned,
                    numeric_cols=lap_cols,
                    present=1.0,
                    n_residues=len(pair_res),
                    reason="reason_present",
                )
            except TimeoutError as exc:
                notes.append(str(exc))
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=len(pair_res),
                    reason="reason_timeout",
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{slot_id} error: {exc}")
                _add_slot(
                    slot_id=slot_id,
                    values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                    numeric_cols=lap_cols,
                    present=0.0,
                    n_residues=len(pair_res),
                    reason="reason_error",
                )

    if preset == "heavy_stratified" and secondary_partition != "none":
        if not heavy_allowed:
            sec_classes: List[str]
            if secondary_partition == "dssp":
                sec_classes = ["H", "E", "C"]
            else:
                sec_classes = ["POS", "NEG", "POLAR", "HYDRO"]
            if secondary_k_max:
                sec_classes = sec_classes[:secondary_k_max]
            for i in range(k_max):
                for sec in sec_classes:
                    slot_id = f"strat.primary.s{i}.{sec}"
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                        numeric_cols=lap_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_guardrail_skip",
                    )
            for i, j in itertools.combinations(range(k_max), 2):
                for sec in sec_classes:
                    slot_id = f"strat.pair.s{i}_s{j}.{sec}"
                    _add_slot(
                        slot_id=slot_id,
                        values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                        numeric_cols=lap_cols,
                        present=0.0,
                        n_residues=0,
                        reason="reason_guardrail_skip",
                    )
        else:
            sec_classes: List[str] = []
            sec_labels: Dict[Tuple[str, int, str], str] = {}
            if secondary_partition == "chemotype":
                sec_classes = ["POS", "NEG", "POLAR", "HYDRO"]
                if secondary_k_max:
                    sec_classes = sec_classes[:secondary_k_max]
            elif secondary_partition == "dssp":
                sec_classes = ["H", "E", "C"]
                if secondary_k_max:
                    sec_classes = sec_classes[:secondary_k_max]
                desc_list = [
                    new_topological_features.ResidueDescriptor(
                        chain_id=res.chain_id,
                        residue_number=res.residue_seq,
                        residue_name=res.residue_name,
                        insertion_code=res.insertion_code or " ",
                        raw_descriptor=res.descriptor,
                    )
                    for res in residues
                ]
                sec_labels = _compute_dssp_labels(
                    pdb_path,
                    desc_list,
                    timeout_seconds=dssp_timeout_seconds,
                    slow_threshold=dssp_slow_threshold,
                    notes=notes,
                )
                if not sec_labels:
                    notes.append("DSSP labels unavailable; stratified slots will be padded (reason_error).")
            else:
                notes.append(f"Unknown secondary partition '{secondary_partition}'; padding stratified slots.")

            def _sec_class(res: InterfaceResidue) -> Optional[str]:
                if secondary_partition == "chemotype":
                    return _chemotype4(res.residue_name)
                if secondary_partition == "dssp":
                    key = (res.chain_id, res.residue_seq, res.insertion_code or " ")
                    return sec_labels.get(key)
                return None

            def _strat_subset(chain: Optional[str], sec: str) -> List[InterfaceResidue]:
                out: List[InterfaceResidue] = []
                for res in residues:
                    if chain is not None and res.chain_id != chain:
                        continue
                    if _sec_class(res) == sec:
                        out.append(res)
                return out

            if secondary_partition == "dssp" and not sec_labels:
                for i in range(k_max):
                    for sec in sec_classes:
                        slot_id = f"strat.primary.s{i}.{sec}"
                        _add_slot(
                            slot_id=slot_id,
                            values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                            numeric_cols=lap_cols,
                            present=0.0,
                            n_residues=0,
                            reason="reason_error",
                        )
                for i, j in itertools.combinations(range(k_max), 2):
                    for sec in sec_classes:
                        slot_id = f"strat.pair.s{i}_s{j}.{sec}"
                        _add_slot(
                            slot_id=slot_id,
                            values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                            numeric_cols=lap_cols,
                            present=0.0,
                            n_residues=0,
                            reason="reason_error",
                        )
            else:
                for i in range(k_max):
                    chain_id = slot_chain[i]
                    for sec in sec_classes:
                        slot_id = f"strat.primary.s{i}.{sec}"
                        if chain_id is None:
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=0,
                                reason="reason_missing_slot",
                            )
                            continue
                        subset = _strat_subset(chain_id, sec)
                        if not subset:
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=0,
                                reason="reason_empty_subset",
                            )
                            continue
                        try:
                            raw_df = _timed(
                                lambda subset=subset: _compute_lap_df(
                                    subset,
                                    lap_cols=lap_cols,
                                    graph_mode=lap_graph_mode_primary,
                                    distance_cutoff=lap_distance_cutoff,
                                    k_neighbors=lap_k_neighbors,
                                    max_neighbors=lap_max_neighbors,
                                    edge_weight=lap_edge_weight,
                                    sigma=lap_sigma,
                                    eigs_count=lap_eigs_count,
                                    moment_orders=lap_moment_orders,
                                    heat_times=lap_heat_times,
                                    normalize=lap_normalize,
                                    include_entropy=lap_include_entropy,
                                ),
                                slot_id,
                            )
                            aligned = _align_block(
                                raw_df,
                                expected_ids=id_list,
                                id_to_index=id_to_index,
                                numeric_cols=lap_cols,
                            )
                            _add_slot(
                                slot_id=slot_id,
                                values=aligned,
                                numeric_cols=lap_cols,
                                present=1.0,
                                n_residues=len(subset),
                                reason="reason_present",
                            )
                        except TimeoutError as exc:
                            notes.append(str(exc))
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=len(subset),
                                reason="reason_timeout",
                            )
                        except Exception as exc:  # pragma: no cover
                            notes.append(f"{slot_id} error: {exc}")
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=len(subset),
                                reason="reason_error",
                            )

                for i, j in itertools.combinations(range(k_max), 2):
                    a = slot_chain[i]
                    b = slot_chain[j]
                    for sec in sec_classes:
                        slot_id = f"strat.pair.s{i}_s{j}.{sec}"
                        if a is None or b is None:
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=0,
                                reason="reason_missing_slot",
                            )
                            continue
                        subset = _strat_subset(a, sec) + _strat_subset(b, sec)
                        if not subset:
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=0,
                                reason="reason_empty_subset",
                            )
                            continue
                        try:
                            raw_df = _timed(
                                lambda subset=subset: _compute_lap_df(
                                    subset,
                                    lap_cols=lap_cols,
                                    graph_mode=lap_graph_mode,
                                    distance_cutoff=lap_distance_cutoff,
                                    k_neighbors=lap_k_neighbors,
                                    max_neighbors=lap_max_neighbors,
                                    edge_weight=lap_edge_weight,
                                    sigma=lap_sigma,
                                    eigs_count=lap_eigs_count,
                                    moment_orders=lap_moment_orders,
                                    heat_times=lap_heat_times,
                                    normalize=lap_normalize,
                                    include_entropy=lap_include_entropy,
                                ),
                                slot_id,
                            )
                            aligned = _align_block(
                                raw_df,
                                expected_ids=id_list,
                                id_to_index=id_to_index,
                                numeric_cols=lap_cols,
                            )
                            _add_slot(
                                slot_id=slot_id,
                                values=aligned,
                                numeric_cols=lap_cols,
                                present=1.0,
                                n_residues=len(subset),
                                reason="reason_present",
                            )
                        except TimeoutError as exc:
                            notes.append(str(exc))
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=len(subset),
                                reason="reason_timeout",
                            )
                        except Exception as exc:  # pragma: no cover
                            notes.append(f"{slot_id} error: {exc}")
                            _add_slot(
                                slot_id=slot_id,
                                values=np.zeros((n_rows, len(lap_cols)), dtype=float),
                                numeric_cols=lap_cols,
                                present=0.0,
                                n_residues=len(subset),
                                reason="reason_error",
                            )

    combined = pd.concat(blocks, axis=1)
    combined.insert(0, "ID", id_list)
    slot_map["notes"] = notes
    return combined, notes, slot_map


def _process_lap_only_v2_task(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    log_path: Path,
    slot_map_path: Optional[Path],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        df, notes, slot_map = _run_lap_only_v2_single(
            pdb_path=pdb_path,
            interface_path=interface_path,
            preset=options["preset"],
            k_max=options["k_max"],
            secondary_partition=options["secondary_partition"],
            secondary_k_max=options["secondary_k_max"],
            lap_graph_mode=options["lap_graph_mode"],
            lap_graph_mode_primary=options["lap_graph_mode_primary"],
            lap_distance_cutoff=options["lap_distance_cutoff"],
            lap_k_neighbors=options["lap_k_neighbors"],
            lap_max_neighbors=options["lap_max_neighbors"],
            lap_edge_weight=options["lap_edge_weight"],
            lap_sigma=options["lap_sigma"],
            lap_eigs_count=options["lap_eigs_count"],
            lap_moment_orders=options["lap_moment_orders"],
            lap_heat_times=options["lap_heat_times"],
            lap_include_entropy=options["lap_include_entropy"],
            lap_normalize=options["lap_normalize"],
            max_atoms=options["max_atoms"],
            max_block_seconds=options["max_block_seconds"],
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
        log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILED\nError: {exc}\n", encoding="utf-8")
        return {"error": str(exc)}


@register_feature_module
class PersistenceKPartiteAdvancedLaplacianOnlyV2Module(TopologyFeatureModule):
    module_id = "topology/persistence_k_partite_advanced_laplacian_only/v2"
    module_kind = "topology"
    default_alias = "Laplacian-only k-partite (v2 fixed-width) = preset-dependent"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Laplacian-only k-partite topology with fixed-width slots.",
        description=(
            "Computes Laplacian spectral features on the bipartite interface graph, "
            "expanded into fixed-width k-partite slots (base, per-primary, per-pair, and optional stratified). "
            "Slots are padded deterministically and include missingness channels for robust downstream training."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "preset": "minimal|lean|heavy|heavy_stratified",
            "neighbor_distance": "Interface neighborhood radius in Å (also Laplacian cutoff default).",
            "lap_graph_mode": "Graph mode for base/pair blocks: cross_chain (default) or all.",
            "lap_graph_mode_primary": "Graph mode for per-primary blocks (default: all).",
            "lap_distance_cutoff": "Å radius for Laplacian neighborhood (defaults to neighbor_distance).",
            "lap_k_neighbors": "Optional deterministic k-NN cap for Laplacian neighborhood.",
            "lap_max_neighbors": "Cap on Laplacian neighborhood size (includes target residue).",
            "lap_edge_weight": "Edge weighting: gaussian (default), inverse, or binary.",
            "lap_sigma": "Sigma for Gaussian edge weights (defaults to neighbor_distance/2).",
            "lap_eigs_count": "Number of smallest nonzero eigenvalues to keep (padded).",
            "lap_moment_orders": "List of spectral moment orders to emit.",
            "lap_heat_times": "List of heat kernel times to emit traces for.",
            "lap_include_entropy": "Include spectral entropy over nonzero eigenvalues.",
            "lap_normalize": "sym (default symmetric normalized) or rw (random-walk) Laplacian.",
            "k_max": "Number of primary slots (chains) in fixed schema; missing slots padded.",
            "secondary_partition": "none|dssp|chemotype (heavy_stratified only).",
            "secondary_k_max": "Max secondary classes used (DSSP fixed 3-state; chemotype fixed 4-class).",
            "max_atoms": "Guardrail: if atom_count > max_atoms, stratified blocks are padded.",
            "max_block_seconds": "Time cap per block; on timeout, slot is padded with reason_timeout.",
            "dssp_timeout_seconds": "Timeout for DSSP calls (heavy_stratified only).",
            "dssp_slow_threshold": "Log DSSP calls exceeding this time (s).",
            "write_slot_maps": "If true, write work/topology_slot_maps/<pdb_id>.slot_map.json for debugging.",
            "max_topology_dim_allowed": "Optional safety: warn/fail if expected worst-case topo_dim exceeds this limit.",
            "fail_fast_topology_dim": "If true, raise error when expected worst-case topo_dim exceeds limit; otherwise warn only.",
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "preset": "lean",
            "neighbor_distance": 8.0,
            "lap_graph_mode": "cross_chain",
            "lap_graph_mode_primary": "all",
            "lap_distance_cutoff": None,
            "lap_k_neighbors": None,
            "lap_max_neighbors": 128,
            "lap_edge_weight": "gaussian",
            "lap_sigma": None,
            "lap_eigs_count": LAP_EIGS_DEFAULT,
            "lap_moment_orders": LAP_MOMENT_DEFAULT,
            "lap_heat_times": LAP_HEAT_DEFAULT,
            "lap_include_entropy": True,
            "lap_normalize": "sym",
            "k_max": 2,
            "secondary_partition": "none",
            "secondary_k_max": 4,
            "max_atoms": 25000,
            "max_block_seconds": 60.0,
            "dssp_timeout_seconds": 10.0,
            "dssp_slow_threshold": 2.0,
            "write_slot_maps": False,
            "max_topology_dim_allowed": None,
            "fail_fast_topology_dim": False,
            "jobs": 16,
        },
        notes={
            "schema_version": SCHEMA_VERSION,
            "lap_feature_dim_default": LAP_FEATURE_DIM_DEFAULT,
            "slot_missingness_dim": SLOT_MISSINGNESS_DIM,
        },
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params.get("lap_distance_cutoff") is None:
            self.params["lap_distance_cutoff"] = float(self.params["neighbor_distance"])
        if self.params.get("lap_sigma") is None:
            self.params["lap_sigma"] = float(self.params["neighbor_distance"]) / 2.0

    def generate_topology(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        interface_dir: Path,
        work_dir: Path,
        log_dir: Path,
        sort_artifacts: bool = True,
        round_decimals: Optional[int] = None,
        slow_threshold: float = 5.0,
    ):
        params = self.params
        preset = str(params.get("preset", "lean")).strip().lower()
        if preset not in {"minimal", "lean", "heavy", "heavy_stratified"}:
            raise ValueError("preset must be minimal|lean|heavy|heavy_stratified")

        secondary_partition = str(params.get("secondary_partition", "none")).strip().lower()
        if preset != "heavy_stratified":
            secondary_partition = "none"
        if secondary_partition not in {"none", "dssp", "chemotype"}:
            raise ValueError("secondary_partition must be none|dssp|chemotype")

        k_max = int(params.get("k_max") or 2)
        secondary_k_max = int(params.get("secondary_k_max") or 4)
        neighbor_distance = float(params["neighbor_distance"])

        lap_graph_mode = str(params.get("lap_graph_mode", "cross_chain")).strip().lower()
        lap_graph_mode_primary = str(params.get("lap_graph_mode_primary", "all")).strip().lower()
        lap_distance_cutoff = float(params.get("lap_distance_cutoff") or neighbor_distance)
        lap_k_neighbors = params.get("lap_k_neighbors")
        if lap_k_neighbors is not None:
            lap_k_neighbors = int(lap_k_neighbors)
        lap_max_neighbors = int(params.get("lap_max_neighbors") or 128)
        lap_edge_weight = str(params.get("lap_edge_weight", "gaussian")).strip().lower()
        lap_sigma = float(params.get("lap_sigma") or (neighbor_distance / 2.0))
        lap_eigs_count = int(params.get("lap_eigs_count") or LAP_EIGS_DEFAULT)
        lap_moment_orders = tuple(params.get("lap_moment_orders") or LAP_MOMENT_DEFAULT)
        lap_heat_times = tuple(params.get("lap_heat_times") or LAP_HEAT_DEFAULT)
        lap_include_entropy = bool(params.get("lap_include_entropy", True))
        lap_normalize = str(params.get("lap_normalize", "sym")).strip().lower()

        max_atoms = int(params.get("max_atoms") or 25000)
        max_block_seconds = float(params.get("max_block_seconds") or 60.0)
        slow_threshold = float(params.get("slow_threshold") or slow_threshold)
        dssp_timeout_seconds = float(params.get("dssp_timeout_seconds") or 10.0)
        dssp_slow_threshold = float(params.get("dssp_slow_threshold") or 0.0)
        write_slot_maps = bool(params.get("write_slot_maps", False))
        max_topology_dim_allowed = params.get("max_topology_dim_allowed")
        fail_fast_topology_dim = bool(params.get("fail_fast_topology_dim", False))

        pdb_list = list(pdb_paths)
        topology_dir = work_dir / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology_log_dir = log_dir / "topology_logs"
        topology_log_dir.mkdir(parents=True, exist_ok=True)
        slot_map_dir = work_dir / "topology_slot_maps"
        if write_slot_maps:
            slot_map_dir.mkdir(parents=True, exist_ok=True)

        schema_spec = _build_schema_spec(
            preset=preset,
            k_max=k_max,
            secondary_partition=secondary_partition,
            secondary_k_max=secondary_k_max,
            lap_dim=_lap_feature_dim(lap_eigs_count, lap_moment_orders, lap_heat_times),
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
            "preset": preset,
            "k_max": k_max,
            "secondary_partition": secondary_partition,
            "secondary_k_max": secondary_k_max,
            "lap_graph_mode": lap_graph_mode,
            "lap_graph_mode_primary": lap_graph_mode_primary,
            "lap_distance_cutoff": lap_distance_cutoff,
            "lap_k_neighbors": lap_k_neighbors,
            "lap_max_neighbors": lap_max_neighbors,
            "lap_edge_weight": lap_edge_weight,
            "lap_sigma": lap_sigma,
            "lap_eigs_count": lap_eigs_count,
            "lap_moment_orders": lap_moment_orders,
            "lap_heat_times": lap_heat_times,
            "lap_include_entropy": lap_include_entropy,
            "lap_normalize": lap_normalize,
            "max_atoms": max_atoms,
            "max_block_seconds": max_block_seconds,
            "slow_threshold": slow_threshold,
            "dssp_timeout_seconds": dssp_timeout_seconds,
            "dssp_slow_threshold": dssp_slow_threshold,
            "sort_artifacts": sort_artifacts,
            "round_decimals": round_decimals,
        }

        worker_count = max(1, int(params.get("jobs") or 1))
        if worker_count <= 1:
            for pdb_path, iface_path, output_path, log_path, slot_path in tasks:
                try:
                    result = _process_lap_only_v2_task(
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
                        _process_lap_only_v2_task,
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
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        preset = params.get("preset")
        if preset is not None:
            value = str(preset).strip().lower()
            if value not in {"minimal", "lean", "heavy", "heavy_stratified"}:
                raise ValueError("preset must be minimal|lean|heavy|heavy_stratified")
            params["preset"] = value

        for key in ("neighbor_distance", "lap_distance_cutoff", "lap_sigma"):
            value = params.get(key)
            if value is not None:
                params[key] = require_positive_float(value, f"topology.params.{key}")

        for key in ("lap_eigs_count", "lap_max_neighbors", "k_max", "secondary_k_max", "max_atoms"):
            value = params.get(key)
            if value is not None:
                params[key] = require_positive_int(value, f"topology.params.{key}")

        lap_k_neighbors = params.get("lap_k_neighbors")
        if lap_k_neighbors is not None:
            params["lap_k_neighbors"] = require_positive_int(lap_k_neighbors, "topology.params.lap_k_neighbors")

        for key in ("max_block_seconds", "dssp_timeout_seconds", "dssp_slow_threshold"):
            value = params.get(key)
            if value is not None:
                params[key] = require_positive_float(value, f"topology.params.{key}", allow_zero=True)

        graph_mode = params.get("lap_graph_mode")
        if graph_mode is not None:
            mode = str(graph_mode).strip().lower()
            if mode not in {"cross_chain", "all"}:
                raise ValueError("lap_graph_mode must be cross_chain|all")
            params["lap_graph_mode"] = mode

        graph_mode_primary = params.get("lap_graph_mode_primary")
        if graph_mode_primary is not None:
            mode = str(graph_mode_primary).strip().lower()
            if mode not in {"cross_chain", "all"}:
                raise ValueError("lap_graph_mode_primary must be cross_chain|all")
            params["lap_graph_mode_primary"] = mode

        edge_weight = params.get("lap_edge_weight")
        if edge_weight is not None:
            mode = str(edge_weight).strip().lower()
            if mode not in {"gaussian", "inverse", "binary"}:
                raise ValueError("lap_edge_weight must be gaussian|inverse|binary")
            params["lap_edge_weight"] = mode

        normalize = params.get("lap_normalize")
        if normalize is not None:
            mode = str(normalize).strip().lower()
            if mode not in {"sym", "rw"}:
                raise ValueError("lap_normalize must be sym|rw")
            params["lap_normalize"] = mode

        secondary = params.get("secondary_partition")
        if secondary is not None:
            sec = str(secondary).strip().lower()
            if sec not in {"none", "dssp", "chemotype"}:
                raise ValueError("secondary_partition must be none|dssp|chemotype")
            params["secondary_partition"] = sec

        for key in (
            "lap_include_entropy",
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

        moment_orders = params.get("lap_moment_orders")
        if moment_orders is not None:
            if not isinstance(moment_orders, (list, tuple)):
                raise ValueError("lap_moment_orders must be a list")
            params["lap_moment_orders"] = [int(x) for x in moment_orders]

        heat_times = params.get("lap_heat_times")
        if heat_times is not None:
            if not isinstance(heat_times, (list, tuple)):
                raise ValueError("lap_heat_times must be a list")
            params["lap_heat_times"] = [float(x) for x in heat_times]

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        params = dict(cls._metadata.defaults)
        if params.get("lap_distance_cutoff") is None:
            params["lap_distance_cutoff"] = params.get("neighbor_distance")
        if params.get("lap_sigma") is None and params.get("neighbor_distance") is not None:
            params["lap_sigma"] = float(params["neighbor_distance"]) / 2.0

        template.update(
            {
                "alias": "Laplacian-only k-partite (v2 fixed-width) = preset-dependent",
                "summary": cls._metadata.summary,
                "description": cls._metadata.description,
                "params": params,
            }
        )
        param_comments = {
            "preset": "minimal=base only; lean=base+per-primary+per-pair; heavy=lean (same blocks); heavy_stratified=heavy+secondary stratification",
            "k_max": "Number of primary slots (chains) in fixed schema; missing slots padded with reason_missing_slot",
            "secondary_partition": "none|dssp|chemotype (heavy_stratified only)",
            "secondary_k_max": "Max secondary classes used (DSSP fixed H/E/C; chemotype fixed POS/NEG/POLAR/HYDRO)",
            "lap_graph_mode": "cross_chain (default) for base/pair blocks; all includes intra-chain edges",
            "lap_graph_mode_primary": "all (default) for per-primary blocks; cross_chain enforces bipartite-only",
            "lap_edge_weight": "gaussian (default, sigma ~ neighbor_distance/2), inverse, or binary",
            "lap_k_neighbors": "optional deterministic k-NN cap; otherwise cutoff-based",
            "lap_max_neighbors": "cap neighborhood size (includes target residue)",
            "max_atoms": "Guardrail: if atom_count > max_atoms, stratified blocks padded with reason_guardrail_skip",
            "max_block_seconds": "Time cap per block; on timeout, slot is padded with reason_timeout",
            "write_slot_maps": "If true, write work/topology_slot_maps/<pdb_id>.slot_map.json for debugging",
            "max_topology_dim_allowed": "Optional safety: warn/fail if expected worst-case topo_dim exceeds this limit",
            "fail_fast_topology_dim": "If true, raise error when expected worst-case topo_dim exceeds limit; otherwise warn only",
        }
        if str(params.get("preset", "lean")) == "lean":
            param_comments.update(
                {
                    "preset": "lean (default): base + per-primary + per-pair slots",
                    "secondary_partition": "ignored for lean",
                    "secondary_k_max": "ignored for lean",
                }
            )
        template["param_comments"] = param_comments

        # Build a shared alias that advertises expected dims for each preset.
        dim_map: Dict[str, Optional[int]] = {}
        for preset_name, preset_params in (
            ("minimal", {"preset": "minimal"}),
            ("lean", {"preset": "lean"}),
            ("heavy", {"preset": "heavy"}),
            ("heavy_stratified", {"preset": "heavy_stratified", "secondary_partition": "chemotype"}),
        ):
            spec = _build_schema_spec(
                preset=preset_params["preset"],
                k_max=int(params.get("k_max") or 2),
                secondary_partition=str(preset_params.get("secondary_partition") or params.get("secondary_partition") or "none"),
                secondary_k_max=int(params.get("secondary_k_max") or 4),
                lap_dim=_lap_feature_dim(
                    int(params.get("lap_eigs_count") or LAP_EIGS_DEFAULT),
                    params.get("lap_moment_orders") or LAP_MOMENT_DEFAULT,
                    params.get("lap_heat_times") or LAP_HEAT_DEFAULT,
                ),
            )
            dim_map[preset_name] = spec.get("expected_worst_case_topology_dim")

        alias_summary = (
            "Laplacian-only k-partite "
            f"{dim_map.get('lean')}D (Lean) | {dim_map.get('minimal')}D (Minimal) | "
            f"{dim_map.get('heavy')}D (Heavy) | {dim_map.get('heavy_stratified')}D (Heavy Stratified)"
        )
        template["alias"] = alias_summary

        alternates: List[Dict[str, object]] = []
        default_preset = str(params.get("preset", "lean"))
        for preset_name, preset_params in (
            ("minimal", {"preset": "minimal"}),
            (
                "lean",
                {
                    "preset": "lean",
                    "secondary_partition": "none",
                },
            ),
            (
                "heavy",
                {
                    "preset": "heavy",
                    "secondary_partition": "none",
                },
            ),
            (
                "heavy_stratified",
                {
                    "preset": "heavy_stratified",
                    "secondary_partition": "chemotype",
                },
            ),
        ):
            if preset_name == default_preset:
                continue
            alt_comments = dict(param_comments)
            if preset_name == "minimal":
                alt_comments.update(
                    {
                        "preset": "minimal: base slot only (no per-primary or pair slots)",
                        "k_max": "ignored for minimal (base slot only)",
                        "secondary_partition": "ignored for minimal",
                        "secondary_k_max": "ignored for minimal",
                        "lap_graph_mode_primary": "ignored for minimal (no per-primary slots)",
                        "max_atoms": "ignored for minimal",
                        "max_block_seconds": "ignored for minimal",
                    }
                )
            elif preset_name == "lean":
                alt_comments.update(
                    {
                        "preset": "lean: base + per-primary + per-pair slots",
                        "secondary_partition": "ignored for lean",
                        "secondary_k_max": "ignored for lean",
                    }
                )
            elif preset_name == "heavy":
                alt_comments.update(
                    {
                        "preset": "heavy: same slots as lean (no extra blocks yet)",
                        "secondary_partition": "ignored for heavy",
                        "secondary_k_max": "ignored for heavy",
                    }
                )
            elif preset_name == "heavy_stratified":
                alt_comments.update(
                    {
                        "preset": "heavy_stratified: lean slots + secondary stratified slots",
                    }
                )
            alt_params = {**params, **preset_params}
            spec = _build_schema_spec(
                preset=alt_params["preset"],
                k_max=int(alt_params.get("k_max") or 2),
                secondary_partition=str(alt_params.get("secondary_partition") or "none"),
                secondary_k_max=int(alt_params.get("secondary_k_max") or 4),
                lap_dim=_lap_feature_dim(
                    int(alt_params.get("lap_eigs_count") or LAP_EIGS_DEFAULT),
                    alt_params.get("lap_moment_orders") or LAP_MOMENT_DEFAULT,
                    alt_params.get("lap_heat_times") or LAP_HEAT_DEFAULT,
                ),
            )
            dim = spec.get("expected_worst_case_topology_dim")
            alternates.append(
                {
                    "module": cls.module_id,
                    "alias": alias_summary,
                    "params": alt_params,
                    "param_comments": alt_comments,
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
