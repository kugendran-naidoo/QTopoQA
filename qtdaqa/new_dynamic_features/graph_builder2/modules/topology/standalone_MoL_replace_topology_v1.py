from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from ..base import (
    TopologyFeatureModule,
    build_metadata,
    require_bool,
    require_positive_float,
    require_positive_int,
)
from ..registry import register_feature_module
from ...lib.progress import StageProgress
from ...lib.topology_runner import _INTERFACE_DESCRIPTOR_RE
from ...lib.edge_common import InterfaceResidue
from ...lib.laplacian_moments import (
    LaplacianMomentConfig,
    build_unweighted_adjacency,
    compute_laplacian_moments,
)


def _parse_interface_file(path: Path) -> List[InterfaceResidue]:
    residues: List[InterfaceResidue] = []
    pattern = _INTERFACE_DESCRIPTOR_RE
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
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
                    insertion_code=(match.group("ins") or " ").strip(),
                    residue_name=match.group("resname"),
                    coord=coord,
                )
            )
    return residues


def _weight_function(mode: str, sigma: float):
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
    distance_cutoff: float,
    k_neighbors: int | None,
    max_neighbors: int,
    graph_mode: str,
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


def _build_laplacian(coords: np.ndarray, *, weight_fn, normalize: str) -> np.ndarray:
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


def _fiedler_stats(eigvals: np.ndarray, eigvecs: np.ndarray) -> Tuple[float, float, float]:
    if eigvals.size < 2 or eigvecs.size == 0:
        return 0.0, 0.0, 0.0
    try:
        idx = np.argsort(eigvals)
        # Second smallest eigenvalue index (algebraic connectivity)
        f_idx = idx[1] if idx.size > 1 else idx[0]
        f_vec = eigvecs[:, f_idx]
        abs_mean = float(np.mean(np.abs(f_vec)))
        abs_std = float(np.std(np.abs(f_vec)))
        signs = np.sign(f_vec)
        pos = np.count_nonzero(signs > 0)
        neg = np.count_nonzero(signs < 0)
        total = max(1, pos + neg)
        balance = float(min(pos, neg) / total)
        return abs_mean, abs_std, balance
    except Exception:
        return 0.0, 0.0, 0.0


def _kirchhoff_proxy(eigvals: np.ndarray) -> float:
    eps = 1e-8
    nonzero = eigvals[eigvals > eps]
    if nonzero.size == 0:
        return 0.0
    return float(np.sum(1.0 / nonzero))


def _format_radius_suffix(radius: float) -> str:
    if radius.is_integer():
        return f"r{int(radius):02d}"
    return f"r{str(radius).replace('.', '_')}"


@register_feature_module
class StandaloneMoLReplaceTopologyModule(TopologyFeatureModule):
    module_id = "topology/standalone_MoL_replace_topology/v1"
    module_kind = "topology"
    default_alias = "Laplacian-only spectral topology (32 eigs, entropy+Fiedler+Kirchhoff, multi-scale 6/8/10 Å)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Per-residue Laplacian spectral features (no PH): eigen blocks, stats, moments, heat traces, Fiedler, Kirchhoff; multi-scale defaults to 6/8/10 Å.",
        description=(
            "Builds Laplacian neighborhoods on interface residues and emits spectral summaries only "
            "(no persistent homology). Defaults to unweighted symmetric normalized Laplacian, 32 smallest "
            "nonzero eigenvalues, entropy, raw+centered moments, heat traces, Fiedler stats, and a Kirchhoff proxy. "
            "Defaults to multi-scale radii (6/8/10 Å) to match or exceed PH richness; set lap_multi_radii=null for single-scale."
        ),
        inputs=("interface_file",),
        outputs=("topology_csv",),
        parameters={
            "neighbor_distance": "Neighborhood cutoff (Å) for Laplacian context (single-scale default).",
            "lap_multi_radii": "Optional list of radii (Å) for multi-scale blocks; when set, replaces single-scale cutoff.",
            "lap_k_neighbors": "Optional deterministic k-NN cap per radius.",
            "lap_max_neighbors": "Cap on neighborhood size (includes target residue).",
            "lap_estimator": "exact or slq (SLQ used for moments when above size threshold).",
            "lap_size_threshold": "Node-count threshold for switching to SLQ moments.",
            "lap_slq_probes": "Probe vectors for SLQ.",
            "lap_slq_steps": "Lanczos steps for SLQ.",
            "lap_weight": "unweighted (default), gaussian, or inverse edge weights.",
            "lap_sigma": "Sigma for gaussian weights; defaults to cutoff/2 when gaussian.",
            "lap_normalize": "sym (default) or rw normalized Laplacian.",
            "lap_eigs_count": "Number of smallest nonzero eigenvalues to record (default 32 for richer spectrum).",
            "lap_heat_times": "Heat trace times.",
            "lap_include_entropy": "Include spectral entropy.",
            "lap_include_fiedler": "Include Fiedler vector stats.",
            "lap_include_kirchhoff": "Include Kirchhoff/effective-resistance proxy.",
            "lap_use_centered_moments": "Include centered spectral moments.",
            "lap_profile": "Profile Laplacian timing per structure (logs timing when true).",
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "neighbor_distance": 8.0,
            "lap_multi_radii": (6.0, 8.0, 10.0),
            "lap_k_neighbors": None,
            "lap_max_neighbors": 128,
            "lap_estimator": "exact",
            "lap_size_threshold": 80,
            "lap_slq_probes": 8,
            "lap_slq_steps": 32,
            "lap_weight": "unweighted",
            "lap_sigma": None,
            "lap_normalize": "sym",
            "lap_eigs_count": 32,
            "lap_heat_times": (0.1, 1.0, 5.0),
            "lap_include_entropy": True,
            "lap_include_fiedler": True,
            "lap_include_kirchhoff": True,
            "lap_use_centered_moments": True,
            "lap_profile": False,
            "jobs": 16,
        },
        notes={"dim_default_single_scale": 55, "dim_multi_scale_default_r6_8_10": 165},
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._base_columns = self._build_base_columns(
            eigs_count=int(self.params["lap_eigs_count"]),
            include_entropy=bool(self.params.get("lap_include_entropy", True)),
            include_centered=bool(self.params.get("lap_use_centered_moments", True)),
            include_fiedler=bool(self.params.get("lap_include_fiedler", True)),
            include_kirchhoff=bool(self.params.get("lap_include_kirchhoff", True)),
            heat_times=self.params.get("lap_heat_times") or (0.1, 1.0, 5.0),
        )
        self._feature_dim_single = len(self._base_columns)
        radii = self.params.get("lap_multi_radii")
        if radii:
            radii_list = self._normalize_radii_list(radii)
            self._radii = radii_list
            self._feature_dim = len(radii_list) * self._feature_dim_single
            self._columns = self._expand_columns(radii_list)
        else:
            self._radii = None
            self._feature_dim = self._feature_dim_single
            self._columns = self._base_columns

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        params = dict(cls._metadata.defaults)
        param_comments = {
            "neighbor_distance": "single-scale cutoff (Å) when lap_multi_radii is null; default 8.0",
            "lap_multi_radii": "multi-scale cutoffs; default [6.0, 8.0, 10.0] to concatenate 3 blocks; set null for single-scale",
            "lap_k_neighbors": "optional deterministic k-NN cap; null uses distance cutoff only",
            "lap_max_neighbors": "hard cap on neighborhood size (includes target); guards cost",
            "lap_estimator": "exact for small neighborhoods; slq for faster moments when node_count > size_threshold",
            "lap_size_threshold": "switch to SLQ when node_count exceeds this",
            "lap_slq_probes": "SLQ probes (higher = more accurate, slower)",
            "lap_slq_steps": "SLQ Lanczos steps (higher = more accurate, slower)",
            "lap_weight": "unweighted (stable) default; gaussian uses sigma=cutoff/2; inverse = 1/d",
            "lap_sigma": "only for gaussian weight; default to cutoff/2 when unset",
            "lap_normalize": "sym (default) or rw Laplacian",
            "lap_eigs_count": "default 32 for richness; 24 for leaner/faster",
            "lap_heat_times": "heat trace times; shorter=local, longer=global",
            "lap_include_entropy": "add spectral entropy (on by default)",
            "lap_include_fiedler": "add Fiedler vector stats (on by default)",
            "lap_include_kirchhoff": "add Kirchhoff/effective resistance proxy (on by default)",
            "lap_use_centered_moments": "include centered moments k2-4 (on by default)",
            "lap_profile": "log Laplacian timing per structure when true",
        }
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "alternates": [
                {
                    "module": cls.module_id,
                    "alias": "Laplacian-only spectral topology (single-scale 8 Å, 32 eigs)",
                    "params": {**params, "lap_multi_radii": None},
                    "summary": cls._metadata.summary,
                    "description": cls._metadata.description,
                }
            ],
        }

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        nd = params.get("neighbor_distance")
        if nd is not None:
            params["neighbor_distance"] = require_positive_float(nd, "topology.params.neighbor_distance")

        def _validate_radii(value: Any) -> Tuple[float, ...]:
            if value is None:
                return ()
            if isinstance(value, (int, float)):
                return (float(value),)
            if not isinstance(value, (list, tuple)):
                raise ValueError("topology.params.lap_multi_radii must be a list/tuple of positive floats.")
            radii = []
            for entry in value:
                radii.append(require_positive_float(entry, "topology.params.lap_multi_radii"))
            if not radii:
                raise ValueError("topology.params.lap_multi_radii must not be empty when provided.")
            return tuple(radii)

        radii = _validate_radii(params.get("lap_multi_radii"))
        if radii:
            params["lap_multi_radii"] = radii

        k_neighbors = params.get("lap_k_neighbors")
        if k_neighbors is not None:
            params["lap_k_neighbors"] = require_positive_int(k_neighbors, "topology.params.lap_k_neighbors")
        max_neighbors = params.get("lap_max_neighbors")
        if max_neighbors is not None:
            params["lap_max_neighbors"] = require_positive_int(max_neighbors, "topology.params.lap_max_neighbors")
        size_threshold = params.get("lap_size_threshold")
        if size_threshold is not None:
            params["lap_size_threshold"] = require_positive_int(size_threshold, "topology.params.lap_size_threshold")
        estimator = params.get("lap_estimator")
        if estimator is not None:
            est = str(estimator).strip().lower()
            if est not in {"exact", "slq"}:
                raise ValueError("topology.params.lap_estimator must be 'exact' or 'slq'.")
            params["lap_estimator"] = est
        slq_probes = params.get("lap_slq_probes")
        if slq_probes is not None:
            params["lap_slq_probes"] = require_positive_int(slq_probes, "topology.params.lap_slq_probes")
        slq_steps = params.get("lap_slq_steps")
        if slq_steps is not None:
            params["lap_slq_steps"] = require_positive_int(slq_steps, "topology.params.lap_slq_steps")

        weight = params.get("lap_weight")
        if weight is not None:
            mode = str(weight).strip().lower()
            if mode not in {"unweighted", "gaussian", "inverse"}:
                raise ValueError("topology.params.lap_weight must be unweighted, gaussian, or inverse.")
            params["lap_weight"] = mode
        sigma = params.get("lap_sigma")
        if sigma is not None:
            params["lap_sigma"] = require_positive_float(sigma, "topology.params.lap_sigma")
        normalize = params.get("lap_normalize")
        if normalize is not None:
            mode = str(normalize).strip().lower()
            if mode not in {"sym", "rw"}:
                raise ValueError("topology.params.lap_normalize must be 'sym' or 'rw'.")
            params["lap_normalize"] = mode
        eigs_count = params.get("lap_eigs_count")
        if eigs_count is not None:
            params["lap_eigs_count"] = require_positive_int(eigs_count, "topology.params.lap_eigs_count")

        heat_times = params.get("lap_heat_times")
        if heat_times is not None:
            if not isinstance(heat_times, (list, tuple)):
                raise ValueError("topology.params.lap_heat_times must be a list/tuple of floats.")
            params["lap_heat_times"] = tuple(require_positive_float(t, "topology.params.lap_heat_times") for t in heat_times)

        for key in ("lap_include_entropy", "lap_include_fiedler", "lap_include_kirchhoff", "lap_use_centered_moments", "lap_profile"):
            flag = params.get(key)
            if flag is not None:
                params[key] = require_bool(flag, f"topology.params.{key}")

        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")

        # Resolve sigma default if needed
        if params.get("lap_weight") == "gaussian" and params.get("lap_sigma") is None:
            cutoff = params.get("neighbor_distance")
            if cutoff is not None:
                params["lap_sigma"] = float(cutoff) / 2.0

    def generate_topology(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        interface_dir: Path,
        work_dir: Path,
        log_dir: Path,
        sort_artifacts: bool = True,
    ):
        topology_dir = work_dir / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology_log_dir = log_dir / "topology_logs"
        topology_log_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[Tuple[Path, Path, Path, Path]] = []
        for pdb_path in sorted(pdb_paths):
            try:
                relative = pdb_path.relative_to(dataset_dir)
                relative_path = Path(relative)
            except ValueError:
                relative_path = Path(pdb_path.name)
            topo_parent = topology_dir / relative_path.parent
            log_parent = topology_log_dir / relative_path.parent
            topo_parent.mkdir(parents=True, exist_ok=True)
            log_parent.mkdir(parents=True, exist_ok=True)
            iface_path = interface_dir / relative_path.parent / f"{pdb_path.stem}.interface.txt"
            output_path = topo_parent / f"{pdb_path.stem}.topology.csv"
            log_path = log_parent / f"{pdb_path.stem}.log"
            tasks.append((pdb_path, iface_path, output_path, log_path))

        worker_count = max(1, int(self.params.get("jobs") or 1))
        if tasks:
            worker_count = min(worker_count, len(tasks))
        success = 0
        failures: List[Tuple[Path, Path, str]] = []
        elapsed = 0.0

        progress = StageProgress("Topology (Laplacian-only)", len(tasks), dataset_name=dataset_dir.name)

        if tasks:
            import time

            start = time.perf_counter()
            if worker_count <= 1:
                for pdb_path, iface_path, output_path, log_path in tasks:
                    ok, failure = self._process_single(
                        pdb_path, iface_path, output_path, log_path, sort_artifacts=sort_artifacts
                    )
                    if ok:
                        success += 1
                    elif failure:
                        failures.append(failure)
                    progress.increment()
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(self._process_single, pdb_path, iface_path, output_path, log_path, sort_artifacts): (
                            pdb_path,
                            log_path,
                        )
                        for pdb_path, iface_path, output_path, log_path in tasks
                    }
                    for future in as_completed(future_map):
                        pdb_path, log_path = future_map[future]
                        try:
                            ok, failure = future.result()
                            if ok:
                                success += 1
                            elif failure:
                                failures.append(failure)
                        except Exception as exc:  # pragma: no cover
                            failures.append((pdb_path, log_path, str(exc)))
                            log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
                        finally:
                            progress.increment()
            elapsed = time.perf_counter() - start

        return {
            "output_dir": topology_dir,
            "log_dir": topology_log_dir,
            "success": success,
            "failures": failures,
            "elapsed": elapsed,
            "processed": len(tasks),
        }

    def _process_single(
        self,
        pdb_path: Path,
        interface_path: Path,
        output_path: Path,
        log_path: Path,
        sort_artifacts: bool = True,
    ) -> Tuple[bool, Tuple[Path, Path, str] | None]:
        residues = _parse_interface_file(interface_path)
        if not residues:
            failure = (pdb_path, log_path, "No interface residues parsed")
            log_path.write_text(
                f"PDB: {pdb_path}\nStatus: FAILURE\nError: No interface residues parsed\n", encoding="utf-8"
            )
            return False, failure
        try:
            feat_map, lap_time = self._compute_features(residues)
            df = pd.DataFrame.from_dict(feat_map, orient="index", columns=self._columns)
            df.insert(0, "ID", df.index)
            if sort_artifacts and "ID" in df.columns:
                df = df.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
            df.to_csv(output_path, index=False)
            log_lines = [
                f"PDB: {pdb_path}",
                "Status: SUCCESS",
                f"Residues processed: {len(residues)}",
                f"Output file: {output_path}",
                f"Feature dim: {self._feature_dim}",
            ]
            if lap_time is not None:
                log_lines.append(f"Laplacian profile time (s): {lap_time:.4f}")
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return True, None
        except Exception as exc:  # pragma: no cover
            log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
            return False, (pdb_path, log_path, str(exc))

    def _compute_features(self, residues: List[InterfaceResidue]) -> Tuple[Dict[str, List[float]], Optional[float]]:
        profile = bool(self.params.get("lap_profile", False))
        total_time = 0.0
        if self._radii:
            feature_map: Dict[str, List[float]] = {res.descriptor: [] for res in residues}
            for radius in self._radii:
                block, block_time = self._compute_single_scale(residues, radius, profile=profile)
                for key, values in block.items():
                    feature_map[key].extend(values)
                if profile and block_time is not None:
                    total_time += block_time
            return feature_map, (total_time if profile else None)
        block, block_time = self._compute_single_scale(residues, float(self.params["neighbor_distance"]), profile=profile)
        return block, (block_time if profile else None)

    def _compute_single_scale(
        self,
        residues: List[InterfaceResidue],
        radius: float,
        *,
        profile: bool = False,
    ) -> Tuple[Dict[str, List[float]], Optional[float]]:
        params = self.params
        k_neighbors = params.get("lap_k_neighbors")
        k_neighbors_int = int(k_neighbors) if k_neighbors not in (None, "") else None
        max_neighbors = int(params.get("lap_max_neighbors") or 128)
        size_threshold = int(params.get("lap_size_threshold") or 80)
        estimator = str(params.get("lap_estimator") or "exact").strip().lower()
        slq_probes = int(params.get("lap_slq_probes") or 8)
        slq_steps = int(params.get("lap_slq_steps") or 32)
        heat_times = params.get("lap_heat_times") or (0.1, 1.0, 5.0)
        weight_mode = str(params.get("lap_weight") or "unweighted").strip().lower()
        sigma = float(params.get("lap_sigma") or radius / 2.0)
        normalize = str(params.get("lap_normalize") or "sym").strip().lower()
        eigs_count = int(params.get("lap_eigs_count") or 32)
        include_entropy = bool(params.get("lap_include_entropy", True))
        include_centered = bool(params.get("lap_use_centered_moments", True))
        include_fiedler = bool(params.get("lap_include_fiedler", True))
        include_kirchhoff = bool(params.get("lap_include_kirchhoff", True))

        weight_fn = _weight_function(weight_mode, sigma)

        feature_map: Dict[str, List[float]] = {}
        profile_time = 0.0
        for idx, res in enumerate(residues):
            indices, coords = _build_neighbors(
                residues,
                idx,
                distance_cutoff=radius,
                k_neighbors=k_neighbors_int,
                max_neighbors=max_neighbors,
                graph_mode="cross_chain",
            )
            if coords.size == 0:
                feature_map[res.descriptor] = [0.0] * self._feature_dim_single
                continue

            lap_start = time.perf_counter() if profile else None
            L = _build_laplacian(coords, weight_fn=weight_fn, normalize=normalize)
            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
            except np.linalg.LinAlgError:
                eigvals = np.zeros((0,), dtype=float)
                eigvecs = np.zeros((0, 0), dtype=float)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
            # Exclude zeros for spectra metrics
            nonzero = eigvals[eigvals > 1e-8]
            sorted_eigs = np.sort(nonzero) if nonzero.size else np.array([], dtype=float)
            k = min(eigs_count, sorted_eigs.size)
            eig_block = np.zeros(eigs_count, dtype=float)
            if k > 0:
                eig_block[:k] = sorted_eigs[:k]

            lambda_min = float(sorted_eigs[0]) if sorted_eigs.size else 0.0
            lambda_max = float(sorted_eigs[-1]) if sorted_eigs.size else 0.0
            lambda_2 = float(sorted_eigs[1]) if sorted_eigs.size > 1 else 0.0
            lambda_mean = float(sorted_eigs.mean()) if sorted_eigs.size else 0.0
            lambda_var = float(sorted_eigs.var()) if sorted_eigs.size else 0.0
            spectral_radius = lambda_max
            cond_guarded = float(lambda_max / max(lambda_min, 1e-8)) if lambda_max > 0 else 0.0

            entropy = 0.0
            if include_entropy and sorted_eigs.size:
                probs = sorted_eigs / max(sorted_eigs.sum(), 1e-8)
                nz = probs[probs > 0]
                entropy = float(-np.sum(nz * np.log(nz)))

            # Moments
            raw_moments = [0.0, 0.0, 0.0, 0.0]
            centered_moments = [0.0, 0.0, 0.0]
            if sorted_eigs.size:
                raw_moments = [float(np.mean(sorted_eigs ** order)) for order in (1, 2, 3, 4)]
                if include_centered:
                    centered = sorted_eigs - lambda_mean
                    centered_moments = [float(np.mean(centered ** order)) for order in (2, 3, 4)]
                else:
                    centered_moments = [0.0, 0.0, 0.0]
            elif estimator == "slq":
                # fallback moments via SLQ when eigenvalues are not available (degenerate graphs)
                adj = build_unweighted_adjacency([r.coord for r in residues], [r.chain_id for r in residues], radius)
                config = LaplacianMomentConfig(
                    size_threshold=size_threshold,
                    estimator=estimator,
                    slq_probes=slq_probes,
                    slq_steps=slq_steps,
                )
                raw, centered = compute_laplacian_moments(adj, moment_orders=(1, 2, 3, 4), config=config)
                if raw:
                    raw_moments = [float(x) for x in raw[:4]]
                if include_centered and centered:
                    centered_moments = [float(x) for x in centered[:3]]
                else:
                    centered_moments = [0.0, 0.0, 0.0]

            # Heat traces
            heat = []
            if sorted_eigs.size:
                for t in heat_times:
                    heat.append(float(np.sum(np.exp(-sorted_eigs * float(t)))))
            else:
                heat = [0.0 for _ in heat_times]

            fiedler_abs_mean, fiedler_abs_std, fiedler_balance = (0.0, 0.0, 0.0)
            if include_fiedler and eigvecs.size and eigvals.size >= 2:
                fiedler_abs_mean, fiedler_abs_std, fiedler_balance = _fiedler_stats(eigvals, eigvecs)

            kirchhoff = _kirchhoff_proxy(sorted_eigs) if include_kirchhoff else 0.0

            features: List[float] = [float(len(indices))]
            features.extend(eig_block.tolist())
            features.extend(
                [
                    lambda_min,
                    lambda_max,
                    lambda_2,
                    lambda_mean,
                    lambda_var,
                    spectral_radius,
                    cond_guarded,
                ]
            )
            features.append(entropy if include_entropy else 0.0)
            features.extend(raw_moments)
            features.extend(centered_moments if include_centered else [0.0, 0.0, 0.0])
            features.extend(heat)
            features.extend([fiedler_abs_mean, fiedler_abs_std, fiedler_balance] if include_fiedler else [0.0, 0.0, 0.0])
            features.append(kirchhoff if include_kirchhoff else 0.0)
            feature_map[res.descriptor] = features
            if profile and lap_start is not None:
                profile_time += time.perf_counter() - lap_start
        return feature_map, (profile_time if profile else None)

    @staticmethod
    def _build_base_columns(
        eigs_count: int,
        include_entropy: bool,
        include_centered: bool,
        include_fiedler: bool,
        include_kirchhoff: bool,
        heat_times: Sequence[Any],
    ) -> List[str]:
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
        if include_entropy:
            cols.append("lap_entropy")
        cols.extend([f"lap_moment_k{order}" for order in (1, 2, 3, 4)])
        if include_centered:
            cols.extend([f"lap_moment_centered_k{order}" for order in (2, 3, 4)])
        cols.extend([f"lap_heat_t{str(t).replace('.', '_')}" for t in heat_times])
        if include_fiedler:
            cols.extend(["lap_fiedler_abs_mean", "lap_fiedler_abs_std", "lap_fiedler_sign_balance"])
        if include_kirchhoff:
            cols.append("lap_kirchhoff_proxy")
        return cols

    def _expand_columns(self, radii: Sequence[float]) -> List[str]:
        expanded: List[str] = []
        for radius in radii:
            suffix = _format_radius_suffix(float(radius))
            for col in self._base_columns:
                expanded.append(f"{col}_{suffix}")
        return expanded

    @staticmethod
    def _normalize_radii_list(radii: Sequence[Any]) -> Tuple[float, ...]:
        normed = []
        for r in radii:
            normed.append(float(r))
        return tuple(normed)

    def describe(self) -> Dict[str, Any]:
        desc = super().describe()
        desc["defaults"]["feature_dim"] = self._feature_dim
        return desc
