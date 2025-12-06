from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

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
from ...lib.edge_common import InterfaceResidue
from ...lib.progress import StageProgress
from ...lib.topology_runner import _INTERFACE_DESCRIPTOR_RE, round_topology_frame
from ...lib.new_topological_features import compute_features_for_residues, ResidueDescriptor, TopologicalConfig

PH_DIM_DEFAULT = 140  # 7 element filters * (f0 5 stats + f1 15 stats)
LAP_DIM_DEFAULT = 32  # lap_num_nodes + eigs (16) + 7 stats + entropy + 4 moments + 3 heat traces
FEATURE_DIM_DEFAULT = PH_DIM_DEFAULT + LAP_DIM_DEFAULT


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
                    insertion_code=(match.group("ins") or " ").strip(),
                    residue_name=match.group("resname"),
                    coord=coord,
                )
            )
    return residues


def _ph_column_names(element_filters: Sequence[Sequence[str]]) -> List[str]:
    filters = ["".join(f) if isinstance(f, (list, tuple)) else str(f) for f in element_filters]
    f0_stats = ["death_sum", "death_min", "death_max", "death_mean", "death_std"]
    f1_stats = [
        "len_sum",
        "len_min",
        "len_max",
        "len_mean",
        "len_std",
        "birth_sum",
        "birth_min",
        "birth_max",
        "birth_mean",
        "birth_std",
        "death_sum",
        "death_min",
        "death_max",
        "death_mean",
        "death_std",
    ]
    cols = [f"f0_{flt}_{stat}" for flt in filters for stat in f0_stats]
    cols += [f"f1_{flt}_{stat}" for flt in filters for stat in f1_stats]
    return cols


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
    k_neighbors: int | None,
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

    # Respect max_neighbors (include target)
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
    else:  # symmetric normalized
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
    k_neighbors: int | None,
    max_neighbors: int,
    edge_weight: str,
    sigma: float,
    eigs_count: int,
    moment_orders: Sequence[int],
    heat_times: Sequence[float],
    normalize: str,
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
        return [float(node_count)] + lap_eigs + [0.0] * 7 + [0.0] + [0.0] * len(moment_orders) + [0.0] * len(heat_times)

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    eigvals_sorted = np.sort(eigvals)
    eps = 1e-9
    zero_count = int(np.sum(eigvals_sorted <= eps))
    positive = eigvals_sorted[eigvals_sorted > eps]

    # Per-eigenvalue output (smallest nonzero eigenvalues)
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
    if positive.size > 0:
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


@register_feature_module
class PersistenceLaplacianHybridModule(TopologyFeatureModule):
    module_id = "topology/persistence_laplacian_hybrid/v1"
    module_kind = "topology"
    default_alias = "140D PH (Betti 0 and Betti 1 topology summaries on interface coords) + 32D Laplacian = 172D"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Persistent homology + Laplacian spectral moments per interface residue.",
        description=(
            "Extends the 140D persistent homology descriptors with Laplacian spectral features "
            "computed on local interface residue graphs (cross-chain by default). Laplacian block "
            "includes per-eigen values, spectral stats, entropy, moments, and heat traces."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "neighbor_distance": "Neighbourhood radius in Å for PH computation.",
            "filtration_cutoff": "Maximum filtration value in Å for PH.",
            "min_persistence": "Minimum persistence threshold for PH features.",
            "dedup_sort": "Enable deduplication & sorting of persistence pairs.",
            "element_filters": "Sequence of element subsets considered for PH statistics.",
            "lap_graph_mode": "cross_chain (default) or all to include same-chain neighbors.",
            "lap_distance_cutoff": "Å radius for Laplacian neighborhood (defaults to neighbor_distance).",
            "lap_k_neighbors": "Optional deterministic k-NN cap for Laplacian neighborhood.",
            "lap_edge_weight": "Edge weighting: gaussian (default), inverse, or binary.",
            "lap_sigma": "Sigma for Gaussian edge weights (defaults to neighbor_distance/2).",
            "lap_eigs_count": "Number of smallest nonzero eigenvalues to keep (padded).",
            "lap_moment_orders": "List of spectral moment orders to emit.",
            "lap_heat_times": "List of heat kernel times to emit traces for.",
            "lap_include_entropy": "Include spectral entropy over nonzero eigenvalues.",
            "lap_max_neighbors": "Cap on Laplacian neighborhood size (includes target residue).",
            "lap_normalize": "sym (default symmetric normalized) or rw (random-walk) Laplacian.",
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "neighbor_distance": 8.0,
            "filtration_cutoff": 8.0,
            "min_persistence": 0.01,
            "dedup_sort": False,
            "element_filters": (
                ("C",),
                ("N",),
                ("O",),
                ("C", "N"),
                ("C", "O"),
                ("N", "O"),
                ("C", "N", "O"),
            ),
            "lap_graph_mode": "cross_chain",
            "lap_distance_cutoff": None,
            "lap_k_neighbors": None,
            "lap_edge_weight": "gaussian",
            "lap_sigma": None,
            "lap_eigs_count": 16,
            "lap_moment_orders": (1, 2, 3, 4),
            "lap_heat_times": (0.1, 1.0, 5.0),
            "lap_include_entropy": True,
            "lap_max_neighbors": 128,
            "lap_normalize": "sym",
            "jobs": 16,
        },
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        # Resolve lap_distance_cutoff / sigma defaults against neighbor_distance
        if self.params.get("lap_distance_cutoff") is None:
            self.params["lap_distance_cutoff"] = float(self.params["neighbor_distance"])
        if self.params.get("lap_sigma") is None:
            self.params["lap_sigma"] = float(self.params["neighbor_distance"]) / 2.0

        eigs_count = int(self.params["lap_eigs_count"])
        moment_orders = tuple(self.params.get("lap_moment_orders", (1, 2, 3, 4)))
        heat_times = tuple(self.params.get("lap_heat_times", (0.1, 1.0, 5.0)))
        self._lap_columns = self._build_lap_columns(eigs_count, moment_orders, heat_times)
        self._ph_columns = _ph_column_names(self.params["element_filters"])
        self._feature_dim = len(self._ph_columns) + len(self._lap_columns)

    @staticmethod
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

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        params = dict(template.get("params", {}))
        if params.get("lap_distance_cutoff") is None:
            params["lap_distance_cutoff"] = params.get("neighbor_distance")
        if params.get("lap_sigma") is None and params.get("neighbor_distance") is not None:
            params["lap_sigma"] = float(params["neighbor_distance"]) / 2.0
        template["params"] = params
        param_comments = dict(template.get("param_comments", {}))
        param_comments.setdefault("lap_graph_mode", "cross_chain (default) or all to include same-chain neighbors")
        param_comments.setdefault("lap_edge_weight", "gaussian (default, sigma ~ neighbor_distance/2), inverse, or binary")
        param_comments.setdefault("lap_k_neighbors", "optional deterministic k-NN cap; otherwise cutoff-based")
        param_comments.setdefault("lap_max_neighbors", "cap neighborhood size (includes target residue)")
        template["param_comments"] = param_comments
        template["alias"] = cls.default_alias
        template.setdefault("notes", {})
        template["notes"].update(
            {
                "feature_dim_ph_default": PH_DIM_DEFAULT,
                "feature_dim_lap_default": LAP_DIM_DEFAULT,
                "feature_dim_total_default": FEATURE_DIM_DEFAULT,
            }
        )
        return template

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        for key in ("neighbor_distance", "filtration_cutoff"):
            value = params.get(key)
            if value is not None:
                params[key] = require_positive_float(value, f"topology.params.{key}")
        min_persistence = params.get("min_persistence")
        if min_persistence is not None:
            params["min_persistence"] = require_positive_float(
                min_persistence, "topology.params.min_persistence", allow_zero=True
            )
        dedup_sort = params.get("dedup_sort")
        if dedup_sort is not None:
            params["dedup_sort"] = require_bool(dedup_sort, "topology.params.dedup_sort")
        element_filters = params.get("element_filters")
        if element_filters is not None:
            if isinstance(element_filters, str):
                raise ValueError(
                    "topology.params.element_filters must be a YAML list, not a single string literal."
                )
            if not isinstance(element_filters, (list, tuple)):
                raise ValueError("topology.params.element_filters must be a list/tuple of sequences.")
            normalised = []
            for entry in element_filters:
                if not isinstance(entry, (list, tuple)):
                    raise ValueError("Each element_filters entry must be a list/tuple of element symbols.")
                if not entry:
                    raise ValueError("Element filter sequences must contain at least one symbol.")
                symbols = []
                for symbol in entry:
                    if not isinstance(symbol, str) or not symbol.strip():
                        raise ValueError("Element filter symbols must be non-empty strings.")
                    symbols.append(symbol.strip())
                normalised.append(tuple(symbols))
            params["element_filters"] = tuple(normalised)

        graph_mode = params.get("lap_graph_mode")
        if graph_mode is not None:
            mode = str(graph_mode).strip().lower()
            if mode not in {"cross_chain", "all"}:
                raise ValueError("topology.params.lap_graph_mode must be 'cross_chain' or 'all'.")
            params["lap_graph_mode"] = mode
        lap_distance_cutoff = params.get("lap_distance_cutoff")
        if lap_distance_cutoff is not None:
            params["lap_distance_cutoff"] = require_positive_float(
                lap_distance_cutoff, "topology.params.lap_distance_cutoff"
            )
        lap_k_neighbors = params.get("lap_k_neighbors")
        if lap_k_neighbors is not None:
            params["lap_k_neighbors"] = require_positive_int(lap_k_neighbors, "topology.params.lap_k_neighbors")
        lap_edge_weight = params.get("lap_edge_weight")
        if lap_edge_weight is not None:
            mode = str(lap_edge_weight).strip().lower()
            if mode not in {"gaussian", "inverse", "binary"}:
                raise ValueError("topology.params.lap_edge_weight must be gaussian, inverse, or binary.")
            params["lap_edge_weight"] = mode
        lap_sigma = params.get("lap_sigma")
        if lap_sigma is not None:
            params["lap_sigma"] = require_positive_float(lap_sigma, "topology.params.lap_sigma")
        lap_eigs_count = params.get("lap_eigs_count")
        if lap_eigs_count is not None:
            params["lap_eigs_count"] = require_positive_int(lap_eigs_count, "topology.params.lap_eigs_count")

        for name, label in (("lap_moment_orders", "lap_moment_orders"), ("lap_heat_times", "lap_heat_times")):
            val = params.get(name)
            if val is not None:
                if not isinstance(val, (list, tuple)) or not val:
                    raise ValueError(f"topology.params.{label} must be a non-empty list/tuple.")
                cleaned = []
                for entry in val:
                    cleaned.append(require_positive_float(entry, f"topology.params.{label}"))
                params[name] = tuple(cleaned)

        lap_include_entropy = params.get("lap_include_entropy")
        if lap_include_entropy is not None:
            params["lap_include_entropy"] = require_bool(lap_include_entropy, "topology.params.lap_include_entropy")
        lap_max_neighbors = params.get("lap_max_neighbors")
        if lap_max_neighbors is not None:
            params["lap_max_neighbors"] = require_positive_int(lap_max_neighbors, "topology.params.lap_max_neighbors")
        lap_normalize = params.get("lap_normalize")
        if lap_normalize is not None:
            mode = str(lap_normalize).strip().lower()
            if mode not in {"sym", "rw"}:
                raise ValueError("topology.params.lap_normalize must be 'sym' or 'rw'.")
            params["lap_normalize"] = mode

        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")

    def generate_topology(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        interface_dir: Path,
        work_dir: Path,
        log_dir: Path,
        sort_artifacts: bool = True,
        round_decimals: Optional[int] = None,
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
            topology_output_parent = topology_dir / relative_path.parent
            topology_log_parent = topology_log_dir / relative_path.parent
            topology_output_parent.mkdir(parents=True, exist_ok=True)
            topology_log_parent.mkdir(parents=True, exist_ok=True)
            interface_path = interface_dir / relative_path.parent / f"{pdb_path.stem}.interface.txt"
            topology_output_path = topology_output_parent / f"{pdb_path.stem}.topology.csv"
            topology_log_path = topology_log_parent / f"{pdb_path.stem}.log"
            tasks.append((pdb_path, interface_path, topology_output_path, topology_log_path))

        worker_count = max(1, int(self.params.get("jobs") or 1))
        if tasks:
            worker_count = min(worker_count, len(tasks))
        success = 0
        failures: List[Tuple[Path, Path, str]] = []
        elapsed = 0.0

        progress = StageProgress("Topology (PH+Lap)", len(tasks), dataset_name=dataset_dir.name)

        if tasks:
            import time

            start = time.perf_counter()
            if worker_count <= 1:
                for pdb_path, interface_path, output_path, log_path in tasks:
                    ok, failure = self._process_single(
                        pdb_path,
                        interface_path,
                        output_path,
                        log_path,
                        sort_artifacts=sort_artifacts,
                        round_decimals=round_decimals,
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
                        executor.submit(
                            self._process_single,
                            pdb_path,
                            interface_path,
                            output_path,
                            log_path,
                            sort_artifacts,
                            round_decimals,
                        ): (pdb_path, log_path)
                        for pdb_path, interface_path, output_path, log_path in tasks
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
        round_decimals: Optional[int] = None,
    ) -> Tuple[bool, Tuple[Path, Path, str] | None]:
        residues = _parse_interface_file(interface_path)
        if not residues:
            failure = (pdb_path, log_path, "No interface residues parsed")
            log_path.write_text(
                f"PDB: {pdb_path}\nStatus: FAILURE\nError: No interface residues parsed\n", encoding="utf-8"
            )
            return False, failure

        descriptors = [
            ResidueDescriptor(
                chain_id=res.chain_id,
                residue_number=res.residue_seq,
                residue_name=res.residue_name,
                insertion_code=(res.insertion_code or " "),
                raw_descriptor=res.descriptor,
            )
            for res in residues
        ]

        config = TopologicalConfig(
            neighbor_distance=float(self.params["neighbor_distance"]),
            filtration_cutoff=float(self.params["filtration_cutoff"]),
            min_persistence=float(self.params["min_persistence"]),
            element_filters=self.params["element_filters"],
            workers=None,
            log_progress=False,
            dedup_sort=bool(self.params["dedup_sort"]),
        )

        try:
            ph_df = compute_features_for_residues(pdb_path, descriptors, config)
            if ph_df.empty or "ID" not in ph_df.columns:
                raise ValueError("PH computation returned no data or missing ID column")
            if sort_artifacts:
                ph_df = ph_df.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
            lap_features = self._compute_laplacian_block(residues)
            lap_df = pd.DataFrame.from_dict(lap_features, orient="index", columns=self._lap_columns)
            lap_df.insert(0, "ID", lap_df.index)
            combined = ph_df.merge(lap_df, on="ID", how="left")
            # Fill any missing lap columns with zeros
            for col in self._lap_columns:
                if col not in combined.columns:
                    combined[col] = 0.0
            # Reorder columns to PH block then Laplacian block
            combined = combined[["ID", *self._ph_columns, *self._lap_columns]]
            round_topology_frame(combined, round_decimals)
            combined.to_csv(output_path, index=False)
            log_path.write_text(
                "\n".join(
                    [
                        f"PDB: {pdb_path}",
                        "Status: SUCCESS",
                        f"Residues processed: {len(residues)}",
                        f"Output file: {output_path}",
                        f"Feature dim: {self._feature_dim}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return True, None
        except Exception as exc:  # pragma: no cover
            log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
            return False, (pdb_path, log_path, str(exc))

    def _compute_laplacian_block(self, residues: List[InterfaceResidue]) -> Dict[str, List[float]]:
        params = self.params
        graph_mode = params.get("lap_graph_mode", "cross_chain")
        distance_cutoff = float(params.get("lap_distance_cutoff") or params.get("neighbor_distance"))
        k_neighbors = params.get("lap_k_neighbors")
        k_neighbors_int = int(k_neighbors) if k_neighbors not in (None, "") else None
        max_neighbors = int(params.get("lap_max_neighbors") or 128)
        edge_weight = params.get("lap_edge_weight", "gaussian")
        sigma = float(params.get("lap_sigma") or params.get("neighbor_distance") / 2.0)
        eigs_count = int(params.get("lap_eigs_count") or 16)
        moment_orders = params.get("lap_moment_orders") or (1, 2, 3, 4)
        heat_times = params.get("lap_heat_times") or (0.1, 1.0, 5.0)
        normalize = params.get("lap_normalize", "sym")

        feature_map: Dict[str, List[float]] = {}
        for idx, res in enumerate(residues):
            feats = _laplacian_features(
                residues,
                idx,
                graph_mode=graph_mode,
                distance_cutoff=distance_cutoff,
                k_neighbors=k_neighbors_int,
                max_neighbors=max_neighbors,
                edge_weight=edge_weight,
                sigma=sigma,
                eigs_count=eigs_count,
                moment_orders=moment_orders,
                heat_times=heat_times,
                normalize=normalize,
            )
            feature_map[res.descriptor] = feats
        return feature_map
