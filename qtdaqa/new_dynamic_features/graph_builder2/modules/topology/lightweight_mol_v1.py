from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
from ...lib.edge_common import InterfaceResidue
from ...lib.progress import StageProgress
from ...lib.topology_runner import _INTERFACE_DESCRIPTOR_RE
from ...lib.new_topological_features import compute_features_for_residues, ResidueDescriptor, TopologicalConfig
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


@register_feature_module
class TopologyLightweightMoLModule(TopologyFeatureModule):
    module_id = "topology/lightweight_MoL/v1"
    module_kind = "topology"
    default_alias = "140D PH + 8D unweighted Lap moments (mu1-4, kappa2-4) = Topology 148D (lean MoL)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Persistent homology plus lean unweighted Laplacian moments per interface residue (no per-eigen outputs).",
        description=(
            "Keeps the 140D persistent homology block and adds 8D unweighted normalized-Laplacian moments "
            "(mu1-4, kappa2-4) on local cross-chain bipartite neighborhoods (cutoff 8 A by default). Uses exact "
            "eigenvalues when node_count <= size_threshold (80), otherwise SLQ (probes=8, steps=32). Deterministic; "
            "no heat trace in v1."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "neighbor_distance": "Neighbourhood radius in Å for PH computation.",
            "filtration_cutoff": "Maximum filtration value in Å for PH.",
            "min_persistence": "Minimum persistence threshold for PH features.",
            "dedup_sort": "Enable deduplication & sorting of persistence pairs.",
            "element_filters": "Sequence of element subsets considered for PH statistics.",
            "lap_k_neighbors": "Optional deterministic k-NN cap for Laplacian neighborhood.",
            "lap_max_neighbors": "Cap on Laplacian neighborhood size (includes target residue).",
            "lap_size_threshold": "Node count threshold for switching from exact eigs to SLQ.",
            "lap_estimator": "exact (default) or slq for Laplacian moments when above threshold.",
            "lap_slq_probes": "Number of probe vectors for SLQ estimator.",
            "lap_slq_steps": "Lanczos steps placeholder (kept for interface compatibility).",
            "lap_profile": "Profile Laplacian moment wall time (writes to log).",
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
            "lap_k_neighbors": None,
            "lap_max_neighbors": 128,
            "lap_size_threshold": 80,
            "lap_estimator": "exact",
            "lap_slq_probes": 8,
            "lap_slq_steps": 32,
            "lap_profile": False,
            "jobs": 16,
        },
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._ph_columns = _ph_column_names(self.params["element_filters"])
        self._lap_columns = [
            "lap_mol_num_nodes",
            "lap_mol_mu1",
            "lap_mol_mu2",
            "lap_mol_mu3",
            "lap_mol_mu4",
            "lap_mol_kappa2",
            "lap_mol_kappa3",
            "lap_mol_kappa4",
        ]
        self._feature_dim = len(self._ph_columns) + len(self._lap_columns)

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        template["alias"] = cls.default_alias
        template["summary"] = cls._metadata.summary
        template["description"] = cls._metadata.description
        template["param_comments"] = {
            "lap_k_neighbors": "Optional deterministic k-NN cap; otherwise cutoff-based",
            "lap_max_neighbors": "Cap neighborhood size (includes target residue)",
            "lap_size_threshold": "Node-count threshold for switching from exact eigs to SLQ (default 80)",
            "lap_estimator": "exact (default) or slq",
            "lap_slq_probes": "Probe vectors for SLQ (default 8)",
            "lap_slq_steps": "Lanczos steps placeholder (default 32; kept for compatibility)",
            "lap_profile": "Profile Laplacian moment wall time (writes to log)",
        }
        template["summary"] = cls._metadata.summary
        template["description"] = cls._metadata.description
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
        lap_profile = params.get("lap_profile")
        if lap_profile is not None:
            params["lap_profile"] = require_bool(lap_profile, "topology.params.lap_profile")

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

        progress = StageProgress("Topology (PH+MoL)", len(tasks), dataset_name=dataset_dir.name)

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
            import time

            lap_elapsed = 0.0
            ph_df = compute_features_for_residues(pdb_path, descriptors, config)
            if ph_df.empty or "ID" not in ph_df.columns:
                raise ValueError("PH computation returned no data or missing ID column")
            if sort_artifacts:
                ph_df = ph_df.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
            lap_start = time.perf_counter()
            lap_features = self._compute_lap_block(residues)
            lap_elapsed = time.perf_counter() - lap_start
            lap_df = pd.DataFrame.from_dict(lap_features, orient="index", columns=self._lap_columns)
            lap_df.insert(0, "ID", lap_df.index)
            combined = ph_df.merge(lap_df, on="ID", how="left")
            for col in self._lap_columns:
                if col not in combined.columns:
                    combined[col] = 0.0
            combined = combined[["ID", *self._ph_columns, *self._lap_columns]]
            combined.to_csv(output_path, index=False)
            log_lines = [
                f"PDB: {pdb_path}",
                "Status: SUCCESS",
                f"Residues processed: {len(residues)}",
                f"Output file: {output_path}",
                f"Feature dim: {self._feature_dim}",
            ]
            if self.params.get("lap_profile"):
                log_lines.append(f"Laplacian time (s): {lap_elapsed:.4f}")
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return True, None
        except Exception as exc:  # pragma: no cover
            log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
            return False, (pdb_path, log_path, str(exc))

    def _compute_lap_block(self, residues: List[InterfaceResidue]) -> Dict[str, List[float]]:
        cutoff = float(self.params.get("neighbor_distance") or 0.0)
        k_neighbors = self.params.get("lap_k_neighbors")
        k_neighbors_int = int(k_neighbors) if k_neighbors not in (None, "") else None
        max_neighbors = int(self.params.get("lap_max_neighbors") or 128)
        size_threshold = int(self.params.get("lap_size_threshold") or 80)
        estimator = str(self.params.get("lap_estimator") or "exact").strip().lower()
        slq_probes = int(self.params.get("lap_slq_probes") or 8)
        slq_steps = int(self.params.get("lap_slq_steps") or 32)
        moment_orders = (1, 2, 3, 4)

        config = LaplacianMomentConfig(
            size_threshold=size_threshold,
            estimator=estimator,
            slq_probes=slq_probes,
            slq_steps=slq_steps,
        )

        feature_map: Dict[str, List[float]] = {}
        for idx, res in enumerate(residues):
            candidates: List[Tuple[float, int, str]] = []
            for j, other in enumerate(residues):
                if idx == j or other.chain_id == res.chain_id:
                    continue
                dist = float(np.linalg.norm(res.coord - other.coord))
                if cutoff > 0 and dist > cutoff:
                    continue
                candidates.append((dist, j, other.descriptor))
            candidates.sort(key=lambda item: (item[0], item[2]))
            if k_neighbors_int is not None and k_neighbors_int > 0:
                candidates = candidates[:k_neighbors_int]
            selected = candidates[: max(0, max_neighbors - 1)]
            node_indices = [idx] + [j for _, j, _ in selected]
            node_coords = [residues[j].coord for j in node_indices]
            node_chains = [residues[j].chain_id for j in node_indices]
            adj = build_unweighted_adjacency(node_coords, node_chains, cutoff)
            raw, centered = compute_laplacian_moments(adj, moment_orders=moment_orders, config=config)
            # raw orders [1,2,3,4]; centered orders [2,3,4]
            features = [float(len(node_indices))]
            features.extend(raw[:4])
            if len(centered) >= 3:
                features.extend(centered[:3])
            else:
                features.extend([0.0] * 3)
            feature_map[res.descriptor] = features
        return feature_map
