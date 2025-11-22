from __future__ import annotations

import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .new_topological_features import (
    ResidueDescriptor,
    TopologicalConfig,
    compute_features_for_residues,
)

_INTERFACE_DESCRIPTOR_RE = re.compile(
    r"^c<(?P<chain>[^>]+)>r<(?P<res>-?\d+)>(?:i<(?P<ins>[^>]+)>)?R<(?P<resname>[^>]+)>$"
)


def _load_interface_descriptors(interface_path: Path) -> Tuple[List[ResidueDescriptor], Optional[str]]:
    descriptors: List[ResidueDescriptor] = []
    try:
        with interface_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                residue_token = stripped.split(None, 1)[0]
                match = _INTERFACE_DESCRIPTOR_RE.match(residue_token)
                if not match:
                    return [], f"Unrecognised descriptor '{residue_token}' in {interface_path}"
                descriptors.append(
                    ResidueDescriptor(
                        chain_id=match.group("chain"),
                        residue_number=int(match.group("res")),
                        residue_name=match.group("resname"),
                        insertion_code=(match.group("ins") or " ").strip() or " ",
                        raw_descriptor=residue_token,
                    )
                )
    except FileNotFoundError:
        return [], f"Interface file not found: {interface_path}"
    except OSError as exc:
        return [], f"Failed to read interface file {interface_path}: {exc}"

    if not descriptors:
        return [], f"No interface residues found in {interface_path}"
    return descriptors, None


def run_topology_stage(
    pdb_paths: Iterable[Path],
    dataset_dir: Path,
    interface_dir: Path,
    work_dir: Path,
    log_dir: Path,
    neighbor_distance: float,
    filtration_cutoff: float,
    min_persistence: float,
    element_filters: Sequence[Sequence[str]],
    dedup_sort: bool = False,
    jobs: Optional[int] = None,
) -> Dict[str, object]:
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

    worker_count = max(1, int(jobs)) if jobs else 1
    success = 0
    failures: List[Tuple[Path, Path, str]] = []
    elapsed = 0.0

    config = TopologicalConfig(
        neighbor_distance=float(neighbor_distance),
        filtration_cutoff=float(filtration_cutoff),
        min_persistence=float(min_persistence),
        element_filters=element_filters,
        workers=None,
        log_progress=False,
        dedup_sort=dedup_sort,
    )

    if tasks:
        start = time.perf_counter()
        if worker_count <= 1:
            for pdb_path, interface_path, output_path, log_path in tasks:
                success += _process_single_topology_task(
                    pdb_path, interface_path, output_path, log_path, config, failures
                )
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_meta = {
                    executor.submit(
                        _process_topology_in_subprocess, pdb_path, interface_path, output_path, config
                    ): (
                        pdb_path,
                        interface_path,
                        output_path,
                        log_path,
                    )
                    for pdb_path, interface_path, output_path, log_path in tasks
                }
                for future in as_completed(future_to_meta):
                    pdb_path, interface_path, output_path, log_path = future_to_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover
                        failures.append((pdb_path, log_path, str(exc)))
                        log_path.write_text(f"Status: FAILURE\nError: {exc}\n", encoding="utf-8")
                    else:
                        if result["error"]:
                            failures.append((pdb_path, log_path, result["error"]))
                            log_path.write_text(
                                f"PDB: {pdb_path}\nStatus: FAILURE\nError: {result['error']}\n", encoding="utf-8"
                            )
                        else:
                            success += 1
                            log_path.write_text(
                                "\n".join(
                                    [
                                        f"PDB: {pdb_path}",
                                        "Status: SUCCESS",
                                        f"Residues processed: {result['residue_count']}",
                                        f"Output file: {output_path}",
                                    ]
                                )
                                + "\n",
                                encoding="utf-8",
                            )
        elapsed = time.perf_counter() - start

    return {
        "output_dir": topology_dir,
        "log_dir": topology_log_dir,
        "success": success,
        "failures": failures,
        "elapsed": elapsed,
        "processed": len(tasks),
    }


def _process_single_topology_task(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    log_path: Path,
    config: TopologicalConfig,
    failures: List[Tuple[Path, Path, str]],
) -> int:
    descriptors, error = _load_interface_descriptors(interface_path)
    if error:
        failures.append((pdb_path, log_path, error))
        log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {error}\n", encoding="utf-8")
        return 0

    try:
        frame = compute_features_for_residues(pdb_path, descriptors, config)
        frame.to_csv(output_path, index=False)
        log_path.write_text(
            "\n".join(
                [
                    f"PDB: {pdb_path}",
                    "Status: SUCCESS",
                    f"Residues processed: {len(descriptors)}",
                    f"Output file: {output_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 1
    except Exception as exc:  # pragma: no cover
        failures.append((pdb_path, log_path, str(exc)))
        log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
        return 0


def _process_topology_in_subprocess(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    config: TopologicalConfig,
) -> Dict[str, object]:
    descriptors, error = _load_interface_descriptors(interface_path)
    if error:
        return {"residue_count": 0, "error": error}
    frame = compute_features_for_residues(pdb_path, descriptors, config)
    frame.to_csv(output_path, index=False)
    return {"residue_count": len(descriptors), "error": None}
