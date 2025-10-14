"""Improved interface residue extraction utilities.

This module rewrites the legacy ``calculate_interface`` helpers with clearer
logic, stronger validation, and a friendlier parallel workflow.  The core idea
is unchanged: identify residues whose alpha-carbons lie within a cutoff
distance of residues on other chains and write the results to disk.  However,
the implementation now favours readability and debuggability over terse,
opaque constructs.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
from Bio.PDB import PDBParser


@dataclass(frozen=True)
class ResidueInfo:
    """Information about a residue's alpha carbon."""

    chain_id: str
    residue_id: int
    residue_name: str
    insertion_code: str
    coord: np.ndarray

    def serialise(self) -> str:
        """Serialise the residue for human-readable text output."""
        coord_str = " ".join(f"{value:.6f}" for value in self.coord)
        if self.insertion_code:
            return (
                f"c<{self.chain_id}>r<{self.residue_id}>i<{self.insertion_code}>"
                f"R<{self.residue_name}> {coord_str}"
            )
        return f"c<{self.chain_id}>r<{self.residue_id}>R<{self.residue_name}> {coord_str}"


def _load_structure_alpha_carbons(pdb_path: Path) -> List[ResidueInfo]:
    """Parse *pdb_path* and return the alpha-carbon coordinates for each residue."""
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")
    if pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"Expected a .pdb file, received: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    residues: List[ResidueInfo] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetatm_flag, seq_id, insertion_code = residue.get_id()
                if hetatm_flag.strip() or not residue.has_id("CA"):
                    continue

                atom = residue["CA"]
                residues.append(
                    ResidueInfo(
                        chain_id=chain.id,
                        residue_id=int(seq_id),
                        residue_name=residue.get_resname(),
                        insertion_code=insertion_code.strip(),
                        coord=np.asarray(atom.get_coord(), dtype=float),
                    )
                )
    return residues


def _group_by_chain(residues: Iterable[ResidueInfo]) -> dict[str, List[ResidueInfo]]:
    by_chain: dict[str, List[ResidueInfo]] = {}
    for residue in residues:
        by_chain.setdefault(residue.chain_id, []).append(residue)
    return by_chain


def _find_interactions_for_pair(
    residues_a: Sequence[ResidueInfo],
    residues_b: Sequence[ResidueInfo],
    cutoff_sq: float,
) -> Tuple[set[int], set[int]]:
    """Return indices of residues within the cutoff for a pair of chains."""
    coords_a = np.stack([r.coord for r in residues_a], axis=0)
    coords_b = np.stack([r.coord for r in residues_b], axis=0)

    # Broadcasting vectors keeps the code short and the implementation fast.
    deltas = coords_a[:, None, :] - coords_b[None, :, :]
    distances_sq = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
    close_pairs = np.argwhere(distances_sq <= cutoff_sq)

    indices_a = {int(i) for i, _ in close_pairs}
    indices_b = {int(j) for _, j in close_pairs}
    return indices_a, indices_b


def find_interface_residues(residues: Sequence[ResidueInfo], cutoff: float) -> List[ResidueInfo]:
    """Return residues whose alpha-carbons are within *cutoff* Å of another chain."""
    if not residues:
        return []

    per_chain = _group_by_chain(residues)
    cutoff_sq = float(cutoff) * float(cutoff)

    interface_indices: dict[str, set[int]] = {chain: set() for chain in per_chain}
    chains = sorted(per_chain.keys())

    for idx_a in range(len(chains)):
        residues_a = per_chain[chains[idx_a]]
        for idx_b in range(idx_a + 1, len(chains)):
            residues_b = per_chain[chains[idx_b]]
            if not residues_a or not residues_b:
                continue

            indices_a, indices_b = _find_interactions_for_pair(residues_a, residues_b, cutoff_sq)
            interface_indices[chains[idx_a]].update(indices_a)
            interface_indices[chains[idx_b]].update(indices_b)

    interface_residues: List[ResidueInfo] = []
    for chain_id in chains:
        for index in sorted(interface_indices[chain_id]):
            interface_residues.append(per_chain[chain_id][index])

    return sorted(
        interface_residues,
        key=lambda res: (res.chain_id, res.residue_id, res.insertion_code, res.residue_name),
    )


def write_interface_file(residues: Sequence[ResidueInfo], output_path: Path) -> None:
    """Write interface residues to *output_path* in a simple text format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for residue in residues:
            handle.write(residue.serialise())
            handle.write("\n")


def process_pdb_file(
    pdb_path: Path,
    output_path: Path,
    cutoff: float = 10.0,
    *,
    logger: Optional[LoggingLike] = None,
) -> int:
    """Process a single PDB file and write its interface information."""

    log = logger or NullLogger()
    log.debug(f"Processing PDB file: {pdb_path}")

    residues = _load_structure_alpha_carbons(pdb_path)
    log.debug(f"Parsed {len(residues)} residues from {pdb_path.name}")

    interface_residues = find_interface_residues(residues, cutoff)
    log.debug(f"Identified {len(interface_residues)} interface residues in {pdb_path.name}")

    write_interface_file(interface_residues, output_path)
    log.debug(f"Wrote interface data to {output_path}")
    return len(interface_residues)


def process_directory(
    pdb_dir: Path,
    output_dir: Path,
    cutoff: float = 10.0,
    *,
    workers: int | None = None,
    output_suffix: str = "",
    output_extension: str = "txt",
) -> Mapping[Path, int]:
    """Process every ``*.pdb`` in *pdb_dir* and write output files under *output_dir*.

    The work is parallelised with :class:`concurrent.futures.ProcessPoolExecutor`.
    ``workers`` defaults to ``None``, which lets Python choose an appropriate
    number of processes.

    Returns a mapping of ``pdb_path`` → ``interface_count`` so callers can inspect
    the results or surface statistics.
    """

    pdb_dir = Path(pdb_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not pdb_dir.is_dir():
        raise NotADirectoryError(f"PDB directory does not exist: {pdb_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_suffix:
        output_suffix = f".{output_suffix.strip('.')}"
    output_extension = output_extension.strip(".") or "txt"

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        return {}

    results: MutableMapping[Path, int] = {}
    failures: MutableMapping[Path, Exception] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(
                process_pdb_file,
                pdb_path,
                output_dir / f"{pdb_path.stem}{output_suffix}.{output_extension}",
                cutoff,
            ): pdb_path
            for pdb_path in pdb_files
        }

        for future in concurrent.futures.as_completed(future_to_path):
            pdb_path = future_to_path[future]
            try:
                results[pdb_path] = future.result()
            except Exception as exc:  # pragma: no cover - surfaced to caller
                failures[pdb_path] = exc

    if failures:
        failure_messages = ", ".join(f"{path.name}: {exc}" for path, exc in failures.items())
        raise RuntimeError(f"Failed to process one or more PDB files: {failure_messages}")

    return results


__all__ = [
    "ResidueInfo",
    "find_interface_residues",
    "write_interface_file",
    "process_pdb_file",
    "process_directory",
    "LoggingLike",
    "NullLogger",
]
class LoggingLike(Protocol):
    """Minimal protocol for objects that mirror ``logging.Logger``."""

    def debug(self, message: str, *args, **kwargs) -> None: ...


class NullLogger:
    """Fallback logger that ignores all messages."""

    def debug(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - trivial
        pass
