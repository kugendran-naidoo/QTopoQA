#!/usr/bin/env python3
"""Extract topology features using the legacy topo_feature implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from Bio.PDB import PDBParser
import pandas as pd

import sys
sys.path.insert(0, "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa/src")
from topo_feature import topo_fea

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/datasets/training/Dockground_MAF2"),
        help="Directory containing the Dockground_MAF2 PDBs (default: %(default)s)",
    )
    parser.add_argument(
        "--neighbor-distance",
        type=float,
        default=6.0,
        help="Neighborhood radius in Å (default: %(default)s)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Persistence cutoff for H0 bars (default: %(default)s)",
    )
    parser.add_argument(
        "--min-persistence",
        type=float,
        default=0.01,
        help="Minimum persistence to keep bars (default: %(default)s)",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=["all"],
        help="Element filters (default: %(default)s)",
    )
    parser.add_argument(
        "--residues-per-file",
        type=int,
        default=0,
        help="If >0, limit to this many residues per file (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_old_method/work/topology"),
        help="Directory to write CSV outputs (default: %(default)s)",
    )
    return parser.parse_args(list(argv))


def select_residues(pdb_path: Path, limit: int) -> list[str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", str(pdb_path))
    descriptors: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                seq, icode = residue.id[1], residue.id[2] if residue.id[2].strip() else " "
                desc = f"c<{chain.id}>r<{seq}>"
                if icode.strip():
                    desc += f"i<{icode}>"
                desc += f"R<{residue.get_resname()}>"
                descriptors.append(desc)
                if limit > 0 and len(descriptors) >= limit:
                    return descriptors
    return descriptors


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    print("Configuration:")
    print(f"  neighbor_distance: {args.neighbor_distance}")
    print(f"  cutoff: {args.cutoff}")
    print(f"  min_persistence: {args.min_persistence}")
    print(f"  element_filters: {args.elements}")
    print(f"  residues_per_file: {args.residues_per_file}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(args.dataset_dir.rglob("*.pdb"))
    parser = PDBParser(QUIET=True)

    for pdb_path in pdb_files:
        residues = select_residues(pdb_path, args.residues_per_file)
        if not residues:
            print(f"Skipping {pdb_path}: no residues")
            continue

        fea = topo_fea(
            str(pdb_path),
            neighbor_dis=args.neighbor_distance,
            e_set=args.elements,
            res_list=residues,
            Cut=args.cutoff,
        )
        df = fea.cal_fea()
        output_path = output_dir / f"{pdb_path.stem}.topology.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
