#!/usr/bin/env python3
import random
import sys
from pathlib import Path

sys.path.insert(0, "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa/src")

from Bio.PDB import PDBParser
import pandas as pd

from topo_feature import topo_fea
from lib.new_topological_features import (
    ResidueDescriptor,
    TopologicalConfig,
    compute_features_for_residues,
)

random.seed(0)

DATASET_ROOT = Path("/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/datasets/training/Dockground_MAF2")
PDB_SAMPLES = sorted(DATASET_ROOT.rglob("*.pdb"))[:5]

NEIGHBOR_DISTANCE = 6.0
CUTOFF = 8.0
MIN_PERSISTENCE = 0.01
ELEMENT_FILTERS = ["all"]
RESIDUES_PER_FILE = 5

parser = PDBParser(QUIET=True)


def select_residue_descriptors(pdb_path: Path) -> list[str]:
    structure = parser.get_structure("struct", str(pdb_path))
    descriptors: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                desc = f"c<{chain.id}>r<{residue.id[1]}>R<{residue.get_resname()}>"
                descriptors.append(desc)
                if len(descriptors) >= RESIDUES_PER_FILE:
                    return descriptors
    return descriptors


def compare_single(pdb_path: Path) -> bool:
    residues = select_residue_descriptors(pdb_path)
    if len(residues) < RESIDUES_PER_FILE:
        print(f"Skipping {pdb_path}: only {len(residues)} residues found")
        return True

    df_old = topo_fea(
        str(pdb_path),
        neighbor_dis=NEIGHBOR_DISTANCE,
        e_set=ELEMENT_FILTERS,
        res_list=residues,
        Cut=CUTOFF,
    ).cal_fea()

    config = TopologicalConfig(
        neighbor_distance=NEIGHBOR_DISTANCE,
        filtration_cutoff=CUTOFF,
        min_persistence=MIN_PERSISTENCE,
        element_filters=ELEMENT_FILTERS,
        workers=None,
        log_progress=False,
    )
    descriptors = [ResidueDescriptor.from_string(r) for r in residues]
    df_new = compute_features_for_residues(pdb_path, descriptors, config)

    if df_old.equals(df_new):
        print(f"Match for {pdb_path}")
        return True

    print(f"Mismatch for {pdb_path}")
    print(df_old.compare(df_new, align_axis=0))
    return False


if __name__ == "__main__":
    all(compare_single(path) for path in PDB_SAMPLES)

