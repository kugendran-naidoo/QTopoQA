#!/usr/bin/env python3
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

# Allow importing the legacy implementation
sys.path.insert(
    0, "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa/src"
)

from topo_feature import topo_fea  # noqa: E402
from lib.new_topological_features import (  # noqa: E402
    ResidueDescriptor,
    TopologicalConfig,
    compute_features_for_residues,
)

random.seed(0)

DATASET_ROOT = Path(
    "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/datasets/training/Dockground_MAF2"
)
PDB_SAMPLES = sorted(DATASET_ROOT.rglob("*.pdb"))[:5]

NEIGHBOR_DISTANCE = 6.0
CUTOFF = 8.0
MIN_PERSISTENCE = 0.01
ELEMENT_FILTERS = ["all"]
RESIDUES_PER_FILE = 5
TOLERANCE = 1e-8

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

    df_old_aligned = df_old.set_index("ID").sort_index()
    df_new_aligned = df_new.set_index("ID").sort_index()

    if not df_old_aligned.columns.equals(df_new_aligned.columns):
        raise ValueError("Column names/order differ between outputs")
    if len(df_old_aligned) != len(df_new_aligned):
        raise ValueError("Row counts differ between outputs")

    numeric_cols = df_old_aligned.select_dtypes(include=[np.number]).columns
    abs_diff = (df_old_aligned[numeric_cols] - df_new_aligned[numeric_cols]).abs()
    max_diff = abs_diff.to_numpy().max(initial=0.0)

    print(
        f"{pdb_path} -> max abs diff: {max_diff:.3e}",
        end="; ",
        flush=True,
    )

    exceeds = abs_diff > TOLERANCE
    if exceeds.any().any():
        print(f"differences above {TOLERANCE:.1e}")
        rows, cols = np.where(exceeds.to_numpy())
        for r_idx, c_idx in zip(rows, cols):
            residue_id = df_old_aligned.index[r_idx]
            column = numeric_cols[c_idx]
            old_val = df_old_aligned.iloc[r_idx, list(numeric_cols).index(column)]
            new_val = df_new_aligned.iloc[r_idx, list(numeric_cols).index(column)]
            diff_val = abs_diff.iloc[r_idx, c_idx]
            print(
                f"  residue {residue_id}, column '{column}': "
                f"old={old_val:.12g}, new={new_val:.12g}, diff={diff_val:.3e}"
            )
        return False

    print(f"all differences <= {TOLERANCE:.1e}")
    return True


if __name__ == "__main__":
    all(compare_single(path) for path in PDB_SAMPLES)

