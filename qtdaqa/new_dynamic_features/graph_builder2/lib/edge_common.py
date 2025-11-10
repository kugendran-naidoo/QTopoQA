from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser


@dataclass
class InterfaceResidue:
    descriptor: str
    chain_id: str
    residue_seq: int
    insertion_code: str
    residue_name: str
    coord: np.ndarray


class StructureCache:
    def __init__(self, pdb_path: Path):
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", str(pdb_path))

    def get_residue(self, chain_id: str, residue_seq: int, insertion_code: str):
        insertion = insertion_code or " "
        try:
            return self.structure[0][chain_id][(" ", residue_seq, insertion)]
        except KeyError:
            return None
