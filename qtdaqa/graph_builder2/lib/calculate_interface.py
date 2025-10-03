import os
import numpy as np
from Bio import PDB
from joblib import Parallel, delayed
from itertools import combinations


class calc_interface:

    def __init__(self, inputfile, interface_cutoff=10):

        # pdb file
        self.inputfile = inputfile

        # interface residues distance to alpha-Carbon
        # default 10 angstrom 
        self.cut = interface_cutoff

    def extract_alpha_carbon_info(self):

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", self.inputfile)

        alpha_carbon_info = []

        # parse PDB file
        # extract alpha carbon info including atomic coordinates of "CA" atoms
        # chain_id, res_id, res_name, ins_code, 3D coords (x,y,z)
        # example 'A', 1, 'ALA' '', 13.456, 13.789, 9.123
        for model in structure:
            for chain in model:
                for residue in chain:

                    if residue.get_id()[0] == " " and residue.has_id("CA"):

                        atom = residue["CA"]
                        chain_id = chain.id
                        res_id = residue.get_id()[1]
                        ins_code = residue.get_id()[2].strip()
                        coords = atom.get_coord()
                        res_name = residue.get_resname()
                        alpha_carbon_info.append((chain_id, res_id, res_name, ins_code, coords))

        return alpha_carbon_info

    def calculate_interface_index(self, alpha_carbon_info):

        interface_index = set()
        for (chain_id1, res_id1, res_name1, ins_code1, coords1), (
            chain_id2,
            res_id2,
            res_name2,
            ins_code2,
            coords2,
        ) in combinations(alpha_carbon_info, 2):
            distance = np.linalg.norm(np.array(coords1) - np.array(coords2))
            if chain_id1 == chain_id2:
                continue
            if distance < self.interface_cutoff:
                interface_index.add((chain_id1, res_id1, res_name1, ins_code1, tuple(coords1)))
                interface_index.add((chain_id2, res_id2, res_name2, ins_code2, tuple(coords2)))

        return sorted(interface_index, key=lambda x: (x[0], int(x[1]), x[2]))

    def write_interface_info(self, alpha_carbon_info, outfile):

        interface_info = self.calculate_interface_index(alpha_carbon_info)

        with open(outfile, "w") as f:
            for chain_id, res_id, res_name, ins_code, coord in interface_info:
                if ins_code == "":
                    f.write(f"c<{chain_id}>r<{res_id}>R<{res_name}> {' '.join(map(str, coord))}\n")
                else:
                    f.write(
                        f"c<{chain_id}>r<{res_id}>i<{ins_code}>R<{res_name}> {' '.join(map(str, coord))}\n"
                    )

    def find_and_write(self, outfile):

        # extract info from pdb file
        alpha_carbon_info = self.extract_alpha_carbon_info()

        # write out interface features
        self.write_interface_info(alpha_carbon_info, outfile)


def interface_batch(pdb_dir, ca_dir, n):
    model_list = [file.split(".")[0] for file in os.listdir(pdb_dir)]
    Parallel(n_jobs=n)(
        delayed(
            lambda model: calc_interface(os.path.join(pdb_dir, f"{model}.pdb"), interface_cutoff=10).find_and_write(
                os.path.join(ca_dir, f"{model}.txt")
            )
        )(model)
        for model in model_list
    )
