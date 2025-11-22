import os
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings(
    "ignore",
    message=".*does not seem to be an mmCIF file",
    module="Bio.PDB.DSSP",
)


class node_fea:
    def __init__(self, model_name, pdb_dir, vertice_dir, topo_dir):
        self.model_name = model_name
        self.vertice_dir = vertice_dir
        self.topo_dir = topo_dir
        self.pdb_dir = pdb_dir

    def sequence_three_letter_one_hot(self, seq_list: list):
        mapping = {
            "ALA": 0,
            "CYS": 1,
            "ASP": 2,
            "GLU": 3,
            "PHE": 4,
            "GLY": 5,
            "HIS": 6,
            "ILE": 7,
            "LYS": 8,
            "LEU": 9,
            "MET": 10,
            "ASN": 11,
            "PRO": 12,
            "GLN": 13,
            "ARG": 14,
            "SER": 15,
            "THR": 16,
            "VAL": 17,
            "TRP": 18,
            "TYR": 19,
            "UNK": 20,
        }
        length = len(seq_list)
        one_hot_array = np.zeros((length, 21))
        for idx, item in enumerate(seq_list):
            item = item.upper()
            if item not in mapping:
                item = "UNK"
            col_idx = mapping[item]
            one_hot_array[idx, col_idx] = 1
        return one_hot_array

    def ss8_one_hot(self, ss8_list: list):
        mapping = {
            "H": [1, 0, 0, 0, 0, 0, 0, 0],
            "B": [0, 1, 0, 0, 0, 0, 0, 0],
            "E": [0, 0, 1, 0, 0, 0, 0, 0],
            "G": [0, 0, 0, 1, 0, 0, 0, 0],
            "I": [0, 0, 0, 0, 1, 0, 0, 0],
            "T": [0, 0, 0, 0, 0, 1, 0, 0],
            "S": [0, 0, 0, 0, 0, 0, 1, 0],
            "-": [0, 0, 0, 0, 0, 0, 0, 1],
        }
        length = len(ss8_list)
        one_hot_array = np.zeros((length, 8))
        for idx, item in enumerate(ss8_list):
            if item not in mapping:
                item = "-"
            one_hot_array[idx, :] = mapping[item]
        return one_hot_array

    def run_dssp(self, pdb_file) -> pd.DataFrame:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self.model_name, pdb_file)
        model = structure[0]
        try:
            dssp = DSSP(model, pdb_file, file_type="PDB", dssp="mkdssp")
        except TypeError:
            dssp = DSSP(model, pdb_file, file_type="PDB", dssp="mkdssp")

        key_list = list(dssp.keys())
        three_letter_list = []
        descriptor_list = []
        ss8_list = []
        rasa_list = []
        phi_list = []
        psi_list = []

        for key in key_list:
            chain_id, res_id = key
            _, resseq, icode = res_id
            residue = model[chain_id][res_id]
            residue_name = residue.get_resname()
            if icode == " ":
                descriptor = f"c<{chain_id}>r<{resseq}>R<{residue_name}>"
            else:
                descriptor = f"c<{chain_id}>r<{resseq}>i<{icode}>R<{residue_name}>"
            descriptor_list.append(descriptor)

            ss8, rasa, phi, psi = dssp[key][2:6]
            ss8_list.append(ss8)
            rasa_list.append(rasa)
            phi_list.append(phi)
            psi_list.append(psi)
            three_letter_list.append(residue_name)

        ss8_array = self.ss8_one_hot(ss8_list)
        ss8_df = pd.DataFrame(ss8_array, columns=[f"SS8_{i}" for i in range(8)])
        one_hot_df = pd.DataFrame(
            self.sequence_three_letter_one_hot(three_letter_list), columns=[f"AA_{i}" for i in range(21)]
        )

        scaler_phi = MinMaxScaler()
        phi_array = np.array(phi_list).reshape(-1, 1)
        phi_list_norm = scaler_phi.fit_transform(phi_array).flatten()

        scaler_psi = MinMaxScaler()
        psi_array = np.array(psi_list).reshape(-1, 1)
        psi_list_norm = scaler_psi.fit_transform(psi_array).flatten()

        feature_df = pd.DataFrame(
            list(zip(descriptor_list, rasa_list, phi_list_norm, psi_list_norm)),
            columns=["ID", "rasa", "phi", "psi"],
        )
        result_df = pd.concat([feature_df, ss8_df, one_hot_df], axis=1)
        return result_df

    def get_topo_col(self):
        e_set = [["C"], ["N"], ["O"], ["C", "N"], ["C", "O"], ["N", "O"], ["C", "N", "O"]]
        e_set_str = ["".join(element) if isinstance(element, list) else element for element in e_set]
        fea_col0 = [f"{obj}_{stat}" for obj in ["death"] for stat in ["sum", "min", "max", "mean", "std"]]
        col_0 = [f"f0_{element}_{fea}" for element in e_set_str for fea in fea_col0]
        fea_col1 = [f"{obj}_{stat}" for obj in ["len", "birth", "death"] for stat in ["sum", "min", "max", "mean", "std"]]
        col_1 = [f"f1_{element}_{fea}" for element in e_set_str for fea in fea_col1]
        topo_col = col_0 + col_1
        return topo_col

    def calculate_fea(self):
        pdb_file = os.path.join(self.pdb_dir, self.model_name + ".pdb")
        vertice_file = os.path.join(self.vertice_dir, self.model_name + ".txt")
        topo_file = os.path.join(self.topo_dir, self.model_name + ".csv")

        basic_fea_df = self.run_dssp(pdb_file)
        vertice_df = pd.read_csv(vertice_file, sep=" ", names=["ID", "co_1", "co_2", "co_3"])

        topo_df = pd.read_csv(topo_file)
        topo_df["ID"] = topo_df["ID"].str.split(" ").apply(lambda x: x[0])

        id_basic_df = pd.merge(vertice_df["ID"], basic_fea_df, on="ID", how="inner")

        topo_col = self.get_topo_col()
        scaler = MinMaxScaler()
        topo_df.loc[:, topo_col] = scaler.fit_transform(topo_df[topo_col])
        merge_df = pd.merge(id_basic_df, topo_df[["ID"] + topo_col], on="ID", how="inner")

        basic_cols = [
            "rasa",
            "phi",
            "psi",
            *[f"SS8_{i}" for i in range(8)],
            *[f"AA_{i}" for i in range(21)],
        ]

        return merge_df, basic_cols + topo_col
