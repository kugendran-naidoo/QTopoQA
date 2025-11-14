"""
Local, dependency-free copy of the original graph construction logic
used by topoqa/src/graph.py and topoqa/src/utils.py so that this package
does not import from the original codebase at runtime.

This module reproduces:
- create_graph_compat(...): writes <model>.pt graphs byte-for-byte compatible
  with the original implementation (given identical inputs).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import PDB
from sklearn.preprocessing import MinMaxScaler
from torch_geometric import data as DATA

try:  # pragma: no cover
    from .pdb_utils import create_pdb_parser
except ImportError:  # pragma: no cover
    from pdb_utils import create_pdb_parser  # type: ignore


def get_topo_col() -> List[str]:
    e_set = [["C"], ["N"], ["O"], ["C", "N"], ["C", "O"], ["N", "O"], ["C", "N", "O"]]
    e_set_str = ["".join(element) if isinstance(element, list) else element for element in e_set]
    fea_col0 = [f"{obj}_{stat}" for obj in ["death"] for stat in ["sum", "min", "max", "mean", "std"]]
    col_0 = [f"f0_{element}_{fea}" for element in e_set_str for fea in fea_col0]
    fea_col1 = [f"{obj}_{stat}" for obj in ["len", "birth", "death"] for stat in ["sum", "min", "max", "mean", "std"]]
    col_1 = [f"f1_{element}_{fea}" for element in e_set_str for fea in fea_col1]
    topo_col = col_0 + col_1
    return topo_col


def get_all_col() -> List[str]:
    basic_col = [
        "rasa",
        "phi",
        "psi",
        *[f"SS8_{i}" for i in range(8)],
        *[f"AA_{i}" for i in range(21)],
    ]
    topo_col = get_topo_col()
    col = basic_col + topo_col
    return col


class inter_chain_dis:
    @staticmethod
    def Calculate_distance(Coor_df: pd.DataFrame, arr_cutoff: List[str]):
        Num_atoms = len(Coor_df)
        Distance_matrix_real = np.zeros((Num_atoms, Num_atoms), dtype=float)
        Distance_matrix = np.ones((Num_atoms, Num_atoms), dtype=float)
        chain_list = list(Coor_df["ID"].str[2])
        for i in range(Num_atoms):
            for j in range(i, Num_atoms):
                if chain_list[i] == chain_list[j]:
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                    continue
                x_i = float(Coor_df["co_1"][i])
                y_i = float(Coor_df["co_2"][i])
                z_i = float(Coor_df["co_3"][i])

                x_j = float(Coor_df["co_1"][j])
                y_j = float(Coor_df["co_2"][j])
                z_j = float(Coor_df["co_3"][j])
                dis = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
                if dis <= float(arr_cutoff[0]) or dis >= float(arr_cutoff[1]):
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                else:
                    Distance_matrix[i][j] = 1.0
                    Distance_matrix[j][i] = 1.0
                    Distance_matrix_real[i][j] = dis
                    Distance_matrix_real[j][i] = dis

        return Distance_matrix, Distance_matrix_real


def get_pointcloud_type(descriptor1: str, descriptor2: str, model: PDB.Structure.Structure, e1: str, e2: str):
    import re

    c_pattern = r"c<([^>]+)>"
    r_pattern = r"r<([^>]+)>"
    i_pattern = r"i<([^>]+)>"

    c_match1 = re.search(c_pattern, descriptor1)
    r_match1 = re.search(r_pattern, descriptor1)
    i_match1 = re.search(i_pattern, descriptor1)
    c_match2 = re.search(c_pattern, descriptor2)
    r_match2 = re.search(r_pattern, descriptor2)
    i_match2 = re.search(i_pattern, descriptor2)

    c_content1 = c_match1.group(1) if c_match1 else None
    r_content1 = int(r_match1.group(1)) if r_match1 else None
    i_content1 = i_match1.group(1) if i_match1 else " "
    c_content2 = c_match2.group(1) if c_match2 else None
    r_content2 = int(r_match2.group(1)) if r_match2 else None
    i_content2 = i_match2.group(1) if i_match2 else " "

    res_id1 = (" ", r_content1, i_content1)
    res1 = model[c_content1][res_id1]
    res_id2 = (" ", r_content2, i_content2)
    res2 = model[c_content2][res_id2]

    def _atoms(res, sel):
        if sel == "all":
            return [[float(a.get_coord()[0]), float(a.get_coord()[1]), float(a.get_coord()[1])] for a in res.get_atoms()]
        return [
            [float(a.get_coord()[0]), float(a.get_coord()[1]), float(a.get_coord()[1])] for a in res.get_atoms() if a.get_name()[0] == sel
        ]

    atom_coords1 = np.array(_atoms(res1, e1))
    atom_coords2 = np.array(_atoms(res2, e2))
    return atom_coords1, atom_coords2


def distance_of_two_points(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_dis_histogram(descriptor1: str, descriptor2: str, model, e1: str = "all", e2: str = "all") -> np.ndarray:
    point_cloud1, point_cloud2 = get_pointcloud_type(descriptor1, descriptor2, model, e1, e2)
    number_1 = len(point_cloud1)
    number_2 = len(point_cloud2)
    if number_1 == 0 or number_2 == 0:
        return np.zeros(10, dtype=int)
    dis_list = sorted(
        [distance_of_two_points(point_cloud1[ind_1], point_cloud2[ind_2]) for ind_1 in range(number_1) for ind_2 in range(number_2)]
    )
    dis_list = np.array(dis_list)
    bins = np.arange(1, 11, 1)
    bins = np.append(bins, np.inf)
    hist, _ = np.histogram(dis_list, bins=bins)
    return hist


def get_atom_dis(vertice_df: pd.DataFrame, model, edge: Tuple[int, int]) -> List[int]:
    hist = get_dis_histogram(vertice_df["ID"][edge[0]], vertice_df["ID"][edge[1]], model)
    return hist.tolist()


def get_element_index_dis_atom(
    mat_re: np.ndarray, mat: np.ndarray, num: float, vertice_df_filter: pd.DataFrame, model
) -> Tuple[List[List[int]], np.ndarray]:
    arr_index: List[List[int]] = []
    edge_atrr: List[List[float]] = []

    for i in range(len(mat)):
        for j in range(i + 1, len(mat[i])):
            if float(mat[i][j]) == num:
                hists = get_atom_dis(vertice_df_filter, model, [i, j])
                edge_atrr.append([mat_re[i][j]] + hists)
                edge_atrr.append([mat_re[i][j]] + hists)
                arr_index.append([i, j])
                arr_index.append([j, i])

    edge_atrr = np.array(edge_atrr)
    scaler = MinMaxScaler()
    if edge_atrr.size > 0:
        edge_atrr = scaler.fit_transform(edge_atrr)
    return arr_index, edge_atrr


def create_graph_compat(model_name: str, node_dir: str, vertice_dir: str, List_cutoff: List[str], graph_dir: str, pdb_dir: str) -> None:
    node_file = os.path.join(node_dir, model_name + ".csv")
    vertice_file = os.path.join(vertice_dir, model_name + ".txt")
    pdb_file = os.path.join(pdb_dir, model_name + ".pdb")
    try:
        fea_df = pd.read_csv(node_file)
        vertice_df = pd.read_csv(vertice_file, sep=" ", names=["ID", "co_1", "co_2", "co_3"])
        vertice_df_filter = pd.merge(fea_df, vertice_df, how="left", on="ID")[["ID", "co_1", "co_2", "co_3"]]
        fea_col = get_all_col()

        parser = create_pdb_parser()
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]

        for i in range(len(List_cutoff)):
            curr_cutoff = List_cutoff[i].split("-")
            dis, dis_real = inter_chain_dis.Calculate_distance(vertice_df_filter, curr_cutoff)
            edge_index, edge_attr = get_element_index_dis_atom(dis_real, dis, 1.0, vertice_df_filter, model)

            fea = fea_df[fea_col].values
            GCNData = DATA.Data(
                x=torch.tensor(fea, dtype=torch.float32),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            )
            graph_path = os.path.join(graph_dir, model_name + ".pt")
            os.makedirs(graph_dir, exist_ok=True)
            torch.save(GCNData, graph_path)
    except Exception as e:
        print(f"error in protein {model_name}: {e}")
