import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional
from .utils import get_all_col,inter_chain_dis,get_element_index_dis_atom
from torch_geometric import data as DATA
import torch
from joblib import Parallel,delayed
from Bio import PDB


    



def create_graph(
    model_name,
    node_dir,
    vertice_dir,
    List_cutoff,
    graph_dir,
    pdb_dir,
    *,
    dump_edges: bool = False,
    edge_dir: Optional[str] = None,
):
    node_file = os.path.join(node_dir,model_name+'.csv')
    vertice_file = os.path.join(vertice_dir,model_name+'.txt')
    pdb_file = os.path.join(pdb_dir,model_name+'.pdb')   
    try:
        fea_df = pd.read_csv(node_file)
        # label_df=pd.read_csv(label_file)
        vertice_df = pd.read_csv(vertice_file,sep=' ',names=['ID','co_1','co_2','co_3'])
        vertice_df_filter = pd.merge(fea_df,vertice_df,how='left',on='ID')[['ID','co_1','co_2','co_3']]
        fea_col = get_all_col()
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein",pdb_file)
        model=structure[0]

        edge_dir_path: Optional[Path] = None
        edge_file_path: Optional[Path] = None
        if dump_edges and edge_dir is not None:
            edge_dir_path = Path(edge_dir)
            edge_dir_path.mkdir(parents=True, exist_ok=True)
            edge_file_path = edge_dir_path / f"{model_name}.edges.csv"
            if edge_file_path.exists():
                edge_file_path.unlink()

        vertex_ids = vertice_df_filter["ID"].tolist()

        for i in range(len(List_cutoff)):
            curr_cutoff = List_cutoff[i].split("-")
            dis,dis_real=inter_chain_dis.Calculate_distance(vertice_df_filter,curr_cutoff)
            edge_index,edge_attr=get_element_index_dis_atom(dis_real,dis,1.0,vertice_df_filter,model)

            fea = fea_df[fea_col].values
            edge_tensor = torch.tensor(edge_attr,dtype=torch.float32)
            edge_index_tensor = torch.LongTensor(edge_index)
            if edge_index_tensor.numel() == 0:
                edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index_tensor = edge_index_tensor.transpose(1, 0)

            if dump_edges and edge_dir_path is not None and edge_file_path is not None:
                src_idx = [int(pair[0]) for pair in edge_index]
                dst_idx = [int(pair[1]) for pair in edge_index]
                src_id = [vertex_ids[idx] if 0 <= idx < len(vertex_ids) else "" for idx in src_idx]
                dst_id = [vertex_ids[idx] if 0 <= idx < len(vertex_ids) else "" for idx in dst_idx]
                row_count = len(src_idx)
                if row_count and edge_tensor.dim() == 2 and edge_tensor.size(0) == row_count and edge_tensor.size(1) > 0:
                    feature_cols = [f"edge_attr_{k:02d}" for k in range(edge_tensor.size(1))]
                    feature_array = edge_tensor.detach().cpu().numpy().copy()
                    edge_df = pd.DataFrame(feature_array, columns=feature_cols)
                else:
                    edge_df = pd.DataFrame(index=range(row_count))
                raw_distance = [float(dis_real[src][dst]) for src, dst in zip(src_idx, dst_idx)]
                edge_df.insert(0, "distance_raw", raw_distance)
                edge_df.insert(0, "dst_id", dst_id)
                edge_df.insert(0, "dst_idx", dst_idx)
                edge_df.insert(0, "src_id", src_id)
                edge_df.insert(0, "src_idx", src_idx)
                edge_df.insert(0, "cutoff", List_cutoff[i])
                edge_df.insert(0, "model", model_name)
                header_needed = not edge_file_path.exists()
                mode = "w" if header_needed else "a"
                edge_df.to_csv(edge_file_path, index=False, mode=mode, header=header_needed)

            GCNData =DATA.Data(x=torch.tensor(fea,dtype=torch.float32),
                                        edge_index=edge_index_tensor,
                                        edge_attr=edge_tensor)
            # GCNData.__setitem__('model_name', [dataname+'&'+model_name])
            graph_path = os.path.join(graph_dir,model_name+'.pt')
            torch.save(GCNData,graph_path)
    except Exception as e:
        print(f'error in protein {model_name}: {e}')
