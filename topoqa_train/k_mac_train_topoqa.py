
import torch
import numpy as np
# Prefer modern DataLoader path for PyG ≥2.0
from torch_geometric.loader import DataLoader


from gat_5_edge1 import GNN_edge1_edgepooling
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import random 
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from torch_geometric.data import Data, Batch
import pandas as pd
from pytorch_lightning.callbacks import Callback
import argparse


def get_args():
    """
    define and parse all training arguments with default values optimized for performance.

    Returns:
      argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Training script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows default values in help
    )
     # ==================== DATA PATHS (REQUIRED) ====================
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Path to directory containing processed training and validation graph data (required)')
    parser.add_argument('--train_label_file', type=str, required=True,
                        help='Path to CSV/file containing train sample labels (required)')
    parser.add_argument('--val_label_file', type=str, required=True,
                        help='Path to CSV/file containing validation sample labels (required)')
    # ==================== MODEL ARCHITECTURE HYPERPARAMETERS ====================
    parser.add_argument('--attention_head', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--pooling_type', type=str, default='mean',
                        help='Type of pooling operation. Options are: "mean" (mean pooling),"max" (max pooling),"add" (sum pooling)')
    # ==================== TRAINING HYPERPARAMETERS ====================
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of samples per training batch')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Initial learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=32,
                        help='Number of batches to accumulate gradients before performing a backward pass and optimizer step')
    parser.add_argument('--seed',type=int,default=222,
                        help='random seed')
    # ==================== OUTPUT & LOGGING SETTINGS ====================
    parser.add_argument('--save_dir', type=str, default='./experiments',
                        help='Directory to save models')
    
    return parser.parse_args()









args = get_args()
BATCH_SIZE=args.batch_size
Epochs=args.num_epochs
_accumulate_grad_batches=args.accumulate_grad_batches
_ckpt_out_dir=args.save_dir
_CURRENT_TIME = datetime.now().strftime("%H_%M_%S")


class GraphDataset(Dataset):
    def __init__(self, file_paths, label_map):
        self.file_paths = file_paths
        self.label_map = label_map  # MODEL -> dockq (float)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph_path = self.file_paths[idx]
        graph_data = torch.load(graph_path)
        # derive MODEL name from filename stem
        model_name = os.path.splitext(os.path.basename(graph_path))[0]
        # remove stale/invalid batch attribute if present
        if hasattr(graph_data, 'batch'):
            try:
                b = getattr(graph_data, 'batch')
                # If batch is None or not a tensor, drop it so Batch can rebuild
                import torch as _t
                if b is None or not isinstance(b, _t.Tensor):
                    delattr(graph_data, 'batch')
            except Exception:
                try:
                    delattr(graph_data, 'batch')
                except Exception:
                    pass
        # attach label from CSV (dockq regression target)
        if model_name in self.label_map:
            y = float(self.label_map[model_name])
            graph_data.y = torch.tensor(y, dtype=torch.float32)
        else:
            # ensure .y exists to avoid AttributeError later
            graph_data.y = torch.tensor(float('nan'), dtype=torch.float32)
        return graph_data




def collate_fn(batch):
    # Model expects a list with length == num_net (1 here)
    # Use Batch to ensure .batch vector is created; rebuild if missing
    B = Batch.from_data_list(batch)
    try:
        bvec = getattr(B, 'batch', None)
        if bvec is None:
            # Rebuild node-to-graph assignment
            N = None
            if getattr(B, 'num_nodes', None) is not None:
                N = int(B.num_nodes)
            elif getattr(B, 'x', None) is not None:
                N = int(B.x.size(0))
            elif getattr(B, 'edge_index', None) is not None and B.edge_index.numel() > 0:
                N = int(B.edge_index.max().item() + 1)
            else:
                N = 0
            B.batch = torch.zeros(N, dtype=torch.long)
    except Exception:
        # As a last resort, set a singleton batch
        N = int(B.x.size(0)) if getattr(B, 'x', None) is not None else 0
        B.batch = torch.zeros(N, dtype=torch.long)
    return [B]

def get_loader(graph_path, model_list, label_map, num_workers: int = 0):
    graph_list = [os.path.join(graph_path, model + '.pt') for model in model_list]
    dataset = GraphDataset(graph_list, label_map)
    # macOS: avoid multiprocessing workers to prevent re-import side effects and FD exhaustion
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    return data_loader


# train
def _build_loaders():
    print('loading train data')
    g_path = args.graph_dir
    train_label = args.train_label_file
    train_df = pd.read_csv(train_label)
    train_list = list(train_df['MODEL'])
    train_label_map = dict(zip(train_df['MODEL'], train_df['dockq']))
    g_list = [file.split('.')[0] for file in os.listdir(g_path)]
    model_list_tr = sorted(set(train_list) & set(g_list))
    train_loader = get_loader(g_path, model_list_tr, train_label_map, num_workers=0)

    print('loading val data')
    val_label = args.val_label_file
    val_df = pd.read_csv(val_label)
    val_list = list(val_df['MODEL'])
    val_label_map = dict(zip(val_df['MODEL'], val_df['dockq']))
    model_list_val = sorted(set(val_list) & set(g_list))
    val_loader = get_loader(g_path, model_list_val, val_label_map, num_workers=0)
    return train_loader, val_loader





def run_experiment():
    # wandb.init()
    seed=args.seed
    set_seed(seed)
    class CustomModelSaver(Callback):
        def __init__(self, save_last_filename):
            self.save_last_filename = save_last_filename

        def on_train_end(self, trainer, pl_module):
            last_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, 'last.ckpt')
            if os.path.exists(last_checkpoint_path):
                new_last_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, self.save_last_filename)
                os.rename(last_checkpoint_path, new_last_checkpoint_path)
    custom_saver = CustomModelSaver(save_last_filename=f'topo_last.ckpt')
    init_lr=args.learning_rate
    # wandb_logger = WandbLogger(id=_CURRENT_TIME,
    #                        offline=False,
    #                        log_model=False)
    checkpoint_callback = ModelCheckpoint(
    dirpath=_ckpt_out_dir,
    filename='topo'+'_{epoch}_{val_loss:.5f}',
    monitor='val_loss',
    save_top_k=3,
    save_last=True,
    mode='min')
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=Epochs,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, custom_saver],
        accumulate_grad_batches=_accumulate_grad_batches)

    # Build loaders only in main process to avoid re-import in workers
    train_loader, val_loader = _build_loaders()

    model = GNN_edge1_edgepooling(init_lr,args.pooling_type,'zuhe',num_net=1,edge_dim=11,heads=args.attention_head)
    trainer.fit(model, train_loader, val_loader)

    torch.cuda.empty_cache()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   

     
if __name__ == '__main__':
    # wandb.agent(function=run_experiment)
    run_experiment()
