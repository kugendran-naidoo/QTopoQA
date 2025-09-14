
import torch
import numpy as np
from torch_geometric.data import DataLoader


from gat_5_edge1 import GNN_edge1_edgepooling
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import random 
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from torch_geometric.data import Data
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
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph_path = self.file_paths[idx]
        graph_data = torch.load(graph_path)
        return [graph_data]




def collate_fn(batch):
    return Data.from_data_list(batch)

def get_loader(graph_path,model_list):
    graph_list = [os.path.join(graph_path,model+'.pt') for model in model_list]
    dataset = GraphDataset(graph_list)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False,num_workers=4)
    return data_loader


# train
print('loading train data')
g_path = args.graph_dir
train_label=args.train_label_file
train_list = list(pd.read_csv(train_label)['MODEL'])
g_list=[file.split('.')[0] for file in os.listdir(g_path)]
model_list = list(set(train_list) & set(g_list))
train_loader = get_loader(g_path,model_list)

print('loading val data')
val_label=args.val_label_file
val_list = list(pd.read_csv(val_label)['MODEL'])
model_list = list(set(val_list) & set(g_list))
val_loader=get_loader(g_path,model_list)





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
        accelerator='gpu',
        devices=[0],
        # accelerator='cpu',
        max_epochs=Epochs,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback,custom_saver],
        accumulate_grad_batches=_accumulate_grad_batches) 

    model = GNN_edge1_edgepooling(init_lr,args.pooling_type,'zuhe',num_net=1,edge_dim=11,heads=args.attention_head)
    trainer.fit(model,train_loader,val_loader)

    torch.cuda.empty_cache()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   

     
if __name__ == '__main__':
    # wandb.agent(function=run_experiment)
    run_experiment()



