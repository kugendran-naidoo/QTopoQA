import torch
import torchmetrics
from torch import optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import pandas as pd
import torch.nn as nn
try:
    from torch_geometric.loader import DataLoader  # PyG >= 2.0
except Exception:
    from torch_geometric.data import DataLoader  # fallback
import numpy as np
import wandb
import random
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pytorch_lightning as pl
from gat_with_edge import GATv2ConvWithEdgeEmbedding1




class GNN_edge1_edgepooling(pl.LightningModule):
    def __init__(self,init_lr,pooling_type,mode,num_net=5,hidden_dim=32,edge_dim=1,output_dim=64,n_output=1,heads=8):
        super().__init__()
        self.mode=mode
        self.init_lr=init_lr
        self.pooling_type=pooling_type
        self.opt = "adam"
        self.weight_decay=0.002
        self.criterion=torchmetrics.MeanSquaredError()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.heads = heads

        self.num_net=num_net
        if self.mode=='ori':
            num_feature_xd=32
        elif self.mode=='all':
            num_feature_xd=52
        elif self.mode=='zuhe':
            num_feature_xd=172
        elif self.mode=='esm':
            num_feature_xd=1252


        self.edge_embed=nn.ModuleList([torch.nn.Linear(edge_dim,hidden_dim) for _ in range(self.num_net)])
        self.embed=nn.ModuleList([torch.nn.Linear(num_feature_xd,hidden_dim) for _ in range(self.num_net)])
        self.conv1=nn.ModuleList([GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=0.25,concat=False) \
                                  for _ in range(self.num_net)])
        self.conv2=nn.ModuleList([GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=0.25,concat=False) \
                                  for _ in range(self.num_net)])       
        self.conv3=nn.ModuleList([GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=0.25,concat=False) \
                                  for _ in range(self.num_net)])
        
        self.fc_edge = nn.Linear(hidden_dim,hidden_dim//2)
        self.protein_fc_1=nn.ModuleList([nn.Linear(hidden_dim+hidden_dim//2,output_dim) for _ in range(num_net)])
        
        # combined layers
        
        self.fc1 = nn.Linear(output_dim,64)
        self.out = nn.Linear(64,n_output)
        
        self.validation_step_outputs = {}
        self.test_step_outputs = {}

    def forward(self, data_data):

        for i,module in enumerate(zip(self.embed,self.conv1,self.conv2,self.conv3,self.protein_fc_1,self.edge_embed)):
            data11=data_data[i]
            x,edge_index,edge_attr,batch = data11.x,data11.edge_index,data11.edge_attr,data11.batch

            # Ensure a valid batch vector exists (some saved graphs may carry a stale None)
            if batch is None:
                try:
                    if x is not None:
                        n_nodes = x.size(0)
                    elif edge_index is not None and edge_index.numel() > 0:
                        n_nodes = int(edge_index.max().item()) + 1
                    else:
                        n_nodes = 0
                    dev = None
                    if edge_index is not None:
                        dev = edge_index.device
                    elif x is not None:
                        dev = x.device
                    batch = torch.zeros(n_nodes, dtype=torch.long, device=dev)
                except Exception:
                    # final fallback on CPU
                    n_nodes = int(x.size(0)) if x is not None else 0
                    batch = torch.zeros(n_nodes, dtype=torch.long)

            
            protein_embed=module[0]
            protein_gat1=module[1]
            protein_gat2=module[2]
            protein_gat3=module[3]
            protein_fc1=module[4]
            edge_embed=module[5]
            
        
            x=protein_embed(x)
            edge_attr = edge_embed(edge_attr)
            x,edge_attr=protein_gat1(x,edge_index,edge_attr)
            # print(edge_attr.shape)
            x=torch.nn.functional.elu(x)
            edge_attr=torch.nn.functional.elu(edge_attr)
            x,edge_attr=protein_gat2(x,edge_index,edge_attr)
            x=torch.nn.functional.elu(x)
            edge_attr=torch.nn.functional.elu(edge_attr)

            
            
            edge_batch_index = batch[edge_index[0]]


            if self.pooling_type == 'add':
                x = global_add_pool(x,batch)
                edge_attr = global_add_pool(edge_attr,edge_batch_index)
            elif self.pooling_type == 'mean':
                x = global_mean_pool(x,batch)
                edge_attr = global_mean_pool(edge_attr,edge_batch_index)

            elif self.pooling_type == 'max':
                x = global_max_pool(x,batch)
                edge_attr = global_max_pool(edge_attr,edge_batch_index)
                return x,edge_attr
            

            edge_attr = self.fc_edge(edge_attr)
            x_edge = torch.cat((x,edge_attr),dim=1)
            x_edge=protein_fc1(x_edge)
            x_edge=self.relu(x_edge)

            if i == 0:
                x11 = x_edge*1/self.num_net
            else:
                x11 = x11 + x_edge*1/self.num_net
        
        xc=self.fc1(x11)
        xc=self.relu(xc)

        out=self.out(xc)
        out=self.sigmoid(out)

        return out            
      

    def configure_optimizers(self):
        if self.opt == 'adam':
            print('USING ADAM')
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.init_lr)
        elif self.opt == 'adamw':
            optimizer = optim.AdamW(self.parameters(),
                                    lr=self.init_lr)
        else:
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.init_lr)
        return optimizer

  
    def training_step(self, train_batch):
        batch_targets = train_batch[0].y
        batch_scores = self.forward(train_batch)
        batch_targets = batch_targets.unsqueeze(1)

        
        train_mse = self.criterion(batch_scores, batch_targets)

        self.log('train_mse', train_mse, on_step=False, on_epoch=True, sync_dist=True,batch_size=16)

        return train_mse
    
    def validation_step(self, val_batch):
        batch_targets = val_batch[0].y
        batch_scores = self.forward(val_batch)
        batch_targets = batch_targets.unsqueeze(1)

        val_mse = self.criterion(batch_scores, batch_targets)
        val_loss = val_mse 
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True,batch_size=16)
        if 'scores' not in self.validation_step_outputs:
            self.validation_step_outputs['scores'] = []
        self.validation_step_outputs['scores'].append(batch_scores)
        if 'true_scores' not in self.validation_step_outputs:
            self.validation_step_outputs['true_scores'] = []
        self.validation_step_outputs['true_scores'].append(batch_targets)        
        # return {'scores': batch_scores, 'true_score':batch_targets}
    def on_validation_epoch_end(self):
        scores = torch.cat([x for x in self.validation_step_outputs['scores']],dim=0)
        true_scores = torch.cat([x for x in self.validation_step_outputs['true_scores']],dim=0)
        scores=scores.view(-1).cpu().data.numpy();true_scores=true_scores.view(-1).cpu().data.numpy()
        # filter NaNs
        mask = np.isfinite(scores) & np.isfinite(true_scores)
        scores = scores[mask]
        true_scores = true_scores[mask]
        # guard zero-variance or too few samples
        if scores.size < 2 or np.std(scores) == 0 or np.std(true_scores) == 0:
            correlation = 0.0
            spearman_corr = 0.0
        else:
            correlation = np.corrcoef(scores, true_scores)[0, 1]
            spearman_corr,_=stats.spearmanr(scores, true_scores)
        self.log('val_pearson_corr',correlation)
        self.log('val_spearman_corr',spearman_corr)
        self.validation_step_outputs.clear()
    def test_step(self, test_batch):
        batch_targets = test_batch[0].y
        batch_name=test_batch[0].model_name
        batch_scores = self.forward(test_batch)
        if 'scores' not in self.test_step_outputs:
            self.test_step_outputs['scores'] = []
        self.test_step_outputs['scores'].append(batch_scores)
        if 'true_scores' not in self.test_step_outputs:
            self.test_step_outputs['true_scores'] = []
        self.test_step_outputs['true_scores'].append(batch_targets)  
        if 'name' not in self.test_step_outputs:
            self.test_step_outputs['name'] = []
        self.test_step_outputs['name'].append(batch_name)  
    # def test_epoch_end(self,outputs):
    def on_test_epoch_end(self):
        scores = torch.cat([x for x in self.test_step_outputs['scores']],dim=0)
        true_scores = torch.cat([x for x in self.test_step_outputs['true_scores']],dim=0)
        scores=scores.view(-1).cpu().data.numpy();true_scores=true_scores.view(-1).cpu().data.numpy()
        # filter NaNs
        mask = np.isfinite(scores) & np.isfinite(true_scores)
        scores = scores[mask]
        true_scores = true_scores[mask]
        test_model_list = [item[0].split('&')[1] for output in self.test_step_outputs['name'] for item in output]
        data_name=self.test_step_outputs['name'][0][0][0].split('&')[0].upper()
        result_df=pd.DataFrame({'MODEL':test_model_list,'DockQ_wave':true_scores,\
                                'pred_dockq_wave':scores})

        dockq_losses=[]
        pearson_corrs=[]
        spearman_corrs=[]
        pdb_list=list(set([x.split('_')[0] for x in result_df['MODEL']]))
        for i in pdb_list:
            ###result only with pdb i
            mask=result_df['MODEL'].str.startswith(i)
            curr_df=result_df[mask]

            ###cal dockq loss
            max_dockq=curr_df['DockQ_wave'].max()
            max_index = curr_df['pred_dockq_wave'].idxmax() ###打分模型top1对应的行
            model_dockq=curr_df.loc[max_index]['DockQ_wave']
            dockq_loss=max_dockq-model_dockq
            dockq_losses.append(dockq_loss)
            curr_pred=np.array(curr_df['pred_dockq_wave'])
            curr_true=np.array(curr_df['DockQ_wave'])
            # filter NaNs and guard zero-variance
            m = np.isfinite(curr_pred) & np.isfinite(curr_true)
            cp = curr_pred[m]; ct = curr_true[m]
            if cp.size < 2 or np.std(cp) == 0 or np.std(ct) == 0:
                pearson_corr = 0.0
                spearman_corr = 0.0
            else:
                pearson_corr=np.corrcoef(cp,ct)[0,1]
                spearman_corr,_=stats.spearmanr(cp, ct)
            pearson_corrs.append(pearson_corr);spearman_corrs.append(spearman_corr)
        ####cal correlation coefficient        
        dockq_pred=np.array(result_df['pred_dockq_wave'])
        dockq_true=np.array(result_df['DockQ_wave'])
        m2 = np.isfinite(dockq_pred) & np.isfinite(dockq_true)
        dp = dockq_pred[m2]; dt = dockq_true[m2]
        if dp.size < 2 or np.std(dp) == 0 or np.std(dt) == 0:
            pearson_corr = 0.0
            spearman_corr = 0.0
        else:
            pearson_corr=np.corrcoef(dp,dt)[0,1]
            spearman_corr,_=stats.spearmanr(dp,dt)

        ####cal mse mae (on filtered arrays)
        mse = mean_squared_error(dp, dt) if dp.size > 0 else 0.0
        mae = mean_absolute_error(dp, dt) if dp.size > 0 else 0.0

        ####cal mean dockq loss
        mean_dockq_loss=np.mean(dockq_losses);std_dockq_loss=np.std(dockq_losses)
        mean_pearson_corr=np.mean(pearson_corrs);mean_spearman_corr=np.mean(spearman_corrs)

        self.log(data_name+'_peason_correlation', pearson_corr)
        self.log(data_name+'_spearman_correlation', spearman_corr)
        self.log(data_name+'_mean_dockq_loss',mean_dockq_loss)
        self.log(data_name+'_std_dockq_loss',std_dockq_loss)
        self.log(data_name+'_mean_pearson_corr',mean_pearson_corr)
        self.log(data_name+'_mean_spearman_corr',mean_spearman_corr)
        self.log(data_name+'_mse',mse)
        self.log(data_name+'_mae',mae)


        self.test_step_outputs.clear()
        return pearson_corr
    

