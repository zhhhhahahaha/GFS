import torch
import torch.nn as nn

from GFS.algo.Predictor import FMLayer
from GFS.algo.Predictor import Transformer
from GFS.algo.Predictor import MLP

# Support different types of row embedding module
# * Sum_MLP (add up all the columns' value and pass into MLP)
# * Con_MLP (concate all the column's value and pass into MLP)
# * FM (factorization machine)
# * DeepFM (Use DeepFM-like row embedding)
# output format [length * row_emb_num * embedding_size]
class RowEmb_mod(nn.Module):
    def __init__(self, 
                 num_fields, 
                 embedding_size, 
                 dropout, 
                 mode_list=None,
                 # parameter for ftransformer 
                 num_blocks=3, 
                 num_heads=8, 
                 valid_feat=None):
        super(RowEmb_mod, self).__init__()
        self.mode_list = mode_list
        self.row_embed_list = nn.ModuleList()
        self.num_fields = num_fields
        for mode in self.mode_list:
            if mode == 'Sum_MLP':
                self.row_embed_list.append(nn.ModuleDict({
                    'mlp': MLP(embedding_size, embedding_size, 4*embedding_size, dropout)
                }))
            elif mode == 'Con_MLP':
                self.row_embed_list.append(nn.ModuleDict({
                    'mlp': MLP(num_fields*embedding_size, embedding_size, 4*num_fields*embedding_size, dropout)
                }))
            elif mode == 'FM':
                fm_output_dim = (num_fields * (num_fields - 1)) // 2
                self.row_embed_list.append(nn.ModuleDict({
                    'fm': FMLayer(self.num_fields),
                    'fm_out': nn.Linear(fm_output_dim, embedding_size),
                }))
            elif mode == 'DeepFM':
                fm_output_dim = (num_fields * (num_fields - 1)) // 2
                self.num_fields = len(valid_feat) if valid_feat else num_fields
                self.row_embed_list.append(nn.ModuleDict({
                    'fm': FMLayer(self.num_fields),
                    'fm_out': nn.Linear(fm_output_dim, embedding_size),
                    'mlp': MLP(num_fields*embedding_size, embedding_size, 4*num_fields*embedding_size, dropout),
                }))
            elif mode == 'fttransformer':
                self.cls = nn.Parameter(torch.randn(1, 1, embedding_size))
                self.row_embed_list.append(nn.ModuleDict({
                    'transformer': Transformer(
                        dim = embedding_size,
                        depth = num_blocks,
                        heads = num_heads,
                        attn_dropout = dropout,
                        ff_dropout = dropout,
                        include_first_norm=False,
                    ),
                }))         
    
    def forward(self, x_emb):
        # x_emb [batch_size * columns * embedding_size]
        output = []
        for i, row_embed in enumerate(self.row_embed_list):
            if self.mode_list[i] == 'Sum_MLP':
                output.append(row_embed['mlp'](x_emb.sum(dim=1)).unsqueeze(1))
            elif self.mode_list[i] == 'Con_MLP':
                output.append(row_embed['mlp'](x_emb.reshape(x_emb.shape[0], -1)).unsqueeze(1))
            elif self.mode_list[i] == 'FM':
                if x_emb.shape[1] == 1:
                    output.append(torch.zeros(x_emb.shape[0], 1, x_emb.shape[2], device=x_emb.device))
                    continue
                output.append(row_embed['fm_out'](row_embed['fm'](x_emb)).unsqueeze(1))
            elif self.mode_list[i] == 'DeepFM':
                dnn_out = row_embed['mlp'](x_emb.reshape(x_emb.shape[0], -1))
                fm_out = row_embed['fm_out'](row_embed['fm'](x_emb))
                output.append((dnn_out+fm_out).unsqueeze(1))
            elif self.mode_list[i] == 'fttransformer':
                tran_out = torch.cat([self.cls.expand(x_emb.shape[0], 1, -1), x_emb], 1)
                tran_out = row_embed['transformer'](tran_out)
                output.append(tran_out[:, 0].unsqueeze(1))
        output = torch.concat(output, dim=1)
        return output
    
# module deal with continuous feature   
class Con_mod(nn.Module):
    def __init__(self, num_cols, embedding_size):
        super(Con_mod, self).__init__()
        self.con_w = nn.Parameter(torch.Tensor(num_cols, embedding_size))
        nn.init.uniform_(self.con_w, -0.01, 0.01)

        self.con_b = nn.Parameter(torch.Tensor(num_cols, embedding_size))
        nn.init.zeros_(self.con_b)
        self.ReLU = nn.ReLU()
    
    def forward(self, con_batch):
        # con_batch [batch_size * columns]
        con_emb = torch.mul(con_batch.unsqueeze(-1), self.con_w.unsqueeze(0)) + self.con_b.unsqueeze(0)
        con_emb = self.ReLU(con_emb)
        return con_emb

# module deal with aggregation
class Agg_func(nn.Module):
    def __init__(self, embedding_size, dropout, mode=None, avg_d=None, aggregators=None, scalers=None):
        super(Agg_func, self).__init__()
        self.mode = mode
        if mode == 'PNA':
            self.scalers = [SCALERS[scale] for scale in scalers]
            self.posttrans = MLP(in_features=len(aggregators)*len(scalers)*embedding_size,
                                out_features=embedding_size,
                                hidden_size= 4*len(aggregators)*len(scalers)*embedding_size, 
                                dropout=dropout)
            self.avg_d = avg_d

    def forward(self, x, d):
        # x: [num_nodes, len(aggregators) * embedding_size]
        # d: [num_nodes]
        if self.mode =='PNA':
            h = torch.cat([scale(x, D=d, avg_d=self.avg_d) for scale in self.scalers], dim=1) # [batch_size, len(aggregators) * len(scalers) * embedding_size]
            output = self.posttrans(h)

        return output


# Scalar function for PNA
# /**Scalar**/
def scale_identity(h, D=None, avg_d=None):
    return h

def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (torch.log(D + 1) / avg_d).unsqueeze(1)

def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    scaler = (avg_d / torch.log(D+1)).unsqueeze(1)
    scaler = torch.where(torch.isinf(scaler), torch.tensor(0.0).to(scaler.device), scaler)

    return h * scaler

SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation}