import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

import sys
sys.path.append('..')

# /** MLP **/
class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, dropout=0.0):
        super(MLP, self).__init__()
        if isinstance(hidden_size, list):
            last_size = in_features
            self.trunc = []
            for h in hidden_size:
                self.trunc.append(
                    nn.Sequential(
                        nn.Linear(last_size, h),
                        nn.LayerNorm(h, elementwise_affine=False, eps=1e-8),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
                last_size = h
            self.trunc.append(nn.Linear(last_size, out_features))
            self.trunc = nn.Sequential(*self.trunc)
        elif np.isscalar(hidden_size):
            self.trunc = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-8),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-8),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, out_features)
            )
            
    def forward(self, x):
        return self.trunc(x)

# /** DeepFM **/
class FMLayer(nn.Module):
    def __init__(self, num_field):
        super(FMLayer, self).__init__()
        self.num_field = num_field
        self.indices = torch.triu_indices(self.num_field, self.num_field, 1)
        self.output_dim = (num_field * (num_field - 1)) // 2
        
    def forward_diff(self, x1, x2):
        inter = torch.matmul(x1, x2.transpose(-2, -1))
        return inter[:, self.indices[0], self.indices[1]].view(inter.size(0), -1)
        
    def forward(self, x):
        inter = torch.matmul(x, x.transpose(-2, -1))
        return inter[:, self.indices[0], self.indices[1]].view(inter.size(0), -1)
    
class DeepFM(nn.Module):
    def __init__(self, num_fields, embedding_size, hidden_size, dropout_prob):
        super(DeepFM, self).__init__()
        self.num_fields = num_fields

        self.fm = FMLayer(self.num_fields)
        self.fm_out = nn.Linear(self.fm.output_dim, 1)
        
        self.l = MLP(self.num_fields * embedding_size, 1, hidden_size, dropout_prob)

    def forward(self, x_emb):
        # x_emb [batch_size * columns * embedding_size]
        dnn_out = self.l(x_emb.reshape(x_emb.shape[0], -1))
        fm_out = self.fm_out(self.fm(x_emb))
        output = dnn_out + fm_out
        return torch.sigmoid(output.view(-1))

# /** FT-transformer **/
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        include_norm = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim) if include_norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_dropout,
        ff_dropout,
        heads=8,
        dim_head=16,
        include_first_norm=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout,
                          include_norm = (i > 0 or include_first_norm)),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)