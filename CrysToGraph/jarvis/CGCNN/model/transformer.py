import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing

from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    PairTensor,
)
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)

from .model_utils import tg_batch_to_batch_edge_index


def gelu(x):
    return 0.5 * x * (1. + torch.erf(x / math.sqrt(2.)))


class GlobalTransformerLayer(MessagePassing):
    def __init__(self,
                in_channels: int,
                hidden_channels : int,
                heads: int,
                bias: bool = True,
                residual: bool = True,
                edge_dim: int = 0,
                dropout: float = 0.0,
                **kwargs,
                ):
        """
        """
        kwargs.setdefault('aggr', 'mean')
        super().__init__(node_dim=0, **kwargs)
        
        self.d_model = in_channels
        self.d_k = hidden_channels
        self.heads = heads
        self.biased = bias
        self.residual = residual
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.incorp_edge = True if edge_dim != 0 else False
        
        assert type(hidden_channels) is int
        assert type(bias) is bool
        
        self.W_Q = Linear(self.d_model, self.heads * self.d_k, 
                             bias=True, weight_initializer='glorot')
        self.W_K = Linear(self.d_model, self.heads * self.d_k, 
                             bias=False, weight_initializer='glorot')
        self.W_V = Linear(self.d_model, self.heads * self.d_k, 
                             bias=True, weight_initializer='glorot')
        self.bias = nn.Parameter(torch.Tensor(self.d_model))
        
        self.drop_W = nn.Dropout(dropout)
        self.drop_attn = nn.Dropout(dropout)
        
        if self.incorp_edge:
            self.embed_edge = Linear(self.edge_dim, self.d_model,
                                       bias=True, weight_initializer='glorot')
        
        self.L_O = nn.Linear(self.heads * self.d_k, self.d_model, bias=True)
        self.LNN1 = nn.LayerNorm(self.d_model)
        self.LNN2 = nn.LayerNorm(self.d_model)
        
        self.FCN1 = nn.Linear(self.d_model, 2*self.d_model)
        self.FCN2 = nn.Linear(2*self.d_model, self.d_model)
        self.drop_n = nn.Dropout(self.dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
#         super().reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_K.reset_parameters()
        self.L_O.reset_parameters()
        
        self.LNN1.reset_parameters()
        self.LNN2.reset_parameters()
        
        self.FCN1.reset_parameters()
        self.FCN2.reset_parameters()
        
        if self.incorp_edge: 
            self.embed_edge.reset_parameters()
            
        zeros(self.bias)
        
    def forward(self,
               x: Tensor,
               batch_index: Tensor,
               edge_index: Adj,
               edge_attr: OptTensor = None,
               ):
        H, C = self.heads, self.d_k
        n_nodes = x.shape[0]
        
        assert x.dim() == 2

#         edge embeddings as positional encoding
        e = self.embed_edge(edge_attr) if self.incorp_edge else None

#         with no self loop
        if self.incorp_edge:
            edge_index, e = remove_self_loops(
                edge_index, e)
            x = x + self.propagate(edge_index, x=x, edge_attr=e)
            
#         calculate multihead attention
        Q = self.drop_W(self.W_Q(x)).view(-1, H, C)
        K = self.drop_W(self.W_K(x)).view(-1, H, C)
        V = self.drop_W(self.W_V(x)).view(-1, H, C)

#         with self loop
#         the fill_value of edge_attr does not matter, 
#         edge_attr will never be calculated
#         scaled dot product attention and output context
        if self.incorp_edge:
            edge_index, e = add_self_loops(
                edge_index, e, fill_value='mean',
                num_nodes=n_nodes)
        batch_edge_index, self.attn_mask = tg_batch_to_batch_edge_index(batch_index)
        batch_edge_index = batch_edge_index.to(Q.device)
        self.attn_mask = self.attn_mask.to(Q.device)
        
        V = self.edge_updater(batch_edge_index, Q=Q, K=K, V=V)
        
#         concat multihead attention and convert to d_model
#         completed
        h = V.transpose(0,1).contiguous().view(n_nodes, self.heads * self.d_k)
        h = self.L_O(h) # h shaped as x
        
        h = self.LNN1(h)
        if self.residual: h = h + x
        x = h
        
        h = self.FCN1(h)
        h = self.drop_n(gelu(h))
        h = self.FCN2(h)
        h = self.LNN2(h)
        if self.residual: h = h + x
        
        if self.biased:
            h = h + self.bias
        
        return h
    
    def edge_update(self,
                   Q_i: Tensor,
                   K_i: Tensor,
                   V_i: Tensor,
                   index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        Q, K, V = Q_i.transpose(0,1), K_i.transpose(0,1), V_i.transpose(0,1)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        attn = softmax(attn, index, ptr, size_i, dim=-1)
        attn = attn * self.attn_mask
#        attn = self.drop_attn(attn)
        
        return torch.matmul(attn, V)
        
    def message(self,
               x_i: Tensor,
               edge_attr: Tensor):
        return edge_attr + 0 * x_i
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.d_model}, '
                f'{self.d_k}, heads={self.heads})')

