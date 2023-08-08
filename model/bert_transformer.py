# new_transformer.py
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


def gelu(x):
    return 0.5 * x * (1. + torch.erf(x / math.sqrt(2.)))


class MultiHeadAttentionLayer(MessagePassing):
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        hidden_channels: int,
        heads: int = 1,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.d_model = in_channels
        self.d_k = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        self.W_Q = Linear(self.d_model, self.heads * self.d_k)
        self.W_K = Linear(self.d_model, self.heads * self.d_k)
        self.W_V = Linear(self.d_model, self.heads * self.d_k)
        if edge_dim is not None:
            self.W_e = Linear(self.edge_dim, self.heads * self.d_k, bias=False)
            self.K_e = Linear(self.d_k * 3, self.d_k)
            self.V_e = Linear(self.d_k * 3, self.d_k)
            self.FCE = Linear(self.heads * self.d_k, self.edge_dim)
#         else:
#             self.W_e = self.register_parameter('W_e', None)
#             self.K_e = self.register_parameter('K_e', None)
#             self.V_e = self.register_parameter('V_e', None)
#             self.FCE = self.register_parameter('FCE', None)

        self.L_skip = Linear(self.d_model, self.heads * self.d_k,
                                   bias=bias)
        self.FCN = Linear(self.heads * self.d_k, self.d_model)
        self.LNN = nn.LayerNorm(self.d_model)
        self.LNE = nn.LayerNorm(self.edge_dim)

        self.reset_parameters()

    def reset_parameters(self):
#         super().reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()
        if self.edge_dim:
            self.W_e.reset_parameters()
            self.K_e.reset_parameters()
            self.V_e.reset_parameters()
            self.FCE.reset_parameters()
        self.L_skip.reset_parameters()
        self.FCN.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        """Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.d_k

        Q = self.W_Q(x).view(-1, H, C)
        K = self.W_K(x).view(-1, H, C)
        V = self.W_V(x).view(-1, H, C)
        
        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, Q=Q, K=K, V=V,
                             edge_attr=edge_attr, size=None)
        if self.edge_dim is not None:
            self.edge_attr = self.edge_attr.view(-1, self.heads * self.d_k)
            self.edge_attr = self.LNE(self.FCE(self.edge_attr)) + edge_attr

        out = out.view(-1, self.heads * self.d_k)
        out = self.LNN(self.FCN(out)) + x
        
        return out, self.edge_attr
        
    def message(self, Q_i: Tensor, K_i: Tensor, K_j: Tensor, V_i: Tensor, V_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if self.W_e is not None:
            assert edge_attr is not None
            edge_attr = self.W_e(edge_attr).view(-1, self.heads,
                                                      self.d_k)
            K_j = self.K_e(torch.cat([K_i, K_j, edge_attr], dim=-1))

        alpha = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.d_k)
        alpha = softmax(alpha, index, ptr, size_i)
#        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = V_j
        if edge_attr is not None:
            out = self.V_e(torch.cat([V_i, out, edge_attr], dim=-1))
            self.edge_attr = edge_attr * alpha.view(-1, self.heads, 1)

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.d_model}, '
                f'{self.d_k}, heads={self.heads})')


class PoswiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, edge_dim):
        super().__init__()
        self.fcn1 = nn.Linear(d_model, 4 * d_model)
        self.fcn2 = nn.Linear(4 * d_model, d_model)
        self.fce1 = nn.Linear(edge_dim, 4 * edge_dim)
        self.fce2 = nn.Linear(4 * edge_dim, edge_dim)
        self.sp = nn.Softplus()
        self.bnn = nn.BatchNorm1d(d_model)
        self.bne = nn.BatchNorm1d(edge_dim)

    def forward(self, x, e):
        x = self.bnn(self.fcn2(gelu(self.fcn1(x)))) + x
        e = self.bne(self.fce2(gelu(self.fce1(e)))) + e
        return self.sp(x), self.sp(e)


class TransformerConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        hidden_channels: int,
        heads: int = 1,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__()
        self.multi_head_attn = MultiHeadAttentionLayer(
                in_channels, hidden_channels, heads, dropout, edge_dim, bias, **kwargs)
        self.ffn = PoswiseFeedForwardNetwork(in_channels, edge_dim)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        x, edge_attr = self.multi_head_attn(x, edge_index, edge_attr, return_attention_weights)
        x, edge_attr = self.ffn(x, edge_attr)
        return x, edge_attr

