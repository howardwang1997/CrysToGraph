import torch
import numpy as np
import scipy as sp
from torch import nn
import dgl
from dgl import DGLGraph

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


def dgl_get_tg_batch_tensor(g):
    gs = dgl.unbatch(g)
    batch = torch.Tensor(g.num_nodes()).long()
    pointer = 0
    for i in range(len(gs)):
        l = gs[i].num_nodes() 
        batch[pointer: pointer+l] = i
        pointer += l
    return batch

def dgl_get_tg_batch_edge_index(g):
    gs = dgl.unbatch(g)
    batch_edge = torch.Tensor(2, g.num_nodes()).long()
    pointer = 0
    for i in range(len(gs)):
        l = gs[i].num_nodes() 
        batch_edge[0, pointer: pointer+l] = pointer
        batch_edge[1, pointer: pointer+l] = torch.tensor([i for i in range(pointer, pointer+l)])
        pointer += l
    return batch_edge

def tg_batch_to_batch_edge_index(batch):
    batch_edge = torch.Tensor(2, batch.shape[0]).long()
    attn_mask = torch.zeros(batch.shape[0], batch.shape[0]).long()
    pointer = 0
    start = 0
    index = 0
    for i in range(batch.shape[0]):
        if batch[i] != index:
            attn_mask[start:pointer, start:pointer] = 1
            start = pointer
            index += 1
        batch_edge[0,i] = start
        batch_edge[1,i] = pointer
        pointer += 1
    attn_mask[start:, start:] = 1
    return batch_edge, attn_mask

def get_finetune_model_params(model, lr, weight_decay):
    params = list(model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if not ('fc' in n or 'gts' in n)], 'weight_decay': weight_decay, 'lr': 1e-6},
        {'params': [p for n, p in params if 'gts' in n], 'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in params if 'fc' in n], 'weight_decay': weight_decay, 'lr': lr*10}
    ]

    return param_group