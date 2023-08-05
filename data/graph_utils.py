# graph_utils.py
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


class GaussianFilter(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    from crystal
    from CGCNN
    """
    def __init__(self, dmin=0, dmax=6, step=0.2, var=None, cuda=True):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax+step, step)
        if cuda and torch.cuda.is_available():
            self.filter = self.filter.cuda()
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return torch.exp(-(distances.unsqueeze(-1) - self.filter)**2 /
                      self.var**2)
    
    
class CrystalGraphWrapper():
    def __init__(self, 
                 g: Union[DGLGraph, OptTensor]=None, 
                 gf: Optional[list] = [],
                 line_graph: bool = False):
        """
        
        """
        self.gf_g = [GaussianFilter(0, 6, 0.4),
                     GaussianFilter(-0.8, 3.7, 0.3),
                     GaussianFilter(-2.25, 2.25, 0.3)]
        self.gf_lg = [GaussianFilter(0, 6, 0.4),
                      GaussianFilter(-0.8, 3.7, 0.3),
                      GaussianFilter(-2.25, 2.25, 0.3),
                      GaussianFilter(-1.4, 1.5, 0.1)]
        
        if len(gf) == 0:
            if line_graph: gf = self.gf_lg
            else: gf = self.gf_g
        
        if g is not None:
            self.x, self.edge_index, self.edge_attr, \
            self.batch_index = self.unwrap(g, gf, line_graph) 
    
    def unwrap(self, g: DGLGraph, gf: list=[], line_graph: bool=False):
        """
        
        """
        if not line_graph:
            x_name = 'atom_features'
            e_name = 'spherical'
        else:
            x_name = 'spherical'
            e_name = 'h'
            
        if len(gf) == 0:
            if line_graph: gf = self.gf_lg
            else: gf = self.gf_g
        
        x = g.ndata[x_name]
        edge_attr = g.edata[e_name]
        edge_index = torch.vstack(g.edges())
        batch_index = dgl_get_tg_batch_tensor(g)
        
        if not line_graph:
            edge_attr = torch.hstack([gf[i].expand(edge_attr[:,i]) for i in range(3)]).detach()
        else:
            x = torch.hstack([gf[i].expand(x[:,i]) for i in range(3)]).detach()
            edge_attr = gf[3].expand(edge_attr).detach()
        
        return x, edge_index, edge_attr, batch_index
    
    def get(self):
        return self.x, self.edge_index, self.edge_attr, self.batch_index
        
        
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

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        DGL internal method
    """

    return dgl.laplacian_pe(g, pos_enc_dim, padding=True)

def random_walk_positional_encoding(g, pos_enc_dim):
    return dgl.random_walk_pe(g, pos_enc_dim)

def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.
    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, t = batch
    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch 
