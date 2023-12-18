# NN.py
import torch
from torch import nn
from torch_geometric import nn as tgnn
from torch import Tensor
from torch_geometric.typing import OptTensor

import dgl
from dgl import nn as dglnn
import dgl.function as fn
from .model_utils import dgl_get_tg_batch_tensor

from .transformer import GlobalTransformerLayer

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


class CrysToGraphNet(nn.Module):
    """
    Contrastive pre-training of convolutions.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                atom_fea_len=64, line_fea_len=30, n_conv=3, h_fea_len=128, n_fc=3, n_gt=1,
                embeddings=None, module=None, norm=False, drop=0.0):
        super(CrysToGraphNet, self).__init__()
        self.embeddings = embeddings
        self.embedded = True
        if self.embeddings is None:
            self.embeddings = nn.Linear(orig_atom_fea_len, atom_fea_len)
            self.embedded = False
        else:
            atom_fea_len = orig_atom_fea_len
            
        self.gf = [GaussianFilter(0, 8, 0.2),
                   GaussianFilter(0, 3.2, 0.2),
                   GaussianFilter(-3.2, 3.2, 0.4),
                   GaussianFilter(-1.4, 1.5, 0.1)]

        self.embeddings_to_hidden = nn.Linear(atom_fea_len, h_fea_len)
        self.edge_to_nbr = nn.Linear(nbr_fea_len, nbr_fea_len)
        self.line_to_line = nn.Linear(line_fea_len, line_fea_len)

        if module is None:
            self.convs = nn.ModuleList([tgnn.CGConv(channels=h_fea_len,
                                                   dim=nbr_fea_len,
                                                   batch_norm=True)
                                       for _ in range(n_conv)]) # need modifying with more types of conv layers
        else:
            if isinstance(module, tuple):
                self.convs, self.line_convs = module
            else:
                self.convs = module

        self.pe_to_hidden = nn.Linear(40, 256)

        self.gts = nn.Sequential(*[GlobalTransformerLayer(256, 32, 8, edge_dim=76)
                                   for _ in range(n_gt)])
        self.conv_sp = nn.Softplus()
            
        self.conv_to_fc = nn.Linear(h_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        self.fcs = nn.ModuleList([nn.Linear(in_features=h_fea_len,
                                            out_features=h_fea_len,
                                            bias=True)
                                  for _ in range(n_fc-1)])
        self.softpluses = nn.ModuleList([nn.Softplus()
            for _ in range(n_fc-1)])
        self.fc_out = nn.Linear(h_fea_len, 1, bias=True)
        self.fin_sp = nn.Softplus()

        if norm:
            self.ln_fc = nn.LayerNorm(h_fea_len)
            self.bn = nn.BatchNorm1d(h_fea_len)
            self.bne = nn.BatchNorm1d(nbr_fea_len)
        self.drop = nn.Dropout(drop)
        
    def forward(self, data, contrastive=True):
        atom_fea = data[0].ndata['atom_features']
        nbr_fea = data[0].edata['spherical']
        nbr_fea = torch.hstack([self.gf[i].expand(nbr_fea[:,i]) for i in range(len(self.gf)-1)] + [(data[0].edata['spherical'][:,0] > 8).view(-1,1)]).float()
        nbr_fea_idx = torch.vstack(data[0].edges())
        crystal_atom_idx = dgl_get_tg_batch_tensor(data[0]).long()

        atom_fea = self.embeddings[atom_fea.T[0].long()].float()
        atom_fea = self.embeddings_to_hidden(atom_fea)
        nbr_fea = self.edge_to_nbr(nbr_fea)

        pe = self.pe_to_hidden(data[0].ndata['pe'])

        line_fea = data[1].edata['h']
        line_fea = self.gf[-1].expand(line_fea).float()
        line_fea_idx = torch.vstack(data[1].edges())
        line_fea = self.line_to_line(line_fea)

        for idx in range(len(self.convs)):
            atom_fea, nbr_fea = self.do_mp(self.convs[idx], self.line_convs[idx],
                                           atom_fea, nbr_fea_idx, nbr_fea, line_fea_idx, line_fea, idx)
        if hasattr(self, 'bn'):
            atom_fea = self.bn(atom_fea)
            nbr_fea = self.bne(nbr_fea)

        atom_fea = atom_fea + pe
        for idx in range(len(self.gts)):
            atom_fea = self.conv_sp(self.do_gt(self.gts[idx], atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea))

        crys_fea = tgnn.pool.global_mean_pool(atom_fea, crystal_atom_idx.cuda())
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.conv_to_fc(crys_fea)
        if hasattr(self, 'bn'): crys_fea = self.ln_fc(crys_fea)

        for fc, sp in zip(self.fcs, self.softpluses):
            crys_fea = sp(crys_fea)
            crys_fea = fc(crys_fea)

        crys_fea = self.fin_sp(crys_fea)

        crys_fea = self.drop(crys_fea)
        out_h = self.fc_out(crys_fea)
        out = out_h
                
        return out

    def do_mp(self, conv_n, conv_l, atom_fea, nbr_fea_idx, nbr_fea, line_fea_idx, line_fea, idx):
        nbr_fea, line_fea = conv_l(nbr_fea, line_fea_idx, line_fea)
        atom_fea, nbr_fea = conv_n(atom_fea, nbr_fea_idx, nbr_fea)
        return atom_fea, nbr_fea

    def do_gt(self, gt_layer, atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea):
        return gt_layer(atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea)
    
