# NN.py
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as tgnn
from torch import Tensor
from torch_geometric.typing import OptTensor

import dgl
from dgl import nn as dglnn
import dgl.function as fn
from model_utils import dgl_get_tg_batch_tensor

from pooling import GlobalAttentionPooling
from transformer import GlobalTransformerLayer

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
    

# class PreTrainingOnNodes(nn.Module):
#     """
#     Pre-training on nodes for atom representations
#     """
#     def __init__(self, atom_fea_len, nbr_fea_len, vocab_len,
#                  n_conv=3, module=None, atoms=None):
#         super().__init__()
#         self.embeddings = nn.Embedding(vocab_len, atom_fea_len)
#         self.atoms = atoms
        
#         if module is None:
#             self.convs = nn.ModuleList([tgnn.CGConv(channels=atom_fea_len,
#                                                    dim=nbr_fea_len,
#                                                    batch_norm=True)
#                                         for _ in range(n_conv)])
#         else:
#             self.convs = module
            
#         self.pooler = nn.Linear(atom_fea_len, atom_fea_len)
#         self.actv1 = nn.Tanh()
#         self.cls = nn.Linear(atom_fea_len, vocab_len)
        
#     def forward(self, data):
#         atom_fea = data.x
#         nbr_fea = data.edge_attr
#         nbr_fea_idx = data.edge_index.long()
        
#         atom_fea = self.embeddings(atom_fea.T[0])
        
#         for conv_func in self.convs:
#             atom_fea = conv_func(atom_fea, nbr_fea_idx, nbr_fea)
        
#         out = self.pooler(atom_fea)
#         out = self.actv1(out)
#         out = self.cls(out)
#         return out
    

class ACGCNNConv(nn.Module):
    """Xie and Grossman graph convolution function.
    10.1103/PhysRevLett.120.145301
    """

    def __init__(
        self,
        node_features: int = 64,
        edge_features: int = 32,
        return_messages: bool = False,
    ):
        """Initialize torch modules for CGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.return_messages = return_messages

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.linear_src = nn.Linear(node_features, 2 * node_features)
        self.linear_dst = nn.Linear(node_features, 2 * node_features)
        self.linear_edge = nn.Linear(edge_features, 2 * node_features)
        self.bn_message = nn.BatchNorm1d(2 * node_features)

        # final batchnorm
        self.bn = nn.BatchNorm1d(node_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """CGCNN convolution defined in Eq 5.
        10.1103/PhysRevLett.120.14530
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # compute edge messages -- coalesce W_f and W_s from the paper
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))
        g.ndata["h_src"] = self.linear_src(node_feats)
        g.ndata["h_dst"] = self.linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("h_src", "h_dst", "h_nodes"))
        m = g.edata.pop("h_nodes") + self.linear_edge(edge_feats)
        m = self.bn_message(m)

        # split messages into W_f and W_s terms
        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        h_f, h_s = torch.chunk(m, 2, dim=1)
        m = torch.sigmoid(h_f) * F.softplus(h_s)
        g.edata["m"] = m

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=fn.copy_e("m", "z"), reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        out = F.softplus(node_feats + h)

        if self.return_messages:
            return out, m

        return out
    
    
class ContrastivePreTraining(nn.Module):
    """
    Contrastive pre-training of convolutions.
    """
    def __init__(self, orig_atom_fea_len, 
                atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                pooling='global_mean', pooling_dim=10, p_h=1, p_h_dim=40, embeddings=None, module=None):
        super(ContrastivePreTraining, self).__init__()
        self.embeddings = embeddings
        self.embedded = True
        if self.embeddings is None:
            self.embeddings = nn.Linear(orig_atom_fea_len, atom_fea_len)
            self.embedded = False
        else:
            atom_fea_len = orig_atom_fea_len

        self.embeddings_to_hidden = nn.Linear(atom_fea_len, h_fea_len)
        self.edge_to_hidden = nn.Linear(nbr_fea_len, h_fea_len)
        self.line_to_hidden = nn.Linear(line_fea_len, h_fea_len)
            
        if module is None:
            self.convs = nn.ModuleList([tgnn.CGConv(channels=atom_fea_len,
                                                   dim=nbr_fea_len,
                                                   batch_norm=True)
                                       for _ in range(n_conv)]) # need modifying with more types of conv layers
        else:
            self.convs = module
        
        if 'global' in pooling:
            self.pooling_method = 'conventional'
        else:
            self.pooling_method = 'exotic'
            module_list = [nn.Linear(pooling_dim, p_h_dim)]
            for i in range(p_h - 1):
                module_list.append(nn.Linear(p_h_dim, p_h_dim))
            module_list.append(nn.Linear(p_h_dim, 1))
            self.pooling_fc = nn.Sequential(*module_list)
            self.pooling_dim = pooling_dim
        self.pooling = pooling
            
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        self.contr_out = nn.Linear(h_fea_len, h_fea_len)
        
    def forward(self, data):
        atom_fea = data.x
        nbr_fea = data.edge_attr
        nbr_fea_idx = data.edge_index
        crystal_atom_idx = data.batch
        
        if not self.embedded:
            atom_fea = self.embeddings(atom_fea.T[0].long())
        else:
            atom_fea = self.embeddings(atom_fea.T[0].long()).detach()
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea_idx, nbr_fea)
        if self.pooling_method == 'conventional':
            crys_fea = self.global_mean_pooling(atom_fea, crystal_atom_idx)
        elif self.pooling_method == 'exotic':
            crys_fea = self.exotic_pooling(atom_fea, crystal_atom_idx, nbr_fea_idx,
                                           method=self.pooling, pooling_dim=self.pooling_dim, edge_attr = nbr_fea)
#             crys_fea = torch.vstack([self.pooling_fc(crys_fea[i:i+self.pooling_dim, :].T).T
#                                      for i in range(min(crystal_atom_idx), max(crystal_atom_idx)+1)])
        
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.conv_to_fc(crys_fea)
#         contrastive starts here
        crys_fea = self.conv_to_fc_softplus(crys_fea)
    
        out = self.contr_out(crys_fea)
                
        return out

    def global_mean_pooling(self, atom_fea, crystal_atom_idx, **kwargs):
        idxs = [i for i in range(min(crystal_atom_idx), max(crystal_atom_idx)+1)]
        idx_count = torch.bincount(crystal_atom_idx)[min(crystal_atom_idx): max(crystal_atom_idx+1)]
        idx = [[idxs[i]]*idx_count[i] for i in range(len(idxs))]

        idxx = []
        idxx = iinidx = 0
        for i in range(len(atom_fea)):
            idx[idxx][iinidx] = i
            iinidx += 1
            if iinidx  == idx_count[idxx]:
                iinidx = 0
                idxx += 1

        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in idx]
        return torch.cat(summed_fea, dim=0)
    
    def exotic_pooling(self, atom_fea, crystal_atom_idx, edge_index, 
                       method='SAG', pooling_dim=10, edge_attr:OptTensor=None):
        method = method.lower()
        assert method in ['sag', 'topk']
        if method == 'sag':
            pooler = tgnn.SAGPooling(in_channels=atom_fea.shape[-1], ratio=pooling_dim)
        else:
            pooler = tgnn.TopKPooling(in_channels=atom_fea.shape[-1], ratio=pooling_dim)
        
        atom_fea = pooler(x=atom_fea, edge_index=edge_index, edge_attr=edge_attr, batch=crystal_atom_idx)[0]
#        atom_fea = self._padding(atom_fea, pooling_dim, crystal_atom_idx)
        atom_fea = self._mean(atom_fea, pooling_dim, crystal_atom_idx)
        return atom_fea

    def _mean(self, x, dim, crystal_atom_idx):
        idxs = [i for i in range(min(crystal_atom_idx), max(crystal_atom_idx)+1)]
        idx_count = torch.bincount(crystal_atom_idx)[min(crystal_atom_idx): max(crystal_atom_idx+1)]
        idx = [[idxs[i]]*idx_count[i] for i in range(len(idxs))]
        idx = []
        for i in range(len(idxs)):
            if idx_count[i] < dim:
                count = idx_count[i]
            else:
                count = dim
            idx.append([idxs[i]] * count)
        summed_fea = [torch.mean(x[idx_map], dim=0, keepdim=True)
                      for idx_map in idx]
        return torch.cat(summed_fea, dim=0)

    def _padding(self, x, dim, crystal_atom_idx):
        maximum = max(crystal_atom_idx)
        minimum = min(crystal_atom_idx)
        batch_size = maximum - minimum + 1
        padded = torch.zeros((batch_size * dim, x.shape[-1])).to(x.device)
        
        idx_count = torch.bincount(crystal_atom_idx)[minimum: maximum + 1]
        sum_atoms = 0
        for i in range(batch_size):
            if idx_count[i] >= dim:
                idx_count[i] = dim
                padded[i * dim: (i + 1) * dim] = x[sum_atoms: sum_atoms + dim]
            else:
                padded[i * dim: i * dim + idx_count[i]] = x[sum_atoms: sum_atoms + idx_count[i]]
            sum_atoms += idx_count[i]
            
        return padded
    

class Finetuning(nn.Module):
    """
    Contrastive pre-training of convolutions.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                atom_fea_len=64, line_fea_len=30, n_conv=3, h_fea_len=128, n_h=1, n_fc=3,
                pooling='global_mean', pooling_dim=10, p_h=1, p_h_dim=40, embeddings=None, module=None, norm=False, drop=0.0):
        super(Finetuning, self).__init__()
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
        self.pe_to_hidden = nn.Linear(46, 256)

        self.global_transformer = GlobalTransformerLayer(256, 32, 8, edge_dim=76)
        self.global_transformer_2 = GlobalTransformerLayer(256, 32, 8, edge_dim=76)
#        self.global_transformer_3 = GlobalTransformerLayer(256, 32, 8, edge_dim=76)
        self.conv_sp = nn.Softplus()
        
        if 'global' in pooling:
            self.pooling_method = 'conventional'
        elif 'sag' in pooling or 'SAG' in pooling:
            self.pooling_method = 'conventional'
            self.sag = tgnn.SAGPooling(in_channels=h_fea_len, GNN=tgnn.GCNConv)
        elif 'attention' in pooling:
            self.pooling_method = 'attention'
            self.global_attention_pooling = GlobalAttentionPooling(h_fea_len)
        else:
            self.pooling_method = 'exotic'
            module_list = [nn.Linear(pooling_dim, p_h_dim)]
            for i in range(p_h - 1):
                module_list.append(nn.Linear(p_h_dim, p_h_dim))
            module_list.append(nn.Linear(p_h_dim, 1))
            self.pooling_fc = nn.Sequential(*module_list)
            self.pooling_dim = pooling_dim
        self.pooling = pooling
            
        self.conv_to_fc = nn.Linear(h_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.lnn = nn.LayerNorm(nbr_fea_len)
        
        self.contr_out = nn.Linear(h_fea_len, h_fea_len)
        
        self.fcs = nn.ModuleList([nn.Linear(in_features=h_fea_len,
                                            out_features=h_fea_len,
                                            bias=True)
                                  for _ in range(n_fc-1)])
        self.softpluses = nn.ModuleList([nn.Softplus()
            for _ in range(n_fc-1)])
        self.fc_out = nn.Linear(h_fea_len, 1, bias=True)
        self.actv = nn.LeakyReLU()
        self.fin_sp = nn.Softplus()

        if norm:
            self.ln = nn.LayerNorm(h_fea_len)
            self.bn = nn.BatchNorm1d(h_fea_len)
        self.sigm = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
        
    def forward(self, data, contrastive=True):
        atom_fea = data[0].ndata['atom_features']
#        nbr_fea = torch.hstack([data[0].edata['spherical'], data[0].edata['count'].unsqueeze(-1)])
        nbr_fea = data[0].edata['spherical']
        nbr_fea = torch.hstack([self.gf[i].expand(nbr_fea[:,i]) for i in range(len(self.gf)-1)] + [(data[0].edata['spherical'][:,0] > 8).view(-1,1)]).float()
        nbr_fea_idx = torch.vstack(data[0].edges())
        crystal_atom_idx = dgl_get_tg_batch_tensor(data[0]).long()

        if not self.embedded:
            try:
                atom_fea = self.embeddings(atom_fea.T[0].long())
            except TypeError:
                atom_fea = self.embeddings[atom_fea.T[0].long()].float()
        else:
            try:
                atom_fea = self.embeddings(atom_fea.T[0].long()).detach()
            except TypeError:
                atom_fea = self.embeddings[atom_fea.T[0].long()].detach().float()
       
        atom_fea = self.embeddings_to_hidden(atom_fea)
        nbr_fea = self.edge_to_nbr(nbr_fea)
#         pe = torch.cat([data[0].ndata['position'], data[0].ndata['fraction'], data[0].ndata['rw_pe'], data[0].ndata['lap_pe']], dim=-1)
        pe = self.pe_to_hidden(data[0].ndata['pe'])

        line_fea = data[1].edata['h']
        line_fea = self.gf[-1].expand(line_fea).float()
        line_fea_idx = torch.vstack(data[1].edges())
        line_fea = self.line_to_line(line_fea)

        for idx in range(len(self.convs)):
#            nbr_fea, line_fea = self.line_convs[idx](nbr_fea, line_fea_idx, line_fea)
#            nbr_fea = self.lnn(nbr_fea)
#            atom_fea, nbr_fea = self.convs[idx](atom_fea, nbr_fea_idx, nbr_fea)
#            atom_fea, nbr_fea = self.convs[idx](atom_fea, nbr_fea_idx, self.lnn(nbr_fea))
#            pass
            atom_fea, nbr_fea = self.mp(self.convs[idx], self.line_convs[idx], 
                     atom_fea, nbr_fea_idx, nbr_fea, line_fea_idx, line_fea)
        if hasattr(self, 'bn'): atom_fea = self.ln(atom_fea)

        atom_fea = atom_fea + pe
        atom_fea = self.conv_sp(self.gt(self.global_transformer, atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea))
        atom_fea = self.conv_sp(self.gt(self.global_transformer_2, atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea))
#        atom_fea = self.conv_sp(self.gt(self.global_transformer_3, atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea))

        if self.pooling_method == 'conventional':
            crys_fea = self.global_mean_pooling(atom_fea, crystal_atom_idx)
        elif self.pooling_method == 'attention':
            crys_fea = self.global_attention_pooling(atom_fea, crystal_atom_idx)
        elif self.pooling_method == 'exotic':
            crys_fea = self.exotic_pooling(atom_fea, crystal_atom_idx, nbr_fea_idx,
                                           method=self.pooling, pooling_dim=self.pooling_dim, edge_attr = nbr_fea)
#             crys_fea = torch.vstack([self.pooling_fc(crys_fea[i:i+self.pooling_dim, :].T).T
#                                      for i in range(min(crystal_atom_idx), max(crystal_atom_idx)+1)])
        
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.conv_to_fc(crys_fea)
#         contrastive starts here
        if contrastive:
            crys_fea_c = self.conv_to_fc_softplus(crys_fea)
            out_c = self.contr_out(crys_fea_c)
#         if hasattr(self, 'bn'): crys_fea = self.bn(crys_fea)

        for fc, sp in zip(self.fcs, self.softpluses):
            crys_fea = sp(crys_fea)
            crys_fea = fc(crys_fea)

        crys_fea = self.fin_sp(crys_fea)

        crys_fea = self.drop(crys_fea)
        out_h = self.fc_out(crys_fea)
        if contrastive:
            out = (out_c, out_h)
        else:
            out = out_h
                
        return out

    def mp(self, conv_n, conv_l, atom_fea, nbr_fea_idx, nbr_fea, line_fea_idx, line_fea):
        nbr_fea, line_fea = conv_l(nbr_fea, line_fea_idx, line_fea)
        atom_fea, line_fea = conv_n(atom_fea, nbr_fea_idx, self.lnn(nbr_fea))
        return atom_fea, nbr_fea

    def gt(self, gt_layer, atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea):
        return gt_layer(atom_fea, crystal_atom_idx, nbr_fea_idx, nbr_fea)

    def global_mean_pooling(self, atom_fea, crystal_atom_idx, **kwargs):
        idxs = [i for i in range(int(min(crystal_atom_idx)), int(max(crystal_atom_idx)+1))]
        idx_count = torch.bincount(crystal_atom_idx)[min(crystal_atom_idx): max(crystal_atom_idx+1)]
        idx = [[idxs[i]]*idx_count[i] for i in range(len(idxs))]

        idxx = []
        idxx = iinidx = 0
        for i in range(len(atom_fea)):
            idx[idxx][iinidx] = i
            iinidx += 1
            if iinidx  == idx_count[idxx]:
                iinidx = 0
                idxx += 1

        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in idx]
        return torch.cat(summed_fea, dim=0)

    def get_batch(self, atom_fea, crystal_atom_idx, **kwargs):
        idxs = [i for i in range(int(min(crystal_atom_idx)), int(max(crystal_atom_idx)+1))]
        idx_count = torch.bincount(crystal_atom_idx)[min(crystal_atom_idx): max(crystal_atom_idx+1)]
        idx = [[idxs[i]]*idx_count[i] for i in range(len(idxs))]

        idxx = []
        idxx = iinidx = 0
        for i in range(len(atom_fea)):
            idx[idxx][iinidx] = i
            iinidx += 1
            if iinidx  == idx_count[idxx]:
                iinidx = 0
                idxx += 1

        return idxx
    
    def exotic_pooling(self, atom_fea, crystal_atom_idx, edge_index, 
                       method='SAG', pooling_dim=10, edge_attr:OptTensor=None):
        method = method.lower()
        assert method in ['sag', 'topk']
        if method == 'sag':
            pooler = tgnn.SAGPooling(in_channels=atom_fea.shape[-1], ratio=pooling_dim)
        else:
            pooler = tgnn.TopKPooling(in_channels=atom_fea.shape[-1], ratio=pooling_dim)
        
        atom_fea = pooler(x=atom_fea, edge_index=edge_index, edge_attr=edge_attr, batch=crystal_atom_idx)[0]
#        atom_fea = self._padding(atom_fea, pooling_dim, crystal_atom_idx)
        atom_fea = self._mean(atom_fea, pooling_dim, crystal_atom_idx)
        return atom_fea

    def _mean(self, x, dim, crystal_atom_idx):
        idxs = [i for i in range(min(crystal_atom_idx), max(crystal_atom_idx)+1)]
        idx_count = torch.bincount(crystal_atom_idx)[min(crystal_atom_idx): max(crystal_atom_idx+1)]
        idx = [[idxs[i]]*idx_count[i] for i in range(len(idxs))]
        idx = []
        for i in range(len(idxs)):
            if idx_count[i] < dim:
                count = idx_count[i]
            else:
                count = dim
            idx.append([idxs[i]] * count)
        summed_fea = [torch.mean(x[idx_map], dim=0, keepdim=True)
                      for idx_map in idx]
        return torch.cat(summed_fea, dim=0)

    def _padding(self, x, dim, crystal_atom_idx):
        maximum = max(crystal_atom_idx)
        minimum = min(crystal_atom_idx)
        batch_size = maximum - minimum + 1
        padded = torch.zeros((batch_size * dim, x.shape[-1])).to(x.device)
        
        idx_count = torch.bincount(crystal_atom_idx)[minimum: maximum + 1]
        sum_atoms = 0
        for i in range(batch_size):
            if idx_count[i] >= dim:
                idx_count[i] = dim
                padded[i * dim: (i + 1) * dim] = x[sum_atoms: sum_atoms + dim]
            else:
                padded[i * dim: i * dim + idx_count[i]] = x[sum_atoms: sum_atoms + idx_count[i]]
            sum_atoms += idx_count[i]
            
        return padded

    
