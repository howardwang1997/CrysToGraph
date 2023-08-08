# augmentation
import random
import math
from tqdm import tqdm
import torch
import dgl

from crystal import compute_bond_cosines, convert_spherical

import pymatgen
from pymatgen.core import Lattice, Structure, Molecule, Species

from crystal import Crystal, CrystalDataset, detect_radius

class Augmentation:
    def __init__(self, crystal_dataset, to_simple=False):
        """
        mask token is set as element(118), 'Og'
        """
        self.cd = crystal_dataset
        self.av = self.cd.atom_vocab
        self.length = self.cd.len()
        self.to_simple = to_simple
        if self.length == 0:
            raise ValueError('Please use processed crystal dataset!')
        self.processed = self.cd.processed_file_names
        
    def mask_atom(self, ratio=0.15, mask_rate=0.8, random_rate=0.1, minimum=8, suffix='am',
                  max_nbr=8, max_radius=6, detect_nbr=True):
        for idx in tqdm(range(self.length)):
            new_crystal = atom_mask(self.cd.get_crystal(idx), ratio, mask_rate, random_rate, minimum, 
                                    self.av, self.cd.embedded, max_nbr, max_radius, detect_nbr, self.cd.gf, self.to_simple)
            self.cd.dump_crystal(new_crystal, idx, suffix)
            
    def delete_bond(self, ratio=0.1, minimum=10, suffix='bd'):
        for idx in tqdm(range(self.length)):
            new_crystal = bond_delete(self.cd.get_crystal(idx), ratio, minimum, self.av)
            self.cd.dump_crystal(new_crystal, idx, suffix)
            
    def remove_subgraph(self, ratio=0.1, minimum=8, suffix='sr',
                        max_nbr=8, max_radius=6, detect_nbr=True):
        for idx in tqdm(range(self.length)):
            new_crystal = subgraph_remove(self.cd.get_crystal(idx), ratio, minimum, 
                                          self.av, self.cd.embedded, max_nbr, max_radius, detect_nbr, self.cd.gf, self.to_simple)
            self.cd.dump_crystal(new_crystal, idx, suffix)
    
    def expand_supercell(self, multiples=(3,3,3), suffix='se',
                         max_nbr=8, max_radius=6, detect_nbr=True):
        for idx in tqdm(range(self.length)):
            new_crystal = supercell_expand(self.cd.get_crystal(idx), multiples,
                                          self.av, self.cd.embedded, max_nbr, max_radius, detect_nbr, self.cd.gf, self.to_simple)
            self.cd.dump_crystal(new_crystal, idx, suffix)
        
        
def is_masked(crystal):
    return hasattr(crystal, 'masked_list')

def atom_mask(x, ratio=0.15, mask_rate=0.8, random_rate=0.1, minimum=8, 
              atom_vocab=None, embedding=False, max_nbr=8, max_radius=6, detect_nbr=True, gf=None, to_simple=False):
    """
    x: Crystal object
    rate: masked ratio in total atoms
    random: random masked ratio in masked atoms
    """
    if minimum * ratio < 1: minimum = math.ceil(1 / ratio)
    sa = _make_minimum_supercell(x.structure, minimum)
    if atom_vocab is None:
        atom_vocab = x.atom_vocab
        
    n = len(sa.species)
    n_select = round(n * ratio)
    list_of_chosen = random.sample([i for i in range(n)], n_select)
    masked_target = atom_vocab.vocab.forward([sa.species[i].symbol for i in list_of_chosen])
    mask_atom = Species('Og')
    for i in list_of_chosen:
        if random.random() <= mask_rate:
            sa.replace(i, mask_atom)
        elif random.random() >= 1 - random_rate:
            random_atom = Species(random.choice(atom_vocab.symbols))
            sa.replace(i, random_atom)
    
    xa = Crystal(structure=sa, 
                 idx=x.idx, 
                 mpid=x.mpid, 
#                  exfoliable_1d=x.exfoliable_1d, 
                 exfoliable_2d=x.exfoliable_2d, 
                 dimension=x.dimension,
                 target=x.target,
                 atom_vocab=atom_vocab)
    xa.generate_graph(atom_vocab, embedding, max_nbr, max_radius, detect_nbr, gf, to_simple=to_simple)
    
    xa.masked_list = list_of_chosen
    xa.masked_target = masked_target
    return xa 

def bond_delete(x, ratio=0.1, minimum=10, atom_vocab=None):
    if minimum * ratio < 1: minimum = math.ceil(1 / ratio)
    g = x.graph[0].clone()
    add_edge = False
    
    n_bonds = g.num_edges()
    n_select = round(n_bonds * ratio)
    list_of_chosen = random.sample([i for i in range(n_bonds)], n_select)
    list_of_remain = [i for i in range(n_bonds)]
    for i in list_of_chosen:
        list_of_remain.remove(i)
    gnd = g.ndata
    ged = g.edata
    
    nbr_starts, nbr_ends = g.edges()
    nbr_starts = nbr_starts[list_of_remain]
    nbr_ends = nbr_ends[list_of_remain]
    
    max_idx_atoms = len(gnd['atom_features'])-1
    if max_idx_atoms not in nbr_starts:
        add_edge = True
        nbr_starts = torch.cat([nbr_starts, torch.Tensor([max_idx_atoms])]).long()
        nbr_ends = torch.cat([nbr_ends, torch.Tensor([max_idx_atoms])]).long()
        gr = torch.vstack([ged['r'][list_of_remain], torch.tensor([0.001,0,0])])
        gs = torch.vstack([ged['spherical'][list_of_remain], convert_spherical(torch.tensor([0.001,0,0]))])
    
    graph = dgl.graph((nbr_starts, nbr_ends))
    for k in gnd.keys():
        graph.ndata[k] = gnd[k]
    if add_edge:
        for k in range(1):
            graph.edata['r'] = gr
            graph.edata['spherical'] = gs
    else:
        for k in ged.keys():
            graph.edata[k] = ged[k][list_of_remain]
        
    lg = graph.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    
    xa = Crystal(structure=x.structure, 
                 idx=x.idx, 
                 mpid=x.mpid, 
#                  exfoliable_1d=x.exfoliable_1d, 
                 exfoliable_2d=x.exfoliable_2d, 
                 dimension=x.dimension,
                 target=x.target,
                 atom_vocab=atom_vocab)
    xa.graph = graph, lg
    
    return xa

def bond_delete_torchgeo(x, ratio=0.1, minimum=10, atom_vocab=None):
    if minimum * ratio < 1: minimum = math.ceil(1 / ratio)
    g = x.graph.clone()
    
    n_bonds = len(g.edge_index[0])
    n_select = round(n_bonds * ratio)
    list_of_chosen = random.sample([i for i in range(n_bonds)], n_select)
    list_of_remain = [i for i in range(n_bonds)]
    for i in list_of_chosen:
        list_of_remain.remove(i)
    g.edge_index = g.edge_index[:,list_of_remain]
    g.edge_attr = g.edge_attr[list_of_remain, :]
    
    xa = Crystal(structure=x.structure, 
                 idx=x.idx, 
                 mpid=x.mpid, 
#                  exfoliable_1d=x.exfoliable_1d, 
                 exfoliable_2d=x.exfoliable_2d, 
                 dimension=x.dimension,
                 target=x.target,
                 atom_vocab=atom_vocab)
    xa.graph = g
    
    return xa

def subgraph_remove(x, ratio=0.1, minimum=8, 
                    atom_vocab=None, embedding=False, max_nbr=8, max_radius=6, detect_nbr=True, gf=None, to_simple=False):
    if minimum * ratio < 1: minimum = math.ceil(1 / ratio)
    sa = _make_minimum_supercell(x.structure, minimum)
    if atom_vocab is None:
        atom_vocab = x.atom_vocab
        
    n = len(sa.species)
    n_remove = round(n * ratio)
    list_of_removal = _find_subgraph(sa, n_remove)
    sa.remove_sites(list_of_removal)
    
    xa = Crystal(structure=sa, 
                 idx=x.idx, 
                 mpid=x.mpid, 
#                  exfoliable_1d=x.exfoliable_1d, 
                 exfoliable_2d=x.exfoliable_2d, 
                 dimension=x.dimension,
                 target=x.target,
                 atom_vocab=atom_vocab)
    xa.generate_graph(atom_vocab, embedding, max_nbr, max_radius, detect_nbr, gf, to_simple=to_simple)
    
    return xa

def supercell_expand(x, multiples=(3,3,3), 
                     atom_vocab=None, embedding=False, max_nbr=8, max_radius=6, detect_nbr=True, gf=None, to_simple=False):
    sa = x.structure.copy()
    if atom_vocab is None:
        atom_vocab = x.atom_vocab
        
    sa.make_supercell(multiples)
    
    xa = Crystal(structure=sa, 
                 idx=x.idx, 
                 mpid=x.mpid, 
#                  exfoliable_1d=x.exfoliable_1d, 
                 exfoliable_2d=x.exfoliable_2d, 
                 dimension=x.dimension,
                 target=x.target,
                 atom_vocab=atom_vocab)
    xa.generate_graph(atom_vocab, embedding, max_nbr, max_radius, detect_nbr, gf, to_simple=to_simple)
    
    return xa

def _make_minimum_supercell(s, minimum):
    sa = s.copy()
    n = len(sa.species)
    if n < minimum:
        m = math.ceil(math.pow(minimum / n, 1/3))
        sa.make_supercell((m,m,m))
    return sa

# def _recurrent_find_subgraph(s, n_find, find=[]):
#     if n_find <= 0:
#         return find
#     else:
#         if find == []:
#             n = random.randint(0, len(s.species))
#             find.append(n)
#         else:
#             nbrs = []
#             radius = detect_radius(s)
#             for a in find:
#                 nbr = list(map(lambda x: x[2], s.get_all_neighbors(radius)[a]))
#                 nbrs.extend(nbr)
#             nbrs = list(filter(lambda x: x not in find, nbrs))
#             find.append(random.choice(nbrs))
#         return _find_subgraph(s, n_find-1, find)
    
def _find_subgraph(s, n_find, find=None):
    radius = detect_radius(s)
    if find is None: find = []
    nbrs = []
    
    for i in range(n_find):
        if find == []:
            n = random.randint(0, len(s.species)-1)
            find.append(n)
        else:
            nbr = list(map(lambda x: x[2], s.get_all_neighbors(radius)[find[-1]]))
            nbrs.extend(nbr)
            nbrs = list(filter(lambda x: x not in find, nbrs))
            
            if nbrs == []:
                n = random.randint(0, len(s.species)-1)
                while n in find:
                    n = random.randint(0, len(s.species)-1)
                find.append(n)
            else:
                find.append(random.choice(nbrs))
    return find
