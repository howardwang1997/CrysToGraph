# crystal.py
import joblib
import os
import json
import functools
from tqdm import tqdm
import warnings
import numpy as np
import collections
import random
import math

from typing import Any, Dict, Union, Optional, Tuple, List
from bisect import bisect_right, bisect_left
import dgl

import pymatgen
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.io.cif import CifParser

import torch
from torch_geometric.data import Data, Dataset
import dgl

from atom import AtomVocab, EmptyAtomVocab
from graph_utils import laplacian_positional_encoding as lpe
from graph_utils import random_walk_positional_encoding as rwpe
from graph_utils import prepare_line_graph_batch

class GaussianFilter(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin=0, dmax=6, step=0.2, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)
    
    
class Crystal:
    def __init__(self,
                 structure=None, 
                 idx=None, 
                 mpid=None, 
#                  exfoliable_1d=False, 
                 exfoliable_2d=False, 
                 dimension=3,
                 target=None, 
                 atom_vocab=None,
                 min_atoms=1):
        self.structure = structure
        self.idx = idx
        self.mpid = mpid
#         self.exfoliable_1d = exfoliable_1d
        self.exfoliable_2d = exfoliable_2d
        self.dimension = dimension
        self.target = target
        self.atom_vocab = atom_vocab
        self.graph = None
        self._masked = False
        self.min_atoms = min_atoms
        if len(structure) < self.min_atoms:
            self.structure.make_supercell((2,2,2))
    
    def define(self,
               structure=None, 
               idx=None, 
               mpid=None, 
#                exfoliable_1d=None, 
               exfoliable_2d=None, 
               dimension=None,
               target=None, 
               atom_vocab=None):
        if structure is not None: 
            self.structure = structure
            if len(structure) < self.min_atoms:
                self.structure.make_supercell((2,2,2))
            print('structure updated')
        if idx is not None: 
            self.idx = idx
            print('idx updated')
        if mpid is not None: 
            self.mpid = mpid
            print('mpid updated')
#         if exfoliable_1d is not None: 
#             self.exfoliable_1d = exfoliable_1d
#             print('exfoliable_1d updated')
        if exfoliable_2d is not None: 
            self.exfoliable_2d = exfoliable_2d
            print('exfoliable_2d updated')
        if dimension is not None: 
            self.dimension = dimension
            print('dimension updated')
        if target is not None: 
            self.target = target
            print('target updated')
        if atom_vocab is not None: 
            self.atom_vocab = atom_vocab
            print('atom_vocab updated')
            
    @property
    def masked(self):
        return self._masked
    
    @property
    def masked_list(self):
        return self._masked_list
    
    @masked_list.setter
    def masked_list(self, list_of_masked):
        self._masked = True
        self._masked_list = list_of_masked
    
    @property
    def masked_target(self):
        return self._masked_target
    
    @masked_target.setter
    def masked_target(self, targets):
        self._masked_target = targets
    
    def generate_graph(self, atom_vocab=None, embedding=False, max_nbr=12, max_radius=8, detect_nbr=True, gf=None, to_simple=False):
        if self.structure is None:
            raise ValueError('No cyrstal structure defined!')
            
        if atom_vocab is not None:
            warnings.warn('Updating atom vocab!')
            self.atom_vocab = atom_vocab
        
        data = structure2dglgraph(structure=self.structure,
                              atom_vocab=self.atom_vocab,
                              embedding=embedding,
                              max_nbr=max_nbr,
                              max_radius=max_radius,
                              detect_nbr=detect_nbr,
                              gf=gf,
                              to_simple=to_simple)
        self.graph = data
        return data
    
    def dump(self, idx=None, suffix='', root='./crystal_dataset/', processed_dir='processed/'):
        if idx is None:
            idx = self.idx
        if suffix != '':
            name = '%d_%s.xt' % (idx, suffix)
        else:
            name = '%d.xt'
        joblib.dump(self, '%s%s%s.xt' % (root, suffix, name))
        
    
class CrystalDataset(Dataset):
    def __init__(self, root='./crystal_dataset/', atom_vocab=None, labels=None, min_atoms=1, to_simple=False,
                 names=None, transform=None, pre_transform=None, pre_filter=None, detect_nbr=True, process=True,
                 embedded=False, 
                 raw_dir='raw/', processed_dir='processed/'):
        
        if root[-1] is not '/': root = '%s/' % root
        self.root = root
        self.names = names
        self.length = 0
        self.raw = None
        self.processed = None
        self.detect_nbr = detect_nbr
        self.labels = labels
        self.do_process = process
#         parameters from config here for gf
        self.gf = GaussianFilter()
        self.min_atoms = min_atoms
        self.to_simple = to_simple
    
        if raw_dir[-1] is not '/': raw_dir = '%s/' % raw_dir
        if processed_dir[-1] is not '/': processed_dir = '%s/' % processed_dir
        if names is None:
            self.length = len(os.listdir(self.raw_dir))
            self.names = ['%d.cif' % i for i in range(self.length)]
            
        self.masked_labels = False
        self.labels = False
        if atom_vocab is not None: self.atom_vocab = atom_vocab
        self.embedded = embedded
        self.prepare_batch = prepare_line_graph_batch
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        self.raw = ['%s/%s' % (self.raw_dir, name) for name in self.names]
        return self.raw

    @property
    def processed_file_names(self):
        self.processed = ['%s/%d.xt' % (self.processed_dir, i) for i in range(self.length)]
        return self.processed
    
    def process(self, names=None, structure_format='cif', embedding=False):
        if not self.do_process: return 0
        
        self.embedded = embedding
        idx = 0
        atom_vocab = self.atom_vocab
        
        if atom_vocab is None:
            if hasattr(self, 'atom_vocab'):
                atom_vocab = self.atom_vocab
            else:
                atom_vocab = self.get_atom_vocab()
        
        if self.names is None:
            raise NameError('No structure initiated!')
        else:
            if self.raw is not None:
                raw = self.raw
            else:
                raw = self.raw_file_names
        
        for f in tqdm(raw):
            s = self.read_raw_data(f, structure_format=structure_format)
            c = Crystal(structure=s, idx=idx, atom_vocab=atom_vocab, min_atoms=self.min_atoms)
#             parameters from config here for generate_graph()
            c.generate_graph(gf=self.gf, embedding=embedding, to_simple=self.to_simple, detect_nbr=self.detect_nbr)

            self.dump_crystal(c, idx)
            idx += 1
        self.length = idx
        
        print('Crystal raw data processed!')
        self.dump()
        
    def get_atom_vocab(self, structure_format='cif', has_mask_token=True, mask_token='Og', atom_vocab=None):
        if atom_vocab is not None:
            av = atom_vocab
            print('Atom vocab loaded!')
        else:
            if self.names is None:
                raise NameError('No structure initiated!')
            else:
                if self.raw is not None:
                    raw = self.raw
                else:
                    raw = self.raw_file_names

            print('Getting atom vocab, please wait.')
            av = EmptyAtomVocab()
    #         empty = True

            for f in tqdm(raw):
                s = self.read_raw_data(f, structure_format=structure_format)
                species = list(set(s.species))
                for element in species:
                    n = element.number
                    if n not in av.numbers:
                        av = av.add_atom(n)

            print('Atom vocab generated!')
        if has_mask_token:
            av.add_atom(mask_token)
        self.atom_vocab = av
        return av
        
    def set_atom_vocab(self, atom_vocab=None):
        self.atom_vocab = atom_vocab
    
    def len(self):
        return self.length    
    
    def read_raw_data(self, name, structure_format='cif'):
        if structure_format.lower() == 'cif':
            cp = CifParser(name)
            s = cp.get_structures()[0]
        elif structure_format.lower() == 'json':
            d = json.loads(name)
            s = Structure.from_dict(d)
        else:
            raise TypeError('Structure file type %s not supported.' % structure_format)
        
        return s
    
    def get_crystal(self, idx, suffix=''):
        xt = joblib.load('%s/%d.xt%s' % (self.processed_dir, idx, suffix))
        return xt
    
    def set_labels(self, labels_list):
        self.labels_list = labels_list
        self.labels = True
    
    def dump_crystal(self, c, idx, suffix=''):
        joblib.dump(c, '%s/%d.xt%s' % (self.processed_dir, idx, suffix))
        
    def dump(self, name='crystal_dataset.jbl'):
        joblib.dump(self, '%s/%s' % (self.processed_dir, name))
    
    def set_masked_labels(self):
        self.masked_labels = True
        
    def set_labels(self, labels_list):
        self.labels_list = labels_list
        self.labels = True
    
    def get(self, idx):
        xt = self.get_crystal(idx)
        graph = xt.graph
        
        if self.masked_labels:
            labels = torch.empty(graph.x.shape[0]).fill_(-1)
            masked_list = xt.masked_list
            masked_labels = xt.masked_target
            for i in range(len(masked_list)):
                labels[masked_list[i]] = masked_labels[i]
            label = labels
        elif self.labels:
            label = self.labels_list[idx]
            
        if type(graph) is tuple:
            return graph, label
        else:
            graph.y = label
            return graph
    
    
class ProcessedCrystalDataset(CrystalDataset):
    def __init__(self, root='./crystal_dataset/', atom_vocab=None, labels=None, suffix='', atom_numbers=None, threshold=None, idx_map=None,
                 names=None, transform=None, pre_transform=None, pre_filter=None, embedded=False,
                 raw_dir='raw/', processed_dir='processed/'):
        
        if root[-1] is not '/': root = '%s/' % root
        self.root = root
        self.names = names
        self.length = 0
        self.raw = None
        self.processed = None
#         parameters from config here for gf
        self.gf = GaussianFilter()
        self.labels = labels
    
        if raw_dir[-1] is not '/': raw_dir = '%s/' % raw_dir
        if processed_dir[-1] is not '/': processed_dir = '%s/' % processed_dir
        if names is None:
            self.length = len(os.listdir(self.raw_dir))
            self.names = ['%d.cif' % i for i in range(self.length)]
            
        self.suffix = suffix
        self.masked_labels = False
        self.labels = False
        self.atom_vocab = atom_vocab
        self.embedded = embedded
        self.prepare_batch = prepare_line_graph_batch
        
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def process(self):
        pass
    

class MixedProcessedCrystalDataset(ProcessedCrystalDataset):
    def __init__(self, root='./crystal_dataset/', atom_vocab=None, labels=None, suffixes=[], batch_size=10,
                 names=None, transform=None, pre_transform=None, pre_filter=None, embedded=False,
                 raw_dir='raw/', processed_dir='processed/'):
        
        if root[-1] is not '/': root = '%s/' % root
        self.root = root
        self.names = names
        self.length = 0
        self.raw = None
        self.processed = None
#         parameters from config here for gf
        self.gf = GaussianFilter()
        self.labels = labels
    
        if raw_dir[-1] is not '/': raw_dir = '%s/' % raw_dir
        if processed_dir[-1] is not '/': processed_dir = '%s/' % processed_dir
        if names is None:
            self.length = len(os.listdir(self.raw_dir))
            self.names = ['%d.cif' % i for i in range(self.length)]
            
        self.suffixes = suffixes
        self.n_yield = len(suffixes)
        self.batch_size = batch_size
        self.masked_labels = False
        self.labels = False
        self.atom_vocab = atom_vocab
        self.embedded = embedded
        self.prepare_batch = prepare_line_graph_batch
        
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def get(self, idx):
        if hasattr(self, 'idx_map'):
            idx = self.idx_map[idx]
        xts = self.get_crystals(idx)
        
        graphs = [xt.graph for xt in xts]
        
        if self.labels:
            label = self.labels_list[idx]
            if type(graphs[0]) is tuple:
                return label, tuple(graphs)
            for i in range(len(graphs)):
                graphs[i].y = label
        
        return tuple(graphs)
    

class ProcessedDGLCrystalDataset(torch.utils.data.Dataset):
    def __init__(self, root='./crystal_dataset/', atom_vocab=None, labels=None, suffix='',
                 names=None, transform=None, pre_transform=None, pre_filter=None, embedded=False,
                 raw_dir='raw/', processed_dir='processed/', load_data=True):
        
        if root[-1] is not '/': root = '%s/' % root
        self.root = root
        self.names = names
        self.length = 0
        self.raw = None
        self.processed = None
#         parameters from config here for gf
        self.gf = GaussianFilter()
        self.labels = labels
        self.raw_dir = root + raw_dir
        self.processed_dir = root + processed_dir
    
        if raw_dir[-1] is not '/': raw_dir = '%s/' % raw_dir
        if processed_dir[-1] is not '/': processed_dir = '%s/' % processed_dir
        if names is None:
            self.length = len(os.listdir(self.raw_dir))
            self.names = ['%d.cif' % i for i in range(self.length)]
            
        self.suffix = suffix
        self.masked_labels = False
        self.labels = False
        self.atom_vocab = atom_vocab
        self.embedded = embedded
        self.prepare_batch = prepare_line_graph_batch
        self.load_data = False
        if load_data:
            self.load_all_data()

        self.gfs =  [GaussianFilter(0, 8, 0.2),
                     GaussianFilter(0, 3.2, 0.2),
                     GaussianFilter(-1.6, 1.6, 0.2),
                     GaussianFilter(-1.4, 1.5, 0.1)]
        
        super().__init__()
        
    @property
    def raw_file_names(self):
        self.raw = ['%s/%s' % (self.raw_dir, name) for name in self.names]
        return self.raw

    @property
    def processed_file_names(self):
        self.processed = ['%s/%d.xt%s' % (self.processed_dir, i, self.suffix) for i in range(self.length)]
        return self.processed
    
    def process(self):
        pass
        
    def set_atom_vocab(self, atom_vocab=None):
        self.atom_vocab = atom_vocab
    
    def __len__(self):
        return self.length    
    
    def get_crystal(self, idx):
        xt = joblib.load('%s/%d.xt%s' % (self.processed_dir, idx, self.suffix))
        return xt
        
    def dump(self, name='crystal_dataset_processed.jbl'):
        joblib.dump(self, '%s/%s' % (self.processed_dir, name))
    
    def set_masked_labels(self):
        self.masked_labels = True
        
    def set_labels(self, labels_list):
        self.labels_list = labels_list
        self.labels = True

    def load_all_data(self, line_graph=True):
        self.load_data = True
        if line_graph == True:
            try:
                self.all_data = joblib.load('%sall_data%s.jbl' % (self.processed_dir, self.suffix))
            except FileNotFoundError:
                self.all_data = []
                for i in tqdm(range(self.length)):
                    self.all_data.append(self.get_crystal(i).graph)
                joblib.dump(self.all_data, '%sall_data%s.jbl' % (self.processed_dir, self.suffix))
            
        else:
            try:
                self.all_data = joblib.load('%sall_data_no_line%s.jbl' % (self.processed_dir, self.suffix))
            except FileNotFoundError:
                self.all_data = []
                for i in tqdm(range(self.length)):
                    self.all_data.append(self.get_crystal(i).graph[0])
                joblib.dump(self.all_data, '%sall_data_no_line%s.jbl' % (self.processed_dir, self.suffix))
    
#     @functools.lru_cache(maxsize=30000)
    def __getitem__(self, idx):
        if self.labels:
            label = self.labels_list[idx]
        
        if self.load_data:
            graph = self.all_data[idx]
        else:
            xt = self.get_crystal(idx)
            graph = xt.graph
        
        if type(graph) is tuple:
            return graph[0], graph[1], label
        else:
            return graph, label

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)

        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        return batched_graph, batched_line_graph, torch.tensor(labels)
        

def load_dataset(name='crystal_dataset', root='./crystal_dataset/', processed_dir='processed/', in_root=True):
    if in_root:
        path = '%s%s%s' % (root, processed_dir, name)
    else:
        path = name
        
    return joblib.load(path)

def load_crystal(idx=None, suffix='', name='', root='./crystal_dataset/', processed_dir='processed/'):
    if idx is not None:
        name = '%d' % idx
#         if suffix != '':
#             name = '%s_%s' % (name, suffix)
    else:
        name = name
    
    return joblib.load('%s%s%s.xt%s' % (root, processed_dir, name, suffix))


def manipulate_crystal(idx=None, suffix='', name='', root='./crystal_dataset/', processed_dir='processed/',
                       mpid=None, 
#                        exfoliable_1d=None, 
                       exfoliable_2d=None,
                       dimension=None,
                       target=None):
    xt = load_crystal(idx, suffix, name, root, processed_dir)
    xt.define(mpid=mpid,
#               exfoliable_1d=exfoliable_1d,
              exfoliable_2d=exfoliable_2d,
              dimension=dimension,
              target=target)
    xt.dump(idx=idx, suffix=suffix, root=root, processed_dir=processed_dir)

    
def structure2tensor(structure, atom_vocab, max_nbr=8, max_radius=6, detect_nbr=True, gf=None, cuda=True):
    """
    atom_emb: (n_atom, emb_size) shaped torch.Tensor
    nbf_fea: (n_atom, nbr, nbr_fea_size) shaped torch.Tensor
    nbr_fea_idx: (n_atom, nbr) shaped torch.Tensor
    """
    atom_emb = torch.vstack([atom_vocab.get_atom_embedding(structure[i].specie.number)
                            for i in range(len(structure))]) #CGCNN

    if detect_nbr == True: radius = detect_radius(structure, max_nbr=max_nbr, max_radius=max_radius)
    else: radius = max_radius
        
    all_nbrs = structure.get_all_neighbors(radius)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    
    for nbr in all_nbrs:
        nbr_fea_idx.append(list(map(lambda x: x[2], nbr)))
        nbr_fea.append(list(map(lambda x: x[1], nbr)))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    
    if gf is None:
        gf = GaussianFilter(dmin=0, dmax=max_radius, step=0.2, var=np.var(nbr_fea))
    nbr_fea = torch.Tensor(gf.expand(nbr_fea))
    nbr_fea_idx = torch.Tensor(nbr_fea_idx).long()
    
    if cuda:
        atom_emb = atom_emb.cuda()
        nbr_fea_idx = nbr_fea_idx.cuda()
        nbr_fea = nbr_fea.cuda()
        
    return atom_emb, nbr_fea, nbr_fea_idx


def structure2tggraph(structure, atom_vocab, embedding=False, max_nbr=8, max_radius=6, detect_nbr=True, gf=None, cuda=True):
    """
    torch_geometric Data object
    """
    if embedding:
        atom_emb = torch.vstack([atom_vocab.get_atom_embedding(structure[i].specie.number)
                                for i in range(len(structure))]) #torch_geometric
    else:
        atom_tokens = [atom.symbol for atom in structure.species]
        atom_emb = torch.Tensor([atom_vocab.vocab.lookup_indices(atom_tokens)]).T
        
    pos = torch.Tensor(structure.cart_coords)

    if detect_nbr == True: radius = detect_radius(structure, max_nbr=max_nbr, max_radius=max_radius)
    else: radius = max_radius
        
    all_nbrs = structure.get_all_neighbors(radius)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_starts, nbr_ends, nbr_fea = [], [], []
    count = 0
    
    for nbr in all_nbrs:
        nbr_starts.extend([count]*len(nbr))
        nbr_ends.extend(list(map(lambda x: x[2], nbr)))
        nbr_fea.extend(list(map(lambda x: x[1], nbr)))
        count += 1
        
    nbr_starts, nbr_ends, nbr_fea = np.array(nbr_starts), np.array(nbr_ends), np.array(nbr_fea)
    
    if gf is None:
        gf = GaussianFilter(dmin=0, dmax=max_radius, step=0.2, var=np.var(nbr_fea))
    nbr_fea = torch.Tensor(gf.expand(nbr_fea))
    nbr_starts = torch.Tensor(nbr_starts).long()
    nbr_ends = torch.Tensor(nbr_ends).long()
    nbr_idx = torch.vstack((nbr_starts, nbr_ends))
    
    if cuda:
        atom_emb = atom_emb.cuda()
        pos = pos.cuda()
        nbr_idx = nbr_fea_idx.cuda()
        nbr_fea = nbr_fea.cuda()
    
    return Data(x=atom_emb, edge_index=nbr_idx, edge_attr=nbr_fea, pos=pos)


def structure2dglgraph(structure, atom_vocab, embedding=False, to_simple=False, max_nbr=12, max_radius=8, detect_nbr=True, gf=None, cuda=True):
    """
    dgl graph object
    """
    if embedding:
        atom_emb = torch.vstack([atom_vocab.get_atom_embedding(structure[i].specie.number)
                                for i in range(len(structure))]) #torch_geometric
    else:
        atom_tokens = [atom.symbol for atom in structure.species]
        atom_emb = torch.Tensor([atom_vocab.vocab.lookup_indices(atom_tokens)]).T
        
    pos = torch.Tensor(structure.cart_coords)
    frac = torch.Tensor(structure.frac_coords)
    lattice = structure.lattice
    volume = lattice.volume
    matrix = lattice.matrix

    gfs =  [GaussianFilter(0, 8, 0.2),
            GaussianFilter(0, 3.2, 0.2),
            GaussianFilter(-1.6, 1.6, 0.2),
            GaussianFilter(-1.4, 1.5, 0.1)]


    if detect_nbr == True: radius = detect_radius(structure, max_nbr=max_nbr, max_radius=max_radius)
    else: radius = max_radius
        
    all_nbrs = structure.get_all_neighbors(radius)
    nbr_starts, nbr_ends = [], []
    cart_starts, cart_ends = [], []
    nbr_count = []
    count = 0

    increment = 0
    while min([len(nbr) for nbr in all_nbrs]) < max_nbr:
        radius += 2 
        all_nbrs = structure.get_all_neighbors(radius)

        increment += 1
        print('INCREMENT %d in raius, radius is %d' % (increment, radius))
    
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    for nbr in all_nbrs:
        """
        [:max_nbr]
        added on 12/07/2023
        to limit the number of maximum neighbors
        """
        nbr_ends.extend(list([count]*len(nbr))[:max_nbr])
        nbr_starts.extend(list(map(lambda x: x[2], nbr))[:max_nbr])
        cart_starts.extend(list([pos[count] for _ in range(len(nbr))])[:max_nbr])
        cart_ends.extend(list(map(lambda x: x.coords, nbr))[:max_nbr])

#        nbrs_list = [0] * (max_nbr+1)
#        nbrs_list[min(len(nbr), max_nbr)] += 1
#        nbr_count.append(nbrs_list)

        count += 1
        
    max_idx_atoms = len(atom_emb)-1
    if max_idx_atoms not in nbr_ends:
        nbr_starts.append(max_idx_atoms)
        nbr_ends.append(max_idx_atoms)
        cart_starts.append(pos[max_idx_atoms])
        cart_ends.append(pos[max_idx_atoms])
        print('MODIFIED')
        
    nbr_starts, nbr_ends = np.array(nbr_starts), np.array(nbr_ends)
    
#     if gf is None:
#         gf = GaussianFilter(dmin=0, dmax=max_radius, step=0.2, var=np.var(nbr_fea))
#     nbr_fea = torch.Tensor(gf.expand(nbr_fea))
    nbr_starts = torch.Tensor(nbr_starts).long()
    nbr_ends = torch.Tensor(nbr_ends).long()
#     nbr_idx = torch.vstack((nbr_starts, nbr_ends))
    
    graph = dgl.graph((nbr_starts, nbr_ends))
    
#     graph.edata['r'] = torch.vstack([
#         pos[nbr_ends[i]] - pos[nbr_starts[i]] for i in range(len(nbr_starts))
#     ])
    """
    changed on 29/07/23
    """
    graph.edata['r'] = torch.tensor(np.vstack(cart_ends) - np.vstack(cart_starts)) 

    lg = graph.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    
    graph.ndata['atom_features'] = atom_emb
#    graph.ndata['nbr_count'] = torch.tensor(nbr_count)
#    graph.ndata['lattice_mat'] = torch.Tensor([
#                                              matrix for i in range(len(atom_emb))])
#    graph.ndata['V'] = torch.Tensor([
#                                    volume for i in range(len(atom_emb))])
#    graph.edata['spherical'] = convert_spherical(
#        torch.vstack([
#        pos[nbr_ends[i]] - pos[nbr_starts[i]] 
#        for i in range(len(nbr_starts))
#    ]))
    graph.edata['spherical'] = convert_spherical(graph.edata['r'])

    graph.ndata['pe'] = torch.cat([pos, frac, lpe(graph, 20), rwpe(graph, 20)], dim=-1)
#    graph.edata['nbr_fea'] = torch.hstack([gfs[i].expand(graph.edata['spherical'][:,i]) for i in range(3)] + [(graph.edata['spherical'][:,0] > 8).view(-1,1)]).float()
    graph.edata.pop('r')
#    graph.edata.pop('spherical')
    
    line_fea = lg.edata['h']
    line_fea = torch.nan_to_num(line_fea, 0.0)
#    line_fea = gfs[-1].expand(line_fea)
#    lg.edata['line_fea'] = line_fea
    lg.ndata.pop('r')
#    lg.edata.pop('h')

    """
    implemented on 26/06/2023, for only line graph training.
    explicitly model the distances between centres of bonds, and the cosine of two bonds.
    """
        
    return graph ,lg

def convert_spherical(euclidean):
    if len(euclidean.shape) == 1:
        x, y, z = euclidean[0], euclidean[1], euclidean[2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            r = 0.001
        if x == 0:
            x = 0.001
        theta = torch.arccos(z / r)
        phi = torch.arctan(y / x)
        out = torch.tensor([r, theta, phi])
    elif len(euclidean.shape) == 2:
        cuda = torch.cuda.is_available()
        device = euclidean.device
        if cuda:
            euclidean = euclidean.cuda()
        x, y, z = euclidean[:,0], euclidean[:,1], euclidean[:,2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        if 0 in r or 0 in x:
            for i in range(len(r)):
                if r[i] == 0:
                    r[i] = 0.001
                if x[i] == 0:
                    x[i] = 0.001
        theta = torch.arccos(z / r)
        phi = torch.arctan(y / x)
        out = torch.vstack([r, theta, phi])
        out = out.T.to(device)
    
    return out   

def _discarded_detect_radius(structure, max_nbr=8, min_radius=2, max_radius=5, step=0.1):
    """
    discarded on 22/06/2023
    """
    m = []
    nsp = len(structure.species)
    for i in range(int(min_radius/step), int(max_radius/step)):
        n = structure.get_all_neighbors(i*step, include_index=True)
        l = [len(n[j]) for j in range(nsp)]
        m.append(l)
    matrix = np.array(m)
    
    radius = int(min_radius/step)
    for row in range(len(matrix)):
        if sum(matrix[row] > max_nbr) > 0:
            radius = radius + row - 1
            break
    radius *= step
    return radius

def detect_radius(structure, max_nbr=12, min_radius=2, max_radius=8, step=0.1):
    """
    implemented on 22/06/2023, debuged and more efficient
    """
    m = []
    nsp = len(structure.species)
    for i in range(int(min_radius/step), int(max_radius/step)):
        n = structure.get_all_neighbors(i*step, include_index=True)
        l = [len(n[j]) for j in range(nsp)]
        if max(l) > max_nbr:
            break
        m.append(max(l))
    
    radius = min_radius + (len(m) - 1) * step
    return radius

def compute_bond_cosines(edges):
    """
    from alignn.graphs
    """
    r1 = -edges.src['r']
    r2 = edges.dst['r']
    bot = (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )

    if (bot == 0).sum() > 0: bot += 0.0001
    bond_cosine = torch.sum(r1 * r2, dim=1) / bot
    
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {'h': bond_cosine}
