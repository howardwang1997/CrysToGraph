# crystal.py
import joblib
import os
import json
from tqdm import tqdm
import warnings
import numpy as np

from typing import Tuple, List
import dgl

from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.io.cif import CifParser

import torch
from torch_geometric.data import Data, Dataset

from .atom import EmptyAtomVocab
from .graph_utils import laplacian_positional_encoding as lpe
from .graph_utils import random_walk_positional_encoding as rwpe
from .graph_utils import prepare_line_graph_batch
from .graph_utils import compute_bond_cosines, detect_radius, convert_spherical
    
    
class Crystal:
    def __init__(self,
                 structure=None, 
                 idx=None, 
                 mpid=None,
                 exfoliable_2d=False, 
                 dimension=3,
                 target=None, 
                 atom_vocab=None,
                 min_atoms=1):
        self.structure = structure
        self.idx = idx
        self.mpid = mpid
        self.exfoliable_2d = exfoliable_2d
        self.dimension = dimension
        self.target = target
        self.atom_vocab = atom_vocab
        self.graph = None
        self._masked = False
        self.min_atoms = min_atoms
        if len(structure) < self.min_atoms:
            self.structure.make_supercell((2, 2, 2))
    
    def define(self,
               structure=None, 
               idx=None, 
               mpid=None,
               exfoliable_2d=None, 
               dimension=None,
               target=None, 
               atom_vocab=None):
        if structure is not None: 
            self.structure = structure
            if len(structure) < self.min_atoms:
                self.structure.make_supercell((2, 2, 2))
            print('structure updated')
        if idx is not None: 
            self.idx = idx
            print('idx updated')
        if mpid is not None: 
            self.mpid = mpid
            print('mpid updated')
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
    
    def generate_graph(self, atom_vocab=None, embedding=False, max_nbr=12, max_radius=8, detect_nbr=True):
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
                                  detect_nbr=detect_nbr)
        self.graph = data
        return data
    
    def dump(self, idx=None, suffix='', root='./crystal_dataset/'):
        if idx is None:
            idx = self.idx
        if suffix != '':
            name = '%d_%s.xt' % (idx, suffix)
        else:
            name = '%d.xt'
        joblib.dump(self, '%s%s%s.xt' % (root, suffix, name))
        
    
class CrystalDataset(Dataset):
    def __init__(self, root='./crystal_dataset/', atom_vocab=None, min_atoms=1, embeddings=None,
                 names=None, transform=None, pre_transform=None, pre_filter=None, detect_nbr=False, process=True,
                 raw_dir='raw/', processed_dir='processed/'):
        
        if root[-1] != '/':
            root = '%s/' % root
        self.root = root
        self.names = names
        self.length = 0
        self.raw = None
        self.processed = None
        self.detect_nbr = detect_nbr
        self.do_process = process
        self.min_atoms = min_atoms
        self.embeddings = embeddings
        if self.embeddings is None:
            self.embedded = False
    
        if raw_dir[-1] != '/':
            raw_dir = '%s/' % raw_dir
        if processed_dir[-1] != '/':
            processed_dir = '%s/' % processed_dir
        if names is None:
            self.length = len(os.listdir(self.raw_dir))
            self.names = ['%d.cif' % i for i in range(self.length)]

        if atom_vocab is not None:
            self.atom_vocab = atom_vocab
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        self.raw = ['%s/%s' % (self.raw_dir, name) for name in self.names]
        return self.raw

    @property
    def processed_file_names(self):
        self.processed = ['%s/%d.xt' % (self.processed_dir, i) for i in range(self.length)]
        return self.processed
    
    def process(self):
        if not self.do_process:
            return 0

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
            s = self.read_raw_data(f, structure_format='cif')
            c = Crystal(structure=s, idx=idx, atom_vocab=atom_vocab, min_atoms=self.min_atoms)
            c.generate_graph(detect_nbr=self.detect_nbr)

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
    
    def dump_crystal(self, c, idx, suffix=''):
        joblib.dump(c, '%s/%d.xt%s' % (self.processed_dir, idx, suffix))
        
    def dump(self, name='crystal_dataset.jbl'):
        joblib.dump(self, '%s/%s' % (self.processed_dir, name))
    
    def get(self, idx):
        pass


class ProcessedDGLCrystalDataset(torch.utils.data.Dataset):
    def __init__(self, root='./crystal_dataset/', atom_vocab=None, suffix='',
                 names=None, embedded=False, raw_dir='raw/', processed_dir='processed/', load_data=True):
        
        if root[-1] != '/':
            root = '%s/' % root
        self.root = root
        self.names = names
        self.length = 0
        self.raw = None
        self.processed = None
        self.raw_dir = root + raw_dir
        self.processed_dir = root + processed_dir
    
        if raw_dir[-1] != '/':
            self.raw_dir = '%s/' % self.raw_dir
        if processed_dir[-1] != '/':
            self.processed_dir = '%s/' % self.processed_dir
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
        
        super().__init__()
        
    @property
    def raw_file_names(self):
        self.raw = ['%s%s' % (self.raw_dir, name) for name in self.names]
        return self.raw

    @property
    def processed_file_names(self):
        self.processed = ['%s%d.xt%s' % (self.processed_dir, i, self.suffix) for i in range(self.length)]
        return self.processed
    
    def process(self):
        pass
        
    def set_atom_vocab(self, atom_vocab=None):
        self.atom_vocab = atom_vocab
    
    def __len__(self):
        return self.length    
    
    def get_crystal(self, idx):
        xt = joblib.load('%s%d.xt%s' % (self.processed_dir, idx, self.suffix))
        return xt
        
    def dump(self, name='crystal_dataset_processed.jbl'):
        joblib.dump(self, '%s%s' % (self.processed_dir, name))
    
    def set_masked_labels(self):
        self.masked_labels = True
        self.labels = True
        try:
            self.labels_list = joblib.load('%smasked_target.jbl' % self.processed_dir)
            self.masked_list = joblib.load('%smasked_list.jbl' % self.processed_dir)
        except FileNotFoundError:
            self.labels_list = []
            self.masked_list = []
            for i in range(self.length):
                self.labels_list.append(self.get_crystal(i).masked_target)
                self.masked_list.append(self.get_crystal(i).masked_list)
                joblib.dump(self.labels_list, '%smasked_target.jbl' % self.processed_dir)
                joblib.dump(self.masked_list, '%smasked_list.jbl' % self.processed_dir)
        
    def set_labels(self, labels_list):
        self.labels_list = labels_list
        self.labels = True

    def load_all_data(self, line_graph=True):
        self.load_data = True
        if line_graph:
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
        assert self.labels
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
    else:
        name = name
    
    return joblib.load('%s%s%s.xt%s' % (root, processed_dir, name, suffix))


def manipulate_crystal(idx=None, suffix='', name='', root='./crystal_dataset/', processed_dir='processed/',
                       mpid=None,
                       exfoliable_2d=None,
                       dimension=None,
                       target=None):
    xt = load_crystal(idx, suffix, name, root, processed_dir)
    xt.define(mpid=mpid,
              exfoliable_2d=exfoliable_2d,
              dimension=dimension,
              target=target)
    xt.dump(idx=idx, suffix=suffix, root=root, processed_dir=processed_dir)


def structure2dglgraph(structure, atom_vocab, embedding=False, max_nbr=12, max_radius=8, detect_nbr=False):
    """
    dgl graph object
    """
    if embedding:
        atom_emb = torch.vstack([atom_vocab.get_atom_embedding(structure[i].specie.number)
                                for i in range(len(structure))])  # torch_geometric
    else:
        atom_tokens = [atom.symbol for atom in structure.species]
        atom_emb = torch.Tensor([atom_vocab.vocab.lookup_indices(atom_tokens)]).T
        
    pos = torch.Tensor(structure.cart_coords)
    frac = torch.Tensor(structure.frac_coords)

    if detect_nbr:
        radius = detect_radius(structure, max_nbr=max_nbr, max_radius=max_radius)
    else:
        radius = max_radius
        
    all_nbrs = structure.get_all_neighbors(radius)
    nbr_starts, nbr_ends = [], []
    cart_starts, cart_ends = [], []
    count = 0

    increment = 0
    while min([len(nbr) for nbr in all_nbrs]) < max_nbr:
        radius += 2 
        all_nbrs = structure.get_all_neighbors(radius)

        increment += 1
        print('INCREMENT %d in raius, radius is %d' % (increment, radius))
    
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    for nbr in all_nbrs:
        nbr_ends.extend(list([count]*len(nbr))[:max_nbr])
        nbr_starts.extend(list(map(lambda x: x[2], nbr))[:max_nbr])
        cart_starts.extend(list([pos[count] for _ in range(len(nbr))])[:max_nbr])
        cart_ends.extend(list(map(lambda x: x.coords, nbr))[:max_nbr])

        count += 1
        
    max_idx_atoms = len(atom_emb)-1
    if max_idx_atoms not in nbr_ends:
        nbr_starts.append(max_idx_atoms)
        nbr_ends.append(max_idx_atoms)
        cart_starts.append(pos[max_idx_atoms])
        cart_ends.append(pos[max_idx_atoms])
        print('MODIFIED')
        
    nbr_starts, nbr_ends = np.array(nbr_starts), np.array(nbr_ends)

    nbr_starts = torch.Tensor(nbr_starts).long()
    nbr_ends = torch.Tensor(nbr_ends).long()
    
    graph = dgl.graph((nbr_starts, nbr_ends))

    graph.edata['r'] = torch.tensor(np.vstack(cart_ends) - np.vstack(cart_starts)) 

    lg = graph.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    
    graph.ndata['atom_features'] = atom_emb
    graph.edata['spherical'] = convert_spherical(graph.edata['r'])

    graph.ndata['pe'] = torch.cat([pos, frac, lpe(graph, 20), rwpe(graph, 20)], dim=-1)
    graph.edata.pop('r')

    lg.edata['h'] = torch.nan_to_num(lg.edata['h'], 0.0)
    lg.ndata.pop('r')
        
    return graph, lg
