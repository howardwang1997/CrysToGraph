# atom.py
import warnings

from mendeleev import element

from collections import OrderedDict
from torchtext.vocab import vocab

import torch
from torch.nn import Embedding


class Atom:
    def __init__(self, atom, indice=None, embedding=None):
        if type(atom) is str:
            self.number = element(atom).atomic_number
        else:
            self.number = atom
        self.symbol = element(self.number).symbol
        self.indice = indice
        self.embedding = embedding
    
    
class AtomVocab:
    def __init__(self, atoms):
        if type(atoms[0]) is str:
            self.symbols = atoms
            self.numbers = [element(i).atomic_number for i in atoms]
        else:
            self.numbers = atoms
            
        self.numbers = sorted(self.numbers)
        self._update(self.numbers)
        
    def _update(self, new_numbers):
        self.numbers = new_numbers
        self.symbols = [element(i).symbol for i in self.numbers]
        self.ordered_dict = OrderedDict(zip(self.symbols, self.numbers))
        self.vocab = vocab(self.ordered_dict, min_freq=1)
        
        return self
    
    def _store_atoms(self):
        self.atoms = dict(zip(self.symbols, 
                              [Atom(i, 
                                    self.vocab.__getitem__(i), 
                                    self.embeddings(torch.LongTensor([self.vocab.__getitem__(i)]))) for i in self.symbols]))
    
    def add_atom(self, atom):
        if type(atom) is str:
            a_add = element(atom).atomic_number
        else:
            a_add = atom
        
        if a_add not in self.numbers:
            self.numbers.append(a_add)
            self.numbers = sorted(self.numbers)
            return self._update(self.numbers)
        else:
            warnings.warn('Atom %s exist in the AtomVocab! Not added!' % element(a_add).names)
            return self
    
    def delete_atom(self, atom):
        if type(atom) is str:
            a_del = element(atom).atomic_number
        else:
            a_del = atom
        
        if a_del in self.numbers:
            self.numbers.remove(a_del)
            return self._update(self.numbers)
        else:
            warnings.warn('Atom %s not exist in the AtomVocab! Not deleted!' % element(a_del).names)
            return self
    
    def generate_embeddings(self, embedding_dim, mask_token=True):
        if mask_token:
            self.add_atom(118)
        self.embeddings = Embedding(len(self.numbers), embedding_dim)
        self._store_atoms()
        return self.embeddings
    
    def get_atom_embedding(self, atom):
        if type(atom) is str:
            a = element(atom).atomic_number
        else:
            a = atom
            
        emb = self.embeddings(torch.Tensor(self.vocab.forward([element(atom).symbol])).int())
        return emb
    
    
class EmptyAtomVocab(AtomVocab):
    def __init__(self):
        self.numbers = []
        self._update(self.numbers)
        
    def add_atom(self, atom):
        add = super().add_atom(atom)
        return AtomVocab(self.numbers)
