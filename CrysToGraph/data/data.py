# data.py
import json
from tqdm import tqdm
import os

import pymatgen
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser, CifWriter

mp_key = ''

def init_folders():
    try:
        os.mkdir('crystal_dataset')
    except FileExistsError:
        pass
    try:
        os.mkdir('crystal_dataset/raw')
    except FileExistsError:
        pass
    try:
        os.mkdir('crystal_dataset/processed')
    except FileExistsError:
        pass
    print('Folders initiated!')

def set_mp_key(key=''):
    global mp_key
    if key is '':
        key = input('Please input Materials Project key here.\n')
    mp_key = key

def exist_mp_key():
    return mp_key is not ''

def mids_to_dicts(mids, key=''):
    if not exist_mp_key: 
        set_mp_key(key)
        
    ds = []
    with MPRester(mp_key) as mpr:
        for i in tqdm(range(len(mids))):
            mid = mids[i]
            s = mpr.get_structure_by_material_id(mid)
            d = s.as_dict()
            ds.append(d)
    
    return ds        

def mids_to_cifs(mids, keys='', dataset_name='dataset'):
    if not exist_mp_key: 
        set_mp_key(key)
        
    try:
        os.mkdir(dataset_name)
    except FileExistsError:
        pass
    
    with MPRester(mp_key) as mpr:
        for i in tqdm(range(len(mids))):
            mid = mids[i]
            s = mpr.get_structure_by_material_id(mid)
            w = CifWriter(s)
            n = '%s/%s.cif' % (dataset_name, mid)
            w.write_file(n)

def dicts_to_cifs(dicts, dataset_name='dataset'):
    try:
        os.mkdir(dataset_name)
    except FileExistsError:
        pass
    
    for i in tqdm(range(len(dicts))):
        d = dicts[i]
        s = Structure.from_dict(d)
        w = CifWriter(s)
        n = '%s/%s.cif' % (dataset_name, i)
        w.write_file(n)

def read_cif(path):
    n = path
    p = CifParser(n)
    s = p.get_structures()[0]
    return s

def cifs_to_dicts(names, dataset_name='dataset'):
    ds = []
    
    for i in tqdm(range(len(names))):
        name = names[i]
        n = '%s/%s.cif' % (dataset_name, name)
        p = CifParser(n)
        s = p.get_structures()[0]
        d = s.as_dict()
        ds.append(d)
        
    return ds

def get_matbench_json_data(root, name):
    with open('%s%s' % (root, name)) as f:
        for line in f:
            mp = json.loads(line)
    return mp

def read_matbench_json(mp):
    index = mp['index']
    structures = []
    targets = []
    
    for i in index:
        structures.append(mp['data'][i][0])
        targets.append(mp['data'][i][1])
    return (index, structures, targets)

def yield_matbench_json(mp):
    index = mp['index']
    for i in index:
        si = mp['data'][i][0]
        ti = mp['data'][i][1]
        yield(i, si, ti)
