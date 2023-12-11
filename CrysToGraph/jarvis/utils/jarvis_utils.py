import json
import pandas as pd

from jarvis_utils.core.atoms import Atoms
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core import Structure
from sklearn.model_selection import KFold
from jarvis_constant import DATASETS_LEN


def jarvis_dataset_to_mp(dataset, label, name, save=False):
    data = [d['atoms'] for d in dataset]
    labels = [d[label] for d in dataset]
    converter = JarvisAtomsAdaptor()
    data = [converter.get_structure(Atoms.from_dict(d)).as_dict() for d in data]
    data_matbench = [[data[i], labels[i]] for i in range(len(data))]
    index = list(range(len(data)))
    assert len(data) == len(labels)

    final = {
        "index": index,
        "columns": ['structure', label],
        "data": data_matbench
    }
    if save:
        with open('jarvis_datasets/mb_'+name+'.json', 'w+') as f:
            json.dump(final, f)
    return final


def _get_list_name(name, length, index):
    r_len = len(str(length))
    all_names = [name + '-' + str(i+1).rjust(r_len, '0') for i in index]
    return all_names


def make_splits(name, length, n_folds=5, random_seed=42):
    kf = KFold(n_folds, random_state=random_seed)
    index = list(range(length))
    dataset_split = {}
    for i, (train_index, test_index) in enumerate(kf.split(index)):
        f = {'train': _get_list_name(name, length, train_index), 'test': _get_list_name(name, length, test_index)}
        dataset_split.update({f'fold_{i}': f})
    dataset = {name: dataset_split}
    return dataset


def make_validation(n_folds=5, random_seed=42, save=False):
    validation = {}
    for k, v in DATASETS_LEN.items():
        validation.update(make_splits(k, v, n_folds, random_seed))
    metadata = {'n_split': n_folds, 'random_state': random_seed, 'shuffle': False}

    final = {'metadata': metadata, 'splits': validation}
    if save:
        with open('jarvis_datasets/metadata_validation.json', 'w+') as f:
            json.dump(final, f)
    return final


def load_dataset(dataset, name):
    data = [Structure.from_dict(d[0]) for d in dataset['data']]
    labels = [d[1] for d in dataset]
    length = len(dataset)
    index = list(range(length))

    all_names = _get_list_name(name, length, index)
    pd_idx = pd.Series(all_names, name='mbid')
    inputs = pd.Series(data, name=dataset['columns'][0], index=pd_idx)
    outputs = pd.Series(labels, name=dataset['columns'][1], index=pd_idx)

    return inputs, outputs
