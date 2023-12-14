import json
import torch
import numpy as np
from torch.nn import L1Loss, MSELoss

from jarvis_constant import DATASETS_LEN, DATASETS_RESULTS
from jarvis_utils import load_dataset


class Task:
    def __init__(self, name):
        self.benchmark_name = 'matbench_v0.1'
        self.dataset_name = name
        self.metadata = {
            'input_type': 'structure',
            'n_samples': DATASETS_LEN[name],
            'nun_entries': DATASETS_LEN[name],
            'task_type': 'regression',
            'file_type': 'json.gz',
            'columns': {'structure': 'Structure.', 'target': 'Target.'},
            'mad': 1,
            'target': 'target',
            'unit': 'TBA',
            'bibtex_refs': 'TBA',
            'description': 'TBA',
            'reference': 'TBA',
            'url': 'TBA',
            'hash': 'TBA'
        }
        self.folds_map = {0: 'fold_0', 1: 'fold_1', 2: 'fold_2', 3: 'fold_3', 4: 'fold_4'}
        self.folds = list(range(5))
        self.folds_nums = list(range(5))
        self.folds_keys = list(self.folds_map.values())
        self.info = 'TBA'
        self.path = f'jarvis_datasets/mb_{name}.json'
        self.loaded = False
        self.inputs = self.outputs = self.splits = None

    def record(self, fold, predictions):
        predictions = torch.tensor(predictions).view(-1)

        fold_key = self.folds_map[fold]
        keys = self.splits[fold_key]['test']
        test_outputs = torch.tensor(self.outputs[keys]).view(-1)
        mae = L1Loss()(predictions, test_outputs).item()
        mse = MSELoss()(predictions, test_outputs).item()
        rmse = np.sqrt(mse)
        print(f'TASK {self.dataset_name} FOLD {fold} RESULTS: MAE = {mae:.4f}, RMSE = {rmse:.4f}')

        predictions = predictions.tolist()
        try:
            with open('jarvis_MB_results.json') as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = DATASETS_RESULTS
        all_results[self.dataset_name][self.folds_map[fold]] = predictions
        with open('jarvis_MB_results.json', 'w+') as f:
            json.dump(all_results, f)

    def load(self):
        with open(self.path) as f:
            dataset = json.load(f)
        self.inputs, self.outputs = load_dataset(dataset, self.dataset_name)
        with open('jarvis_datasets/metadata_validation.json') as f:
            self.splits = json.load(f)['splits'][self.dataset_name]

    def get_train_and_val_data(self, fold):
        fold_key = self.folds_map[fold]
        keys = self.splits[fold_key]['train']
        return self.inputs[keys], self.outputs[keys]

    def get_test_data(self, fold):
        fold_key = self.folds_map[fold]
        keys = self.splits[fold_key]['test']
        return self.inputs[keys], self.outputs[keys]