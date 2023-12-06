import json
import os
import random
from tqdm import tqdm
import joblib
import argparse

from jarvis.core.atoms import Atoms


def make_dir(work):
    if work[-1] == '/':
        work = work[:-1]
    dirs = [
        work,
        f'{work}/train',
        f'{work}/val',
        f'{work}/train/raw',
        f'{work}/train/processed',
        f'{work}/val/raw',
        f'{work}/val/processed',
    ]
    for path in dirs:
        try:
            os.mkdir(path)
            print(f'{path} created.')
        except FileExistsError:
            print(f'{path} exists.')


def split_dataset(dataset, ratio=0.9):
    length = len(dataset)
    all_data = [i for i in range(length)]
    t = int(length * ratio)
    train = random.sample(all_data, t)
    val = [i for i in all_data if i not in train]

    return train, val


def process_jarvis_dataset(dataset, work, label, train_idx, val_idx):
    data_t, data_v = [dataset[i] for i in train_idx], [dataset[i] for i in val_idx]
    l_t, l_v = [data[label] for data in data_t], [data[label] for data in data_v]
    for i in tqdm(range(len(data_t))):
        a = Atoms.from_dict(data_t[i])
        a.write_cif(f'{work}/train/raw/{i}.cif')
    for i in tqdm(range(len(data_v))):
        a = Atoms.from_dict(data_v[i])
        a.write_cif(f'{work}/val/raw/{i}.cif')
    joblib.dump((train_idx, val_idx), f'{work}/tvs.jbl')
    joblib.dump(l_t, f'{work}/train/labels.jbl')
    joblib.dump(l_v, f'{work}/val/labels.jbl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset to be processed, must be jarvis dataset.')
    parser.add_argument('--work', type=str, help='target directory.')
    parser.add_argument('--label', type=str, help='label name')
    parser.add_argument('--ratio', type=float, default=0.9, help='ratio of train set')
    parser.add_argument('--split', type=str, help='path to split file')
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    make_dir(args.work)

    if args.split:
        train_idx, val_idx = joblib.load(args.split)
    else:
        train_idx, val_idx = split_dataset(dataset, args.ratio)

    process_jarvis_dataset(dataset, args.work, args.label, train_idx, val_idx)


if __name__ == '__name__':
    main()
