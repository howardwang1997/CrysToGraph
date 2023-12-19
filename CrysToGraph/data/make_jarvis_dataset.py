import json
import os
import random
from tqdm import tqdm
import joblib
import argparse
import shutil

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


def split_dataset(dataset, ratio=0.9, seed=42):
    length = len(dataset)
    len_train = int(length * ratio)
    random.seed(seed)
    all_idx = list(range(len(dataset)))
    random.shuffle(all_idx)
    train_idx, val_idx = all_idx[:len_train], all_idx[len_train:]

    return train_idx, val_idx


def process_jarvis_dataset(dataset, work, label, train_idx, val_idx, label_2=''):
    second_label = False
    data_t, data_v = [dataset[i] for i in train_idx], [dataset[i] for i in val_idx]
    l_t, l_v = [data[label] for data in data_t], [data[label] for data in data_v]
    error_idx_t, error_idx_v = [], []
    if label_2 != '':
        second_label = True
        l2_t, l2_v = [data[label_2] for data in data_t], [data[label_2] for data in data_v]
    for i in tqdm(range(len(data_t))):
        try:
            a = Atoms.from_dict(data_t[i]['atoms'])
            a.write_cif(f'{work}/train/raw/{i}.cif')
        except TypeError:
            error_idx_t.append(i)
    for i in tqdm(range(len(data_v))):
        try:
            a = Atoms.from_dict(data_v[i]['atoms'])
            a.write_cif(f'{work}/val/raw/{i}.cif')
        except TypeError:
            error_idx_v.append(i)
    joblib.dump((train_idx, val_idx), f'{work}/tvs.jbl')
    print(f'{len(error_idx_t)} samples in train and {len(error_idx_v)} samples in val not converted!')

    joblib.dump(error_idx_t, f'{work}/train/error_samples.jbl')
    joblib.dump(error_idx_v, f'{work}/val/error_samples.jbl')
    error_idx_t.sort(reverse=True)
    error_idx_v.sort(reverse=True)
    for i in error_idx_t:
        l_t.pop(i)
    for i in error_idx_v:
        l_v.pop(i)
    joblib.dump(l_t, f'{work}/train/labels.jbl')
    joblib.dump(l_v, f'{work}/val/labels.jbl')
    if second_label:
        for i in error_idx_t:
            l2_t.pop(i)
        for i in error_idx_v:
            l2_v.pop(i)
        joblib.dump(l2_t, f'{work}/train/labels_2.jbl')
        joblib.dump(l2_v, f'{work}/val/labels_2.jbl')


def remove_errors(raw_path, error_samples):
    if raw_path[-1] == '/':
        raw_path = raw_path[:-1]
    total_length = os.listdir(raw_path)
    accumulate = 0

    for i in tqdm(range(len(total_length))):
        if i in error_samples:
            accumulate += 1
        if accumulate == 0:
            continue
        else:
            shutil.copy(f'{raw_path}/{i}.cif', f'{raw_path}/{i-accumulate}.cif')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset to be processed, must be jarvis dataset.')
    parser.add_argument('--work', type=str, help='target directory.')
    parser.add_argument('--label', type=str, help='label name')
    parser.add_argument('--label_2', type=str, help='another label name', default='')
    parser.add_argument('--ratio', type=float, default=0.9, help='ratio of train set')
    parser.add_argument('--split', type=str, help='path to split file')
    parser.add_argument('--random_seed', type=int, help='random seed for dataset split', default=42)
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    make_dir(args.work)

    if args.split:
        train_idx, val_idx = joblib.load(args.split)
    else:
        train_idx, val_idx = split_dataset(dataset, args.ratio, args.random_seed)

    process_jarvis_dataset(dataset, args.work, args.label, train_idx, val_idx, label_2=args.label_2)
    remove_errors(f'{args.work}/train/raw', joblib.load(f'{args.work}/train/error_samples.jbl'))
    remove_errors(f'{args.work}/val/raw', joblib.load(f'{args.work}/val/error_samples.jbl'))


if __name__ == '__main__':
    main()
