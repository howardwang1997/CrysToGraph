import json

from .jarvis_constant import DATASETS_MAP
from .jarvis_utils import jarvis_dataset_to_mp, make_validation


def main():
    for k, v in DATASETS_MAP.items():
        with open(v['path']) as f:
            dataset = json.load(f)
        _ = jarvis_dataset_to_mp(dataset, v['label'], k, save=True)

    _ = make_validation(random_seed=42, save=True)


if __name__ == '__main__':
    main()
