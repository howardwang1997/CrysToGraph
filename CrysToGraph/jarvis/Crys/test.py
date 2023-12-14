import os
import joblib
import argparse
import shutil

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from matbench.bench import MatbenchBenchmark

from data import CrystalDataset
from train import Trainer
from model.NN import CrysToGraphNet
from model.bert_transformer import TransformerConvLayer
from model.scheduler import WarmupMultiStepLR

mb = MatbenchBenchmark(autoload=False)
mb = mb.from_preset('matbench_v0.1', 'structure')

parser = argparse.ArgumentParser(description='Run CrysToGraph on matbench.')
parser.add_argument('checkpoint', type=str)
parser.add_argument('--task', type=str, default='dielectric')
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--n_conv', type=int, default=5)
parser.add_argument('--atom_fea_len', type=int, default=156)
parser.add_argument('--nbr_fea_len', type=int, default=76)
parser.add_argument('--batch_size', type=int, default=10)
args = parser.parse_args()

for task in mb.tasks:
    name = task.dataset_name
    if args.task not in name:
        continue
    task.load()

    classification = task.metadata['task_type'] == 'classification'
    input_type = task.metadata['input_type']

    # hyperparameters
    atom_fea_len = args.atom_fea_len
    nbr_fea_len = args.nbr_fea_len
    batch_size = args.batch_size

    if atom_fea_len == 156:
        embeddings_path = 'embeddings_86_64catcgcnn.pt'
    else:
        embeddings_path = ''

    # mkdir
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

    for fold in task.folds:
        if args.fold >= 0 and args.fold <= 5:
            if fold != args.fold:
                continue

        # define atom_vocab, dataset, model, trainer
        embeddings = torch.load(embeddings_path).cuda()
        atom_vocab = joblib.load('atom_vocab.jbl')
        module = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=76, dropout=0.0) for _ in range(args.n_conv)]), \
                 nn.ModuleList([TransformerConvLayer(76, 24, 8, edge_dim=30, dropout=0.0) for _ in range(args.n_conv)])
        ctgn = CrysToGraphNet(atom_fea_len, nbr_fea_len, embeddings=embeddings, h_fea_len=256, n_conv=args.n_conv, n_fc=2, module=module, norm=True)
        ctgn.load_state_dict(torch.load(args.checkpoint))
        trainer = Trainer(ctgn, name='%s_%d' % (name, fold), classification=classification)

        # predict
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        cd = CrystalDataset(root=name,
                            atom_vocab=atom_vocab,
                            inputs=test_inputs,
                            outputs=test_outputs)
        test_loader = DataLoader(cd, batch_size=batch_size, shuffle=False, collate_fn=cd.collate_line_graph)
        predictions = trainer.predict(test_loader=test_loader)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()