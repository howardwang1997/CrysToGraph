import os
import json
import time
from tqdm import tqdm
import joblib

import torch
from torch import nn, optim
from torch_geometric import nn as tgnn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader


import data as c2data
import crystal
from augmentation import Augmentation
import augmentation
from NN import WrappedCGCNN, PreTrainingOnNodes
from pretrain import AtomRepresentationPretraining, Normalizer

# make dir and get data from matbench
try:
    os.mkdir('/home/howardwang/Documents/crystal_dataset')
    os.mkdir('crystal_dataset/raw')
    os.mkdir('crystal_dataset/processed')
except FileExistsError:
    pass

data_root = './crystal_dataset/'
data_name = 'matbench_mp_e_form.json'

# Read processed dataset
cd = crystal.load_dataset()

# start the pretraining
atom_fea_len = 9
nbr_fea_len = cd.get_crystal(0).graph.edge_attr.shape[1] # add the attr!

model = nn.ModuleList([tgnn.CGConv(channels=32,
                                                   dim=nbr_fea_len,
                                                   batch_norm=True)
                                        for _ in range(3)])

nbr_fea_len = cd.get_crystal(0).graph.edge_attr.shape[1] # add the attr!
vocab_len = len(cd.atom_vocab.numbers) # add the len attr to AtomVocab!
pton = PreTrainingOnNodes(atom_fea_len, nbr_fea_len, vocab_len, module=model)
print(pton)

t = time.time()
pcd = crystal.ProcessedCrystalDataset(suffix='am')
pcd.set_atom_vocab(cd.atom_vocab)
pcd.set_masked_labels()
arp = AtomRepresentationPretraining(model=pton)
trainloader = DataLoader(pcd, batch_size=100, shuffle=True) # need to make masked labels
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(pton.parameters(), lr=0.02, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
epochs = 30

arp.train(train_loader=trainloader, 
          criterion=criterion, 
          optimizer=optimizer, 
          scheduler=scheduler,
          epochs=epochs)
print('training time =', time.time()-t)

_, emb = arp.get_atomic_representation()
loss = arp.loss_list
joblib.dump(loss, 'output/loss_38.jbl')

torch.save(emb, 'config/atom_representations_85_38.tch')
