import os
import time
import joblib

import torch
from torch import nn, optim
from torch_geometric import nn as tgnn
from torch.utils.data import DataLoader

from .data import crystal
from .model.NN import PreTrainingOnNodes
from .train import AtomRepresentationPretraining

# make dir and get data from matbench
try:
    os.mkdir('/home/howardwang/Documents/datasets/overall/')
    os.mkdir('/home/howardwang/Documents/datasets/overall/raw/')
    os.mkdir('/home/howardwang/Documents/datasets/overall/processed/')
except FileExistsError:
    pass

data_root = '/home/howardwang/Documents/datasets/overall/'
atom_vocab = joblib.load('/home/howardwang/Documents/datasets/overall/atom_vocab.jbl')

# start the pretraining
atom_fea_len = 64
nbr_fea_len = 76
batch_size = 32

model = nn.ModuleList([tgnn.CGConv(channels=64,
                                   dim=nbr_fea_len,
                                   batch_norm=True)
                       for _ in range(3)])

vocab_len = len(atom_vocab.numbers) # add the len attr to AtomVocab!
pton = PreTrainingOnNodes(atom_fea_len, nbr_fea_len, vocab_len, module=model)
print(pton)

t = time.time()
pcd = crystal.ProcessedDGLCrystalDataset(root=data_root, suffix='am', load_data=False)
pcd.set_atom_vocab(atom_vocab)
pcd.set_masked_labels()
arp = AtomRepresentationPretraining(model=pton)
trainloader = DataLoader(pcd, batch_size=batch_size, shuffle=True, collate_fn=pcd.collate_masked_graph, num_workers=10)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(pton.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
epochs = 60

arp.train(train_loader=trainloader, 
          criterion=criterion, 
          optimizer=optimizer, 
          scheduler=scheduler,
          epochs=epochs)
print('training time =', time.time()-t)

_, emb = arp.get_atomic_representation()
loss = arp.loss_list
joblib.dump(loss, 'output/loss_arp_0.jbl')

torch.save(emb, 'config/atom_representations_86_0.pt')
