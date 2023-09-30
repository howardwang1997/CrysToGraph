import os
import time
import joblib

import torch
from torch import nn, optim
from torch_geometric import nn as tgnn
from torch.utils.data import DataLoader

from data import crystal
from model.nt_xent import NTXentLoss as ntxl
from model.NN import ContrastivePreTraining
from train import GraphConvPretraining
from model.bert_transformer import TransformerConvLayer

# make dir and get data from matbench
try:
    os.mkdir('/home/howardwang/Documents/datasets/eform/')
    os.mkdir('/home/howardwang/Documents/datasets/eform/raw/')
    os.mkdir('/home/howardwang/Documents/datasets/eform/processed/')
except FileExistsError:
    pass

data_root = '/home/howardwang/Documents/datasets/eform/'
atom_vocab = joblib.load('/home/howardwang/Documents/datasets/atom_vocab.jbl')
embeddings_path = '/home/howardwang/Documents/datasets/embeddings_86_64catcgcnn.pt'
embeddings = torch.load(embeddings_path).cuda()

# start the pretraining
atom_fea_len = 156
nbr_fea_len = 76
batch_size = 32

module = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=76, dropout=0.0) for _ in range(3)]), \
         nn.ModuleList([TransformerConvLayer(76, 24, 8, edge_dim=30, dropout=0.0) for _ in range(3)])


cpt = ContrastivePreTraining(atom_fea_len, nbr_fea_len, embeddings=embeddings, h_fea_len=256, n_conv=3,
                             module=module, norm=True)
gcp = GraphConvPretraining(model=cpt)
optimizer = optim.AdamW(cpt.parameters(), lr=0.0002, betas=(0.9, 0.99), weight_decay=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)
epochs = 100

t = time.time()
pcdc = crystal.ProcessedCrystalDatasetContrastive(root=data_root, suffixes=['', 'am', 'mr']
                                                  , load_data=True, random_suffix=True)
pcdc.set_atom_vocab(atom_vocab)
trainloader = DataLoader(pcdc, batch_size=batch_size, shuffle=True, collate_fn=pcdc.collate_line_graph, num_workers=10)
ntxent_criterion = ntxl(batch_size=batch_size, temperature=0.1, use_cosine_similarity=False)

gcp.train(train_loader=trainloader,
          criterion=ntxent_criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          epochs=epochs,
          verbose_freq=800)
print('training time =', time.time()-t)

loss = gcp.loss_list
joblib.dump(loss, 'output/loss_gcp_0.jbl')
torch.save(gcp.model.cpu(), 'config/contrastive_pretrained.pt')
torch.save(gcp.model.state_dict(), 'config/contrastive_pretrained_sd.pt')
