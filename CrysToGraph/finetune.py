import os
import time
import joblib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import crystal
from train import FineTuningWithDGL

from model.bert_transformer import TransformerConvLayer
from model.scheduler import WarmupMultiStepLR
from model.NN import Finetuning
from model.model_utils import get_finetune_model_params

# make dir and get data from matbench
try:
    os.mkdir('/home/howardwang/Documents/datasets/train/')
    os.mkdir('/home/howardwang/Documents/datasets/train/raw/')
    os.mkdir('/home/howardwang/Documents/datasets/train/processed/')
except FileExistsError:
    pass

data_root = '/home/howardwang/Documents/datasets/train/'
atom_vocab = joblib.load('config/atom_vocab.jbl')

# process dataset, OPTIONAL
cd = crystal.CrystalDataset(root=data_root, atom_vocab=atom_vocab)

# Read processed dataset
checkpoint = torch.load('config/contrastive_pretrained_sd.pt')
embeddings = torch.load('embeddings_86_64catcgcnn.pt')

# start the pretraining
atom_fea_len = 156
nbr_fea_len = 76 
vocab_len = len(atom_vocab.numbers) # add the len attr to AtomVocab!

t = time.time()

module = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=76, dropout=0.0) for _ in range(3)]), \
         nn.ModuleList([TransformerConvLayer(76, 24, 8, edge_dim=30, dropout=0.0) for _ in range(3)])

batch_size = 32
embeddings = embeddings.cuda()
ft = Finetuning(atom_fea_len, nbr_fea_len, embeddings=embeddings, h_fea_len=256, n_conv=3, n_fc=2, n_gt=1,
                module=module, norm=True, drop=0.0)
ft.load_state_dict(checkpoint, strict=False)
mtcp = FineTuningWithDGL(model=ft)
# optimizer = optim.SGD(ft.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.AdamW(get_finetune_model_params(ft, 1e-4, 0), lr=0.0001, betas=(0.9, 0.99), weight_decay=0)
scheduler = WarmupMultiStepLR(optimizer, [50], gamma=0.1)
epochs = 100

mpcd = crystal.ProcessedDGLCrystalDataset(root=data_root)
mpcd.set_atom_vocab(atom_vocab)

target = joblib.load('%slabels.jbl' % data_root)
mpcd.set_labels(target)

trainloader = DataLoader(mpcd, batch_size=batch_size, shuffle=True, collate_fn=mpcd.collate_line_graph)

mtcp.train(mixed_train_loader=trainloader,
           optimizer=optimizer,
           epochs=epochs,
           verbose_freq=700,
           grad_accum=8)
mtcp.save_model('config/finetune.pt')
mtcp.save_state_dict('config/finerune_sd.pt')

print('training time =', time.time()-t)


#EVAL starts from here
from torch.nn import L1Loss, MSELoss
from math import sqrt
pcd = crystal.ProcessedDGLCrystalDataset(root='/home/howardwang/Documents/datasets/val/', atom_vocab=atom_vocab)
l = joblib.load('/home/howardwang/Documents/datasets/val/labels.jbl')
pcd.set_labels(l)
MAE = L1Loss()
MSE = MSELoss()
mae_list = []
mse_list = []

testloader = DataLoader(pcd, batch_size=30, shuffle=False, collate_fn=pcd.collate_line_graph)

model = mtcp.model
model.eval()
for _, data in enumerate(testloader):
    data = tuple([data[0].to(torch.device('cuda:0')), data[1].to(torch.device('cuda:0')), torch.unsqueeze(data[2], -1)])
    out = model(data).cpu()
    mae = float(MAE(out, data[-1]))
    mse = float(MSE(out, data[-1]))
    mae_list.append(mae)
    mse_list.append(mse)
#    print('%d\tMAE: %.4f\tRMSE: %.4f' % (_, mae, rmse))

mae = sum(mae_list) / len(mae_list)
rmse = sqrt(sum(mse_list) / len(mse_list))
print('AVERAGE\tMAE: %.4f\tRMSE: %.4f' % (mae, rmse))
