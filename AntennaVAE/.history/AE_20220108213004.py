# %%
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List
from torch.nn.parameter import Parameter
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import collections
import pandas as pd 
import numpy as np
import random 
from torch.utils.data import DataLoader, random_split
import scanpy as sc

from model import Encoder, Decoder, OutputLayer
from loss_function import ZINB, maximum_mean_discrepancy
from utils import plot_train

# %%
# Data
path_pool = [
    r"./data/mtx_0.mtx", 
    r'./data/GSE65525/GSM1599497_ES_d2_LIFminus.csv', 
    r'./data/GSE65525/GSM1599498_ES_d4_LIFminus.csv', 
    r'./data/GSE65525/GSM1599499_ES_d7_LIFminus.csv'
    ]
# path_mtx = path_pool[0]
data_test= sc.read_csv(path_pool[1])
# data_test = sc.read_mtx(path_mtx)

# %%
# PREVIEW
load_data = data_test.T.copy()
sc.pp.filter_genes(load_data, min_cells=50)
print(data_test.shape, load_data.shape)

# %%
batch_size = 32
all_data = np.array(load_data.X)

seed = 222
random.seed(seed)

m = len(all_data)
print(m)

train_data, test_data = random_split(dataset=all_data, lengths=[int(m - m * 0.2), int(m * 0.2) + 1])

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# %%
a = torch.isnan(torch.tensor(all_data))
print(torch.sum(a))

# %%
net_struc = [all_data.shape[1], 1024, 512, 256, 128, 64]

encoder = Encoder(features=net_struc)
decoder = Decoder(features=net_struc[1:][::-1])
output_layer = OutputLayer(net_struc[0], net_struc[1])
# loss_fn = MMD_LOSS()
loss_fn = nn.MSELoss()

# Define Optimizer
lr = 0.0005

#Random seed
torch.manual_seed(seed)
param_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(param_to_optimize, lr=lr, weight_decay=1e-05)

# %%
# Training
def train_epoch(encoder, decoder, dataloader, optimizer):
    encoder.train()
    decoder.train()
    output_layer.train()
    train_loss = 0.0
    for sc_data_batch in dataloader:
        # Encode
        encoded_data = encoder(sc_data_batch)
        # Decode
        decoded_data = decoder(encoded_data)

        # Compute params for DCA
        mean_param, pi_param, theta_param = output_layer(decoded_data)

        # Evaluate loss
        zinb = ZINB(pi_param, theta=theta_param)
        zinb_loss = (zinb.loss(mean_param, sc_data_batch))
        mmd_loss = maximum_mean_discrepancy(mean_param, sc_data_batch)
        loss = zinb_loss * 0.5 + mmd_loss * 0.5
        
        # Backword 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss / len(dataloader.dataset)

# %%
# Testing func
def test_epoch(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    output_layer.eval()
    test_loss = 0.0
    with torch.no_grad(): # Don't track gradients
        for sc_data_batch in dataloader:
            # Encode
            encoded_data = encoder(sc_data_batch)
            # Decode
            decoded_data = decoder(encoded_data)
            # Compute params for DCA
            mean_param, pi_param, theta_param = output_layer(decoded_data)

            zinb = ZINB(pi_param, theta=theta_param)

            zinb_loss = (zinb.loss(mean_param, sc_data_batch))
            mmd_loss = maximum_mean_discrepancy(mean_param, sc_data_batch)
            loss = zinb_loss * 0.5 + mmd_loss * 0.5
            
            test_loss += loss.item()
    return test_loss / len(dataloader.dataset)

# %%
num_epochs = 50
all_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss = train_epoch(encoder,decoder, train_loader,optim)
   test_loss = test_epoch(encoder,decoder,test_loader)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,test_loss))
   all_loss['train_loss'].append(train_loss)
   all_loss['val_loss'].append(test_loss)
   if epoch and (epoch + 1) % 5 == 0:
      plot_train(all_loss)

# %%
torch.lgamma(torch.tensor(2))


