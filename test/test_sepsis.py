# In[]
import sys, os
sys.path.append('../src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import pandas as pd 
import numpy as np
import random 
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from model import Encoder, Decoder, OutputLayer
from loss_function import *

from dataset import dataset
from scipy import sparse

import utils
import model
import train
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.preprocessing import LabelEncoder
from metrics import *
import anndata as ad

class dataset(Dataset):

    def __init__(self, counts, anno, time_point, batch_id, group_id):

        assert not len(counts) == 0, "Count is empty"
        # normalize the count
        self.libsizes = np.tile(np.sum(counts, axis = 1, keepdims = True), (1, counts.shape[1]))
        # is the tile necessary?
        
        self.counts_norm = counts/self.libsizes * 100
        self.counts_norm = np.log1p(self.counts_norm)
        self.counts = torch.FloatTensor(counts)

        # further standardize the count
        self.counts_stand = torch.FloatTensor(StandardScaler().fit_transform(self.counts_norm))
        self.anno = torch.Tensor(anno)
        self.libsizes = torch.FloatTensor(self.libsizes)
        self.time_point = torch.Tensor(time_point)
        self.batch_id = torch.Tensor(batch_id)
        self.group_id = torch.Tensor(group_id)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        if self.anno is not None:
            sample = {"batch_id": self.batch_id[idx], "time_point": self.time_point[idx], "group_id": self.group_id[idx], "count": self.counts[idx,:], "count_stand": self.counts_stand[idx,:], "index": idx, "anno": self.anno[idx], "libsize": self.libsizes[idx]}
        else:
            sample = {"batch_id": self.batch_id[idx], "time_point": self.time_point[idx], "group_id": self.group_id[idx],  "count": self.counts[idx,:], "count_stand": self.counts_stand[idx,:], "index": idx, "libsize": self.libsizes[idx]}
        return sample

# In[]
le = LabelEncoder()

# format: "Cohort Name": ([batch ids], time_point, UTI-label)
batch_dict = {
            #   'Control': ([4, 6], 0, 0),
              'Leuk-UTI': ([3, 5], 0),
              'Int-URO': ([7, 11], 1),
              'URO': ([13, 15], 2),
            #   'Bac-SEP': ([31, 33], 2, 0),
            #   'ICU-SEP': ([19, 29], 2, 0),
             }

# Use all data but ICU-NoSEP, and use "group-id" to label whether this cohort includes UTI symptom,

dir = r'../data/scp_gex_matrix/processed_sepsis_7533/'
batchsize = 8
ngenes = 7533
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

sc_datasets = []
sc_datasets_raw = []
train_loaders = []
test_loaders = []
train_loaders_raw = []
test_loaders_raw = []
n = 0
for name, labels in batch_dict.items():
    idxes = labels[0]
    time_point = labels[1]
    UTI_label = labels[2]
    time_points = []
    counts_rnas = []
    group_ids = []
    batch_ids = []
    annos = []
    for idx in idxes:
        counts_rna = np.array(sparse.load_npz(os.path.join(dir, '{}/mtx_{}_batch_{}.npz'.format(name,name,idx))).todense())
        anno = pd.read_csv(os.path.join(dir, '{}/meta_{}_batch{}.csv'.format(name, name,idx)))["Cell_Type"]
        assert counts_rna.shape[0] == anno.shape[0]
        anno = le.fit_transform(anno)
        annos.append(anno)
        counts_rnas.append(counts_rna)
        time_points.append([time_point] * counts_rna.shape[0])
        group_ids.append([UTI_label] * counts_rna.shape[0])
        batch_ids.append([n] * counts_rna.shape[0])
        sc_dataset_raw =  dataset(counts = counts_rna,anno = anno, time_point = [time_point] * counts_rna.shape[0], 
                         group_id = [UTI_label] * counts_rna.shape[0], 
                         batch_id = [n] * counts_rna.shape[0])
        sc_datasets_raw.append(sc_dataset_raw)
        train_loaders_raw.append(DataLoader(sc_dataset_raw, batch_size = batchsize, shuffle = True))
        test_loaders_raw.append(DataLoader(sc_dataset_raw, batch_size = len(sc_dataset_raw), shuffle = False))
        print(name, idx, 'finished')
        n += 1
    sc_dataset = dataset(counts = counts_rna,anno = np.concatenate(annos), time_point = np.concatenate(time_points), 
                         group_id = np.concatenate(group_ids), 
                         batch_id = np.concatenate(batch_ids))
    sc_datasets.append(sc_dataset)
    train_loaders.append(DataLoader(sc_dataset, batch_size = batchsize, shuffle = True))
    test_loaders.append(DataLoader(sc_dataset, batch_size = len(sc_dataset), shuffle = False))
    
nbatches = len(sc_datasets_raw)


# In[]
import scanpy as sc
from matplotlib import rcParams

counts_norms = []
annos = []
batch_ids = []
counts = 0
for name, labels in batch_dict.items():
    idxes = labels[0]
    time_point = labels[1]
    UTI_label = labels[2]
    for idx in idxes:
        counts_norms.append(sc_datasets_raw[counts].counts_norm)
        annos.append(pd.read_csv(os.path.join(dir, '{}/meta_{}_batch{}.csv'.format(name, name,idx)))["Cell_Type"].values.squeeze())
        # batch_ids.append(np.array(["time:" + str(time_point) + ", batch" + str(UTI_label) for x in range(counts_norms[-1].shape[0])]))
        batch_ids.append(np.array([name for x in range(counts_norms[-1].shape[0])]))
        assert annos[-1].shape[0] == counts_norms[-1].shape[0]
        counts += 1
adata = ad.AnnData(np.concatenate(counts_norms, axis = 0))

adata.obs['Cell_Type'] = np.concatenate(annos)
adata.obs['batch_id'] = np.concatenate(batch_ids)
adata.obs['Cell_Type'] = adata.obs['Cell_Type'].astype("category")
adata.obs['batch_id'] = adata.obs['batch_id'].astype("category")

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
sc.tl.umap(adata)

save = None
rcParams['figure.figsize'] = 14, 10
sc.pl.umap(adata, color = ['batch_id'], save = save)
sc.pl.umap(adata, color = ['Cell_Type'], save = save)

# sc.tl.leiden(adata, resolution = 0.1)
# sc.pl.umap(adata, color = ['leiden'], save=save)

# In[]
# initialize the model
import importlib 
importlib.reload(train)
ldim = 32
lr = 5e-4
model_dict = {}
model_dict["encoder"] = model.Encoder(features = [ngenes, 256, 32, ldim], dropout_rate = 0, negative_slope = 0.2).to(device)
model_dict["decoder"] = model.Decoder(features = [ldim, 32, 256, ngenes], dropout_rate = 0, negative_slope = 0.2).to(device)
# initialize the optimizer
param_to_optimize = [
    {'params': model_dict["encoder"].parameters()},
    {'params': model_dict["decoder"].parameters()}
]

optim_ae = torch.optim.Adam(param_to_optimize, lr=lr)

# use Circle loss to distinguish different time points
contrastive_loss = CircleLoss(m=0.25, gamma= 80)
# contrastive_loss = TripletLoss(margin=0.3)
# contrastive_loss = torch.nn.CrossEntropyLoss()
# contrastive_loss = SupConLoss()
# contrastive_loss = SupervisedContrastiveLoss()
lamb_contr_time = 1e-2
lamb_contr_group = 1e-2
lamb_mmd = 0
lamb_recon = 0
n_epoches = 30
time_dim = 5
group_dim = 5
losses = train.train_epoch_mmd(model_dict = model_dict, train_data_loaders = train_loaders_raw, test_data_loaders = test_loaders_raw, 
                      optimizer = optim_ae, n_epoches = n_epoches, interval = 10, lamb_mmd = lamb_mmd, lamb_recon = lamb_recon, lamb_contr_time = lamb_contr_time, lamb_contr_group = lamb_contr_group,
                      lamb_pi = 1e-5, use_zinb = True, contr_loss=contrastive_loss, 
                      time_dim = time_dim, group_dim = group_dim)


# In[]
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.3, random_state = 0) 

zs = []
# batch_ids = [0, 6]
# test_loaders_sub = [test_loaders_raw[i] for i in batch_ids]
test_loaders_sub = test_loaders_raw
nbatches_sub = len(test_loaders_sub)
annos_sub = annos
time_sub = []
group_sub = []
for data_batch in zip(*test_loaders_sub):
    with torch.no_grad():
        for idx, x in enumerate(data_batch):
            z = model_dict["encoder"](x["count_stand"].to(device))
            mu, pi, theta = model_dict["decoder"](z)
            zs.append(z.cpu().detach().numpy())
            time_sub.append(x["time_point"])
            group_sub.append(x["group_id"])

# use the shared dimensions
x_umap_shared = umap_op.fit_transform(np.concatenate(zs, axis = 0)[:, (time_dim + group_dim):])
# use the time dimensions
x_umap_time = umap_op.fit_transform(np.concatenate(zs, axis = 0)[:, : time_dim])
# x_umap_time = np.concatenate(zs, axis = 0)[:, :time_dim]
# use the group dimensions
x_umap_group = umap_op.fit_transform(np.concatenate(zs, axis = 0)[:, time_dim:(time_dim + group_dim)])
# x_umap_group = np.concatenate(zs, axis = 0)[:, time_dim:(time_dim + group_dim)]

# separate into batches
x_umaps_shared = []
x_umaps_time = []
x_umaps_group = []

for batch in range(nbatches_sub):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_shared.append(x_umap_shared[start_pointer:end_pointer,:])
        x_umaps_time.append(x_umap_time[start_pointer:end_pointer,:])
        x_umaps_group.append(x_umap_group[start_pointer:end_pointer,:])
    elif batch == (nbatches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps_shared.append(x_umap_shared[start_pointer:,:])
        x_umaps_time.append(x_umap_time[start_pointer:,:])
        x_umaps_group.append(x_umap_group[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_shared.append(x_umap_shared[start_pointer:end_pointer,:])
        x_umaps_time.append(x_umap_time[start_pointer:end_pointer,:])
        x_umaps_group.append(x_umap_group[start_pointer:end_pointer,:])

# utils.plot_latent(x_umaps_shared, annos = batch_ids, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_shared, annos = annos_sub, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps_time, annos = batch_ids, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_time, annos = time_sub, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps_group, annos = batch_ids, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_group, annos = group_sub, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# %%
