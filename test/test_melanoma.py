# In[]
from random import random
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import scdisinfact
import loss_function as loss_func
import utils
import bmk

import anndata as ad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

from umap import UMAP
import seaborn
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')


# In[] Read in the dataset
# version 1, all cells
data_dir = "../data/melanoma/"
result_dir = "melanoma/"
counts = sparse.load_npz(data_dir + "counts_2000.npz")
genes = np.loadtxt(data_dir + "genes_2000.txt", dtype = np.object)
barcodes = np.loadtxt(data_dir + "barcodes.txt", dtype = np.object)
counts = pd.DataFrame(counts.todense(), index = genes.squeeze(), columns = barcodes.squeeze())
meta_cells = pd.read_csv(data_dir + "meta_cells.csv", index_col = 0)
# 48 samples in total, encode label into indices
batch_ids, batch_names = pd.factorize(meta_cells["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze())
response_ids, response_names = pd.factorize(meta_cells["characteristics: response"].values.squeeze())

# NOTE: no normalization and log transform, they are all in the model
counts_array = []
meta_cells_array = []
datasets_array = []
for batch_id, batch_name in enumerate(batch_names):
    # if batch_name in ["Pre_P1", "Pre_P2", "Post_P1_2", "Post_P4"]:
        counts_array.append(counts.loc[:, batch_ids == batch_id].values.T)
        meta_cells_array.append(meta_cells[batch_ids == batch_id])
        response = response_ids[batch_ids == batch_id]
        datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = meta_cells_array[-1]["Cluster number"].values.squeeze(), diff_labels = [response], batch_id = batch_ids[batch_ids == batch_id]))

# version 2, only CD8+ T cells
counts_sub = counts.loc[:, meta_cells["CD8+T annotation"] != "Other"]
meta_cells_sub = meta_cells[meta_cells["CD8+T annotation"] != "Other"]
# 48 samples in total, encode label into indices
batch_sub_ids, batch_sub_names = pd.factorize(meta_cells_sub["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze())
response_sub_ids, response_sub_names = pd.factorize(meta_cells_sub["characteristics: response"].values.squeeze())

counts_sub_array = []
meta_cells_sub_array = []
datasets_sub_array = []
for batch_id, batch_name in enumerate(batch_sub_names):
    # if batch_name in ["Pre_P1", "Pre_P2", "Post_P1_2", "Post_P4"]:
        counts_sub_array.append(counts_sub.loc[:, batch_sub_ids == batch_id].values.T)
        meta_cells_sub_array.append(meta_cells_sub[batch_sub_ids == batch_id])
        response = response_sub_ids[batch_sub_ids == batch_id]
        datasets_sub_array.append(scdisinfact.dataset(counts = counts_sub_array[-1], anno = meta_cells_sub_array[-1]["Cluster number"].values.squeeze(), diff_labels = [response], batch_id = batch_sub_ids[batch_sub_ids == batch_id]))


# In[] Test with all
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 

x_umap = umap_op.fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
# separate into batches
x_umaps = []
for batch, _ in enumerate(counts_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (len(batch_names) - 1):
        start_pointer = start_pointer + counts_array[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + counts_array[batch - 1].shape[0]
        end_pointer = start_pointer + counts_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

save_file = None

utils.plot_latent(x_umaps, annos = [x["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["characteristics: response"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["Cluster number"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["Cluster number"].values.squeeze() for x in meta_cells_array], mode = "separate", save = save_file, figsize = (10,140), axis_label = "UMAP", markerscale = 6)

# In[] training the model
import importlib 
importlib.reload(scdisinfact)
# m, gamma = 0.3, 0.5
# contr_loss = loss_func.CircleLoss(m = m, gamma = gamma)
# contr_loss = loss_func.SupervisedContrastiveLoss(temperature = 0.07)

# reconstruction, mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [0.01, 1.0, 0.1, 1, 1e-5]
# lambs = [1, 0.00, 0.0, 0.0, 0, 0, 0.0]
Ks = [12, 4]

model1 = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs, seed = 0, device = device)
# model1 = scdisinfact.scdisinfact_ae(datasets = datasets_array, Ks = Ks, batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs[0:5] + [lambs[6]], contr_loss = contr_loss, seed = 0, device = device)
losses = model1.train(nepochs = 1000)
# torch.save(model1.state_dict(), result_dir + "model.pth")
# model1.load_state_dict(torch.load(result_dir + "model.pth"))

# In[] Plot the loss curve
plt.rcParams["font.size"] = 20
loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests = losses
iters = np.arange(1, len(loss_tests)+1)

fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_tests, "-*", label = 'Total loss')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_gl_d_tests, "-*", label = 'Group Lasso diff')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for dataset in datasets_array:
    with torch.no_grad():
        z_c, _ = model1.Enc_c(torch.concat([dataset.counts_stand, dataset.batch_id[:, None]], dim = 1).to(model1.device))
        # z_c = model1.Enc_c(dataset.counts_stand.to(model1.device))

        z_ds.append([])
        for Enc_d in model1.Enc_ds:
            z_d, _ = Enc_d(torch.concat([dataset.counts_stand, dataset.batch_id[:, None]], dim = 1).to(model1.device))
            # z_d = Enc_d(dataset.counts_stand.to(model1.device))
            z_ds[-1].append(z_d.cpu().detach().numpy())
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch, _ in enumerate(datasets_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (len(datasets_array) - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f'plots_{Ks}_{lambs[1]}_{lambs[2]}_{lambs[3]}_{lambs[4]}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


# utils.plot_latent(zs = z_cs_umaps, annos = [x["Cluster number"].values.squeeze() for x in meta_cells_array], mode = "separate", axis_label = "UMAP", figsize = (10,140), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["Cluster number"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15, 7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cluster number"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["characteristics: response"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)


# In[] Test with only T cells, didn't see clear separation of clusters
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 

x_umap = umap_op.fit_transform(np.concatenate(counts_sub_array, axis = 0))
# separate into batches
x_umaps = []
for batch, batch_name in enumerate(batch_sub_names):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts_sub_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (len(batch_sub_names) - 1):
        start_pointer = start_pointer + counts_sub_array[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + counts_sub_array[batch - 1].shape[0]
        end_pointer = start_pointer + counts_sub_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

save_file = None

utils.plot_latent(x_umaps, annos = [x["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", save = result_dir + "batches_sub.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["characteristics: response"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", save = result_dir + "conditions_sub.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["Cluster number"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", save = result_dir + "celltype_sub.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["Cluster number"].values.squeeze() for x in meta_cells_sub_array], mode = "separate", save = save_file, figsize = (10,140), axis_label = "UMAP", markerscale = 6)

# In[] training the model
import importlib 
importlib.reload(scdisinfact)
# reconstruction, mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [0.01, 1.0, 0.1, 1, 1e-5]
# lambs = [1, 0.00, 0.0, 0.0, 0, 0, 0.0]
Ks = [12, 4]

model1 = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs, seed = 0, device = device)
# model1 = scdisinfact.scdisinfact_ae(datasets = datasets_array, Ks = Ks, batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs[0:5] + [lambs[6]], contr_loss = contr_loss, seed = 0, device = device)
losses = model1.train(nepochs = 300)
torch.save(model1.state_dict(), result_dir + "model_sub.pth")
model1.load_state_dict(torch.load(result_dir + "model_sub.pth"))

# In[] Plot the loss curve
plt.rcParams["font.size"] = 20
loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests = losses
iters = np.arange(1, len(loss_tests)+1)

fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_tests, "-*", label = 'Total loss')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_gl_d_tests, "-*", label = 'Group Lasso diff')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for dataset in datasets_sub_array:
    with torch.no_grad():
        z_c, _ = model1.Enc_c(torch.concat([dataset.counts_stand, dataset.batch_id[:, None]], dim = 1).to(model1.device))
        z_ds.append([])
        for Enc_d in model1.Enc_ds:
            z_d, _ = Enc_d(torch.concat([dataset.counts_stand, dataset.batch_id[:, None]], dim = 1).to(model1.device))
            z_ds[-1].append(z_d.cpu().detach().numpy())
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch, batch_name in enumerate(batch_sub_names):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (len(batch_sub_names) - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f'plots_{Ks}_{lambs[1]}_{lambs[2]}_{lambs[3]}_{lambs[4]}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umaps, annos = [x["CD8+T annotation"].values.squeeze() for x in meta_cells_sub_array], mode = "separate", axis_label = "UMAP", figsize = (10,70), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["CD8+T annotation"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", axis_label = "UMAP", figsize = (15,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["CD8+T annotation"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["characteristics: response"].values.squeeze() for x in meta_cells_sub_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 5)


# %%
