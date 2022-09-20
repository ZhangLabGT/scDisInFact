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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

from umap import UMAP
import seaborn
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

from anndata import AnnData

# In[] Read in the dataset, filtering out the batches that we don't need
# checked total cell count correct: 1,462,702 cells, genes: 5000
data_dir = "../data/covid/batch_processed/"
data_dir = "../data/covid/batch_raw/"
result_dir = "covid/"
result_dir = "covid_raw/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
batches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]
patient_meta = pd.read_excel("../data/covid/patient_meta.xlsx")
# genes = pd.read_csv(data_dir + "filtered_gene_5000.csv", index_col = 0).values.squeeze()
genes = pd.read_csv(data_dir + "covid_gene.tsv", header = None).values.squeeze()

counts_array = []
meta_cells_array = []
for batch_id in batches:
    # TODO: read genes and barcodes
    # counts_array.append(sparse.load_npz(data_dir + f"raw_filtered_batch{batch_id}.npz").toarray())
    counts_array.append(sparse.load_npz(data_dir + f"mtx_batch{batch_id}_raw.npz").toarray())

    # NOTE: COLUMNS: cellName, sampleID, celltype, majorType, PatientID, Batches, City, Age, Sex, Sample type, CoVID-19 severity, Sample time, 
    # Sampling day (Days after symptom onset), SARS-CoV-2, Single cell sequencing platform, BCR single cell sequencing, CR single cell sequencing, 
    # Outcome, Comorbidities, COVID-19-related medication and anti-microbials Leukocytes [G/L], Neutrophils [G/L], Lymphocytes [G/L], Unpublished
    meta_cells_array.append(pd.read_csv(data_dir + f"meta_batch{batch_id}.csv", index_col = 0))
    print(batch_id)
    print(counts_array[-1].shape[0])
    print(meta_cells_array[-1].shape[0])
    assert counts_array[-1].shape[0] == meta_cells_array[-1].shape[0]
    print(f'Batch ID: {batch_id}, number of cells: {counts_array[-1].shape[0]}')

# NOTE: In the paper there are in total 5 conditions: 
# 1. healthy donor (n = 25), 
# 2. disease progress severe (n = 54), 
# 3. disease progress moderate/mild (n = 22), 
# 4. recovered from severe (n = 38)
# 5. recovered from moderate/mild (n = 57)
# IMPORTENT factors: Batch/city, Age[grouped], Sex, severity + sample time (5 conditions), Comorbidities, COVID-19-related medication and anti-microbials
# The first two sections of the paper talk about fresh and frozen pbmc sample type, 
# and its association with age, sex, stage, severity, and sample time, sample type (frozen/fresh PBMC).
# one city can have multiple batch, one batch consists of different patient, one patient with multiple sampling time.
# city >> sample batch >> patient/day >> sample. Sample is the smallest concept, each sample should be considered as one true batch.
# the paper mainly consider the sample type and sample time as the technical effect.
counts = np.concatenate(counts_array, axis = 0)
meta_cells = pd.concat(meta_cells_array, axis = 0)
meta_cells.index = meta_cells["cellName"].values
meta_cells = meta_cells.iloc[:, 1:]
adata = AnnData(X= sparse.csr_matrix(counts))
adata.obs = meta_cells
adata.var.index = genes

# In[]
# NOTE: how to treat the batches? If uses the sample, one sample will only have one unique condition [issue with MMD]. Use sequencing batch, sample variation within batch?
# TODO: only uses one sequencing batch, check if the distribution of different patients under the same condition should be the same.

# totally 42 samples
# patient_meta_sub = patient_meta[(patient_meta["Sample type"] == "fresh PBMC") & (patient_meta["Age"] != "unknown") & (patient_meta["Sample time"] != "convalescence")]
adata_sub = adata[(adata.obs["Sample type"] == "fresh PBMC") & (adata.obs["Age"] != "unknown") & (adata.obs["Sample time"] != "convalescence")]
# subsampling
adata_sub = adata_sub[::10,:]

batch_ids, batch_names = pd.factorize(adata_sub.obs["Batches"].values.squeeze())
severity_ids, severity_names = pd.factorize(adata_sub.obs["CoVID-19 severity"].values.squeeze())
sex_ids, sex_names = pd.factorize(adata_sub.obs["Sex"])

counts_array = []
meta_cells_array = []
datasets_array = []
for batch_id, batch_name in enumerate(batch_names):
    adata_batch = adata_sub[batch_ids == batch_id, :]
    counts_array.append(adata_batch.X.toarray())
    meta_cells_array.append(adata_batch.obs)
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [severity_ids[batch_ids == batch_id], sex_ids[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))

# In[]
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 

x_umap = umap_op.fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
# separate into batches
x_umaps = []
for batch, _ in enumerate(counts_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (len(counts_array) - 1):
        start_pointer = start_pointer + counts_array[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + counts_array[batch - 1].shape[0]
        end_pointer = start_pointer + counts_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

save_file = None

utils.plot_latent(x_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["CoVID-19 severity"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions1.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["Sex"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions2.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["majorType"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["majorType"].values.squeeze() for x in meta_cells_array], mode = "separate", save = result_dir + "separate.png", figsize = (10, 40), axis_label = "Latent", markerscale = 6)

# In[]
import importlib 
importlib.reload(scdisinfact)
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [1e-2, 1.0, 0.1, 1, 1e-6]
Ks = [12, 4, 4]

model1 = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs, seed = 0, device = device)
losses = model1.train(nepochs = 300, recon_loss = "NB")
torch.save(model1.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model1.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth"))

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
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch, _ in enumerate(datasets_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (len(datasets_array) - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f'plots_{Ks}_{lambs}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = [x["majorType"].values.squeeze() for x in meta_cells_array], mode = "separate", axis_label = "UMAP", figsize = (10,40), save = (result_dir + comment+"common_dims_separate.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["majorType"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["majorType"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_1_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["CoVID-19 severity"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_1_condition.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [x["majorType"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_2_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [x["Sex"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_2_condition.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

 # %%
