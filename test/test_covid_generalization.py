# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")

import scdisinfact
import utils
import bmk

import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

from anndata import AnnData
from sklearn.decomposition import PCA

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Load the dataset, treat each dataset as a batch, as the authors of each data claim that there were minor batch effect
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_dir = "../data/covid_integrated/"
result_dir = "results_covid/dropout/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

GxC1 = sparse.load_npz(data_dir + "GxC1.npz")
GxC2 = sparse.load_npz(data_dir + "GxC2.npz")
GxC3 = sparse.load_npz(data_dir + "GxC3.npz")
# be careful with the ordering
meta_c1 = pd.read_csv(data_dir + "meta_arunachalam_2020.txt", sep = "\t", index_col = 0)
meta_c2 = pd.read_csv(data_dir + "meta_lee_2020.txt", sep = "\t", index_col = 0)
meta_c3 = pd.read_csv(data_dir + "meta_wilk_2020.txt", sep = "\t", index_col = 0)

meta = pd.concat([meta_c1, meta_c2, meta_c3], axis = 0)
genes = pd.read_csv(data_dir + "genes_shared.txt", index_col = 0).values.squeeze()
# process age
age = meta.age.values.squeeze().astype(object)
age[meta["age"] < 40] = "40-"
age[(meta["age"] >= 40)&(meta["age"] < 65)] = "40-65"
# senior or not, class is not the main issue
# age[meta["age"] < 65] = "65-"
age[meta["age"] >= 65] = "65+"
meta["age"] = age

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

# no mmd batches
data_dict = scdisinfact.create_scdisinfact_dataset(counts_array, meta_cells_array, 
                                                   condition_key = ["disease_severity", "age"], 
                                                   batch_key = "dataset")

# In[] Visualize the original count matrix

umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0)
counts = np.concatenate([x.counts for x in data_dict["datasets"]], axis = 0)
counts_norm = counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6) * 100
counts_norm = np.log1p(counts_norm)
x_umap = umap_op.fit_transform(counts_norm)

utils.plot_latent(x_umap, annos = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "conditions1.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umap, annos = np.concatenate([x["age"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "conditions2.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "separate", save = result_dir + "celltype_l1.png", figsize = (10, 15), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")


# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import importlib 
importlib.reload(scdisinfact)
#----------------------------------------------------------------------------
# # reference setting
# reg_mmd_comm = 1e-4
# reg_mmd_diff = 1e-4
# reg_gl = 1
# reg_tc = 0.5
# reg_class = 1
# # loss_kl explode, 1e-5 is too large
# reg_kl = 1e-5
# reg_contr = 0.01
# # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
# lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
# Ks = [8, 4, 4]

# batch_size = 64
# # kl term explode when nepochs = 70
# nepochs = 50
# interval = 10
# lr = 5e-4

#----------------------------------------------------------------------------

# argument
reg_mmd_comm = eval(sys.argv[1])
reg_mmd_diff = eval(sys.argv[2])
reg_gl = eval(sys.argv[3])
reg_tc = eval(sys.argv[4])
reg_class = eval(sys.argv[5])
reg_kl = eval(sys.argv[6])
reg_contr = eval(sys.argv[7])
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
nepochs = eval(sys.argv[8])
lr = eval(sys.argv[9])
batch_size = eval(sys.argv[10])
interval = 10
Ks = [8, 4, 4]

#----------------------------------------------------------------------------
model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = reg_contr)
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
model.eval()

# In[] Plot results
z_cs = []
z_ds = []
zs = []

with torch.no_grad():
    for dataset in data_dict["datasets"]:
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds.append([x.cpu().detach().numpy() for x in z_d])
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))


comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l1.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l2"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "joint", axis_label = "UMAP", figsize = (17,7), save = (result_dir + comment+"common_dims_celltypes_l2.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l3"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l3.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "separate", axis_label = "UMAP", figsize = (10,21), save = (result_dir + comment+"common_dims_celltypes_l1_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]

print("#-------------------------------------------------------")
print("#")
print("# Test generalization -- in sample")
print("#")
print("#-------------------------------------------------------")

data_dir = "../data/covid_integrated/"
result_dir = "results_covid/dropout/generalization_is/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


# train test split
np.random.seed(0)
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta_cells in zip(data_dict["datasets"], data_dict["meta_cells"]):
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size], generator = torch.Generator().manual_seed(0))
    
    permute_idx = np.random.permutation(np.arange(len(dataset)))
    train_idx = permute_idx[:int(0.9 * len(dataset))]
    test_idx = permute_idx[int(0.9 * len(dataset)):]
    # datasets_array_train.append(torch.utils.data.Subset(dataset, train_idx))
    # datasets_array_test.append(torch.utils.data.Subset(dataset, test_idx))

    dataset_train = scdisinfact.scdisinfact_dataset(counts = dataset[train_idx]["counts"], 
                                                    counts_norm = dataset[train_idx]["counts_norm"],
                                                    size_factor = dataset[train_idx]["size_factor"],
                                                    diff_labels = dataset[train_idx]["diff_labels"],
                                                    batch_id = dataset[train_idx]["batch_id"],
                                                    mmd_batch_id = dataset[train_idx]["mmd_batch_id"]
                                                    )

    dataset_test = scdisinfact.scdisinfact_dataset(counts = dataset[test_idx]["counts"], 
                                                   counts_norm = dataset[test_idx]["counts_norm"],
                                                   size_factor = dataset[test_idx]["size_factor"],
                                                   diff_labels = dataset[test_idx]["diff_labels"],
                                                   batch_id = dataset[test_idx]["batch_id"],
                                                   mmd_batch_id = dataset[test_idx]["mmd_batch_id"]
                                                   )
    
    datasets_array_train.append(dataset_train)
    datasets_array_test.append(dataset_test)
    meta_cells_array_train.append(meta_cells.iloc[train_idx,:])
    meta_cells_array_test.append(meta_cells.iloc[test_idx,:])

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}

# In[]

model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = reg_contr)
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
model.eval()

# In[] Plot results
LOSS_RECON_TRAIN = 0
LOSS_KL_TRAIN = 0
LOSS_MMD_COMM_TRAIN = 0
LOSS_MMD_DIFF_TRAIN = 0
LOSS_CLASS_TRAIN = 0
LOSS_CONTR_TRAIN = 0
LOSS_TC_TRAIN = 0
LOSS_GL_D_TRAIN = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]

    # load count data
    for x in datasets_array_train:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    idx = np.random.choice(counts_norm.shape[0], 1000, replace = False)
    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm[idx,:].to(model.device), batch_ids = batch_id[idx,:].to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id[idx,:].to(model.device))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor[idx].to(model.device), \
        count = counts[idx,:].to(model.device), batch_id = mmd_batch_id[idx].to(model.device), diff_labels = [x[idx].to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TRAIN += losses[0].item()
    LOSS_KL_TRAIN += losses[1].item()
    LOSS_MMD_COMM_TRAIN += losses[2].item()
    LOSS_MMD_DIFF_TRAIN += losses[3].item()
    LOSS_CLASS_TRAIN += losses[4].item()
    LOSS_CONTR_TRAIN += losses[5].item()
    LOSS_TC_TRAIN += losses[6].item()
    LOSS_GL_D_TRAIN += losses[7].item()

print("Train:")
print(f"LOSS RECON: {LOSS_RECON_TRAIN}")
print(f"LOSS KL: {LOSS_KL_TRAIN}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TRAIN}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TRAIN}")
print(f"LOSS CLASS: {LOSS_CLASS_TRAIN}")
print(f"LOSS CONTR: {LOSS_CONTR_TRAIN}")
print(f"LOSS TC: {LOSS_TC_TRAIN}")
print(f"LOSS GL D: {LOSS_GL_D_TRAIN}")

del counts_norm, batch_id, mmd_batch_id, size_factor, counts, diff_labels, dict_inf, dict_gen, losses

# In[]
z_cs_train = []
z_ds_train = []
zs_train = []

for dataset in datasets_array_train:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds_train.append([x.cpu().detach().numpy() for x in z_d])
        z_cs_train.append(z_c.cpu().detach().numpy())
        zs_train.append(np.concatenate([z_cs_train[-1]] + z_ds_train[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds_train], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds_train], axis = 0)))

comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_celltypes_l1_train.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_batches_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond2_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_celltypes_l1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_cond1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff1_batches_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_celltypes_l1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_cond2_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff2_batches_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# In[] NOTE: Check, even when training together, the loss is very different, potential bug
# NOTE: could be the issue of standard scaler in the preprocessing step between train and test
LOSS_RECON_TEST = 0
LOSS_KL_TEST = 0
LOSS_MMD_COMM_TEST = 0
LOSS_MMD_DIFF_TEST = 0
LOSS_CLASS_TEST = 0
LOSS_CONTR_TEST = 0
LOSS_TC_TEST = 0
LOSS_GL_D_TEST = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]

    # load count data
    for x in datasets_array_test:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    # select subset index
    idx = np.random.choice(counts_norm.shape[0], 1000, replace = False)
    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm[idx,:].to(model.device), batch_ids = batch_id[idx,:].to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id[idx,:].to(model.device))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor[idx].to(model.device), \
        count = counts[idx,:].to(model.device), batch_id = mmd_batch_id[idx].to(model.device), diff_labels = [x[idx].to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TEST += losses[0].item()
    LOSS_KL_TEST += losses[1].item()
    LOSS_MMD_COMM_TEST += losses[2].item()
    LOSS_MMD_DIFF_TEST += losses[3].item()
    LOSS_CLASS_TEST += losses[4].item()
    LOSS_CONTR_TEST += losses[5].item()
    LOSS_TC_TEST += losses[6].item()
    LOSS_GL_D_TEST += losses[7].item()

print("\nTEST:")
print(f"LOSS RECON: {LOSS_RECON_TEST}")
print(f"LOSS KL: {LOSS_KL_TEST}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TEST}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TEST}")
print(f"LOSS CLASS: {LOSS_CLASS_TEST}")
print(f"LOSS CONTR: {LOSS_CONTR_TEST}")
print(f"LOSS TC: {LOSS_TC_TEST}")
print(f"LOSS GL D: {LOSS_GL_D_TEST}")

del counts_norm, batch_id, mmd_batch_id, size_factor, counts, diff_labels, dict_inf, dict_gen, losses

# In[]
z_cs_test = []
z_ds_test = []
zs_test = []

for dataset in datasets_array_test:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds_test.append([x.cpu().detach().numpy() for x in z_d])
        z_cs_test.append(z_c.cpu().detach().numpy())
        zs_test.append(np.concatenate([z_cs_test[-1]] + z_ds_test[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_test, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds_test], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds_test], axis = 0)))

comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_celltypes_l1_test.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff1_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff2_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# In[]
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + z_cs_test))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + z_ds_test], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"

for x in meta_cells_array_test:
    x["mode"] = "test"

meta_cells = pd.concat(meta_cells_array_train + meta_cells_array_test, axis = 0, ignore_index = True)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["predicted.celltype.l1"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"celltype_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells["disease_severity"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"disease_severity_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells["age"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"age_joint.png" if result_dir else None, markerscale = 6, s = 5)


# In[]
print("#-------------------------------------------------------")
print("#")
print("# Test generalization -- out of sample")
print("#")
print("#-------------------------------------------------------")

data_dir = "../data/covid_integrated/"
result_dir = "results_covid/dropout/generalization_oos/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# construct training and testing data
np.random.seed(0)
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta_cells in zip(data_dict["datasets"], data_dict["meta_cells"]):
    # X33, healthy, 40-, wilk_2020
    test_idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] == "lee_2020")).values
    train_idx = ~test_idx

    if np.sum(train_idx) > 0:
        dataset_train = scdisinfact.scdisinfact_dataset(counts = dataset[train_idx]["counts"], 
                                                        counts_norm = dataset[train_idx]["counts_norm"],
                                                        size_factor = dataset[train_idx]["size_factor"],
                                                        diff_labels = dataset[train_idx]["diff_labels"],
                                                        batch_id = dataset[train_idx]["batch_id"],
                                                        mmd_batch_id = dataset[train_idx]["mmd_batch_id"]
                                                        )
        datasets_array_train.append(dataset_train)
        meta_cells_array_train.append(meta_cells.iloc[train_idx,:])
    
    if np.sum(test_idx) > 0:
        dataset_test = scdisinfact.scdisinfact_dataset(counts = dataset[test_idx]["counts"], 
                                                    counts_norm = dataset[test_idx]["counts_norm"],
                                                    size_factor = dataset[test_idx]["size_factor"],
                                                    diff_labels = dataset[test_idx]["diff_labels"],
                                                    batch_id = dataset[test_idx]["batch_id"],
                                                    mmd_batch_id = dataset[test_idx]["mmd_batch_id"]
                                                    )
        datasets_array_test.append(dataset_test)
        meta_cells_array_test.append(meta_cells.iloc[test_idx,:])

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}

# In[]
model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)


model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = reg_contr)
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
model.eval()

# In[] Plot results
LOSS_RECON_TRAIN = 0
LOSS_KL_TRAIN = 0
LOSS_MMD_COMM_TRAIN = 0
LOSS_MMD_DIFF_TRAIN = 0
LOSS_CLASS_TRAIN = 0
LOSS_CONTR_TRAIN = 0
LOSS_TC_TRAIN = 0
LOSS_GL_D_TRAIN = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]

    # load count data
    for x in datasets_array_train:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    idx = np.random.choice(counts_norm.shape[0], 1000, replace = False)
    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm[idx,:].to(model.device), batch_ids = batch_id[idx,:].to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id[idx,:].to(model.device))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor[idx].to(model.device), \
        count = counts[idx,:].to(model.device), batch_id = mmd_batch_id[idx].to(model.device), diff_labels = [x[idx].to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TRAIN += losses[0].item()
    LOSS_KL_TRAIN += losses[1].item()
    LOSS_MMD_COMM_TRAIN += losses[2].item()
    LOSS_MMD_DIFF_TRAIN += losses[3].item()
    LOSS_CLASS_TRAIN += losses[4].item()
    LOSS_CONTR_TRAIN += losses[5].item()
    LOSS_TC_TRAIN += losses[6].item()
    LOSS_GL_D_TRAIN += losses[7].item()

print("Train:")
print(f"LOSS RECON: {LOSS_RECON_TRAIN}")
print(f"LOSS KL: {LOSS_KL_TRAIN}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TRAIN}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TRAIN}")
print(f"LOSS CLASS: {LOSS_CLASS_TRAIN}")
print(f"LOSS CONTR: {LOSS_CONTR_TRAIN}")
print(f"LOSS TC: {LOSS_TC_TRAIN}")
print(f"LOSS GL D: {LOSS_GL_D_TRAIN}")

del counts_norm, batch_id, mmd_batch_id, size_factor, counts, diff_labels, dict_inf, dict_gen, losses

# In[]
z_cs_train = []
z_ds_train = []
zs_train = []

for dataset in datasets_array_train:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds_train.append([x.cpu().detach().numpy() for x in z_d])
        z_cs_train.append(z_c.cpu().detach().numpy())
        zs_train.append(np.concatenate([z_cs_train[-1]] + z_ds_train[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds_train], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds_train], axis = 0)))

comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_celltypes_l1_train.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_batches_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond2_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_celltypes_l1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_cond1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff1_batches_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_celltypes_l1_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_cond2_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_train]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_train]), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff2_batches_train.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# In[] NOTE: Check, even when training together, the loss is very different, potential bug
# NOTE: could be the issue of standard scaler in the preprocessing step between train and test
LOSS_RECON_TEST = 0
LOSS_KL_TEST = 0
LOSS_MMD_COMM_TEST = 0
LOSS_MMD_DIFF_TEST = 0
LOSS_CLASS_TEST = 0
LOSS_CONTR_TEST = 0
LOSS_TC_TEST = 0
LOSS_GL_D_TEST = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]

    # load count data
    for x in datasets_array_test:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)


    # select subset index
    idx = np.random.choice(counts_norm.shape[0], 1000, replace = False)
    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm[idx,:].to(model.device), batch_ids = batch_id[idx,:].to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id[idx,:].to(model.device))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor[idx].to(model.device), \
        count = counts[idx,:].to(model.device), batch_id = mmd_batch_id[idx].to(model.device), diff_labels = [x[idx].to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TEST += losses[0].item()
    LOSS_KL_TEST += losses[1].item()
    LOSS_MMD_COMM_TEST += losses[2].item()
    LOSS_MMD_DIFF_TEST += losses[3].item()
    LOSS_CLASS_TEST += losses[4].item()
    LOSS_CONTR_TEST += losses[5].item()
    LOSS_TC_TEST += losses[6].item()
    LOSS_GL_D_TEST += losses[7].item()

print("\nTEST:")
print(f"LOSS RECON: {LOSS_RECON_TEST}")
print(f"LOSS KL: {LOSS_KL_TEST}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TEST}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TEST}")
print(f"LOSS CLASS: {LOSS_CLASS_TEST}")
print(f"LOSS CONTR: {LOSS_CONTR_TEST}")
print(f"LOSS TC: {LOSS_TC_TEST}")
print(f"LOSS GL D: {LOSS_GL_D_TEST}")

del counts_norm, batch_id, mmd_batch_id, size_factor, counts, diff_labels, dict_inf, dict_gen, losses

# In[] Plot separately does not convey any meaning
z_cs_test = []
z_ds_test = []
zs_test = []

for dataset in datasets_array_test:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds_test.append([x.cpu().detach().numpy() for x in z_d])
        z_cs_test.append(z_c.cpu().detach().numpy())
        zs_test.append(np.concatenate([z_cs_test[-1]] + z_ds_test[-1], axis = 1))

# # UMAP
# umap_op = UMAP(min_dist = 0.1, random_state = 0)
# z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_test, axis = 0))
# z_ds_umap = []
# z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds_test], axis = 0)))
# z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds_test], axis = 0)))

# comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
# if not os.path.exists(result_dir + comment):
#     os.makedirs(result_dir + comment)


# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_celltypes_l1_test.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff1_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array_test]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array_test]), \
#     mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff2_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# In[]
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + z_cs_test))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + z_ds_test], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"

for x in meta_cells_array_test:
    x["mode"] = "test"

meta_cells = pd.concat(meta_cells_array_train + meta_cells_array_test, axis = 0, ignore_index = True)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["predicted.celltype.l1"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"celltype_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells["disease_severity"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"disease_severity_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells["age"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"age_joint.png" if result_dir else None, markerscale = 6, s = 5)

# %%
