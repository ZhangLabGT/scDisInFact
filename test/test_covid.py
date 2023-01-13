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
result_dir = "covid/"
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
age[meta["age"] >= 65] = "65+"
meta["age"] = age

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

# no mmd batches
datasets_array, meta_cells_array, matching_dict = scdisinfact.create_scdisinfact_dataset(counts_array, meta_cells_array, 
                                                                                         condition_key = ["disease_severity", "age"], 
                                                                                         batch_key = "dataset", batch_cond_key = None, meta_genes = genes)

# # In[] Visualize the original count matrix

# umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0)

# x_umap = umap_op.fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
# # separate into batches
# x_umaps = []
# for batch, _ in enumerate(counts_array):
#     if batch == 0:
#         start_pointer = 0
#         end_pointer = start_pointer + counts_array[batch].shape[0]
#         x_umaps.append(x_umap[start_pointer:end_pointer,:])
#     elif batch == (len(counts_array) - 1):
#         start_pointer = start_pointer + counts_array[batch - 1].shape[0]
#         x_umaps.append(x_umap[start_pointer:,:])
#     else:
#         start_pointer = start_pointer + counts_array[batch - 1].shape[0]
#         end_pointer = start_pointer + counts_array[batch].shape[0]
#         x_umaps.append(x_umap[start_pointer:end_pointer,:])

# save_file = None

# utils.plot_latent(x_umaps, annos = [x["dataset"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

# utils.plot_latent(x_umaps, annos = [x["disease_severity"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions1.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

# utils.plot_latent(x_umaps, annos = [x["age"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions2.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

# utils.plot_latent(x_umaps, annos = [x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array], mode = "separate", save = result_dir + "celltype_l1.png", figsize = (10, 15), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")

# utils.plot_latent(x_umaps, annos = [x["predicted.celltype.l2"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype_l2.png", figsize = (15, 7), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")

# utils.plot_latent(x_umaps, annos = [x["predicted.celltype.l3"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype_l3.png", figsize = (11, 7), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")


# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import importlib 
importlib.reload(scdisinfact)
#----------------------------------------------------------------------------
# reference setting
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
# loss_kl explode, 1e-5 is too large
reg_kl = 1e-5
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4, 4]

batch_size = 64
# kl term explode when nepochs = 70
nepochs = 100
interval = 10
lr = 5e-4

#----------------------------------------------------------------------------

# # argument
# reg_mmd_comm = eval(sys.argv[1])
# reg_mmd_diff = eval(sys.argv[2])
# reg_gl = eval(sys.argv[3])
# reg_tc = eval(sys.argv[4])
# reg_class = eval(sys.argv[5])
# reg_kl = eval(sys.argv[6])
# reg_contr = 0.01
# # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
# lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
# nepochs = eval(sys.argv[7])
# lr = eval(sys.argv[8])
# batch_size = eval(sys.argv[9])
# interval = 10
# Ks = [8, 4, 4]

#----------------------------------------------------------------------------

print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))

losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for dataset in datasets_array:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
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


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l1.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l2"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "joint", axis_label = "UMAP", figsize = (17,7), save = (result_dir + comment+"common_dims_celltypes_l2.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l3"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l3.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "separate", axis_label = "UMAP", figsize = (10,21), save = (result_dir + comment+"common_dims_celltypes_l1_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in meta_cells_array]), batches = np.concatenate([x["dataset"].values.squeeze() for x in meta_cells_array]), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]

print("#-------------------------------------------------------")
print("#")
print("# Test generalization -- in sample")
print("#")
print("#-------------------------------------------------------")

data_dir = "../data/covid_integrated/"
result_dir = "covid/generalization_is/"
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
# process age
age = meta.age.values.squeeze().astype(object)
age[meta["age"] < 40] = "40-"
age[(meta["age"] >= 40)&(meta["age"] < 65)] = "40-65"
age[meta["age"] >= 65] = "65+"
meta["age"] = age

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

counts = sparse.vstack(counts_array, format = "csr")
meta_cells = pd.concat(meta_cells_array, axis = 0)

# construct training and testing data
# generate random indices
np.random.seed(0)
permute_idx = np.random.permutation(np.arange(counts.shape[0]))
train_idx = permute_idx[:int(0.9 * counts.shape[0])]
test_idx = permute_idx[int(0.9 * counts.shape[0]):]

# construct count matrix
counts_train = counts[train_idx,:]
counts_test = counts[test_idx,:]
meta_cells_train = meta_cells.iloc[train_idx,:]
meta_cells_test = meta_cells.iloc[test_idx,:]

# no mmd batches
datasets_array_train, meta_cells_array_train, matching_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_cells_train, 
                                                                                                     condition_key = ["disease_severity", "age"], 
                                                                                                     batch_key = "dataset", batch_cond_key = None, meta_genes = genes)

# make sure that the matching_dict matches between train and test, cannot use the wrapped function
# the condition and batch of train must cover the test
cond_ids1 = np.zeros(meta_cells_test.shape[0])
cond_ids2 = np.zeros(meta_cells_test.shape[0])
batch_ids = np.zeros(meta_cells_test.shape[0])
# must match with the training dataset
for idx, batch in enumerate(matching_dict_train["batch_name"]):
    batch_ids[meta_cells_test["dataset"].values.squeeze() == batch] = idx
batch_ids
# condition is only used for loss calculation
for idx, cond in enumerate(matching_dict_train["cond_names"][0]):
    cond_ids1[meta_cells_test["disease_severity"].values.squeeze() == cond] = idx
for idx, cond in enumerate(matching_dict_train["cond_names"][1]):
    cond_ids2[meta_cells_test["age"].values.squeeze() == cond] = idx
# mmd_batch_id is only used for loss calculation
meta_cells_test["batch_cond"] = meta_cells_test[["dataset", "disease_severity", "age"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
batch_cond_ids, batch_cond_names = pd.factorize(meta_cells_test["batch_cond"].values.squeeze(), sort = True)

# meta_cells_test is still the same
# matching_dict matches the training dataset
datasets_test = scdisinfact.scdisinfact_dataset(counts = counts_test.toarray(), anno = None, 
                                            diff_labels = [cond_ids1, cond_ids2], 
                                            batch_id = batch_ids,
                                            mmd_batch_id = batch_cond_ids
                                        )

# In[]
# reference setting
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-5
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4, 4]

batch_size = 64
# kl term explode when nepochs = 70
nepochs = 100
interval = 10
# print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)


losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

# In[] Plot results
LOSS_RECON_TRAIN = 0
LOSS_KL_TRAIN = 0
LOSS_MMD_COMM_TRAIN = 0
LOSS_MMD_DIFF_TRAIN = 0
LOSS_CLASS_TRAIN = 0
LOSS_CONTR_TRAIN = 0
LOSS_TC_TRAIN = 0
LOSS_GL_D_TRAIN = 0

z_cs_train = []
z_ds_train = []
zs_train = []

np.random.seed(0)
for dataset in datasets_array_train:
    with torch.no_grad():
        # whole dataset consumes to much memory, randomly choose 100 cellls
        idx = np.random.choice(dataset.counts_stand.shape[0], 100, replace = False)
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand[idx,:].to(model.device), batch_ids = dataset.batch_id[idx,None].to(model.device), print_stat = True, eval_model = True)
        dict_inf["z_d"] = dict_inf["mu_d"]
        dict_inf["z_c"] = dict_inf["mu_c"]
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = [x for x in dict_inf["mu_d"]], batch_ids = dataset.batch_id[idx,None].to(model.device))
        
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds_train.append([x.cpu().detach().numpy() for x in z_d])
        z_cs_train.append(z_c.cpu().detach().numpy())
        zs_train.append(np.concatenate([z_cs_train[-1]] + z_ds_train[-1], axis = 1))

        # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
        losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = dataset.size_factor[idx].to(model.device), \
            count = dataset.counts[idx,:].to(model.device), batch_id = dataset.mmd_batch_id[idx].to(model.device), diff_labels = [x[idx].to(model.device) for x in dataset.diff_labels], recon_loss = "NB")
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


# In[]
z_cs_train = []
z_ds_train = []
zs_train = []

for dataset in datasets_array_train:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        dict_inf["z_d"] = dict_inf["mu_d"]
        dict_inf["z_c"] = dict_inf["mu_c"]
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

comment = f'figures_{Ks}_{lambs}_{batch_size}/'
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

# In[]
LOSS_RECON_TEST = 0
LOSS_KL_TEST = 0
LOSS_MMD_COMM_TEST = 0
LOSS_MMD_DIFF_TEST = 0
LOSS_CLASS_TEST = 0
LOSS_CONTR_TEST = 0
LOSS_TC_TEST = 0
LOSS_GL_D_TEST = 0

with torch.no_grad():
    # pass through the encoders
    dict_inf = model.inference(counts = datasets_test.counts_stand.to(model.device), batch_ids = datasets_test.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
    dict_inf["z_d"] = dict_inf["mu_d"]
    dict_inf["z_c"] = dict_inf["mu_c"]
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = datasets_test.batch_id[:,None].to(model.device))
    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    
    z_ds_test = [x.cpu().detach().numpy() for x in z_d]
    z_cs_test = z_c.cpu().detach().numpy()
    zs_test = np.concatenate([z_cs_test] + z_ds_test, axis = 1)

    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = datasets_test.size_factor.to(model.device), \
            count = datasets_test.counts.to(model.device), batch_id = datasets_test.mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in datasets_test.diff_labels], recon_loss = "NB")

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

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(z_cs_test)
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(z_ds_test[0]))
z_ds_umap.append(umap_op.fit_transform(z_ds_test[1]))

utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_celltypes_l1_test.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["disease_severity"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["age"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_test["disease_severity"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_test["disease_severity"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff1_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells_test["age"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff2_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + [z_cs_test]))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + [z_ds_test]], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"
meta_cells_test["mode"] = "test"
meta_cells = pd.concat(meta_cells_array_train + [meta_cells_test], axis = 0, ignore_index = True)
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
result_dir = "covid/generalization_oos/"
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
# process age
age = meta.age.values.squeeze().astype(object)
age[meta["age"] < 40] = "40-"
age[(meta["age"] >= 40)&(meta["age"] < 65)] = "40-65"
age[meta["age"] >= 65] = "65+"
meta["age"] = age

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

counts = sparse.vstack(counts_array, format = "csr")
meta_cells = pd.concat(meta_cells_array, axis = 0)

# construct training and testing data
# generate random indices
np.random.seed(0)
# test dataset X32: healthy, 40-
test_idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "40-")).values
train_idx = ~test_idx
# construct count matrix
counts_train = counts[train_idx,:]
counts_test = counts[test_idx,:]
meta_cells_train = meta_cells.iloc[train_idx,:]
meta_cells_test = meta_cells.iloc[test_idx,:]

# no mmd batches
datasets_array_train, meta_cells_array_train, matching_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_cells_train, 
                                                                                                     condition_key = ["disease_severity", "age"], 
                                                                                                     batch_key = "dataset", batch_cond_key = None, meta_genes = genes)

# make sure that the matching_dict matches between train and test, cannot use the wrapped function
# the condition and batch of train must cover the test
cond_ids1 = np.zeros(meta_cells_test.shape[0])
cond_ids2 = np.zeros(meta_cells_test.shape[0])
batch_ids = np.zeros(meta_cells_test.shape[0])
# must match with the training dataset
for idx, batch in enumerate(matching_dict_train["batch_name"]):
    batch_ids[meta_cells_test["dataset"].values.squeeze() == batch] = idx
batch_ids
# condition is only used for loss calculation
for idx, cond in enumerate(matching_dict_train["cond_names"][0]):
    cond_ids1[meta_cells_test["disease_severity"].values.squeeze() == cond] = idx
for idx, cond in enumerate(matching_dict_train["cond_names"][1]):
    cond_ids2[meta_cells_test["age"].values.squeeze() == cond] = idx
# mmd_batch_id is only used for loss calculation
meta_cells_test["batch_cond"] = meta_cells_test[["dataset", "disease_severity", "age"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
batch_cond_ids, batch_cond_names = pd.factorize(meta_cells_test["batch_cond"].values.squeeze(), sort = True)

# meta_cells_test is still the same
# matching_dict matches the training dataset
datasets_test = scdisinfact.scdisinfact_dataset(counts = counts_test.toarray(), anno = None, 
                                            diff_labels = [cond_ids1, cond_ids2], 
                                            batch_id = batch_ids,
                                            mmd_batch_id = batch_cond_ids
                                        )

# In[]
# reference setting
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-5
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4, 4]

batch_size = 64
# kl term explode when nepochs = 70
nepochs = 100
interval = 10
# print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)


losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

# In[] Plot results
LOSS_RECON_TRAIN = 0
LOSS_KL_TRAIN = 0
LOSS_MMD_COMM_TRAIN = 0
LOSS_MMD_DIFF_TRAIN = 0
LOSS_CLASS_TRAIN = 0
LOSS_CONTR_TRAIN = 0
LOSS_TC_TRAIN = 0
LOSS_GL_D_TRAIN = 0

z_cs_train = []
z_ds_train = []
zs_train = []

np.random.seed(0)
for dataset in datasets_array_train:
    with torch.no_grad():
        # whole dataset consumes to much memory, randomly choose 100 cellls
        idx = np.random.choice(dataset.counts_stand.shape[0], 100, replace = False)
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand[idx,:].to(model.device), batch_ids = dataset.batch_id[idx,None].to(model.device), print_stat = True, eval_model = True)
        dict_inf["z_d"] = dict_inf["mu_d"]
        dict_inf["z_c"] = dict_inf["mu_c"]
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = [x for x in dict_inf["mu_d"]], batch_ids = dataset.batch_id[idx,None].to(model.device))
        
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_ds_train.append([x.cpu().detach().numpy() for x in z_d])
        z_cs_train.append(z_c.cpu().detach().numpy())
        zs_train.append(np.concatenate([z_cs_train[-1]] + z_ds_train[-1], axis = 1))

        # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
        losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = dataset.size_factor[idx].to(model.device), \
            count = dataset.counts[idx,:].to(model.device), batch_id = dataset.mmd_batch_id[idx].to(model.device), diff_labels = [x[idx].to(model.device) for x in dataset.diff_labels], recon_loss = "NB")
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


# In[]
z_cs_train = []
z_ds_train = []
zs_train = []

for dataset in datasets_array_train:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        dict_inf["z_d"] = dict_inf["mu_d"]
        dict_inf["z_c"] = dict_inf["mu_c"]
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

comment = f'figures_{Ks}_{lambs}_{batch_size}/'
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

# In[]
LOSS_RECON_TEST = 0
LOSS_KL_TEST = 0
LOSS_MMD_COMM_TEST = 0
LOSS_MMD_DIFF_TEST = 0
LOSS_CLASS_TEST = 0
LOSS_CONTR_TEST = 0
LOSS_TC_TEST = 0
LOSS_GL_D_TEST = 0

with torch.no_grad():
    # pass through the encoders
    dict_inf = model.inference(counts = datasets_test.counts_stand.to(model.device), batch_ids = datasets_test.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
    dict_inf["z_d"] = dict_inf["mu_d"]
    dict_inf["z_c"] = dict_inf["mu_c"]
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = datasets_test.batch_id[:,None].to(model.device))
    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    
    z_ds_test = [x.cpu().detach().numpy() for x in z_d]
    z_cs_test = z_c.cpu().detach().numpy()
    zs_test = np.concatenate([z_cs_test] + z_ds_test, axis = 1)

    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = datasets_test.size_factor.to(model.device), \
            count = datasets_test.counts.to(model.device), batch_id = datasets_test.mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in datasets_test.diff_labels], recon_loss = "NB")

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

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(z_cs_test)
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(z_ds_test[0]))
z_ds_umap.append(umap_op.fit_transform(z_ds_test[1]))

utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_celltypes_l1_test.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["disease_severity"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_test["age"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_test["disease_severity"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff1_cond1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_test["disease_severity"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff1_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_celltypes_l1_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells_test["age"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff2_cond2_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells_test["predicted.celltype.l1"].values.squeeze(), batches = meta_cells_test["dataset"].values.squeeze(), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff2_batches_test.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + [z_cs_test]))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + [z_ds_test]], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"
meta_cells_test["mode"] = "test"
meta_cells = pd.concat(meta_cells_array_train + [meta_cells_test], axis = 0, ignore_index = True)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["predicted.celltype.l1"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"celltype_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells["disease_severity"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"disease_severity_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells["age"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"age_joint.png" if result_dir else None, markerscale = 6, s = 5)

