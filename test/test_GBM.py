# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import scdisinfact
import utils

import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# In[]
data_dir = "../data/GBM_treatment/Fig4/processed/"
result_dir = "GBM_treatment/Fig4_minibatch64/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# NOTE: orig.ident: patient id _ timepoint (should be batches), Patient: patient id, Timepoint: timepoint of sampling, Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cell = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
counts = sp.load_npz(data_dir + "counts_rna.npz")

# condition
treatment_id, treatments = pd.factorize(meta_cell["treatment"].values.squeeze())
# one patient has multiple batches
patient_ids, patient_names = pd.factorize(meta_cell["patient_id"].values.squeeze())
# batches, use samples as batches
sample_ids, sample_names = pd.factorize(meta_cell["sample_id"].values.squeeze())

datasets_array = []
counts_array = []
meta_cells_array = []
for sample_id, sample_name in enumerate(sample_names):
    counts_array.append(counts[sample_ids == sample_id, :].toarray())
    meta_cells_array.append(meta_cell.iloc[sample_ids == sample_id, :])
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [treatment_id[sample_ids == sample_id]], batch_id = patient_ids[sample_ids == sample_id], mmd_batch_id = sample_ids[sample_ids == sample_id]))
    print(len(datasets_array[-1]))
    print(torch.unique(datasets_array[-1].batch_id))
    print(torch.unique(datasets_array[-1].mmd_batch_id))

# # In[]
# umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 

# x_pca = PCA(n_components = 80).fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
# x_umap = umap_op.fit_transform(x_pca)
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

# utils.plot_latent(x_umaps, annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "treatment.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "location.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "gender.png", figsize = (20, 10), axis_label = "Latent", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "mstatus.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

# In[]
import importlib 
importlib.reload(scdisinfact)

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]
nepochs = 200
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size =64, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
# train_joint is more efficient, but does not work as well compared to train
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)


torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth"))

_ = model.eval()

# In[] Plot the loss curve
plt.rcParams["font.size"] = 20
loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests = losses
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

comment = f'plots_{Ks}_{lambs}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umaps, annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"common_patient_id.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"common_treatment.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_location.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_gender.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_mstatus.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umaps[0], annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_patient_id.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_treatment.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_location.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_gender.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_mstatus.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)


# In[]
data_dir = "../data/GBM_treatment/Fig5/processed/"
result_dir = "GBM_treatment/Fig5/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# NOTE: orig.ident: patient id _ timepoint (should be batches), Patient: patient id, Timepoint: timepoint of sampling, Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cell = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
counts = sp.load_npz(data_dir + "counts_rna.npz")

# condition
treatment_id, treatments = pd.factorize(meta_cell["treatment"].values.squeeze())
# batches, use samples as batches
batch_ids, batch_names = pd.factorize(meta_cell["sample_id"].values.squeeze())

datasets_array = []
counts_array = []
meta_cells_array = []
for batch_id, batch_name in enumerate(batch_names):
    counts_array.append(counts[batch_ids == batch_id, :].toarray())
    meta_cells_array.append(meta_cell.iloc[batch_ids == batch_id, :])
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [treatment_id[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))


# # In[]
# umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 

# x_pca = PCA(n_components = 80).fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
# x_umap = umap_op.fit_transform(x_pca)
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

# utils.plot_latent(x_umaps, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "treatment.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "location.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "gender.png", figsize = (20, 10), axis_label = "Latent", markerscale = 6)

# utils.plot_latent(x_umaps, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "mstatus.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

# In[]
import importlib 
importlib.reload(scdisinfact)

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]
nepochs = 200
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 64, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
# train_joint is more efficient, but does not work as well compared to train
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)


torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth"))

_ = model.eval()

# In[] Plot the loss curve
plt.rcParams["font.size"] = 20
loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests = losses
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
for i, j in zip(iters, loss_gl_d_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_recon_tests, "-*", label = 'reconstruction')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_recon_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_class_tests, "-*", label = 'classifier')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_recon_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

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

comment = f'plots_{Ks}_{lambs}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umaps, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"common_patient_id.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"common_treatment.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_location.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_gender.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_mstatus.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umaps[0], annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_patient_id.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_treatment.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_location.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_gender.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_mstatus.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

# %%
