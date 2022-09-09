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

from umap import UMAP
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

from anndata import AnnData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[]
# NOTE: the sepsis dataset include three cohorts. 
# The primary corhort are subjects from Massachusetts General Hospital (MGH) with urinary-tract infection (UTI) admited to Emergency Department (ED).
# Patients are classified as Leuk-UTI (no sepsis, 10 patients), Int-URO (with sepsis, 7 patients), and URO (with sepsis, 10 patients). 
# Two secondary corhorts from Brigham and Women’s Hospital (BWH): 1. bacteremic individuals with sepsis in hospital wards (Bac-SEP).
# 2. individual admitted to the medical intensive care unit (ICU) either with sepsis (ICU-SEP) or without sepsis (ICU-NoSEP)
# for comparison, include healthy control in the end (details check the method part in the paper).
# Two sample types, include CD45+ PBMC (1,000–1,500 cells per subject) and LIN–CD14–HLA-DR+ dendritic cells (300–500 cells per subject).

# # processed dataset
# data_dir = "../data/sepsis/sepsis_batch_processed/"
# result_dir = "sepsis/"
# raw dataset
data_dir = "../data/sepsis/sepsis_batch_raw/split_batches/"
result_dir = "sepsis_raw/2layers/"
# genes = np.loadtxt(data_dir + "genes_5000.txt", dtype = np.object)
genes = np.loadtxt(data_dir + "genes_raw.txt", dtype = np.object)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# read in the dataset
counts_array = []
meta_cell_array = []
for batch_id in range(1, 36):
    if batch_id not in [25, 30]:
        meta_cell = pd.read_csv(data_dir + f"meta_Batch_{batch_id}.csv", index_col = 0)
        # Cell_Type stores the major cell type, Cell_State stores the detailed cell type, 
        # Cohort stores the condition of disease, and biosample_id stores the tissue type, Patient for the patient ID
        meta_cell_array.append(meta_cell)
        counts = sparse.load_npz(data_dir + f"counts_Batch_{batch_id}.npz")
        counts_array.append(counts.toarray())

counts = np.concatenate(counts_array, axis = 0)
adata = AnnData(X = counts)
adata.obs = pd.concat(meta_cell_array, axis = 0)
adata.var.index = genes

# In[] filter the group that we need
# NOTE: we use the primary cohort, including Leuk-UTI, Int-URO, URO, (and control?). The difference between control and Leuk-UTI is the infection of UTI not sepsis.
adata_primary = adata[(adata.obs["Cohort"] == "Leuk-UTI")|(adata.obs["Cohort"] == "Int-URO")|(adata.obs["Cohort"] == "URO"), :]
adata_secondary = adata[(adata.obs["Cohort"] == "Bac-SEP")|(adata.obs["Cohort"] == "ICU-NoSEP")|(adata.obs["Cohort"] == "ICU-SEP"), :]
adata_pbmc = adata_primary[adata_primary.obs["biosample_id"] == "CD45",:]
adata_leuk_uti = adata[adata.obs["Cohort"] == "Leuk-UTI", :]
adata_int_uro = adata[adata.obs["Cohort"] == "Int-URO", :]
adata_uro = adata[adata.obs["Cohort"] == "URO", :]
adata_control = adata[adata.obs["Cohort"] == "Control", :]
adata_bac_sep = adata[adata.obs["Cohort"] == "Bac-SEP", :]
adata_icu_nosep = adata[adata.obs["Cohort"] == "ICU-NoSEP", :]
adata_icu_sep = adata[adata.obs["Cohort"] == "ICU-SEP", :]


# NOTE: Two ways of separating condition in primary cohort
# 1. one condition for UTI, where control -> no UTI, and Leuk-UTI, Int-URO, and URO -> UTI; another condition for sepsis, where Control, Leuk-UTI -> no sepsis, Int-URO, URO
# 2. Another way is to use Leuk-UTI, Int-URO, and URO (control? with no UTI)

# the second way
batch_ids, batch_names = pd.factorize(adata_secondary.obs["Batches"].values.squeeze())
severity_ids, severity_names = pd.factorize(adata_secondary.obs["Cohort"].values.squeeze())

counts_array = []
meta_cells_array = []
datasets_array = []
for batch_id, batch_name in enumerate(batch_names):
    adata_batch = adata_secondary[batch_ids == batch_id, :]
    counts_array.append(adata_batch.X.toarray())
    meta_cells_array.append(adata_batch.obs)
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [severity_ids[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))

    print(len(datasets_array[-1]))
# In[]
'''
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

utils.plot_latent(x_umaps, annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions1.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6, label_inplace = True, text_size = "small")

utils.plot_latent(x_umaps, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6, label_inplace = True, text_size = "small")

# utils.plot_latent(x_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "separate", save = result_dir + "separate.png", figsize = (10, 70), axis_label = "Latent", markerscale = 6)
'''

# In[]
import importlib 
importlib.reload(scdisinfact)
# reference setting
# reg_mmd_comm = 1e-4
# reg_mmd_diff = 1e-4
# reg_gl = 1
# reg_tc = 0.5
# reg_class = 1
# reg_kl = 1e-6
# # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
# lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
# Ks = [8, 4]

# argument
reg_mmd_comm = eval(sys.argv[1])
reg_mmd_diff = eval(sys.argv[2])
reg_gl = eval(sys.argv[3])
reg_tc = eval(sys.argv[4])
reg_class = eval(sys.argv[5])
reg_kl = eval(sys.argv[6])
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]

nepochs = 300
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 8, interval = interval, lr = 5e-4, 
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


# utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "separate", axis_label = "UMAP", figsize = (10,140), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15,10), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = True, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15,10), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = True, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15, 10), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)



# %%
