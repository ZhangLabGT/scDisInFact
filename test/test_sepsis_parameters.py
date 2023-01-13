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
# Load the dataset, treat each sample as a batch
#
# NOTE: the sepsis dataset include three cohorts. 
# The primary corhort are subjects from Massachusetts General Hospital (MGH) with urinary-tract infection (UTI) admited to Emergency Department (ED).
# Patients are classified as Leuk-UTI (no sepsis, 10 patients), Int-URO (with sepsis, 7 patients), and URO (with sepsis, 10 patients). 
# Two secondary corhorts from Brigham and Women’s Hospital (BWH): 1. bacteremic individuals with sepsis in hospital wards (Bac-SEP).
# 2. individual admitted to the medical intensive care unit (ICU) either with sepsis (ICU-SEP) or without sepsis (ICU-NoSEP)
# for comparison, include healthy control in the end (details check the method part in the paper).
# Two sample types, include CD45+ PBMC (1,000–1,500 cells per subject) and LIN–CD14–HLA-DR+ dendritic cells (300–500 cells per subject).
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # processed dataset
# data_dir = "../data/sepsis/sepsis_batch_processed/"
# result_dir = "sepsis/"
# raw dataset
data_dir = "../data/sepsis/sepsis_batch_raw/split_batches/"
result_dir = "sepsis_raw/secondary/batch_batches/"
# genes = np.loadtxt(data_dir + "genes_5000.txt", dtype = np.object)
genes = np.loadtxt(data_dir + "genes_raw.txt", dtype = np.object)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# read in the dataset
counts_array = []
meta_cell_array = []
for batch_id in range(1, 36):
    # not batches correspond to id 25 and 30
    if batch_id not in [25, 30]:
        # one patient can be in multiple batches, for both primary and secondary
        meta_cell = pd.read_csv(data_dir + f"meta_Batch_{batch_id}.csv", index_col = 0)
        # one processing batch have multiple conditions.
        # print(np.unique(meta_cell["Cohort"].values.squeeze()))
        
        # Cell_Type stores the major cell type, Cell_State stores the detailed cell type, 
        # Cohort stores the condition of disease, and biosample_id stores the tissue type, Patient for the patient ID
        meta_cell_array.append(meta_cell)
        counts = sparse.load_npz(data_dir + f"counts_Batch_{batch_id}.npz")
        counts_array.append(counts.toarray())

counts = np.concatenate(counts_array, axis = 0)
adata = AnnData(X = counts)
adata.obs = pd.concat(meta_cell_array, axis = 0)
adata.var.index = genes

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
adata = adata_secondary

# processing batches
batch_ids, batch_names = pd.factorize(adata.obs["Batches"].values.squeeze())
# patients
# batch_ids, batch_names = pd.factorize(adata.obs["donor_id"].values.squeeze())
severity_ids, severity_names = pd.factorize(adata.obs["Cohort"].values.squeeze())

counts_array = []
meta_cells_array = []
datasets_array = []
for batch_id, batch_name in enumerate(batch_names):
    adata_batch = adata[batch_ids == batch_id, :]
    counts_array.append(adata_batch.X.toarray())
    meta_cells_array.append(adata_batch.obs)
    datasets_array.append(scdisinfact.scdisinfact_dataset(counts = counts_array[-1], anno = None, 
                                              diff_labels = [severity_ids[batch_ids == batch_id]], 
                                              batch_id = batch_ids[batch_ids == batch_id]
                                              ))

    print(f"current batch: {batch_id}")
    print(f"number of cells: {len(datasets_array[-1])}")
    print(f"unique conditions: {np.unique(severity_ids[batch_ids == batch_id])}")

# # In[] Check the batch effect: 1. among batches; 2. among patients
# from sklearn.decomposition import PCA
# adata_curr = adata.copy()
# conditions = np.unique(adata_curr.obs["Cohort"].values)
# silhouette_batches = []
# silhouette_patients = []
# for condition in conditions:
#     adata_cond = adata_curr[adata_curr.obs["Cohort"] == condition,:]
#     counts_cond = adata_cond.X
#     label2_cond = adata_cond.obs["Cell_State"].values.squeeze()
#     label1_cond = adata_cond.obs["Cell_Type"].values.squeeze()
#     batch_cond = adata_cond.obs["Batches"].values.squeeze()
#     patient_cond = adata_cond.obs["donor_id"].values.squeeze()
#     # normalize
#     counts_cond = counts_cond/(np.sum(counts_cond, axis = 1, keepdims = True) + 1e-6) * 100
#     counts_cond = np.log1p(counts_cond)
#     # pca
#     pca_cond = PCA(n_components = 50).fit_transform(counts_cond)

#     # # using label 1
#     # silhouette_batches = bmk.silhouette_batch(X = pca_cond, batch_gt = batch_cond, group_gt = label1_cond, verbose = False)
#     # silhouette_patients = bmk.silhouette_batch(X = pca_cond, batch_gt = patient_cond, group_gt = label1_cond, verbose = False)
#     # print("scores with patient as batches {:.4f}".format(silhouette_patients))
#     # print("scores with batch id as batches {:.4f}".format(silhouette_batches))
#     # # using label 2
#     # silhouette_batches = bmk.silhouette_batch(X = pca_cond, batch_gt = batch_cond, group_gt = label2_cond, verbose = False)
#     # silhouette_patients = bmk.silhouette_batch(X = pca_cond, batch_gt = patient_cond, group_gt = label2_cond, verbose = False)
#     # print("scores with patient as batches {:.4f}".format(silhouette_patients))
#     # print("scores with batch id as batches {:.4f}".format(silhouette_batches))

#     # joint
#     silhouette_batches.append(bmk.silhouette_batch(X = pca_cond, batch_gt = batch_cond, group_gt = np.array(["0"] * batch_cond.shape[0]), verbose = False))
#     silhouette_patients.append(bmk.silhouette_batch(X = pca_cond, batch_gt = patient_cond, group_gt = np.array(["0"] * batch_cond.shape[0]), verbose = False))
# print("scores with patient as batches {:.4f}".format(np.mean(np.array(silhouette_patients))))
# print("scores with batch id as batches {:.4f}".format(np.mean(np.array(silhouette_batches))))


# In[] Visualize the original count matrix

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

utils.plot_latent(x_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions1.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype.png", figsize = (10, 7), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")

utils.plot_latent(x_umaps, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype_sub.png", figsize = (11, 7), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")

# utils.plot_latent(x_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "separate", save = result_dir + "separate.png", figsize = (10, 70), axis_label = "UMAP", markerscale = 6)


# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
batch_size = 64

nepochs = 1000
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = batch_size, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_{batch_size}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}.pth", map_location = device))

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

comment = f'plots_{Ks}_{lambs}_{batch_size}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


# utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "separate", axis_label = "UMAP", figsize = (10,140), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)

utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cellstates.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cohorts.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims_celltypes.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims_condition.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[] 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Test prediction accuracy
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pred_conds = [np.where(severity_names == "ICU-NoSEP")[0][0]]
X_scdisinfact_impute = []
ref_batch = 0
z_ds_preds = []
for batch_id, dataset in enumerate(datasets_array):
    X = model.predict_counts(predict_dataset = dataset, predict_conds = pred_conds, predict_batch = ref_batch)
    X_scdisinfact_impute.append(X.detach().cpu().numpy())
X_scdisinfact_impute = np.concatenate(X_scdisinfact_impute)

X_scdisinfact_impute_norm = X_scdisinfact_impute/(np.sum(X_scdisinfact_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scdisinfact_impute_norm = np.log1p(X_scdisinfact_impute_norm)
x_pca_scdisinfact = PCA(n_components = 10).fit_transform(X_scdisinfact_impute_norm)

# plot umap
umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 
x_umap_scdisinfact = umap_op.fit_transform(x_pca_scdisinfact)
# separate into batches
x_umaps_scdisinfact = []
for batch, _ in enumerate(datasets_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + len(datasets_array[batch])
        x_umaps_scdisinfact.append(x_umap_scdisinfact[start_pointer:end_pointer,:])

    elif batch == (len(datasets_array) - 1):
        start_pointer = start_pointer + len(datasets_array[batch-1])
        x_umaps_scdisinfact.append(x_umap_scdisinfact[start_pointer:,:])

    else:
        start_pointer = start_pointer + len(datasets_array[batch-1])
        end_pointer = start_pointer + len(datasets_array[batch])
        x_umaps_scdisinfact.append(x_umap_scdisinfact[start_pointer:end_pointer,:])

utils.plot_latent(zs = x_umaps_scdisinfact, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"predict_celltype.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = x_umaps_scdisinfact, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"predict_cellstate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = x_umaps_scdisinfact, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"predict_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = x_umaps_scdisinfact, annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"predict_cohort.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# In[]

x_pca_scdisinfact = PCA(n_components = 10).fit_transform(X_scdisinfact_impute_norm)

n_neighbors = 100
# graph connectivity score, check the batch mixing of the imputated gene expression data
gc_scdisinfact = bmk.graph_connectivity(X = x_pca_scdisinfact, \
    groups = np.concatenate([x["Cell_Type"].values.squeeze() for x in meta_cells_array], axis = 0), k = n_neighbors)
print('GC (scDisInFact): {:.3f}'.format(gc_scdisinfact))

# silhouette (batch)
silhouette_batch_scdisinfact = bmk.silhouette_batch(X = x_pca_scdisinfact, \
    batch_gt = np.concatenate([x["Batches"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["Cell_Type"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))

# silhouette (condition)
silhouette_condition_scdisinfact = bmk.silhouette_batch(X = x_pca_scdisinfact, \
    batch_gt = np.concatenate([x["Batches"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["Cell_Type"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette condition (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact))


# ARI score, check the separation of cell types? still needed?
nmi_scdisinfact = []
ari_scdisinfact = []

nmi_scgen = []
ari_scgen = []

nmi_scpregan = []
ari_scpregan = []

for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scdisinfact = utils.leiden_cluster(X = x_pca_scdisinfact, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["Cell_Type"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scdisinfact))
    ari_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["Cell_Type"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scdisinfact))

print('NMI (scDisInFact): {:.3f}'.format(max(nmi_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_scdisinfact)))
