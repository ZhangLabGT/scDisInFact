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
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [severity_ids[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))

    print(len(datasets_array[-1]))

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

utils.plot_latent(x_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "conditions1.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype.png", figsize = (10, 7), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")

utils.plot_latent(x_umaps, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltype_sub.png", figsize = (11, 7), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")

# utils.plot_latent(x_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "separate", save = result_dir + "separate.png", figsize = (10, 70), axis_label = "UMAP", markerscale = 6)


# In[]
import importlib 
importlib.reload(scdisinfact)
# reference setting
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]

# argument
# reg_mmd_comm = eval(sys.argv[1])
# reg_mmd_diff = eval(sys.argv[2])
# reg_gl = eval(sys.argv[3])
# reg_tc = eval(sys.argv[4])
# reg_class = eval(sys.argv[5])
# reg_kl = eval(sys.argv[6])
# # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
# lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
# Ks = [8, 4]
batch_size = 8

nepochs = 1000
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = batch_size, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
# train_joint is more efficient, but does not work as well compared to train
model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

_ = model.eval()

# torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_{batch_size}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}.pth", map_location = device))

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
for i, j in zip(iters, loss_class_tests):
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

comment = f'plots_{Ks}_{lambs}_{batch_size}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


# utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "separate", axis_label = "UMAP", figsize = (10,140), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)

utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = True, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["Cell_State"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cellstates.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = True, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cell_Type"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims_celltypes.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Cohort"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims_condition.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["Batches"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[] Test imputation accuracy

counts_joint = []
batch_ids = []
severity_labels = []
batch_labels = []

for dataset in datasets_array:
    counts_joint.append(dataset.counts_stand)
    batch_ids.append(dataset.batch_id[:,None])
    severity_labels.append(dataset.diff_labels[0].numpy())
    batch_labels.append(dataset.batch_id.numpy())

counts_joint = torch.cat(counts_joint, dim = 0)
batch_ids = torch.cat(batch_ids, dim = 0)
severity_labels = np.concatenate(severity_labels, axis = 0)
batch_labels = np.concatenate(batch_labels, axis = 0)

with torch.no_grad():
    # pass through the encoders
    dict_inf = model.inference(counts = counts_joint.to(model.device), batch_ids = batch_ids.to(model.device), print_stat = True, eval_model = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = batch_ids.to(model.device))
    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    

mean_bacsep = torch.mean(z_d[0][severity_names[severity_labels] == "Bac-SEP"], dim = 0)[None,:]
# stim1, batch 2 and 3
mean_icusep = torch.mean(z_d[0][severity_names[severity_labels] == "ICU-SEP"], dim = 0)[None,:]
# stim2, batch 4 and 5
mean_icuctrl = torch.mean(z_d[0][severity_names[severity_labels] == "ICU-NoSEP"], dim = 0)[None,:]


delta_bacsep_icuctrl = mean_icuctrl - mean_bacsep
change_icusep_icuctrl = mean_icuctrl - mean_icusep
change_icuctrl_icuctrl = mean_icuctrl - mean_icuctrl


with torch.no_grad():
    #-------------------------------------------------------------------------------------------------------------------
    #
    # Remove both batch effect and condition effect (Imputation of count matrices under the control condition)
    #
    #-------------------------------------------------------------------------------------------------------------------
    
    # removed of batch effect
    ref_batch = 0.0
    # still use the original batch_id as input, not change the latent embedding
    z_c, _ = model.Enc_c(counts_joint.to(model.device), batch_ids.to(model.device))
    z_d = []
    for Enc_d in model.Enc_ds:
        # still use the original batch_id as input, not change the latent embedding
        _z_d, _ = Enc_d(counts_joint.to(model.device), batch_ids.to(model.device))
        _z_d[severity_labels == "Bac-SEP"] = _z_d[severity_labels == "Bac-SEP"] + delta_bacsep_icuctrl
        _z_d[severity_labels == "ICU-SEP"] = _z_d[severity_labels == "ICU-SEP"] + change_icusep_icuctrl
        _z_d[severity_labels == "ICU-NoSEP"] = _z_d[severity_labels == "ICU-NoSEP"] + change_icuctrl_icuctrl
        z_d.append(_z_d)        

    # change the batch_id into ref batch as input
    z = torch.concat([z_c] + z_d + [torch.tensor([ref_batch] * counts_joint.shape[0], device = model.device)[:,None]], axis = 1)        

    # NOTE: change the batch_id into ref batch as input, change the diff condition into control
    mu_impute, _, _ = model.Dec(torch.concat([z_c] + z_d, dim = 1), torch.tensor([ref_batch] * counts_joint.shape[0], device = model.device)[:,None])
    X_scdisinfact_impute = mu_impute.cpu().detach().numpy() 

# In[] Key gene discovery
model_params = torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}.pth")
# last 21 dimensions are batch ids
inf = np.array(model_params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.))[:genes.shape[0]]
sorted_genes = genes[np.argsort(inf)[::-1]]
key_genes_score = pd.DataFrame(columns = ["genes", "score"])
key_genes_score["genes"] = sorted_genes
key_genes_score["score"] = inf[np.argsort(inf)[::-1]]
key_genes_score.to_csv(result_dir + "key_genes.csv")
key_genes_score.iloc[0:1000,0:1].to_csv(result_dir + "key_gene_list.txt", index = False, header = False)


# In[]
plt.rcParams["font.size"] = 20
key_genes_score = pd.read_csv(result_dir + "key_genes.csv", index_col = 0)

fig, ax = plt.subplots(figsize=(10, 5))
# Show each distribution with both violins and points
sns.violinplot(x="score",data= key_genes_score, palette="Set3", inner="points")
sns.despine(left=True)
fig.suptitle('Scores of genes', fontsize=18, fontweight='bold')
ax.set_xlabel("scores",size = 16,alpha=0.7)


fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(x="score",data = key_genes_score, palette="Set3")
# sns.despine(left=True)
# sns.stripplot(x="score", data = key_genes_score, color="gray", edgecolor="gray", size = 2, alpha = 0.7)
# genes that differentially expressed in MS1 cells from ICU-SEP versus ICU-NoSEP: PLAC8, CLU, CD52, TMEM176B, TMEM176A
key_genes1 = key_genes_score[key_genes_score["genes"].isin(["PLAC8", "CLU", "CD52", "TMEM176B", "TMEM176A"])]
sns.stripplot(x="score", data = key_genes1, color="red", edgecolor="gray", size = 6)
fig.savefig(result_dir + "marker_gene_violin.png", bbox_inches = "tight")


# %%
