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
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings
import time
warnings.filterwarnings("ignore")
import seaborn as sns

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_dir = "../data/GBM_treatment/Fig4/processed/"
result_dir = "GBM_treatment/Fig4_patient_minibatch8/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# orig.ident: patient id _ timepoint (should be batches), 
# Patient: patient id, 
# Timepoint: timepoint of sampling, 
# Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cell = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
meta_cell_seurat = pd.read_csv(data_dir + "meta_cells_seurat.csv", sep = "\t", index_col = 0)
meta_cell["mstatus"] = meta_cell_seurat["mstatus"].values.squeeze()
# meta_cell.loc[(meta_cell["mstatus"] != "Myeloid") & ((meta_cell["mstatus"] != "Oligodendrocytes") & (meta_cell["mstatus"] != "tumor")), "mstatus"] = "Other"
counts = sp.load_npz(data_dir + "counts_rna.npz")

# condition
treatment_id, treatments = pd.factorize(meta_cell["treatment"].values.squeeze())
# batches: 1. consider patient as batch, 2. consider sample as batch
patient_ids, patient_names = pd.factorize(meta_cell["patient_id"].values.squeeze())
sample_ids, sample_names = pd.factorize(meta_cell["sample_id"].values.squeeze())

datasets_array = []
counts_array = []
meta_cells_array = []
for sample_id, sample_name in enumerate(sample_names):
    counts_array.append(counts[sample_ids == sample_id, :].toarray())
    meta_cells_array.append(meta_cell.iloc[sample_ids == sample_id, :])
    datasets_array.append(scdisinfact.scdisinfact_dataset(counts = counts_array[-1], anno = None, 
                                              diff_labels = [treatment_id[sample_ids == sample_id]], 
                                              # use sample_ids instead of patient_ids
                                              batch_id = patient_ids[sample_ids == sample_id],
                                              # batch_id = sample_ids[sample_ids == sample_id], 
                                              mmd_batch_id = sample_ids[sample_ids == sample_id]
                                              ))
    
    print(len(datasets_array[-1]))
    print(torch.unique(datasets_array[-1].batch_id))
    print(torch.unique(datasets_array[-1].mmd_batch_id))


datasets_array, meta_cells_array = scdisinfact.create_scdisinfact_dataset(counts, meta_cell, condition_key = ["treatment"], batch_key = "patient_id", batch_cond_key = "sample_id", meta_genes = genes)

# In[] Visualize the original count matrix

umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 

x_pca = PCA(n_components = 80).fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
x_umap = umap_op.fit_transform(x_pca)
# separate into batches
x_umaps = []
for batch, _ in enumerate(meta_cells_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (len(meta_cells_array) - 1):
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

save_file = None

utils.plot_latent(x_umaps, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "treatment.png", figsize = (11,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "location.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "gender.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "mstatus.png", figsize = (11,7), axis_label = "UMAP", markerscale = 6, s = 2)

# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
start_time = time.time()
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 8, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)
_ = model.eval()

torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth", map_location = device))
end_time = time.time()
print("time used: {:.2f}".format(end_time - start_time))


# # In[] Plot the loss curve
# plt.rcParams["font.size"] = 20
# loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests = losses
# iters = np.arange(1, len(loss_tests)+1)

# fig = plt.figure(figsize = (40, 10))
# ax = fig.add_subplot()
# ax.plot(iters, loss_tests, "-*", label = 'Total loss')
# ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
# ax.set_yscale('log')
# for i, j in zip(iters, loss_tests):
#     ax.annotate("{:.3f}".format(j),xy=(i,j))

# fig.savefig(result_dir + "total_loss.png", bbox_inches = "tight")
# fig = plt.figure(figsize = (40, 10))
# ax = fig.add_subplot()
# ax.plot(iters, loss_gl_d_tests, "-*", label = 'Group Lasso diff')
# ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
# ax.set_yscale('log')
# for i, j in zip(iters, loss_tests):
#     ax.annotate("{:.3f}".format(j),xy=(i,j))

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


utils.plot_latent(zs = z_cs_umaps, annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_patient_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_sample_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_treatment.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_location.png".format()) if result_dir else None, markerscale = 9, s = 3, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_gender.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_mstatus.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umaps[0], annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_patient_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_sample_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_treatment.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_location.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_gender.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_mstatus.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[] 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Key gene detection
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

model_params = torch.load(result_dir + f"model_{Ks}_{lambs}.pth")
# last 21 dimensions are batch ids
inf = model.extract_gene_scores()[0]
sorted_genes = genes[np.argsort(inf)[::-1]]
key_genes_score = pd.DataFrame(columns = ["genes", "score"])
key_genes_score["genes"] = [x.split("_")[1] for x in sorted_genes]
key_genes_score["score"] = inf[np.argsort(inf)[::-1]]
key_genes_score.to_csv(result_dir + "key_genes.csv")
key_genes_score.iloc[0:300,0:1].to_csv(result_dir + "key_gene_list.txt", index = False, header = False)

# Plot key genes
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
# metallothioneins and neuron marker genes: H1FX, MT1X, MT2A, MT1E, MT1H, MT1G, SNAP25, RAB3A, NRGN, NXPH4, SLC17A7
key_genes1 = key_genes_score[key_genes_score["genes"].isin(["H1FX", "MT1X", "MT2A", "MT1E", "MT1H", "MT1G", "SNAP25", "RAB3A", "NRGN", "NXPH4", "SLC17A7"])]
sns.stripplot(x="score", data = key_genes1, color="red", edgecolor="gray", size = 6)
# Myeloid-relevent genes
key_genes2 = key_genes_score[key_genes_score["genes"].isin(["CD68", "CTSD", "CTSB", "CD163", "CYBB", "CCR5", "CTSZ", "SAMHD1", "PLA2G7", "SIGLEC1", "LILRB3", "CCR1", "APOBR"])]
sns.stripplot(x="score", data = key_genes2, color="blue", edgecolor="gray", size = 6)
# key_genes3 = key_genes_score[key_genes_score["genes"].isin(["LEFTY1", "BEX5", "SAXO2", "KCNB1"])]
# sns.stripplot(x="score", data = key_genes3, color="green", edgecolor="gray", size = 3)
fig.savefig(result_dir + "marker_gene_violin.png", bbox_inches = "tight")


# In[] 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Test prediction accuracy
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# z_cs = []
# z_ds = []
# zs = []
# treatment_labels = []
# X_scdisinfact_impute = []
# with torch.no_grad():
#     for batch_id, dataset in enumerate(datasets_array):
#         treatment_labels.append(treatments[dataset.diff_labels[0].numpy()])
#         # pass through the encoders
#         dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
#         # pass through the decoder
#         dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
#         z_c = dict_inf["mu_c"]
#         z_d = dict_inf["mu_d"]
#         z = torch.cat([z_c] + z_d, dim = 1)
#         mu = dict_gen["mu"]
#         z_cs.append(z_c.cpu().detach().numpy())
#         zs.append(z.cpu().detach().numpy())
#         z_ds.append([x.cpu().detach().numpy() for x in z_d])   

# treatment_labels = np.concatenate(treatment_labels, axis = 0)
# z_ds = np.concatenate([x[0] for x in z_ds], axis = 0)

# mean_ctrl = np.mean(z_ds[treatment_labels == "vehicle (DMSO)"], axis = 0)[None,:]
# mean_stim = np.mean(z_ds[treatment_labels == "0.2 uM panobinostat"], axis = 0)[None,:]
# delta_stim_ctrl = mean_ctrl - mean_stim
# delta_ctrl_ctrl = mean_ctrl - mean_ctrl

# with torch.no_grad():
#     # removed of batch effect
#     ref_batch = 0.0
#     for batch_id, dataset in enumerate(datasets_array):
#         # still use the original batch_id as input, not change the latent embedding
#         z_c, _ = model.Enc_c(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
#         z_d = []
#         for Enc_d in model.Enc_ds:
#             # still use the original batch_id as input, not change the latent embedding
#             _z_d, _ = Enc_d(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
#             # control: 0, stimulation: 1
#             _z_d[dataset.diff_labels[0] == 0] = _z_d[dataset.diff_labels[0] == 0] + torch.from_numpy(delta_ctrl_ctrl).to(model.device)
#             _z_d[dataset.diff_labels[0] == 1] = _z_d[dataset.diff_labels[0] == 1] + torch.from_numpy(delta_stim_ctrl).to(model.device)
#             z_d.append(_z_d)        

#         # change the batch_id into ref batch as input
#         z = torch.concat([z_c] + z_d + [torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None]], axis = 1)        

#         # NOTE: change the batch_id into ref batch as input, change the diff condition into control
#         mu_impute, _, _ = model.Dec(torch.concat([z_c] + z_d, dim = 1), torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None])
#         X_scdisinfact_impute.append(mu_impute.cpu().detach().numpy())

# X_scdisinfact_impute = np.concatenate(X_scdisinfact_impute)

# checked, produce the same result as above
pred_conds = [np.where(treatments == "vehicle (DMSO)")[0][0]]
# pred_conds = 0
X_scdisinfact_impute = []

for batch_id, dataset in enumerate(datasets_array):
    X = model.predict_counts(predict_dataset = dataset, predict_conds = pred_conds, predict_batch = 0)
    X_scdisinfact_impute.append(X.detach().cpu().numpy())

X_scdisinfact_impute = np.concatenate(X_scdisinfact_impute)


# In[] Calculate batch mixing score
# normalization
X_scdisinfact_impute_norm = X_scdisinfact_impute/(np.sum(X_scdisinfact_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scdisinfact_impute_norm = np.log1p(X_scdisinfact_impute_norm)
x_pca_scdisinfact = PCA(n_components = 10).fit_transform(X_scdisinfact_impute_norm)

# plot umap
umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 
x_umap_scdisinfact = umap_op.fit_transform(x_pca_scdisinfact)
# separate into batches
x_umaps_scdisinfact = []
for batch, _ in enumerate(meta_cells_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps_scdisinfact.append(x_umap_scdisinfact[start_pointer:end_pointer,:])
    elif batch == (len(meta_cells_array) - 1):
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        x_umaps_scdisinfact.append(x_umap_scdisinfact[start_pointer:,:])
    else:
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps_scdisinfact.append(x_umap_scdisinfact[start_pointer:end_pointer,:])

utils.plot_latent(x_umaps_scdisinfact, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + comment + "predict_batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps_scdisinfact, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + comment + "predict_treatment.png", figsize = (11,7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps_scdisinfact, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + comment + "predict_location.png", figsize = (12, 7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps_scdisinfact, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + comment + "predict_gender.png", figsize = (10, 7), axis_label = "UMAP", markerscale = 6, s = 2)

utils.plot_latent(x_umaps_scdisinfact, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + comment + "predict_mstatus.png", figsize = (11,7), axis_label = "UMAP", markerscale = 6, s = 2)

# In[]
# normalization for scGEN
X_scgen_impute = sp.load_npz("GBM_treatment/Fig4_minibatch8/scGEN/counts_scgen.npz").toarray()
X_scgen_impute = (X_scgen_impute >= 0) * X_scgen_impute
X_scgen_impute_norm = X_scgen_impute/(np.sum(X_scgen_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scgen_impute_norm = np.log1p(X_scgen_impute_norm)
x_pca_scgen = PCA(n_components = 10).fit_transform(X_scgen_impute_norm) 

# plot umap
umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 
x_umap_scgen = umap_op.fit_transform(x_pca_scgen)
# separate into batches
x_umaps_scgen = []
for batch, _ in enumerate(meta_cells_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps_scgen.append(x_umap_scgen[start_pointer:end_pointer,:])
    elif batch == (len(meta_cells_array) - 1):
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        x_umaps_scgen.append(x_umap_scgen[start_pointer:,:])
    else:
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps_scgen.append(x_umap_scgen[start_pointer:end_pointer,:])

utils.plot_latent(x_umaps_scgen, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scGEN/predict_batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_scgen, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scGEN/predict_treatment.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_scgen, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scGEN/predict_location.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps_scgen, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scGEN/predict_gender.png", figsize = (20, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps_scgen, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scGEN/predict_mstatus.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

# In[]
# normalization for scPreGAN
X_scpregan_impute = sp.load_npz("GBM_treatment/Fig4_minibatch8/scPreGAN/counts_scpregan.npz").toarray()
X_scpregan_impute = (X_scpregan_impute >= 0) * X_scpregan_impute
X_scpregan_impute_norm = X_scpregan_impute/(np.sum(X_scpregan_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scpregan_impute_norm = np.log1p(X_scpregan_impute_norm)
x_pca_scpregan = PCA(n_components = 10).fit_transform(X_scpregan_impute_norm) 

# plot umap
umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 
x_umap_scpregan = umap_op.fit_transform(x_pca_scpregan)
# separate into batches
x_umaps_scpregan = []
for batch, _ in enumerate(meta_cells_array):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps_scpregan.append(x_umap_scpregan[start_pointer:end_pointer,:])
    elif batch == (len(meta_cells_array) - 1):
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        x_umaps_scpregan.append(x_umap_scpregan[start_pointer:,:])
    else:
        start_pointer = start_pointer + meta_cells_array[batch - 1].shape[0]
        end_pointer = start_pointer + meta_cells_array[batch].shape[0]
        x_umaps_scpregan.append(x_umap_scpregan[start_pointer:end_pointer,:])

utils.plot_latent(x_umaps_scpregan, annos = [x["sample_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scPreGAN/predict_batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_scpregan, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scPreGAN/predict_treatment.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps_scpregan, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scPreGAN/predict_location.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps_scpregan, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scPreGAN/predict_gender.png", figsize = (20, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps_scpregan, annos = [x["mstatus"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "scPreGAN/predict_mstatus.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

# In[]

x_pca_scgen = PCA(n_components = 10).fit_transform(X_scgen_impute_norm) 
x_pca_scdisinfact = PCA(n_components = 10).fit_transform(X_scdisinfact_impute_norm)
x_pca_scpregan = PCA(n_components = 10).fit_transform(X_scpregan_impute_norm) 

n_neighbors = 100
# graph connectivity score, check the batch mixing of the imputated gene expression data
gc_scdisinfact = bmk.graph_connectivity(X = x_pca_scdisinfact, \
    groups = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), k = n_neighbors)
print('GC (scDisInFact): {:.3f}'.format(gc_scdisinfact))

gc_scgen = bmk.graph_connectivity(X = x_pca_scgen, \
    groups = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), k = n_neighbors)
print('GC (scGEN): {:.3f}'.format(gc_scgen))

gc_scpregan = bmk.graph_connectivity(X = x_pca_scpregan, \
    groups = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), k = n_neighbors)
print('GC (scPreGAN): {:.3f}'.format(gc_scpregan))


# silhouette (batch)
silhouette_batch_scdisinfact = bmk.silhouette_batch(X = x_pca_scdisinfact, \
    batch_gt = np.concatenate([x["patient_id"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))

silhouette_batch_scgen = bmk.silhouette_batch(X = x_pca_scgen, \
    batch_gt = np.concatenate([x["sample_id"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette batch (scGEN): {:.3f}'.format(silhouette_batch_scgen))

silhouette_batch_scpregan = bmk.silhouette_batch(X = x_pca_scpregan, \
    batch_gt = np.concatenate([x["sample_id"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette batch (scPreGAN): {:.3f}'.format(silhouette_batch_scpregan))

# silhouette (condition)
silhouette_condition_scdisinfact = bmk.silhouette_batch(X = x_pca_scdisinfact, \
    batch_gt = np.concatenate([x["treatment"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette condition (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact))

silhouette_condition_scgen = bmk.silhouette_batch(X = x_pca_scgen, \
    batch_gt = np.concatenate([x["treatment"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette condition (scGEN): {:.3f}'.format(silhouette_condition_scgen))

silhouette_condition_scpregan = bmk.silhouette_batch(X = x_pca_scpregan, \
    batch_gt = np.concatenate([x["treatment"].values.squeeze() for x in meta_cells_array], axis = 0), \
        group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), verbose = False)
print('Silhouette condition (scPreGAN): {:.3f}'.format(silhouette_condition_scpregan))


# ARI score, check the separation of cell types? still needed?
nmi_scdisinfact = []
ari_scdisinfact = []

nmi_scgen = []
ari_scgen = []

nmi_scpregan = []
ari_scpregan = []

for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scdisinfact = utils.leiden_cluster(X = x_pca_scdisinfact, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scdisinfact))
    ari_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scdisinfact))

    leiden_labels_scgen = utils.leiden_cluster(X = x_pca_scgen, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scgen.append(bmk.nmi(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scgen))
    ari_scgen.append(bmk.ari(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scgen))

    leiden_labels_scpregan = utils.leiden_cluster(X = x_pca_scpregan, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scpregan.append(bmk.nmi(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scpregan))
    ari_scpregan.append(bmk.ari(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in meta_cells_array], axis = 0), group2 = leiden_labels_scpregan))

print('NMI (scDisInFact): {:.3f}'.format(max(nmi_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_scdisinfact)))

print('NMI (scGEN): {:.3f}'.format(max(nmi_scgen)))
print('ARI (scGEN): {:.3f}'.format(max(ari_scgen)))

print('NMI (scPreGAN): {:.3f}'.format(max(nmi_scpregan)))
print('ARI (scPreGAN): {:.3f}'.format(max(ari_scpregan)))

# scores
scores = pd.DataFrame(columns = ["GC", "NMI", "ARI", "Silhouette (condition)", "Silhouette (batches)", "methods"])
scores["NMI"] = np.array([np.max(nmi_scdisinfact), np.max(nmi_scgen), np.max(nmi_scpregan)])
scores["ARI"] = np.array([np.max(ari_scdisinfact), np.max(ari_scgen), np.max(nmi_scpregan)])
scores["methods"] = np.array(["scDisInFact", "scGEN", "scPreGAN"])
scores["GC"] = np.array([gc_scdisinfact, gc_scgen, gc_scpregan])
scores["Silhouette (condition)"] = np.array([silhouette_condition_scdisinfact, silhouette_condition_scgen, silhouette_condition_scpregan])
scores["Silhouette (batches)"] = np.array([silhouette_batch_scdisinfact, silhouette_batch_scgen, silhouette_batch_scpregan])
scores.to_csv(result_dir + comment + "batch_mixing_scdisinfact.csv")

# In[]
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize = (17,5))
axs = fig.subplots(nrows = 1, ncols = 3)
sns.barplot(data = scores, x = "methods", y = "Silhouette (batches)", ax = axs[0], color = "blue")
show_values(axs[0], )
sns.barplot(data = scores, x = "methods", y = "Silhouette (condition)", ax = axs[1], color = "blue")
show_values(axs[1])
sns.barplot(data = scores, x = "methods", y = "ARI", ax = axs[2], color = "blue")
show_values(axs[2])

axs[0].set_ylim(0.5, 1)
axs[1].set_ylim(0.5, 1)
axs[2].set_ylim(0, 1)
axs[0].set_ylabel("ASW (batch)")
axs[1].set_ylabel("ASW (batch)")
axs[0].set_title("Batch alignment")
axs[1].set_title("Condition alignment")
axs[2].set_title("Cell type separation")

plt.tight_layout()
fig.savefig(result_dir + comment + "prediction_score.png", bbox_inches = "tight")

# In[]
'''
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
'''
# %%
