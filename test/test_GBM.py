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
from adjustText import adjust_text

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
result_dir = "GBM_treatment/Fig4_patient/"
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
meta_cell.loc[(meta_cell["mstatus"] != "Myeloid") & ((meta_cell["mstatus"] != "Oligodendrocytes") & (meta_cell["mstatus"] != "tumor")), "mstatus"] = "Other"
counts = sp.load_npz(data_dir + "counts_rna.npz")

data_dict = scdisinfact.create_scdisinfact_dataset(counts, meta_cell, condition_key = ["treatment"], batch_key = "patient_id", batch_cond_key = "sample_id")

# In[] Visualize the original count matrix

umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0)
counts = np.concatenate([x.counts for x in data_dict["datasets"]], axis = 0)
counts_norm = counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6) * 100
counts_norm = np.log1p(counts_norm)
x_umap = umap_op.fit_transform(counts_norm)

utils.plot_latent(x_umap, annos = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "patients.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)
utils.plot_latent(x_umap, annos = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "treatment.png", figsize = (11,7), axis_label = "UMAP", markerscale = 6, s = 2)
utils.plot_latent(x_umap, annos = np.concatenate([x["location"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "location.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)
utils.plot_latent(x_umap, annos = np.concatenate([x["gender"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "gender.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)
utils.plot_latent(x_umap, annos = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "mstatus.png", figsize = (11,7), axis_label = "UMAP", markerscale = 6, s = 2)

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
reg_kl = 1e-5
reg_contr = 0.01
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
Ks = [8, 4]

batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)
torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
_ = model.eval()


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

comment = f'results_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for dataset in data_dict["datasets"]:
    with torch.no_grad():
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
pca_op = PCA(n_components = 2)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]),\
    mode = "batches", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_patient_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["sample_id"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_sample_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_treatment.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["location"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_location.png".format()) if result_dir else None, markerscale = 9, s = 3, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["gender"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_gender.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_mstatus.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "batches", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_patient_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["sample_id"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (12,7), save = (result_dir + comment+"diff_sample_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_treatment.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["location"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_location.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["gender"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_gender.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_mstatus.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[] 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Key gene detection
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

model_params = torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
# last 21 dimensions are batch ids
inf = model.extract_gene_scores()[0]
sorted_genes = genes[np.argsort(inf)[::-1]]
key_genes_score = pd.DataFrame(columns = ["genes", "score"])
key_genes_score["genes"] = [x.split("_")[1] for x in sorted_genes]
key_genes_score["score"] = inf[np.argsort(inf)[::-1]]
key_genes_score.to_csv(result_dir + comment + "key_genes.csv")
key_genes_score.iloc[0:300,0:1].to_csv(result_dir + comment + "key_gene_list.txt", index = False, header = False)

# Plot key genes
plt.rcParams["font.size"] = 20
key_genes_score = pd.read_csv(result_dir + comment + "key_genes.csv", index_col = 0)
key_genes_score["type"] = "key_genes"


fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(x="score", y = "type", data = key_genes_score, palette="Set3")

# metallothioneins and neuron marker genes: H1FX, MT1X, MT2A, MT1E, MT1H, MT1G, SNAP25, RAB3A, NRGN, NXPH4, SLC17A7
key_genes1 = key_genes_score[key_genes_score["genes"].isin(["H1FX", "MT1X", "MT2A", "MT1E", "MT1H", "MT1G", "SNAP25", "RAB3A", "NRGN", "NXPH4", "SLC17A7"])]
g = sns.stripplot(x="score", y = "type", data = key_genes1, color="red", edgecolor="gray", size = 6)
texts_red = []
for i in range(len(key_genes1)):
    if i <= 20:
        texts_red.append(g.text(y=key_genes1["type"].values[i], x=key_genes1["score"].values[i]+0.001, s=key_genes1["genes"].values[i], horizontalalignment='right', size='medium', color='red', fontsize = 10))

# Myeloid-relevent genes
key_genes2 = key_genes_score[key_genes_score["genes"].isin(["CD68", "CTSD", "CTSB", "CD163", "CYBB", "CCR5", "CTSZ", "SAMHD1", "PLA2G7", "SIGLEC1", "LILRB3", "CCR1", "APOBR"])]
g = sns.stripplot(x="score", y = "type", data = key_genes2, color="blue", edgecolor="gray", size = 6)
texts_blue = []
for i in range(len(key_genes2)):
    if i <= 20:
        texts_blue.append(g.text(y=key_genes2["type"].values[i], x=key_genes2["score"].values[i]+0.001, s=key_genes2["genes"].values[i], horizontalalignment='right', size='medium', color='blue', fontsize = 10))

# adjust_text(, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
adjust_text(texts_red + texts_blue, only_move={'points':'y', 'texts':'y'})

ax.set(ylabel=None, yticks = [])
# key_genes3 = key_genes_score[key_genes_score["genes"].isin(["LEFTY1", "BEX5", "SAXO2", "KCNB1"])]
# sns.stripplot(x="score", data = key_genes3, color="green", edgecolor="gray", size = 3)
fig.savefig(result_dir + comment + "marker_gene_violin(w marker).png", bbox_inches = "tight")

# %%
