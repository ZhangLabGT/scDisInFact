# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.bmk as bmk

import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings
import time
warnings.filterwarnings("ignore")
import seaborn as sns
from adjustText import adjust_text

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", color = "blue", fontweight = "bold", fontsize = 20) 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left", color = "blue", fontweight = "bold", fontsize = 20)

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
result_dir = "results_GBM_treatment/Fig4_patient/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = object)
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

np.save(file = result_dir + "count_umap.npy", arr = x_umap)

# In[]
x_umap = np.load(result_dir + "count_umap.npy")

utils.plot_latent(x_umap, annos = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "patients.png", figsize = (7,5), axis_label = "UMAP", markerscale = 9, s = 2, legend = False)
utils.plot_latent(x_umap, annos = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "treatment.png", figsize = (7,5), axis_label = "UMAP", markerscale = 9, s = 2, legend = False)
utils.plot_latent(x_umap, annos = np.concatenate([x["location"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "location.png", figsize = (7,5), axis_label = "UMAP", markerscale = 9, s = 2, legend = False)
utils.plot_latent(x_umap, annos = np.concatenate([x["gender"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "gender.png", figsize = (7,5), axis_label = "UMAP", markerscale = 9, s = 2, legend = False)
utils.plot_latent(x_umap, annos = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), mode = "annos", save = result_dir + "mstatus.png", figsize = (7,5), axis_label = "UMAP", markerscale = 9, s = 2, legend = False)

# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
Ks = [8, 2]

batch_size = 64
nepochs = 100
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
# model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
# _ = model.eval()
# torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

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

np.save(file = result_dir + comment + "z_cs_umap.npy", arr = z_cs_umap)
np.save(file = result_dir + comment + "z_ds_umap.npy", arr = z_ds_umap[0])

# In[]
z_cs_umap = np.load(file = result_dir + comment + "z_cs_umap.npy")
z_ds_umap = []
z_ds_umap.append(np.load(file = result_dir + comment + "z_ds_umap.npy"))

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
    mode = "batches", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_patient_id.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small", legend = False)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), batches = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_treatment.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small", legend = False)


# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Disentanglement
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# read in scinsight
result_scinsight = "./results_GBM_treatment/Fig4_patient/scinsight/"

W2 = pd.read_csv(result_scinsight + "W2.txt", sep = "\t")
H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t")
H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t")
W11 = pd.read_csv(result_scinsight + "W11.txt", sep = "\t")
W12 = pd.read_csv(result_scinsight + "W12.txt", sep = "\t")
W13 = pd.read_csv(result_scinsight + "W13.txt", sep = "\t")
W14 = pd.read_csv(result_scinsight + "W14.txt", sep = "\t")
W15 = pd.read_csv(result_scinsight + "W15.txt", sep = "\t")
W16 = pd.read_csv(result_scinsight + "W16.txt", sep = "\t")
W17 = pd.read_csv(result_scinsight + "W17.txt", sep = "\t")
W18 = pd.read_csv(result_scinsight + "W18.txt", sep = "\t")
W19 = pd.read_csv(result_scinsight + "W19.txt", sep = "\t")
W110 = pd.read_csv(result_scinsight + "W110.txt", sep = "\t")
W111 = pd.read_csv(result_scinsight + "W111.txt", sep = "\t")

# H1: ctrl, H2: stim; 
x_cond = [W11.values@H1.values, W12.values@H1.values, W13.values@H2.values, W14.values@H1.values, W15.values@H2.values, W16.values@H1.values,
          W17.values@H2.values, W18.values@H1.values, W19.values@H2.values, W110.values@H1.values, W111.values@H2.values]
x_cond = np.concatenate(x_cond, axis = 0)

umap_op = UMAP(min_dist = 0.1, random_state = 0)
w2_umap = umap_op.fit_transform(W2.values)
x_cond_umap = umap_op.fit_transform(x_cond)

meta_scinsight = pd.concat([meta_cell.loc[(meta_cell["patient_id"] == "PW029") & (meta_cell["treatment"] == "vehicle (DMSO)"),:], 
                            meta_cell.loc[(meta_cell["patient_id"] == "PW030") & (meta_cell["treatment"] == "vehicle (DMSO)"),:], 
                            meta_cell.loc[(meta_cell["patient_id"] == "PW030") & (meta_cell["treatment"] == "0.2 uM panobinostat"),:], 
                            meta_cell.loc[(meta_cell["patient_id"] == "PW032") & (meta_cell["treatment"] == "vehicle (DMSO)"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW032") & (meta_cell["treatment"] == "0.2 uM panobinostat"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW034") & (meta_cell["treatment"] == "vehicle (DMSO)"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW034") & (meta_cell["treatment"] == "0.2 uM panobinostat"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW036") & (meta_cell["treatment"] == "vehicle (DMSO)"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW036") & (meta_cell["treatment"] == "0.2 uM panobinostat"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW040") & (meta_cell["treatment"] == "vehicle (DMSO)"),:],
                            meta_cell.loc[(meta_cell["patient_id"] == "PW040") & (meta_cell["treatment"] == "0.2 uM panobinostat"),:]], ignore_index = True, axis = 0)

# utils.plot_latent(zs = w2_umap, annos = meta_scinsight["mstatus"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_celltypes.png" if result_scinsight else None , markerscale = 6, s = 5)
# # utils.plot_latent(zs = w2_umap, annos = meta_scinsight["mstatus"].values.squeeze(), batches = meta_scinsight["patient_id"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,10), save = result_scinsight + "common_celltypes_sep.png" if result_scinsight else None , markerscale = 6, s = 5)
# utils.plot_latent(zs = w2_umap, annos = meta_scinsight["patient_id"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_batches.png" if result_scinsight else None, markerscale = 6, s = 5)
# utils.plot_latent(zs = w2_umap, annos = meta_scinsight["treatment"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_condition.png" if result_scinsight else None, markerscale = 6, s = 5)
# utils.plot_latent(zs = x_cond_umap, annos = meta_scinsight["mstatus"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff1_celltypes.png" if result_scinsight else None, markerscale = 6, s = 5)
# utils.plot_latent(zs = x_cond_umap, annos = meta_scinsight["patient_id"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff1_batch.png" if result_scinsight else None, markerscale = 6, s = 5)
# utils.plot_latent(zs = x_cond_umap, annos = meta_scinsight["treatment"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff1_condition.png" if result_scinsight else None, markerscale = 6, s = 5)

# In[]
#------------------------------------------------------------------------------------------------------------------------------------------
#
# Benchmark, common space, check removal of batch effect (=condition effect), keep cluster information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# removal of batch and condition effect
# 1. scdisinfact
silhouette_batch_scdisinfact = bmk.silhouette_batch(X = np.concatenate(z_cs, axis = 0), batch_gt = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), \
                                                    group_gt = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), verbose = False)
print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))
# 2. scinsight
silhouette_batch_scinsight = bmk.silhouette_batch(X = W2.values, batch_gt = meta_scinsight["patient_id"].values.squeeze(), group_gt = meta_scinsight["mstatus"].values.squeeze(), verbose = False)
print('Silhouette batch (scInsight): {:.3f}'.format(silhouette_batch_scinsight))


# NMI and ARI measure the separation of cell types
# 1. scdisinfact
nmi_cluster_scdisinfact = []
ari_cluster_scdisinfact = []
for resolution in np.arange(0.1, 2, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    print(np.unique(leiden_labels_clusters).shape)
    nmi_cluster_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_clusters))
    ari_cluster_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["mstatus"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_clusters))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_cluster_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_cluster_scdisinfact)))

# 2. scinsight
nmi_cluster_scinsight = []
ari_cluster_scinsight = []
for resolution in np.arange(0.1, 2, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = W2.values, knn_indices = None, knn_dists = None, resolution = resolution)
    print(np.unique(leiden_labels_clusters).shape)
    nmi_cluster_scinsight.append(bmk.nmi(group1 = meta_scinsight["mstatus"].values.squeeze(), group2 = leiden_labels_clusters))
    ari_cluster_scinsight.append(bmk.ari(group1 = meta_scinsight["mstatus"].values.squeeze(), group2 = leiden_labels_clusters))
print('NMI (scInsight): {:.3f}'.format(max(nmi_cluster_scinsight)))
print('ARI (scInsight): {:.3f}'.format(max(ari_cluster_scinsight)))


# In[]
#------------------------------------------------------------------------------------------------------------------------------------------
#
# condition-specific space, check removal of batch effect, removal of cell type effect, keep condition information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# removal of batch effect, removal of cell type effect
silhouette_condition_scdisinfact = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["patient_id"].values.squeeze() for x in data_dict["meta_cells"]]), group_gt = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), verbose = False)
print('Silhouette condition, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact))
silhouette_condition_scinsight = bmk.silhouette_batch(X = x_cond, batch_gt = meta_scinsight["patient_id"].values.squeeze(), group_gt = meta_scinsight["treatment"].values.squeeze(), verbose = False)
print('Silhouette condition, removal of batch effect (scInsight): {:.3f}'.format(silhouette_condition_scinsight))


# keep of condition information
nmi_condition_scdisinfact = []
ari_condition_scdisinfact = []
# NOTE: the range is not good, only two conditions
for resolution in range(-3, 1, 1):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
    print(np.unique(leiden_labels_conditions))
    nmi_condition_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_conditions))
    ari_condition_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["treatment"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_conditions))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact)))

nmi_condition_scinsight = []
ari_condition_scinsight = []
for resolution in range(-3, 1, 1):
    leiden_labels_conditions = bmk.leiden_cluster(X = x_cond, knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
    print(np.unique(leiden_labels_conditions))
    nmi_condition_scinsight.append(bmk.nmi(group1 = meta_scinsight["treatment"].values.squeeze(), group2 = leiden_labels_conditions))
    ari_condition_scinsight.append(bmk.ari(group1 = meta_scinsight["treatment"].values.squeeze(), group2 = leiden_labels_conditions))
print('NMI (scInsight): {:.3f}'.format(max(nmi_condition_scinsight)))
print('ARI (scInsight): {:.3f}'.format(max(ari_condition_scinsight)))


scores_scdisinfact = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (common)", "Silhouette batch (condition & batches)"])
scores_scdisinfact["methods"] = np.array(["scDisInFact"])
scores_scdisinfact["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact)])
scores_scdisinfact["ARI (common)"] = np.array([max(ari_cluster_scdisinfact)])
scores_scdisinfact["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact])
scores_scdisinfact["NMI (condition)"] = np.array([max(nmi_condition_scinsight)])
scores_scdisinfact["ARI (condition)"] = np.array([max(ari_condition_scinsight)])
scores_scdisinfact["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scdisinfact])

scores_scinsight = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (common)", "Silhouette batch (condition & batches)"])
scores_scinsight["methods"] = np.array(["scINSIGHT"])
scores_scinsight["NMI (common)"] = np.array([max(nmi_cluster_scinsight)])
scores_scinsight["ARI (common)"] = np.array([max(ari_cluster_scinsight)])
scores_scinsight["Silhouette batch (common)"] = np.array([silhouette_batch_scinsight])
scores_scinsight["NMI (condition)"] = np.array([max(nmi_condition_scinsight)])
scores_scinsight["ARI (condition)"] = np.array([max(ari_condition_scinsight)])
scores_scinsight["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scinsight])

scores = pd.concat([scores_scdisinfact, scores_scinsight], axis = 0)
scores.to_csv(result_dir + comment + "score_disentangle.csv")

# In[]
from matplotlib.ticker import FormatStrFormatter
plt.rcParams["font.size"] = 15
scores = pd.read_csv(result_dir + comment + "score_disentangle.csv", index_col = 0)
fig = plt.figure(figsize = (15,5))
ax = fig.subplots(nrows = 1, ncols = 3)
sns.barplot(data = scores, x = "methods", y = "NMI (common)", ax = ax[0], width = 0.5)
ax[0].set_ylabel("NMI")
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0].set_title("NMI (shared)")
ax[0].set_xlabel(None)
show_values(ax[0])

sns.barplot(data = scores, x = "methods", y = "ARI (common)", ax = ax[1], width = 0.5)
ax[1].set_ylabel("ARI")
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].set_title("ARI (shared)")
ax[1].set_xlabel(None)
show_values(ax[1])

sns.barplot(data = scores, x = "methods", y = "Silhouette batch (common)", ax = ax[2], width = 0.5)
ax[2].set_ylabel("Silhouette batch")
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].set_title("Silhouette batch\n(shared)")
ax[2].set_ylim(0.8, 1)
ax[2].set_xlabel(None)
show_values(ax[2])

plt.tight_layout()
fig.savefig(result_dir + comment + "barplot_common.png", bbox_inches = "tight")

fig = plt.figure(figsize = (15,5))
ax = fig.subplots(nrows = 1, ncols = 3)
sns.barplot(data = scores, x = "methods", y = "NMI (condition)", ax = ax[0], width = 0.5)
ax[0].set_ylabel("NMI")
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0].set_title("NMI (condition)")
ax[0].set_xlabel(None)
show_values(ax[0])

sns.barplot(data = scores, x = "methods", y = "ARI (condition)", ax = ax[1], width = 0.5)
ax[1].set_ylabel("ARI")
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].set_title("ARI (condition)")
ax[1].set_xlabel(None)
show_values(ax[1])

sns.barplot(data = scores, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2], width = 0.5)
ax[2].set_ylabel("ASW-batch")
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].set_title("Silhouette batch\n(condition & batches)")
ax[2].set_xlabel(None)
show_values(ax[2])

plt.tight_layout()
fig.savefig(result_dir + comment + "barplot_condition.png", bbox_inches = "tight")


fig = plt.figure(figsize = (12,5))
ax = fig.subplots(nrows = 1, ncols = 3)
sns.barplot(data = scores, x = "methods", y = "ARI (common)", ax = ax[0], width = 0.5)
ax[0].set_ylabel("ARI")
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0].set_title("shared-bio factors\nARI")
ax[0].set_xlabel(None)
show_values(ax[0])

sns.barplot(data = scores, x = "methods", y = "Silhouette batch (common)", ax = ax[1], width = 0.5)
ax[1].set_ylabel("ASW-batch")
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].set_title("shared-bio factors\nASW-batch")
ax[1].set_xlabel(None)
show_values(ax[1])


sns.barplot(data = scores, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2], width = 0.5)
ax[2].set_ylabel("ASW-batch")
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].set_title("unshared-bio factors\nASW-batch")
ax[2].set_xlabel(None)
show_values(ax[2])

plt.tight_layout()
fig.savefig(result_dir + comment + "barplot_disentangle.png", bbox_inches = "tight")



# In[] 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Key gene detection
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
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
# texts_red = []
# for i in range(len(key_genes1)):
#     if i <= 20:
#         texts_red.append(g.text(y=key_genes1["type"].values[i], x=key_genes1["score"].values[i]+0.001, s=key_genes1["genes"].values[i], horizontalalignment='right', size='medium', color='red'))

# Myeloid-relevent genes
key_genes2 = key_genes_score[key_genes_score["genes"].isin(["CD68", "CTSD", "CTSB", "CD163", "CYBB", "CCR5", "CTSZ", "SAMHD1", "PLA2G7", "SIGLEC1", "LILRB3", "CCR1", "APOBR"])]
g = sns.stripplot(x="score", y = "type", data = key_genes2, color="blue", edgecolor="gray", size = 6)
# texts_blue = []
# for i in range(len(key_genes2)):
#     if i <= 20:
#         texts_blue.append(g.text(y=key_genes2["type"].values[i], x=key_genes2["score"].values[i]+0.001, s=key_genes2["genes"].values[i], horizontalalignment='right', size='medium', color='blue'))

# # adjust_text(, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
# adjust_text(texts_red + texts_blue, only_move={'points':'y', 'texts':'y'})

ax.set(ylabel=None, yticks = [])
# key_genes3 = key_genes_score[key_genes_score["genes"].isin(["LEFTY1", "BEX5", "SAXO2", "KCNB1"])]
# sns.stripplot(x="score", data = key_genes3, color="green", edgecolor="gray", size = 3)
fig.savefig(result_dir + comment + "marker_gene_violin(w marker).png", bbox_inches = "tight")

# %%
