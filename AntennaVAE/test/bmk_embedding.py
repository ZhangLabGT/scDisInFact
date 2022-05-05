# In[]
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
import model
import loss_function as loss_func
from loss_function import CircleLoss, grouplasso
import utils
import scdistinct
import anndata as ad
import bmk
import matplotlib.pyplot as plt

from umap import UMAP
import seaborn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[]
# NOTE: First load in the model, make sure you put the current model class into scdistinct, put the dataset class also into scdistinct
model1 = scdistinct.scdistinct(datasets = datasets, Ks = [12, 4], batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs, seed = 0, device = device, contr_loss = contr_loss)
# load the state-dict of the model
model1.load_state_dict("xxxx")

z_cs = []
z_ts = []
zs = []
with torch.no_grad():
    for x in datasets:
        z_c = model1.Enc_c(x.counts_stand.to(model1.device))
        z_t = model1.Enc_t(x.counts_stand.to(model1.device))
        z_cs.append(z_c.cpu().detach().numpy())
        z_ts.append(z_t.cpu().detach().numpy())
        zs.append(torch.cat((z_c, z_t), dim = 1).cpu().detach().numpy())

# NOTE: cluster_labels is a list of cell type ids from different batches (format similar to z_cs, z_ts)
cluster_labels = xxx
# NOTE: condition_labels is a list of cell conditions ids from different batches (format similar to z_cs, z_ts)
condition_labels = xxx

# NOTE: a folder in the test directory
result_dir  = "xxxx"
# In[] Calculate scores
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scdistinct
n_neighbors = 30
gc_cluster = bmk.graph_connectivity(X = np.concatenate(z_cs, axis = 0), groups = np.concatenate(cluster_labels, axis = 0), k = n_neighbors)
print('GC cluster (scDistinct): {:.3f}'.format(gc_cluster))

n_neighbors = 30
gc_condition = bmk.graph_connectivity(X = np.concatenate(z_ts, axis = 0), groups = np.concatenate(condition_labels, axis = 0), k = n_neighbors)
print('GC condition (scDistinct): {:.3f}'.format(gc_condition))

# NMI and ARI measure the separation of cell types
# 1. scDistinct
nmi_cluster = []
ari_cluster = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_cluster.append(bmk.nmi(group1 = np.concatenate(cluster_labels), group2 = leiden_labels_clusters))
    ari_cluster.append(bmk.ari(group1 = np.concatenate(cluster_labels), group2 = leiden_labels_clusters))
print('NMI (scDistinct): {:.3f}'.format(max(nmi_cluster)))
print('ARI (scDistinct): {:.3f}'.format(max(ari_cluster)))

nmi_condition = []
ari_condition = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate(z_ts, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_condition.append(bmk.nmi(group1 = np.concatenate(condition_labels), group2 = leiden_labels_conditions))
    ari_condition.append(bmk.ari(group1 = np.concatenate(condition_labels), group2 = leiden_labels_conditions))
print('NMI (scDistinct): {:.3f}'.format(max(nmi_condition)))
print('ARI (scDistinct): {:.3f}'.format(max(ari_condition)))

scores = pd.DataFrame(columns = ["methods", "resolution", "NMI (cluster)", "ARI (cluster)", "NMI (condition)", "ARI (condition)", "GC (cluster)", "GC (condition)"])
scores["NMI (cluster)"] = np.array(nmi_cluster)
scores["ARI (cluster)"] = np.array(ari_cluster)
scores["GC (cluster)"] = np.array([gc_cluster] * len(nmi_cluster))

scores["NMI (condition)"] = np.array(nmi_condition)
scores["ARI (condition)"] = np.array(ari_condition)
scores["GC (condition)"] = np.array([gc_condition] * len(nmi_condition))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
scores["methods"] = np.array(["scDistinct"] * len(ari_condition))
scores.to_csv(result_dir + "score.csv")

# In[] 
# NOTE: Plot barplot of the scores if we have baseline
'''
plt.rcParams["font.size"] = 15
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.4f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

score = pd.read_csv(result_dir + "score.csv")
gc_cluster = np.max(score.loc[score["methods"] == "scDistinct", "GC (cluster)"].values)
gc_condition = np.max(score.loc[score["methods"] == "scDistinct", "GC (condition)"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [gc_cluster, gc_condition], width = 0.4)
barlist[0].set_color('r')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("graph connectivity", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDistinct (cluster)", "scDistinct (condition)"])
_ = ax.set_ylabel("GC", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "GC.pdf", bbox_inches = "tight")    

# NMI
nmi_cluster = np.max(score.loc[score["methods"] == "scDistinct", "NMI (cluster)"].values)
nmi_condition = np.max(score.loc[score["methods"] == "scDistinct", "NMI (condition)"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [nmi_cluster, nmi_condition], width = 0.4)
barlist[0].set_color('r')    
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("NMI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDistinct (cluster)", "scDistinct (condition)"])
_ = ax.set_ylabel("NMI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "NMI.pdf", bbox_inches = "tight")    

# ARI
nmi_cluster = np.max(score.loc[score["methods"] == "scDistinct", "ARI (cluster)"].values)
nmi_condition = np.max(score.loc[score["methods"] == "scDistinct", "ARI (condition)"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [nmi_cluster, nmi_condition], width = 0.4)
barlist[0].set_color('r')    
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("ARI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDistinct (cluster)", "scDistinct (condition)"])
_ = ax.set_ylabel("ARI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "ARI.pdf", bbox_inches = "tight")    
'''
# %%




