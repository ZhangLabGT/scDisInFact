# In[]
from random import random
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
import scdisinfact
import loss_function as loss_func
import utils
import bmk

import anndata as ad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

from umap import UMAP
import seaborn
import warnings
warnings.filterwarnings('ignore')

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.4f}'.format(p.get_height())
            ax.text(_x, _y + 0.1, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def plot_heatmap(weight_matrix, vmax = 0.5, vmin = 0.1, 
                 annot=True, linewidths=0.1,linecolor='white', mask_value= 0.015, 
                 title = None, save = False, path = None,
                 figsize = (15, 8)):
    reshaped = np.array(weight_matrix.detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.).reshape(25,20))
    fig_hp = plt.figure(figsize = figsize)
    ax = fig_hp.add_subplot()
    seaborn.heatmap(reshaped, vmin = vmin, vmax = vmax, 
                    cmap=seaborn.diverging_palette(250, 10, s=75, l=40,n=50, center="light"),
                    annot= annot,
                    linewidths= linewidths,
                    linecolor= linecolor,
                    mask= reshaped < mask_value
                   )
    ax.set_title(title)
    plt.show()
    if save and path:
        fig_hp.savefig(path + '.png', bbox_inches = "tight")
def detect_ls(weight_matrix, show_matrix = False):
    sort_w = torch.sort((weight_matrix.pow(2).sum(dim=0).add(1e-8).pow(1/2.)), descending = True)
    origin_w = weight_matrix.pow(2).sum(dim=0).add(1e-8).pow(1/2.)
    if show_matrix:
        print(origin_w)
        print('!!!')
        print(sort_w)
    n_40 = len([i for i in sort_w[1][:40] if i < 40])
    n_20 = len([i for i in sort_w[1][:40] if i < 20])
    comment = r'First 20: {} /20; First 40: {} /40'.format(n_20, n_40)
    return comment

simulated_lists = [
 'dataset_10000_500_0.4_20_0.5',
#  'dataset_trail',
 'dataset_10000_500_0.1_20_4',
 'dataset_10000_500_0.4_20_4',
 'dataset_10000_500_0.1_20_2',
 'dataset_10000_500_0.1_20_0.5',
 'dataset_10000_500_0.4_20_1',
 'dataset_10000_500_0.1_20_1',
 'dataset_10000_500_0.4_20_2']


# In[]
this_simulated_dir = simulated_lists[-1]
# this_simulated_dir = simulated_lists[-1]
result_dir = './simulated/two_cond/'+this_simulated_dir + "/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

path = '../data/simulated/two_cond/' + this_simulated_dir + '/'
n_batches = 6
counts_rnas = []
labels = []
conditions1 = []
conditions2 = []
batches = []
datasets = []
batch_list = [1, 2, 3, 4, 5, 6]
label_annos = []
label_batches = []
label_conditions1 = []
label_conditions2 = []

np.random.seed(0)
# for batch_id in range(1, n_batches + 1):
for batch_id in batch_list:
    counts_rnas.append(pd.read_csv(path + f'GxC{batch_id}.txt', sep = "\t", header = None).values.T)
    this_anno = pd.read_csv(path + f'cell_label{batch_id}.txt', sep = "\t", index_col = 0).values.squeeze()
    label_annos.append(np.array([('cell type '+str(i)) for i in this_anno]))
    
    labels.append(pd.read_csv(path + f'cell_label{batch_id}.txt', sep = "\t", index_col = 0).values.squeeze())
    label_batches.append(np.array(['batch ' + str(batch_id)] * labels[-1].shape[0]))
    batches.append(np.array([batch_id] * labels[-1].shape[0]))

    # shuffle
    # random_condition = np.random.choice(a = np.array([0, 1, 2]), size = 1)[0]
    # label_conditions.append(np.array(['condition '+str(random_condition)] * labels[-1].shape[0]))
    # conditions.append(np.array([random_condition] * labels[-1].shape[0]))
    # original
    if batch_id in [1, 2]:
        label_conditions1.append(np.array(['condition 0'] * labels[-1].shape[0]))
        conditions1.append(np.array([0] * labels[-1].shape[0]))
    elif batch_id in [3, 4]:
        label_conditions1.append(np.array(['condition 1'] * labels[-1].shape[0]))
        conditions1.append(np.array([1] * labels[-1].shape[0]))    
    else:
        label_conditions1.append(np.array(['condition 2'] * labels[-1].shape[0]))
        conditions1.append(np.array([2] * labels[-1].shape[0]))

    if batch_id in [1, 2, 3]:
        label_conditions2.append(np.array(['condition 0'] * labels[-1].shape[0]))
        conditions2.append(np.array([0] * labels[-1].shape[0]))
    else:
        label_conditions2.append(np.array(['condition 1'] * labels[-1].shape[0]))
        conditions2.append(np.array([1] * labels[-1].shape[0]))
        
    datasets.append(scdisinfact.dataset(counts = counts_rnas[-1], anno = labels[-1], diff_labels = [conditions1[-1], conditions2[-1]], batch_id = batches[-1]))
    print(batch_id, 'Finished')

# In[]
# # check the visualization before integration
# umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 
# counts_norms = []
# annos = []
# conditions = []
# for batch in range(n_batches):
#     counts_norms.append(datasets[batch].counts_norm)
#     annos.append(datasets[batch].anno)
#     conditions.append(datasets[batch].diff_label)
# x_umap = umap_op.fit_transform(np.concatenate(counts_norms, axis = 0))
# # separate into batches
# x_umaps = []
# for batch in range(n_batches):
#     if batch == 0:
#         start_pointer = 0
#         end_pointer = start_pointer + counts_norms[batch].shape[0]
#         x_umaps.append(x_umap[start_pointer:end_pointer,:])
#     elif batch == (n_batches - 1):
#         start_pointer = start_pointer + counts_norms[batch - 1].shape[0]
#         x_umaps.append(x_umap[start_pointer:,:])
#     else:
#         start_pointer = start_pointer + counts_norms[batch - 1].shape[0]
#         end_pointer = start_pointer + counts_norms[batch].shape[0]
#         x_umaps.append(x_umap[start_pointer:end_pointer,:])

# save_file = None

# utils.plot_latent(x_umaps, annos = annos, mode = "modality", save = save_file, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps, annos = conditions, mode = "joint", save = save_file, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# utils.plot_latent(x_umaps, annos = annos, mode = "separate", save = None, figsize = (10,70), axis_label = "Latent")

# In[] training the model
import importlib 
importlib.reload(scdisinfact)
m, gamma = 0.3, 10
# reconstruction, mmd, cross_entropy, contrastive, group_lasso, kl divergence, total correlation
lambs = [1, 0.01, 1.0, 0.0, 0, 1e-5, 0.1]
contr_loss = loss_func.CircleLoss(m = m, gamma = gamma)
# contr_loss = SupervisedContrastiveLoss()
model1 = scdisinfact.scdisinfact(datasets = datasets, Ks = [12, 4, 4], batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs, contr_loss = contr_loss, seed = 0, device = device)
losses = model1.train(nepochs = 100)
# torch.save(model1.state_dict(), result_dir + "model.pth")
# model1.load_state_dict(torch.load(result_dir + "model.pth"))

 # In[] Plot the loss curve
# loss_test, loss_recon, loss_mmd, loss_class, loss_gl_d, loss_gl_c = losses

# iters = np.arange(1, len(loss_gl_c)+1)
# fig = plt.figure(figsize = (40, 10))
# ax = fig.add_subplot()
# ax.plot(iters, loss_gl_c, "-*", label = 'Group Lasso common')
# ax.plot(iters, loss_gl_d, "-*", label = 'Group Lasso diff')
# ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
# ax.set_yscale('log')
# for i, j in zip(iters, loss_gl_c):
#     ax.annotate("{:.3f}".format(j),xy=(i,j))
# for i, j in zip(iters, loss_gl_d):
#     ax.annotate("{:.3f}".format(j),xy=(i,j))

# fig.savefig(result_dir+'/gl_both_ent_loss.png', bbox_inches = "tight")

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for batch_id, dataset in enumerate(datasets):
    with torch.no_grad():
        z_c, _ = model1.Enc_c(torch.concat([dataset.counts_stand, torch.FloatTensor([[batch_id]]).expand(dataset.counts_stand.shape[0], 1)], dim = 1).to(model1.device))
        z_ds.append([])
        for Enc_d in model1.Enc_ds:
            z_d, _ = Enc_d(torch.concat([dataset.counts_stand, torch.FloatTensor([[batch_id]]).expand(dataset.counts_stand.shape[0], 1)], dim = 1).to(model1.device))
            z_ds[-1].append(z_d.cpu().detach().numpy())
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

result_dir = False
comment = r'circle_{}_{}_{}_'.format(m, gamma, lambs[-1])

utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = label_conditions2, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_conditions1, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)

# utils.plot_latent(zs = zs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"all_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
# utils.plot_latent(zs = zs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"all_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)
# utils.plot_latent(zs = zs_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"all_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 5)   


# In[] Calculate AUPRC score
plt.rcParams["font.size"] = 20

simulated_lists = [
 'dataset_10000_500_0.4_20_0.5',
 'dataset_10000_500_0.4_20_1',
 'dataset_10000_500_0.4_20_2',
 'dataset_10000_500_0.4_20_4']

gt = np.zeros((1, 500))
gt[:,:20] = 1
auprc_dict = {}
for dataset_dir in simulated_lists:
    result_dir = './simulated/'+dataset_dir + "/"
    model_params = torch.load(result_dir + "model.pth")
    inf = np.array(model_params["Enc_d.fc.fc_layers.Layer 0.linear.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.))
    auprc_dict[dataset_dir] = bmk.compute_auprc(inf, gt)

# plot bar chart
st_dict = sorted(auprc_dict.items())
x, y = list(zip(*st_dict))
plt_y = np.array(list(y))/0.04
plt_x = [i[18:] for i in np.array(list(x))]
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
ax.bar(plt_x, plt_y, width = 0.4, color = 'bbbb')
show_values_on_bars(ax)
ax.set_xticklabels(["0.5", "1", "2", "4"])
ax.set_xlabel("Perturbation parameter")
ax.set_ylabel("AUPRC Ratio")
fig.savefig("simulated/AUPRC_ratio.png", bbox_inches = "tight")


# In[]
from sklearn.metrics import precision_recall_curve
sim_data = simulated_lists[0]
gt = np.zeros((1, 500))
gt[:,:20] = 1
result_dir = './simulated/'+sim_data + "/"
model_params = torch.load(result_dir + "model.pth")
inf = np.array(model_params["Enc_d.fc.fc_layers.Layer 0.linear.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.))
auprc_dict[dataset_dir] = bmk.compute_auprc(inf, gt)
inf = np.abs(inf)
gt = np.abs(gt)
gt = (gt > 1e-6).astype(int)
inf = (inf - np.min(inf))/(np.max(inf) - np.min(inf) + 1e-12)
prec, recall, thresholds = precision_recall_curve(y_true=gt.reshape(-1,), probas_pred=inf.reshape(-1,), pos_label=1)

prec_random = np.array([20/500] * recall.shape[0]) 
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
ax.plot(recall, prec, "-*", label = "scDisInFact")
ax.plot(recall, prec_random, "-*", label = "Random")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
fig.savefig(result_dir + "AURPC_curve.png", bbox_inches = "tight")

# %%
