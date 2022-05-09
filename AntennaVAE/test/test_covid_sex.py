# In[]
import sys, os
sys.path.append("../src")
import torch
import numpy as np 
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn

import scdisinfact
import loss_function as loss_func
import utils
import bmk


le = LabelEncoder()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

# In[]
data_dir = '../data/covid/processed_covid_sample/'
batch_info = pd.read_csv(data_dir + 'batch_info.csv', index_col = 0)
merge_meta = pd.read_csv('../data/covid/merge_meta.csv', index_col = 0)
pd.set_option('display.max_rows', None)

# In[] Print sample ID for BALF and SPUTUM source
balf_lst = ['S-M074-1','S-M075','S-S086-1','S-M076-1','S-S085-1','S-S087-1','S-S088-1','S-S089-1','S-S090-1','S-S006','S-S008','S-S009']
sputum_lst = ['S-S010-1','S-S011-1','S-S012-1','S-M002-1','S-M003-1','S-S010-2','S-S005-1','S-S001-1','S-S007-1','S-S007-2','S-M003-2','S-M003-3','S-S005-2','S-S011-2','S-M002-2','S-S012-2','S-S002','S-S003','S-S012-3','S-S004-1','S-S004-2','S-S004-3',]
print('BALF',set([i.split('-')[-1] for i in merge_meta[(merge_meta['sampleID'].isin(balf_lst))].index.tolist()]))
print('sputum',set([i.split('-')[-1] for i in merge_meta[(merge_meta['sampleID'].isin(sputum_lst))].index.tolist()]))

# In[] Print sample ID for different severity and sample time
sv = ['severe/critical', 'mild/moderate']
st = ['progression', 'convalescence']
for i in st:
    for j in sv:
        convalescence = [i.split('-')[-1] for i in merge_meta[(merge_meta['Sample time'] == i) & (merge_meta['severity'] == j)].index.tolist()]
        convalescence = list(set(convalescence))
        convalescence= [int(i) for i in convalescence]
        convalescence.sort()
        print(i, j, np.array(convalescence))
        print('         ')


control = [i.split('-')[-1] for i in merge_meta[(merge_meta['Sample time'] == 'control') ].index.tolist()]
control = list(set(control))
control= [int(i) for i in control]
control.sort()
print(np.array(control))

# In[] Read in the dataset
data_dir = '../data/covid/processed_covid_sample/'
result_dir = "./batch_sex_comb_pro_con/"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

counts_rnas = []
labels = []
conditions = []
batches = []
datasets = []
label_annos = []
label_batches = []
label_conditions = []

# label must start with 0
batch_dict = {'critical':[[1, 49, 86, 89], 2],
              'moderate':[[63, 76, 96, 153], 1],
              'control':[[61, 62, 67, 81], 0]
}
batch_dict_only_prog = {'critical':[[1, 2, 80], 1],
              'moderate':[[11, 23, 140], 0],
#               'moderate':[[94, 95, 96], 1], 
#                         95 and 96 are BALF
#               'control':[[61, 62, 67, 81], 0]
}

batch_dict_only_conva = {'critical':[[50, 51, 89], 2],
              'moderate':[[75, 76, 151], 1],
#               'control':[[61, 62, 67, 81], 0]
}
batch_dict_conva_ctrl = {'critical':[[50, 51, 89], 2],
              'moderate':[[75, 76, 151], 1],
              'control':[[62, 67, 30], 0]
}
batch_dict_prog_ctrl = {'critical':[[1, 2, 80], 2],
#               'moderate':[[11, 23, 230], 1],
              'moderate':[[11, 23], 1],
              'control':[[62, 67, 30], 0]
}
batch_dict_cmb = {'critical':[[50, 51, 1, 2], 2],
              'moderate':[[75, 76, 11, 23], 1],
#               'control':[[61, 62, 67, 81], 0]
}
batch_only_balf = {
#     'critical':[[101, 102], 2],
                   'critical':[[1, 2], 2],
#                    'moderate':[[95, 96], 1],
                   'moderate':[[11, 96], 1],
}

batch_sex_prog_cri = {
                   'F':[[1, 2], 0],
                   'M':[[263, 89], 1],
}
batch_sex_con_cri = {
                   'F':[[51, 126], 0],
                   'M':[[40, 50], 1],
}
batch_sex_con_mod = {
                   'F':[[58, 74], 0],
                   'M':[[60, 133], 1],
}
batch_sex_comb = {
                   'F-moderate':[[58, 74], 0],
                   'F-critical':[[51, 126], 0],
                   'M-moderate':[[60, 133], 1],
                   'M-critical':[[40, 50], 1],
}
batch_sex_comb_pro_con = {
                   'F-conv-moderate':[[58, 74], 0],
                   'F-prog-critical':[[1, 2], 0],
                   'M-conv-moderate':[[60, 133], 1],
                   'M-prog-critical':[[263, 89], 1],
}

n = 0
for severity, label in batch_sex_comb_pro_con.items():
    idxes = label[0]
    condition = label[1]
    plt_name = severity
    severity = severity[7:]
    for idx in idxes:
        counts_rnas.append(np.array(sp.load_npz(os.path.join(data_dir, '{}/mtx_batch_{}_{}.npz'.format(severity,idx,severity))).todense()))
        label_annos.append((pd.read_csv(os.path.join(data_dir, '{}/meta_batch_{}_{}.csv'.format(severity,idx,severity)))["majorType"]))
        labels.append(le.fit_transform(pd.read_csv(os.path.join(data_dir, '{}/meta_batch_{}_{}.csv'.format(severity,idx,severity)))["majorType"]))
        
        label_conditions.append(np.array(['sex '+str(condition)+'-'+plt_name[:1]] * counts_rnas[-1].shape[0]))
        conditions.append(np.array([condition] * counts_rnas[-1].shape[0]))
        label_batches.append(np.array(['batch '+str(n)+'-' + plt_name]* counts_rnas[-1].shape[0]))
        batches.append(np.array([n] * counts_rnas[-1].shape[0]))
        
        datasets.append(scdisinfact.dataset(counts = counts_rnas[-1],anno = labels[-1], diff_label = conditions[-1], batch_id = batches[-1]))
        print(severity, idx, 'finished')
        n += 1
    
n_batches = len(datasets)

# In[]
# check the visualization before integration
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 
counts_norms = []
annos = []
conditions = []
for batch in range(n_batches):
    counts_norms.append(datasets[batch].counts_norm)
    annos.append(datasets[batch].anno)
    conditions.append(datasets[batch].diff_label)
x_umap = umap_op.fit_transform(np.concatenate(counts_norms, axis = 0))
# separate into batches
x_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts_norms[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + counts_norms[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + counts_norms[batch - 1].shape[0]
        end_pointer = start_pointer + counts_norms[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

save_file = result_dir

utils.plot_latent(x_umaps, annos = label_annos, mode = "modality", save = save_file + "mod.png", figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = label_conditions, mode = "joint", save = save_file + "clust.png", figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = label_annos, mode = "separate", save = save_file + "sep.png", figsize = (10,35), axis_label = "Latent")
# In[]
m, gamma = 0.3, 0.1
# reconstruction, mmd, cross_entropy, contrastive, group_lasso
# lambda 0. reconstruction, 1. MMD, 2. classification, 3. contrastive, 4. group lasso
lambs = [1, 0.03, 10, 0.1, 0.1]
k_comm = 12
k_diff = 4
contr_loss = loss_func.CircleLoss(m = m, gamma = gamma)
# contr_loss = SupervisedContrastiveLoss()
model1 = scdisinfact.scdisinfact_ae(datasets = datasets, Ks = [k_comm, k_diff], batch_size = 128, interval = 10, lr = 5e-4, lambs = lambs, contr_loss = contr_loss, seed = 0, device = device)
losses = model1.train(nepochs = 100)
torch.save(model1.state_dict(), result_dir + f"model_{k_comm}_{k_diff}_{lambs[1]}_{lambs[2]}_{lambs[3]}_{lambs[4]}.pth")
model1.load_state_dict(torch.load(result_dir + f"model_{k_comm}_{k_diff}_{lambs[1]}_{lambs[2]}_{lambs[3]}_{lambs[4]}.pth"))

# In[] Plot the loss curve
loss_test, loss_recon, loss_mmd, loss_class, loss_gl_d, loss_gl_c = losses

iters = np.arange(1, len(loss_gl_c)+1)
fig = plt.figure(figsize = (40, 10))
ax = fig.add_subplot()
ax.plot(iters, loss_gl_c, "-*", label = 'Group Lasso common')
ax.plot(iters, loss_gl_d, "-*", label = 'Group Lasso diff')
ax.legend(loc = 'upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor = (1.04, 1))
ax.set_yscale('log')
for i, j in zip(iters, loss_gl_c):
    ax.annotate("{:.3f}".format(j),xy=(i,j))
for i, j in zip(iters, loss_gl_d):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

fig.savefig(result_dir+'/gl_both_ent_loss.png', bbox_inches = "tight")

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for x in datasets:
    with torch.no_grad():
        z_c = model1.Enc_c(x.counts_stand.to(model1.device))
        z_d = model1.Enc_d(x.counts_stand.to(model1.device))
    z_cs.append(z_c.cpu().detach().numpy())
    z_ds.append(z_d.cpu().detach().numpy())
    zs.append(torch.cat((z_c, z_d), dim = 1).cpu().detach().numpy())

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = umap_op.fit_transform(np.concatenate(z_ds, axis = 0))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = []
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps.append(z_ds_umap[start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps.append(z_ds_umap[start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps.append(z_ds_umap[start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f'plots_{k_comm}_{k_diff}_{lambs[1]}_{lambs[2]}_{lambs[3]}_{lambs[4]}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,40), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"time_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 5)

utils.plot_latent(zs = zs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"all_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = zs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"all_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = zs_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"all_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 5)   


# %%
