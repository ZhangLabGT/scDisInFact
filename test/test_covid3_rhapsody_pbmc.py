# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import scdisinfact
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

# In[] Read in the dataset, filtering out the batches that we don't need
# here we use cohort 2 because cohort 2 has little batch effect between control and covid-19 patient, whereas cohort 1 has large batch effect between control and covid-19 patient. 
# Since control and covid-19 are from completely different batches (control is publically available dataset), it is hard to guarantee the removal of batch effect between control and covid-19 patient
# In addition, the control of cohort 1 doesn't have sex information
data_dir = "../data/covid19-3/Rhapsody_pbmc/raw/"
result_dir = "covid19-3/Rhapsody_pbmc/raw/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# raw dataset
genes = np.loadtxt(data_dir + "genes.csv", dtype = np.object)
meta_cells = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t")
counts_rna = sparse.load_npz(data_dir + "counts_rna.npz")

# sampleID should be the samples/batches, one donor has 1/2 samples (early, late)
batch_ids, batch_names = pd.factorize(meta_cells["sampleID"].values.squeeze())
# one donor has early and late
batch_ids, batch_names = pd.factorize(meta_cells["donor"].values.squeeze())
severity_ids, severity_names = pd.factorize(meta_cells["group_per_sample"].values.squeeze())
# severity_ids, severity_names = pd.factorize(meta_cells["who_per_sample"].values.squeeze())
stages_ids, stages_names = pd.factorize(meta_cells["disease_stage"].values.squeeze())
sex_ids, sex = pd.factorize(meta_cells["sex"].values.squeeze())

counts_array = []
meta_cells_array = []
datasets_array = []
for batch_id, batch_name in enumerate(batch_names):
    counts_array.append(counts_rna[batch_ids == batch_id, :].toarray())
    meta_cells_array.append(meta_cells.iloc[batch_ids == batch_id, :])
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [severity_ids[batch_ids == batch_id], stages_ids[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))

# In[]
'''
# with 5000 genes, very little batch effect
umap_op = UMAP(n_components = 2, n_neighbors = 100, min_dist = 0.4, random_state = 0) 

x_pca = PCA(n_components = 80).fit_transform(np.concatenate([x.counts_norm for x in datasets_array], axis = 0))
x_umap = umap_op.fit_transform(x_pca)
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

utils.plot_latent(x_umaps, annos = [x["sampleID"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["group_per_sample"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "condition1.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["disease_stage"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "condition2.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["cluster_labels_res.0.4"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "celltypes.png", figsize = (20, 10), axis_label = "Latent", markerscale = 6)
'''
# In[]
import importlib 
importlib.reload(scdisinfact)

reg_mmd_comm = 1e-2
reg_mmd_diff = 1e-2
reg_gl = 1
reg_tc = 0.1
reg_class = 1
reg_kl = 1e-5
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [12, 4, 4]
nepochs = 50
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
for i, j in zip(iters, loss_tests):
    ax.annotate("{:.3f}".format(j),xy=(i,j))

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for dataset in datasets_array:
    with torch.no_grad():
        z_c, z_d, z, mu = model.test_model(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = False)        
        z_ds.append(z_d.cpu().detach().numpy())
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[]]
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

comment = f"plots_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umaps, annos = [x["cluster_labels_res.0.4"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15,10), save = (result_dir + comment+"common_dims_celltypes.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = True, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["group_per_sample"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15,10), save = (result_dir + comment+"common_dims_condition.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = True, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["sampleID"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (15, 10), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["cluster_labels_res.0.4"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"diff_dims_celltypes.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["group_per_sample"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = (result_dir + comment+"diff_dims_condition.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

 # %%
