# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import scdisinfact
import utils

import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# In[]
'''
data_dir = "../data/GBM_treatment/Fig4/raw/"
result_dir = "GBM_treatment/Fig4/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

patient_info = pd.read_excel("../data/GBM_treatment/patient_info.xlsx")
patient_info.index = patient_info["sample_id"].values.squeeze()

# Fig4, according to TableS1, we have 
# 1 vehicle, 1 etoposide for PW029, (SAMPLE ID: PW029-701, PW029-702)
# 2 vehicle, 1 etoposide, 1 panobinostat for PW030, (SAMPLE ID: PW030-701, PW030-702, PW030-703, PW030-704)
# 2 vehicle, 1 etoposide, 1 panobinostat for PW032, (SAMPLE ID: PW032-702, PW032-705, PW032-709 (vehicle, 3->2), PW032-703, PW032-711)
# 2 vehicle, 1 etoposide, 1 panobinostat for PW034, (SAMPLE ID: PW034-701, PW034-702, PW034-703, PW034-705)
# 2 vehicle, 1 etoposide, 1 panobinostat for PW036, (SAMPLE ID: PW036-701, PW036-702, PW036-703, PW036-705)
# 2 vehicle, 1 etoposide, 1 panobinostat for PW040, (SAMPLE ID: PW040-701, PW040-703, PW040-705, PW040-707, PW040-709, PW040-711 (vehicle 5->2), PW040-712)
PW029 = patient_info[patient_info["patient_id"] == "PW029"]
PW030 = patient_info.loc[["PW030-701", "PW030-702", "PW030-703", "PW030-704"],:]
PW032 = patient_info.loc[["PW032-702", "PW032-705", "PW032-709", "PW032-703", "PW032-711"],:]
PW034 = patient_info.loc[["PW034-701", "PW034-702", "PW034-703", "PW034-705"],:]
PW036 = patient_info.loc[["PW036-701", "PW036-702", "PW036-703", "PW036-705"],:]
PW040 = patient_info.loc[["PW040-701", "PW040-703", "PW040-705", "PW040-707", "PW040-709", "PW040-711", "PW040-712"],:]
Fig4 = pd.concat((PW029, PW030, PW032, PW034, PW036, PW040), axis = 0)

# consider one patient as one batch, and the condition correspond to the drug treatment (intend to use control and one drug (panobinostat))
Fig4 = Fig4[(Fig4["treatment"] == "vehicle (DMSO)") | (Fig4["treatment"] == "0.2 uM panobinostat")]
counts_array = []
meta_cells_array = []
for sample_id in Fig4.index.values:
    counts = sp.load_npz(data_dir + sample_id + ".npz")
    meta_cell = pd.read_csv(data_dir + "meta_" + sample_id + ".csv", index_col = 0)
    meta_cells_array.append(meta_cell)
    counts_array.append(counts.toarray())

meta_cell = pd.concat(meta_cells_array, axis = 0)
counts = np.concatenate(counts_array, axis = 0)
# condition
treatment_id, treatments = pd.factorize(meta_cell["treatment"].values.squeeze())
# batches
batch_ids, batch_names = pd.factorize(meta_cell["patient_id"].values.squeeze())

datasets_array = []
counts_array = []
meta_cells_array = []
for batch_id, batch_name in enumerate(batch_names):
    counts_array.append(counts[batch_ids == batch_id, :])
    meta_cells_array.append(meta_cell.iloc[batch_ids == batch_id, :])
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [treatment_id[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))
'''
# In[]
data_dir = "../data/GBM_treatment/Fig4/processed/"
result_dir = "GBM_treatment/Fig4/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# NOTE: orig.ident: patient id _ timepoint (should be batches), Patient: patient id, Timepoint: timepoint of sampling, Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cell = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t")
counts = sp.load_npz(data_dir + "counts_rna.npz")

# condition
treatment_id, treatments = pd.factorize(meta_cell["treatment"].values.squeeze())
# batches
batch_ids, batch_names = pd.factorize(meta_cell["patient_id"].values.squeeze())

datasets_array = []
counts_array = []
meta_cells_array = []
for batch_id, batch_name in enumerate(batch_names):
    counts_array.append(counts[batch_ids == batch_id, :].toarray())
    meta_cells_array.append(meta_cell.iloc[batch_ids == batch_id, :])
    datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = None, diff_labels = [treatment_id[batch_ids == batch_id]], batch_id = batch_ids[batch_ids == batch_id]))


# In[]
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

utils.plot_latent(x_umaps, annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "batches.png", figsize = (17,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "treatment.png", figsize = (12,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "location.png", figsize = (12, 10), axis_label = "Latent", markerscale = 6)

utils.plot_latent(x_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", save = result_dir + "gender.png", figsize = (20, 10), axis_label = "Latent", markerscale = 6)

# In[]
import importlib 
importlib.reload(scdisinfact)
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [1e-2, 1.0, 0.1, 1, 1e-6]
Ks = [12, 4]
nepochs = 1000
model1 = scdisinfact.scdisinfact(datasets = datasets_array, Ks = Ks, batch_size = 128, interval = nepochs/5, lr = 5e-4, lambs = lambs, seed = 0, device = device)
losses = model1.train(nepochs = nepochs, recon_loss = "NB")
torch.save(model1.state_dict(), result_dir + "model.pth")
model1.load_state_dict(torch.load(result_dir + "model.pth"))

# In[] Plot the loss curve
plt.rcParams["font.size"] = 20
loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests = losses
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
        z_c, _ = model1.Enc_c(torch.concat([dataset.counts_stand, dataset.batch_id[:, None]], dim = 1).to(model1.device))
        # z_c = model1.Enc_c(dataset.counts_stand.to(model1.device))

        z_ds.append([])
        for Enc_d in model1.Enc_ds:
            z_d, _ = Enc_d(torch.concat([dataset.counts_stand, dataset.batch_id[:, None]], dim = 1).to(model1.device))
            # z_d = Enc_d(dataset.counts_stand.to(model1.device))
            z_ds[-1].append(z_d.cpu().detach().numpy())
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


utils.plot_latent(zs = z_cs_umaps, annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"common_patient_id.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"common_treatment.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umaps, annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_location.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umaps, annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12, 10), save = (result_dir + comment+"common_gender.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umaps[0], annos = [x["patient_id"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_patient_id.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["treatment"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_treatment.png") if result_dir else None , markerscale = 6, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["location"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_location.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [x["gender"].values.squeeze() for x in meta_cells_array], mode = "joint", axis_label = "UMAP", figsize = (12,10), save = (result_dir + comment+"diff_gender.png".format()) if result_dir else None, markerscale = 6, s = 1, alpha = 0.5)

# %%
