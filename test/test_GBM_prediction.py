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

from sklearn.metrics import r2_score 
import scipy.stats as stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
result_dir = "results_GBM_treatment/Fig4_patient_new/prediction/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# orig.ident: patient id _ timepoint (should be batches), 
# Patient: patient id, 
# Timepoint: timepoint of sampling, 
# Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cells = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
meta_cells_seurat = pd.read_csv(data_dir + "meta_cells_seurat.csv", sep = "\t", index_col = 0)
meta_cells["mstatus"] = meta_cells_seurat["mstatus"].values.squeeze()
meta_cells.loc[(meta_cells["mstatus"] != "Myeloid") & ((meta_cells["mstatus"] != "Oligodendrocytes") & (meta_cells["mstatus"] != "tumor")), "mstatus"] = "Other"
counts = sp.load_npz(data_dir + "counts_rna.npz")

data_dict = scdisinfact.create_scdisinfact_dataset(counts, meta_cells, condition_key = ["treatment"], batch_key = "patient_id", batch_cond_key = "sample_id")

# TODO: train test split

# construct training and testing data
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta in zip(data_dict["datasets"], data_dict["meta_cells"]):

    test_idx = (meta["sample_id"] == "PW034-705").values.squeeze()
    train_idx = ~test_idx
    # # use all as training data
    # train_idx = (meta["sample_id"] != "0").values.squeeze()

    if np.sum(train_idx) > 0:
        dataset_train = scdisinfact.scdisinfact_dataset(counts = dataset[train_idx]["counts"], 
                                                        counts_norm = dataset[train_idx]["counts_norm"],
                                                        size_factor = dataset[train_idx]["size_factor"],
                                                        diff_labels = dataset[train_idx]["diff_labels"],
                                                        batch_id = dataset[train_idx]["batch_id"],
                                                        mmd_batch_id = dataset[train_idx]["mmd_batch_id"]
                                                        )
        datasets_array_train.append(dataset_train)
        meta_cells_array_train.append(meta.iloc[train_idx,:])
    
    if np.sum(test_idx) > 0:
        dataset_test = scdisinfact.scdisinfact_dataset(counts = dataset[test_idx]["counts"], 
                                                    counts_norm = dataset[test_idx]["counts_norm"],
                                                    size_factor = dataset[test_idx]["size_factor"],
                                                    diff_labels = dataset[test_idx]["diff_labels"],
                                                    batch_id = dataset[test_idx]["batch_id"],
                                                    mmd_batch_id = dataset[test_idx]["mmd_batch_id"]
                                                    )
        datasets_array_test.append(dataset_test)
        meta_cells_array_test.append(meta.iloc[test_idx,:])

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}


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
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
Ks = [8, 2]

batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
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

# In[]
# latent embedding of cells in training set
z_cs_train = []
z_ds_train = []

for dataset in data_dict_train["datasets"]:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = False)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]        
        z_cs_train.append(z_c.cpu().detach().numpy())
        z_ds_train.append([x.cpu().detach().numpy() for x in z_d])

# latent embedding of cells in testing set
z_cs_test = []
z_ds_test = []

for dataset in data_dict_test["datasets"]:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = False)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]        
        z_cs_test.append(z_c.cpu().detach().numpy())
        z_ds_test.append([x.cpu().detach().numpy() for x in z_d])

umap_op = UMAP(min_dist = 0.1, random_state = 0)
pca_op = PCA(n_components = 2)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + z_cs_test))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + z_ds_test], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"

for x in meta_cells_array_test:
    x["mode"] = "test"

meta_cells_train_test = pd.concat(meta_cells_array_train + meta_cells_array_test, axis = 0, ignore_index = True)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells_train_test["mstatus"].values.squeeze(), batches = meta_cells_train_test["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"sharedfactor_train_test.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells_train_test["treatment"].values.squeeze(), batches = meta_cells_train_test["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"unsharedfactor_train_test.png" if result_dir else None, markerscale = 6, s = 5)

# In[]
# input matrix, considering two cases: 1. vehicle (DMSO) in the same batch; 2. vehicle (DMSO) in a different batch
# 1. the first case: vehicle (DMSO) in the same batch;
input_idx = ((meta_cells["treatment"] == "vehicle (DMSO)") & (meta_cells["patient_id"] == "PW034")).values.squeeze() # PW034-701, PW034-702
counts_input = counts[input_idx,:].toarray()
meta_input = meta_cells.loc[input_idx,:]

counts_gt = counts[meta_cells["sample_id"] == "PW034-705",:].toarray()
meta_gt = meta_cells.loc[meta_cells["sample_id"] == "PW034-705",:]

# denoising
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["treatment"], 
                                          batch_key = "patient_id", predict_conds = None, predict_batch = None)
# perturbation prediction
counts_scdisinfact = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["treatment"], 
                                          batch_key = "patient_id", predict_conds = ["0.2 uM panobinostat"], predict_batch = "PW034")
# counts_scdisinfact_opt, z_ds_perturbed, z_ds_gt, z_ds_input = model.predict_counts_optrans(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["treatment"], 
#                                           batch_key = "patient_id", predict_conds = ["0.2 uM panobinostat"], predict_batch = "PW034")


# read scgen and scpregan
counts_scgen = sp.load_npz("results_GBM_treatment/Fig4_patient/scGEN/samebatch/counts_scgen.npz").toarray()
counts_scpregan = sp.load_npz("results_GBM_treatment/Fig4_patient/scPreGAN/samebatch/counts_scpregan.npz").toarray()
counts_scgen = counts_scgen * (counts_scgen > 0)
counts_scpregan = counts_scpregan * (counts_scpregan > 0)
# normalization
counts_scdisinfact = counts_scdisinfact/(np.sum(counts_scdisinfact, axis = 1, keepdims = True) + 1e-6) * 100
# counts_predict_opt = counts_scdisinfact_opt/(np.sum(counts_scdisinfact_opt, axis = 1, keepdims = True) + 1e-6) * 100
counts_scgen = counts_scgen/(np.sum(counts_scgen, axis = 1, keepdims = True) + 1e-6) * 100
counts_scpregan = counts_scpregan/(np.sum(counts_scpregan, axis = 1, keepdims = True) + 1e-6) * 100
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6) * 100
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6) * 100
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6) * 100


# # In[]
# # NOTE: Sanity check, if the predict unshared-bio factor mix with ground truth
# z_ds_perturbed = z_ds_perturbed[0].detach().cpu().numpy()
# z_ds_umap = UMAP(random_state = 0, min_dist = 0.1).fit_transform(np.concatenate([z_ds_perturbed, z_ds_gt, z_ds_input], axis = 0))
# utils.plot_latent(z_ds_umap, annos = np.array(["2. perturbed"] * z_ds_perturbed.shape[0] + ["1. gt"] * z_ds_gt.shape[0] + ["3.input"] * z_ds_input.shape[0]), mode = "annos", save = None, figsize = (10,10), axis_label = "UMAP", markerscale = 6)
# utils.plot_latent(z_ds_umap[:z_ds_perturbed.shape[0],:], annos = meta_input["mstatus"].values.squeeze())
# utils.plot_latent(z_ds_umap[z_ds_perturbed.shape[0]:,:], annos = np.array([1]* z_ds_umap[z_ds_perturbed.shape[0]:,:].shape[0]))


# In[] Visualize the predicted counts

# no 1-1 match, check cell-type level scores
unique_celltypes = ["Myeloid", "Oligodendrocytes", "tumor"]
mean_inputs = []
mean_scdisinfacts = []
# mean_scdisinfacts_opt = []
mean_scgens = []
mean_scpregans = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_scdisinfact = np.mean(counts_scdisinfact[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    # mean_scdisinfact_opt = np.mean(counts_scdisinfact_opt[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_scgen = np.mean(counts_scgen[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_scpregan = np.mean(counts_scpregan[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)

    mean_inputs.append(mean_input)
    mean_scdisinfacts.append(mean_scdisinfact)
    # mean_scdisinfacts_opt.append(mean_scdisinfact_opt)
    mean_scgens.append(mean_scgen)
    mean_scpregans.append(mean_scpregan)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_scdisinfacts = np.array(mean_scdisinfacts)
# mean_scdisinfacts_opt = np.array(mean_scdisinfacts_opt)
mean_scgens = np.array(mean_scgens)
mean_scpregans = np.array(mean_scpregans)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_scdisinfacts - mean_gts_denoised) ** 2, axis = 1)
# mses_scdisinfact_opt = np.sum((mean_scdisinfacts_opt - mean_gts_denoised) ** 2, axis = 1)
mses_scgens = np.sum((mean_scgens - mean_gts) ** 2, axis = 1)
mses_scpregans = np.sum((mean_scpregans - mean_gts) ** 2, axis = 1)

# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts[i,:])[0] for i in range(mean_gts.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_scdisinfacts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts.shape[0])])
# pearsons_scdisinfact_opt = np.array([stats.pearsonr(mean_scdisinfacts_opt[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scgen = np.array([stats.pearsonr(mean_scgens[i,:], mean_gts[i,:])[0] for i in range(mean_gts.shape[0])])
pearsons_scpregan = np.array([stats.pearsonr(mean_scpregans[i,:], mean_gts[i,:])[0] for i in range(mean_gts.shape[0])])

# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts[i,:]) for i in range(mean_gts.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_scdisinfacts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts.shape[0])])
# r2_scdisinfact_opt = np.array([r2_score(y_pred = mean_scdisinfacts_opt[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scgen = np.array([r2_score(y_pred = mean_scgens[i,:], y_true = mean_gts[i,:]) for i in range(mean_gts.shape[0])])
r2_scpregan = np.array([r2_score(y_pred = mean_scpregans[i,:], y_true = mean_gts[i,:]) for i in range(mean_gts.shape[0])])

scores1 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE (Ratio)", "Pearson (Ratio)", "R2 (Ratio)", "Method", "Prediction"])
scores1["MSE"] = np.concatenate([mses_scdisinfact, mses_scgens, mses_scpregans], axis = 0)
scores1["MSE (Ratio)"] = np.concatenate([mses_scdisinfact/mses_input, mses_scgens/mses_input, mses_scpregans/mses_input], axis = 0)
scores1["Pearson"] = np.concatenate([pearsons_scdisinfact, pearsons_scgen, pearsons_scpregan], axis = 0)
scores1["Pearson (Ratio)"] = np.concatenate([pearsons_scdisinfact/pearsons_input, pearsons_scgen/pearsons_input, pearsons_scpregan/pearsons_input], axis = 0)
scores1["R2"] = np.concatenate([r2_scdisinfact, r2_scgen, r2_scpregan], axis = 0)
scores1["R2 (Ratio)"] = np.concatenate([r2_scdisinfact/r2_input, r2_scgen/r2_input, r2_scpregan/r2_input], axis = 0)
scores1["Method"] = ["scDisInFact"] * len(r2_scdisinfact) + ["scGen"] * len(r2_scgen) + ["scPreGAN"] * len(r2_scpregan)
scores1["Prediction"] = "treatment\n(w/o batch effect)"

# In[]
# PCA calculation will cost scdisinfact prediction to loss information
x_pca_scdisinfact = np.log1p(np.concatenate([counts_gt_denoised, counts_scdisinfact], axis = 0))
x_umap_scdisinfact = UMAP(min_dist = 0.1, random_state = 0).fit_transform(x_pca_scdisinfact)
np.save(file = result_dir + comment + "predict_scdisinfact(same).npy", arr = x_umap_scdisinfact)
x_umap_scdisinfact = np.load(result_dir + comment + "predict_scdisinfact(same).npy")
# utils.plot_latent(x_umap_scdisinfact, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scdisinfact(samebatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scdisinfact, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), mode = "annos", save = result_dir + comment + "predict_scdisinfact(samebatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scdisinfact, annos = np.array(["2. Gold-standard"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + comment + "predict_scdisinfact_batches(samebatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)


# x_pca_scdisinfact_opt = np.log1p(np.concatenate([counts_gt_denoised, counts_scdisinfact_opt], axis = 0))
# x_umap_scdisinfact_opt = UMAP(min_dist = 0.4, random_state = 0).fit_transform(x_pca_scdisinfact_opt)
# utils.plot_latent(x_umap_scdisinfact_opt, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scdisinfact_opt(samebatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
# utils.plot_latent(x_umap_scdisinfact_opt, annos = np.array(["2. Ground truth"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + "predict_scdisinfact_opt(samebatch).png", figsize = (10,5), axis_label = "UMAP", markerscale = 6)


x_pca_scgen = np.log1p(np.concatenate([counts_gt, counts_scgen], axis = 0))
x_umap_scgen = UMAP(min_dist = 0.1, random_state = 0).fit_transform(x_pca_scgen)
np.save(file = result_dir + comment + "predict_scgen(same).npy", arr = x_umap_scgen)
x_umap_scgen = np.load(result_dir + comment + "predict_scgen(same).npy")
# utils.plot_latent(x_umap_scgen, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scgen(samebatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scgen, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), mode = "annos", save = result_dir + comment + "predict_scgen(samebatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scgen, annos = np.array(["2. Gold-standard"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + comment + "predict_scgen_batches(samebatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)

x_pca_scpregan = np.log1p(np.concatenate([counts_gt, counts_scpregan], axis = 0))
x_umap_scpregan = UMAP(min_dist = 0.1, random_state = 0).fit_transform(x_pca_scpregan)
np.save(file = result_dir + comment + "predict_scpregan(same).npy", arr = x_umap_scpregan)
x_umap_scpregan = np.load(result_dir + comment + "predict_scpregan(same).npy")
# utils.plot_latent(x_umap_scpregan, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scpregan(samebatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scpregan, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), mode = "annos", save = result_dir + comment + "predict_scpregan(samebatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scpregan, annos = np.array(["2. Gold-standard"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + comment + "predict_scpregan_batches(samebatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)

# In[]
# input matrix, considering two cases: 1. vehicle (DMSO) in the same batch; 2. vehicle (DMSO) in a different batch
# the second case
input_idx = ((meta_cells["treatment"] == "vehicle (DMSO)")  & (meta_cells["patient_id"] == "PW030")).values.squeeze()
counts_input = counts[input_idx,:].toarray()
meta_input = meta_cells.loc[input_idx,:]

counts_gt = counts[meta_cells["sample_id"] == "PW034-705",:].toarray()
meta_gt = meta_cells.loc[meta_cells["sample_id"] == "PW034-705",:]

# denoising
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["treatment"], 
                                          batch_key = "patient_id", predict_conds = None, predict_batch = None)
# perturbation prediction
counts_scdisinfact = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["treatment"], 
                                      batch_key = "patient_id", predict_conds = ["0.2 uM panobinostat"], predict_batch = "PW034")

counts_scgen = sp.load_npz("results_GBM_treatment/Fig4_patient/scGEN/diffbatch/counts_scgen.npz").toarray()
counts_scpregan = sp.load_npz("results_GBM_treatment/Fig4_patient/scPreGAN/diffbatch/counts_scpregan.npz").toarray()
counts_scgen = counts_scgen * (counts_scgen > 0)
counts_scpregan = counts_scpregan * (counts_scpregan > 0)

# normalization
counts_scdisinfact = counts_scdisinfact/(np.sum(counts_scdisinfact, axis = 1, keepdims = True) + 1e-6) * 100
counts_scgen = counts_scgen/(np.sum(counts_scgen, axis = 1, keepdims = True) + 1e-6) * 100
counts_scpregan = counts_scpregan/(np.sum(counts_scpregan, axis = 1, keepdims = True) + 1e-6) * 100
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6) * 100
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6) * 100
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6) * 100

# In[] Visualize the predicted counts


# no 1-1 match, check cell-type level scores
# unique_celltypes = np.unique(meta_gt["mstatus"].values)
unique_celltypes = ["Myeloid", "Oligodendrocytes", "tumor"]
mean_inputs = []
mean_scdisinfacts = []
mean_scgens = []
mean_scpregans = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_scdisinfact = np.mean(counts_scdisinfact[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_scgen = np.mean(counts_scgen[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_scpregan = np.mean(counts_scpregan[np.where(meta_input["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["mstatus"].values.squeeze() == celltype)[0],:], axis = 0)

    mean_inputs.append(mean_input)
    mean_scdisinfacts.append(mean_scdisinfact)
    mean_scgens.append(mean_scgen)
    mean_scpregans.append(mean_scpregan)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_scdisinfacts = np.array(mean_scdisinfacts)
mean_scgens = np.array(mean_scgens)
mean_scpregans = np.array(mean_scpregans)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_scdisinfacts - mean_gts_denoised) ** 2, axis = 1)
mses_scgens = np.sum((mean_scgens - mean_gts) ** 2, axis = 1)
mses_scpregans = np.sum((mean_scpregans - mean_gts) ** 2, axis = 1)

# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts[i,:])[0] for i in range(mean_gts.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_scdisinfacts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts.shape[0])])
pearsons_scgen = np.array([stats.pearsonr(mean_scgens[i,:], mean_gts[i,:])[0] for i in range(mean_gts.shape[0])])
pearsons_scpregan = np.array([stats.pearsonr(mean_scpregans[i,:], mean_gts[i,:])[0] for i in range(mean_gts.shape[0])])

# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts[i,:]) for i in range(mean_gts.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_scdisinfacts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts.shape[0])])
r2_scgen = np.array([r2_score(y_pred = mean_scgens[i,:], y_true = mean_gts[i,:]) for i in range(mean_gts.shape[0])])
r2_scpregan = np.array([r2_score(y_pred = mean_scpregans[i,:], y_true = mean_gts[i,:]) for i in range(mean_gts.shape[0])])

scores2 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE (Ratio)", "Pearson (Ratio)", "R2 (Ratio)", "Method", "Prediction"])
scores2["MSE"] = np.concatenate([mses_scdisinfact, mses_scgens, mses_scpregans], axis = 0)
scores2["MSE (Ratio)"] = np.concatenate([mses_scdisinfact/mses_input, mses_scgens/mses_input, mses_scpregans/mses_input], axis = 0)
scores2["Pearson"] = np.concatenate([pearsons_scdisinfact, pearsons_scgen, pearsons_scpregan], axis = 0)
scores2["Pearson (Ratio)"] = np.concatenate([pearsons_scdisinfact/pearsons_input, pearsons_scgen/pearsons_input, pearsons_scpregan/pearsons_input], axis = 0)
scores2["R2"] = np.concatenate([r2_scdisinfact, r2_scgen, r2_scpregan], axis = 0)
scores2["R2 (Ratio)"] = np.concatenate([r2_scdisinfact/r2_input, r2_scgen/r2_input, r2_scpregan/r2_input], axis = 0)
scores2["Method"] = ["scDisInFact"] * len(r2_scdisinfact) + ["scGen"] * len(r2_scgen) + ["scPreGAN"] * len(r2_scpregan)
scores2["Prediction"] = "treatment\n(w/ batch effect)"


# In[]
# PCA calculation will cost scdisinfact prediction to loss information
# x_pca_scdisinfact = PCA(n_components = 30).fit_transform(np.log1p(np.concatenate([counts_scdisinfact, counts_gt_denoised], axis = 0)))
x_pca_scdisinfact = np.log1p(np.concatenate([counts_gt_denoised, counts_scdisinfact], axis = 0))
x_umap_scdisinfact = UMAP(min_dist = 0.1, random_state = 0).fit_transform(x_pca_scdisinfact)
np.save(file = result_dir + comment + "predict_scdisinfact(diff).npy", arr = x_umap_scdisinfact)
x_umap_scdisinfact = np.load(result_dir + comment + "predict_scdisinfact(diff).npy")
# utils.plot_latent(x_umap_scdisinfact, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scdisinfact(diffbatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scdisinfact, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), mode = "annos", save = result_dir + comment + "predict_scdisinfact(diffbatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scdisinfact, annos = np.array(["2. Gold-standard"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + comment + "predict_scdisinfact_batches(diffbatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)

x_pca_scgen = np.log1p(np.concatenate([counts_gt, counts_scgen], axis = 0))
x_umap_scgen = UMAP(min_dist = 0.1, random_state = 0).fit_transform(x_pca_scgen)
np.save(file = result_dir + comment + "predict_scgen(diff).npy", arr = x_umap_scgen)
x_umap_scgen = np.load(result_dir + comment + "predict_scgen(diff).npy")
# utils.plot_latent(x_umap_scgen, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scgen(diffbatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scgen, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), mode = "annos", save = result_dir + comment + "predict_scgen(diffbatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scgen, annos = np.array(["2. Gold-standard"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + comment + "predict_scgen_batches(diffbatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)

x_pca_scpregan = np.log1p(np.concatenate([counts_gt, counts_scpregan], axis = 0))
x_umap_scpregan = UMAP(min_dist = 0.1, random_state = 0).fit_transform(x_pca_scpregan)
np.save(file = result_dir + comment + "predict_scpregan(diff).npy", arr = x_umap_scpregan)
x_umap_scpregan = np.load(result_dir + comment + "predict_scpregan(diff).npy")
# utils.plot_latent(x_umap_scpregan, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), batches = np.array(["Ground truth"] * meta_gt.shape[0] + ["Predict"] * meta_input.shape[0] ), mode = "separate", save = result_dir + "predict_sep_scpregan(diffbatch).png", figsize = (10,10), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scpregan, annos = pd.concat([meta_gt, meta_input], axis = 0)["mstatus"].values.squeeze(), mode = "annos", save = result_dir + comment + "predict_scpregan(diffbatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)
utils.plot_latent(x_umap_scpregan, annos = np.array(["2. Gold-standard"] * meta_gt.shape[0] + ["1. Predict"] * meta_input.shape[0]), mode = "annos", save = result_dir + comment + "predict_scpregan_batches(diffbatch).png", figsize = (8,5), axis_label = "UMAP", markerscale = 6)

# In[]
scores = pd.concat([scores1, scores2], axis = 0)
scores.to_csv(result_dir + comment + "scores_denoised.csv")


# In[]
import seaborn as sns
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (15,5), dpi = 500)
ax = fig.subplots(nrows = 1, ncols = 3)
scores = pd.read_csv(result_dir + comment + "scores_denoised.csv", index_col = 0)

sns.barplot(data = scores, x = "Prediction", hue = "Method", y = "MSE", ax = ax[0], capsize = 0.1)
sns.barplot(data = scores, x = "Prediction", hue = "Method", y = "Pearson", ax = ax[1], capsize = 0.1)
sns.barplot(data = scores, x = "Prediction", hue = "Method", y = "R2", ax = ax[2], capsize = 0.1)
# sns.barplot(data = scores, x = "Prediction", hue = "Method", y = "MSE (Ratio)", ax = ax[3], capsize = 0.1)
# sns.barplot(data = scores, x = "Prediction", hue = "Method", y = "Pearson (Ratio)", ax = ax[4], capsize = 0.1)
# sns.barplot(data = scores, x = "Prediction", hue = "Method", y = "R2 (Ratio)", ax = ax[5], capsize = 0.1)

handles, labels = ax[2].get_legend_handles_labels()

sns.stripplot(data = scores, x = "Prediction", hue = "Method", y = "MSE", ax = ax[0], color = "black", dodge = True)
sns.stripplot(data = scores, x = "Prediction", hue = "Method", y = "Pearson", ax = ax[1], color = "black", dodge = True)
sns.stripplot(data = scores, x = "Prediction", hue = "Method", y = "R2", ax = ax[2], color = "black", dodge = True)
# sns.stripplot(data = scores, x = "Prediction", hue = "Method", y = "MSE (Ratio)", ax = ax[3], color = "black", dodge = True)
# sns.stripplot(data = scores, x = "Prediction", hue = "Method", y = "Pearson (Ratio)", ax = ax[4], color = "black", dodge = True)
# sns.stripplot(data = scores, x = "Prediction", hue = "Method", y = "R2 (Ratio)", ax = ax[5], color = "black", dodge = True)

_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45)
_ = ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 45)
# _ = ax[3].set_xticklabels(ax[0].get_xticklabels(), rotation = 45)
# _ = ax[4].set_xticklabels(ax[1].get_xticklabels(), rotation = 45)
# _ = ax[5].set_xticklabels(ax[2].get_xticklabels(), rotation = 45)

ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].get_legend().remove()
# ax[3].get_legend().remove()
# ax[4].get_legend().remove()
# ax[5].get_legend().remove()
l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon = False)

ax[0].set_xlabel(None)
ax[1].set_xlabel(None)
ax[2].set_xlabel(None)
# ax[3].set_xlabel(None)
# ax[4].set_xlabel(None)
# ax[5].set_xlabel(None)

ax[0].ticklabel_format(axis = "y", style = "scientific", scilimits = (0,0))
ax[0].set_yscale("log")
# ax[1].set_yscale("log")
# ax[2].set_yscale("log")
# ax[3].set_yscale("log")
# ax[4].set_yscale("log")
# ax[5].set_yscale("log")
ax[2].set_ylim(0.0, 1.0)

fig.tight_layout()
fig.savefig(result_dir + comment + "scores_denoised.png", bbox_inches = "tight")




# %%
