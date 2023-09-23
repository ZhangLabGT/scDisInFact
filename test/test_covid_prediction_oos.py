
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
import seaborn as sns

from umap import UMAP
from sklearn.decomposition import PCA
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score 
import scipy.stats as stats

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
# Load the dataset, treat each dataset as a batch, as the authors of each data claim that there were minor batch effect
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_dir = "../data/covid_integrated/"
result_dir = "results_covid/new/prediction_oos/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

GxC1 = sparse.load_npz(data_dir + "GxC1.npz")
GxC2 = sparse.load_npz(data_dir + "GxC2.npz")
GxC3 = sparse.load_npz(data_dir + "GxC3.npz")
# be careful with the ordering
meta_c1 = pd.read_csv(data_dir + "meta_arunachalam_2020.txt", sep = "\t", index_col = 0)
meta_c2 = pd.read_csv(data_dir + "meta_lee_2020.txt", sep = "\t", index_col = 0)
meta_c3 = pd.read_csv(data_dir + "meta_wilk_2020.txt", sep = "\t", index_col = 0)

meta = pd.concat([meta_c1, meta_c2, meta_c3], axis = 0)
genes = pd.read_csv(data_dir + "genes_shared.txt", index_col = 0, sep = "\t").values.squeeze()
# process age
age = meta.age.values.squeeze().astype(object)
age[meta["age"] < 40] = "40-"
age[(meta["age"] >= 40)&(meta["age"] < 65)] = "40-65"
age[meta["age"] >= 65] = "65+"
meta["age"] = age

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

# no mmd batches
data_dict = scdisinfact.create_scdisinfact_dataset(counts_array, meta_cells_array, 
                                                   condition_key = ["disease_severity", "age"], 
                                                   batch_key = "dataset")
                    
# construct training and testing data
np.random.seed(0)
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta_cells in zip(data_dict["datasets"], data_dict["meta_cells"]):
    # test case: X52, moderate, 40-65, lee_2020
    test_idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] == "lee_2020")).values
    # remove all matrices corresponding to condition: moderate, 40-65
    train_idx = ~((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-65")).values

    if np.sum(train_idx) > 0:
        dataset_train = scdisinfact.scdisinfact_dataset(counts = dataset[train_idx]["counts"], 
                                                        counts_norm = dataset[train_idx]["counts_norm"],
                                                        size_factor = dataset[train_idx]["size_factor"],
                                                        diff_labels = dataset[train_idx]["diff_labels"],
                                                        batch_id = dataset[train_idx]["batch_id"],
                                                        mmd_batch_id = dataset[train_idx]["mmd_batch_id"]
                                                        )
        datasets_array_train.append(dataset_train)
        meta_cells_array_train.append(meta_cells.iloc[train_idx,:])
    
    if np.sum(test_idx) > 0:
        dataset_test = scdisinfact.scdisinfact_dataset(counts = dataset[test_idx]["counts"], 
                                                    counts_norm = dataset[test_idx]["counts_norm"],
                                                    size_factor = dataset[test_idx]["size_factor"],
                                                    diff_labels = dataset[test_idx]["diff_labels"],
                                                    batch_id = dataset[test_idx]["batch_id"],
                                                    mmd_batch_id = dataset[test_idx]["mmd_batch_id"]
                                                    )
        datasets_array_test.append(dataset_test)
        meta_cells_array_test.append(meta_cells.iloc[test_idx,:])

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}


# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # reference setting
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
Ks = [8, 2, 2]

batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
_ = model.eval()
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

# In[] Plot results
z_cs = []
z_ds = []

for dataset in data_dict_train["datasets"]:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        mu = dict_gen["mu"]    
        z_ds.append([x.cpu().detach().numpy() for x in z_d])
        z_cs.append(z_c.cpu().detach().numpy())

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
pca_op = PCA(n_components = 2)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))

comment = f'results_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l1.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_dims1_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "batches", axis_label = "PCA", figsize = (12,7), save = (result_dir + comment+"diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_dims2_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (10,7), save = (result_dir + comment+"diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "PCA", figsize = (12,7), save = (result_dir + comment+"diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]
# NOTE: note scenario 1 and 2 can also be achieved with 1 scgen/scpregan model too. The training would be 
# scenario 1: traing with data from healthy 40-, healthy 40-65. Input moderate 40-, predict moderate 40-65 (make it easy, of data from lee_2020)
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Test data of condition: moderate, 40-65, lee_2020. Input data of condition: moderate, 40-, lee_2020. Training data: except moderate, 40-65, lee_2020.")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-") & (meta_cells["dataset"] == "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)
mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores1 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores1["MSE"] = mses_scdisinfact
scores1["MSE input"] = mses_input
scores1["Pearson"] = pearsons_scdisinfact
scores1["Pearson input"] = pearsons_input
scores1["R2"] = r2_scdisinfact
scores1["R2 input"] = r2_input
scores1["Method"] = "scDisInFact"
scores1["Prediction"] = "Age\n(w/o batch effect)"


# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 2. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 40-65, lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
# scenario 2: traing with data from healthy 40-, moderate 40-. Input healthy 40-65, predict moderate 40-65 (make it easy, of data from lee_2020)
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] == "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores2 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores2["MSE"] = mses_scdisinfact
scores2["MSE input"] = mses_input
scores2["Pearson"] = pearsons_scdisinfact
scores2["Pearson input"] = pearsons_input
scores2["R2"] = r2_scdisinfact
scores2["R2 input"] = r2_input
scores2["Method"] = "scDisInFact"
scores2["Prediction"] = "Severity\n(w/o batch effect)"


# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 3. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 65+, lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
# scenario 3: for scgen/scpregan, need a cascade of two model, one predict disease effect, another predict age effect. 
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "65+") & (meta_cells["dataset"] == "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores3 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores3["MSE"] = mses_scdisinfact
scores3["MSE input"] = mses_input
scores3["Pearson"] = pearsons_scdisinfact
scores3["Pearson input"] = pearsons_input
scores3["R2"] = r2_scdisinfact
scores3["R2 input"] = r2_input
scores3["Method"] = "scDisInFact"
scores3["Prediction"] = "Age & Severity\n(w/o batch effect)"


# In[]
# NOTE: include batch effect
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 4. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: moderate, 40-, !lee_2020 (=wilk_2020)")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-") & (meta_cells["dataset"] != "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)
mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores4 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores4["MSE"] = mses_scdisinfact
scores4["MSE input"] = mses_input
scores4["Pearson"] = pearsons_scdisinfact
scores4["Pearson input"] = pearsons_input
scores4["R2"] = r2_scdisinfact
scores4["R2 input"] = r2_input
scores4["Method"] = "scDisInFact"
scores4["Prediction"] = "Age\n(w/ batch effect)"

# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 5. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 40-65, !lee_2020 (=wilk_2020)")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] != "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)
mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores5 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores5["MSE"] = mses_scdisinfact
scores5["MSE input"] = mses_input
scores5["Pearson"] = pearsons_scdisinfact
scores5["Pearson input"] = pearsons_input
scores5["R2"] = r2_scdisinfact
scores5["R2 input"] = r2_input
scores5["Method"] = "scDisInFact"
scores5["Prediction"] = "Severity\n(w/ batch effect)"


# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 6. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 65+, !lee_2020 (=Arunachalam_2020)")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "65+") & (meta_cells["dataset"] != "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores6 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores6["MSE"] = mses_scdisinfact
scores6["MSE input"] = mses_input
scores6["Pearson"] = pearsons_scdisinfact
scores6["Pearson input"] = pearsons_input
scores6["R2"] = r2_scdisinfact
scores6["R2 input"] = r2_input
scores6["Method"] = "scDisInFact"
scores6["Prediction"] = "Age & Severity\n(w/ batch effect)"

# In[]
scores = pd.concat([scores1, scores2, scores3, scores4, scores5, scores6], axis = 0)
scores.to_csv(result_dir + comment + "scores.csv")

# de_pvals = pd.concat([pvals_scores1, pvals_scores2, pvals_scores3, pvals_scores4, pvals_scores5, pvals_scores6], axis = 0)
# de_pvals.to_csv(result_dir + "de_pvals.csv")

# In[]
# Drop-out does improve the performance
scores_scdisinfact = pd.read_csv(result_dir + comment + "scores.csv", index_col = 0)
scores_scgen = pd.read_csv("results_covid/scGEN_oos/scores_scgen.csv", index_col = 0)
scores_scpregan = pd.read_csv("results_covid/scPreGAN_oos/scores_scpregan.csv", index_col = 0)
scores = pd.concat([scores_scdisinfact, scores_scgen, scores_scpregan], axis = 0)
scores.loc[scores["Method"] == "scGEN", "Method"] = "scGen"

scores1 = scores.loc[(scores["Prediction"] == "Severity\n(w/o batch effect)") | (scores["Prediction"] == "Age\n(w/o batch effect)") | (scores["Prediction"] == "Age & Severity\n(w/o batch effect)"),:]
scores1.loc[scores1["Prediction"] == "Severity\n(w/o batch effect)", "Prediction"] = "Severity"
scores1.loc[scores1["Prediction"] == "Age\n(w/o batch effect)", "Prediction"] = "Age"
scores1.loc[scores1["Prediction"] == "Age & Severity\n(w/o batch effect)", "Prediction"] = "Age & Severity"

scores2 = scores.loc[(scores["Prediction"] == "Severity\n(w/ batch effect)") | (scores["Prediction"] == "Age\n(w/ batch effect)") | (scores["Prediction"] == "Age & Severity\n(w/ batch effect)"),:]
scores2.loc[scores2["Prediction"] == "Severity\n(w/ batch effect)", "Prediction"] = "Severity"
scores2.loc[scores2["Prediction"] == "Age\n(w/ batch effect)", "Prediction"] = "Age"
scores2.loc[scores2["Prediction"] == "Age & Severity\n(w/ batch effect)", "Prediction"] = "Age & Severity"

import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize = (20,5), dpi = 200)
ax = fig.subplots(nrows = 1, ncols = 3)
sns.barplot(data = scores1, x = "Prediction", hue = "Method", y = "MSE", ax = ax[0], width = 0.5, capsize = 0.1)
sns.barplot(data = scores1, x = "Prediction", hue = "Method", y = "Pearson", ax = ax[1], width = 0.5, capsize = 0.1)
sns.barplot(data = scores1, x = "Prediction", hue = "Method", y = "R2", ax = ax[2], width = 0.5, capsize = 0.1)
fig.tight_layout()
_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45)
_ = ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 45)
ax[0].get_legend().remove()
ax[1].get_legend().remove()
leg = ax[2].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6)
ax[0].set_xlabel(None)
ax[1].set_xlabel(None)
ax[2].set_xlabel(None)
ax[0].set_ylabel("MSE", fontsize = 25)
ax[1].set_ylabel("Pearson", fontsize = 25)
ax[2].set_ylabel("R2", fontsize = 25)
# ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
# ax[1].yaxis.set_major_locator(plt.MaxNLocator(6))
# ax[2].yaxis.set_major_locator(plt.MaxNLocator(6))
ax[0].ticklabel_format(axis = "y", style = "scientific", scilimits = (0,0))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# remove some outliers
ax[0].set_ylim(0, 0.006)
ax[1].set_ylim(-0.0, 1.1)
ax[2].set_ylim(-0.0, 1.1)
fig.savefig(result_dir + comment + "scores_wo_batcheffect.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20,5), dpi = 200)
ax = fig.subplots(nrows = 1, ncols = 3)
sns.barplot(data = scores2, x = "Prediction", hue = "Method", y = "MSE", ax = ax[0], width = 0.5, capsize = 0.1)
sns.barplot(data = scores2, x = "Prediction", hue = "Method", y = "Pearson", ax = ax[1], width = 0.5, capsize = 0.1)
sns.barplot(data = scores2, x = "Prediction", hue = "Method", y = "R2", ax = ax[2], width = 0.5, capsize = 0.1)
fig.tight_layout()
_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45)
_ = ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 45)
ax[0].get_legend().remove()
ax[1].get_legend().remove()
leg = ax[2].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, edgecolor="black")

ax[0].set_xlabel(None)
ax[1].set_xlabel(None)
ax[2].set_xlabel(None)
ax[0].set_ylabel("MSE", fontsize = 25)
ax[1].set_ylabel("Pearson", fontsize = 25)
ax[2].set_ylabel("R2", fontsize = 25)
# ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
# ax[1].yaxis.set_major_locator(plt.MaxNLocator(6))
# ax[2].yaxis.set_major_locator(plt.MaxNLocator(6))
ax[0].ticklabel_format(axis = "y", style = "scientific", scilimits = (0,0))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# remove some outliers
ax[0].set_ylim(0, 0.006)
ax[1].set_ylim(-0.0, 1.1)
ax[2].set_ylim(-0.0, 1.1)
fig.savefig(result_dir + comment + "scores_w_batcheffect.png", bbox_inches = "tight")




# %%
