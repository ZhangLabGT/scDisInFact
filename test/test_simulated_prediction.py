# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd

sys.path.append("../src")
import scdisinfact
import utils
import bmk
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 

# In[]
sigma = 0.4
n_diff_genes = 100
diff = 8
ngenes = 500
ncells_total = 10000 
n_batches = 2
data_dir = f"../data/simulated/unif/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
result_dir = f"./results_simulated/prediction/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: randomly remove some celltypes?
counts_gt = []
counts_ctrl_healthy = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_severe = []
# cell types
label_annos = []

for batch_id in range(n_batches):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
    counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
    counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
    counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))    

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# In-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")
np.random.seed(0)
counts_gt_test = []
counts_test = []
meta_cells = []
for batch_id in range(n_batches):
    # generate permutation
    permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
    # since there are totally four combinations of conditions, separate the cells into four groups
    chuck_size = int(counts_gt[batch_id].shape[0]/4)
    counts_gt_test.append(counts_gt[batch_id][permute_idx,:])

    counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                       counts_ctrl_severe[batch_id][permute_idx[chuck_size:(2*chuck_size)],:], 
                                       counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                       counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))

    
    meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
    meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
    meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["severe"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
    meta_cell["annos"] = label_annos[batch_id][permute_idx]
    meta_cells.append(meta_cell)

data_dict_full = scdisinfact.create_scdisinfact_dataset(counts_test, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")


# In[]
# # check the visualization before integration
# umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 
# counts = np.concatenate([x.counts for x in data_dict_full["datasets"]], axis = 0)
# counts_norm = counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6) * 100
# counts_norm = np.log1p(counts_norm)
# x_umap = umap_op.fit_transform(counts_norm)

# utils.plot_latent(x_umap, annos = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), mode = "annos", save = result_dir + "batches.png", figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 2)

# utils.plot_latent(x_umap, annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_full["meta_cells"]]), mode = "annos", save = result_dir + "conditions1.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

# utils.plot_latent(x_umap, annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_full["meta_cells"]]), mode = "annos", save = result_dir + "conditions2.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 2)

# utils.plot_latent(x_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), mode = "separate", save = result_dir + "annos.png", figsize = (10, 10), axis_label = "UMAP", markerscale = 6, s = 2, label_inplace = False, text_size = "small")



# In[] training the model
# TODO: track the time usage and memory usage
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
Ks = [8, 4, 4]

batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict_full, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)
# torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth", map_location = device))
_ = model.eval()

# In[] Plot results
# z_cs = []
# z_ds = []
# zs = []

# for dataset in data_dict_full["datasets"]:
#     with torch.no_grad():
#         # pass through the encoders
#         dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
#         # pass through the decoder
#         dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
#         z_c = dict_inf["mu_c"]
#         z_d = dict_inf["mu_d"]
#         z = torch.cat([z_c] + z_d, dim = 1)
#         mu = dict_gen["mu"]    
#         z_ds.append([x.cpu().detach().numpy() for x in z_d])
#         z_cs.append(z_c.cpu().detach().numpy())
#         zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# # UMAP
# umap_op = UMAP(min_dist = 0.1, random_state = 0)
# pca_op = PCA(n_components = 2)
# z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
# z_ds_umap = []
# z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
# z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
# zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))


# comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
# if not os.path.exists(result_dir + comment):
#     os.makedirs(result_dir + comment)


# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_annos.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "separate", axis_label = "UMAP", figsize = (10,10), save = (result_dir + comment+"common_dims_annos_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

# utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
# utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), batches = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
#     mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Training: all input matrices. Predict condition: (ctrl, severe, batch 0), input condition: (stim, severe, batch 0).")
print("#")
print("# -------------------------------------------------------------------------------------------")

counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# ground truth (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
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
scores1["Prediction"] = "Condition 1\n(w/o batch effect)"


# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 2. Training: all input matrices. Predict condition: (ctrl, severe, batch 0), input condition: (ctrl, healthy, batch 0).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 0)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# ground truth (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
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
scores2["Prediction"] = "Condition 2\n(w/o batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 3. Training: all input matrices. Predict condition: (ctrl, severe, batch 0), input condition: (stim, healthy, batch 0).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 0)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# ground truth (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
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

scores3 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores3["MSE"] = mses_scdisinfact
scores3["MSE input"] = mses_input
scores3["Pearson"] = pearsons_scdisinfact
scores3["Pearson input"] = pearsons_input
scores3["R2"] = r2_scdisinfact
scores3["R2 input"] = r2_input
scores3["Method"] = "scDisInFact"
scores3["Prediction"] = "Condition 1&2\n(w/o batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 4. Training: all input matrices. Predict condition: (ctrl, severe, batch 0), input condition: (stim, severe, batch 1).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 1)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# ground truth (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
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
scores4["Prediction"] = "Condition 1\n(w/ batch effect)"


# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 5. Training: all input matrices. Predict condition: (ctrl, severe, batch 0), input condition: (stim, healthy, batch 1).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 1)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# ground truth (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
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
scores5["Prediction"] = "Condition 2\n(w/ batch effect)"
# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 6. Training: all input matrices. Predict condition: (ctrl, severe, batch 0), input condition: (stim, healthy, batch 1).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 1)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# ground truth (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
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
scores6["Prediction"] = "Condition 1&2\n(w/ batch effect)"

# In[]
scores = pd.concat([scores1, scores2, scores3, scores4, scores5, scores6], axis = 0)
scores.to_csv(result_dir + "scores_full.csv")


# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# Out-of-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_gt = []
counts_ctrl_healthy = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_severe = []
# cell types
label_annos = []

for batch_id in range(n_batches):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
    counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
    counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
    counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))  

# NOTE: select counts for each batch
np.random.seed(0)
counts_gt_test = []
counts_test = []
meta_cells = []

for batch_id in range(n_batches):
    # generate permutation
    permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
    # since there are totally four combinations of conditions, separate the cells into four groups
    chuck_size = int(counts_gt[batch_id].shape[0]/4)

    if batch_id == 0:
        counts_gt_test.append(np.concatenate([counts_gt[batch_id][permute_idx[:chuck_size],:], 
                                        counts_gt[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                        counts_gt[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
        # remove (ctrl, severe, batch 0)
        counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                        counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                        counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))

        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta_cell["batch"] = np.array([batch_id] * counts_gt_test[batch_id].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                        label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                        label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
        meta_cells.append(meta_cell)
        
    else:
        counts_gt_test.append(np.concatenate([counts_gt[batch_id][permute_idx[:chuck_size],:], 
                                        counts_gt[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                        counts_gt[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))

        # remove (ctrl, severe, batch 1), don't want to see the count corresponding to condition ctrl & severe
        counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                        counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                        counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
        
        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta_cell["batch"] = np.array([batch_id] * counts_gt_test[batch_id].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                        label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                        label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
        meta_cells.append(meta_cell)

data_dict = scdisinfact.create_scdisinfact_dataset(counts_test, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = 64, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

# train_joint is more efficient, but does not work as well compared to train
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)
torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_oos.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}_oos.pth", map_location = device))
_ = model.eval()

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Training: all input matrices except (ctrl, severe, batch 0). Predict condition: (ctrl, severe, batch 0), input condition: (stim, severe, batch 0).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    
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
scores1["Prediction"] = "Condition 1\n(w/o batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 2. Training: all input matrices except (ctrl, severe, batch 0). Predict condition: (ctrl, severe, batch 0), input condition: (ctrl, healthy, batch 0).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
# input (ctrl, healthy, batch 0)
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 0)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# predict (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    
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
scores2["Prediction"] = "Condition 2\n(w/o batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 3. Training: all input matrices except (ctrl, severe, batch 0). Predict condition: (ctrl, severe, batch 0), input condition: (stim, healthy, batch 0).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
# input (ctrl, healthy, batch 0)
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 0)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# predict (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    
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

scores3 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores3["MSE"] = mses_scdisinfact
scores3["MSE input"] = mses_input
scores3["Pearson"] = pearsons_scdisinfact
scores3["Pearson input"] = pearsons_input
scores3["R2"] = r2_scdisinfact
scores3["R2 input"] = r2_input
scores3["Method"] = "scDisInFact"
scores3["Prediction"] = "Condition 1&2\n(w/o batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 4. Training: all input matrices except (ctrl, severe, batch 0). Predict condition: (ctrl, severe, batch 0), input condition: (stim, severe, batch 1).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
# input (ctrl, healthy, batch 1)
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 1)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# predict (stim, healthy, batch 1)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)

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
scores4["Prediction"] = "Condition 1\n(w/ batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 5. Training: all input matrices except (ctrl, severe, batch 0). Predict condition: (ctrl, severe, batch 0), input condition: (ctrl, healthy, batch 1).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
# input (ctrl, healthy, batch 1)
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 1)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# predict (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)

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
scores5["Prediction"] = "Condition 2\n(w/ batch effect)"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 5. Training: all input matrices except (ctrl, severe, batch 0). Predict condition: (ctrl, severe, batch 0), input condition: (stim, healthy, batch 1).")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
# input (ctrl, healthy, batch 1)
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 1)).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                      batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])

# predict (ctrl, severe, batch 0)
counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    counts_gt.append(dataset.counts[idx,:].numpy())
    meta_gt.append(meta_cells.loc[idx,:])

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                          batch_key = "batch", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["annos"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["annos"].values.squeeze() == celltype)[0],:], axis = 0)

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
scores6["Prediction"] = "Condition 1&2\n(w/ batch effect)"

# In[]
scores = pd.concat([scores1, scores2, scores3, scores4, scores5, scores6], axis = 0)
scores.to_csv(result_dir + f"scores_oos.csv")

# In[]
# -------------------------------------------------------------------------------------------------
#
# Test 1
#
# -------------------------------------------------------------------------------------------------
scores_all = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction", "training", "MSE (ratio)", "Pearson (ratio)", "R2 (ratio)"])
for n_diff_genes in [20, 50, 100]:
    for diff in [2, 4, 8]:
        scdisinfact_dir = f"./results_simulated/prediction/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
        scores_scdisinfact_is = pd.read_csv(scdisinfact_dir + "scores_full.csv", index_col = 0)
        scores_scdisinfact_oos = pd.read_csv(scdisinfact_dir + "scores_oos.csv", index_col = 0)
        scores_scdisinfact_is["training"] = "is"
        scores_scdisinfact_oos["training"] = "oos"
        
        scgen_dir = f"./results_simulated/scgen/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
        scores_scgen_is = pd.read_csv(scgen_dir + "scores_full.csv", index_col = 0)
        scores_scgen_oos = pd.read_csv(scgen_dir + "scores_oos.csv", index_col = 0)
        scores_scgen_is["training"] = "is"
        scores_scgen_oos["training"] = "oos"

        scpregan_dir = f"./results_simulated/scpregan/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
        scores_scpregan_is = pd.read_csv(scpregan_dir + "scores_full.csv", index_col = 0)
        scores_scpregan_oos = pd.read_csv(scpregan_dir + "scores_oos.csv", index_col = 0)
        scores_scpregan_is["training"] = "is"
        scores_scpregan_oos["training"] = "oos"
        # scores_scpregan_is["Method"] = "scPreGAN"
        # scores_scpregan_oos["Method"] = "scPreGAN"

        scores = pd.concat([scores_scdisinfact_is, scores_scdisinfact_oos, scores_scgen_is, scores_scgen_oos, scores_scpregan_is, scores_scpregan_oos], axis = 0)

        # scores.loc[scores["Prediction"] == "condition effect (condition 1)", "Prediction"] = "Condition 1\n(w/o batch effect)"
        # scores.loc[scores["Prediction"] == "condition effect (condition 2)", "Prediction"] = "Condition 2\n(w/o batch effect)"
        # scores.loc[scores["Prediction"] == "condition effect (condition 1) + batch effect", "Prediction"] = "Condition 1\n(w/ batch effect)"
        # scores.loc[scores["Prediction"] == "condition effect (condition 2) + batch effect", "Prediction"] = "Condition 2\n(w/ batch effect)"

        scores["MSE (ratio)"] = scores["MSE"].values/scores["MSE input"]
        scores["Pearson (ratio)"] = scores["Pearson"].values/scores["Pearson input"]
        scores["R2 (ratio)"] = scores["R2"].values/scores["R2 input"]

        scores_all = pd.concat([scores_all, scores], axis = 0)

scores_is = scores_all[scores_all["training"] == "is"]
scores_oos = scores_all[scores_all["training"] == "oos"]



# In[]
scores1 = scores_is.loc[(scores_is["Prediction"] == "Condition 1\n(w/o batch effect)") | (scores_is["Prediction"] == "Condition 2\n(w/o batch effect)") | (scores_is["Prediction"] == "Condition 1&2\n(w/o batch effect)"),:]
scores1.loc[scores1["Prediction"] == "Condition 1\n(w/o batch effect)", "Prediction"] = "Treatment"
scores1.loc[scores1["Prediction"] == "Condition 2\n(w/o batch effect)", "Prediction"] = "Severity"
scores1.loc[scores1["Prediction"] == "Condition 1&2\n(w/o batch effect)", "Prediction"] = "Treatment\n& Severity"

scores2 = scores_is.loc[(scores_is["Prediction"] == "Condition 1\n(w/ batch effect)") | (scores_is["Prediction"] == "Condition 2\n(w/ batch effect)") | (scores_is["Prediction"] == "Condition 1&2\n(w/ batch effect)"),:]
scores2.loc[scores2["Prediction"] == "Condition 1\n(w/ batch effect)", "Prediction"] = "Treatment"
scores2.loc[scores2["Prediction"] == "Condition 2\n(w/ batch effect)", "Prediction"] = "Severity"
scores2.loc[scores2["Prediction"] == "Condition 1&2\n(w/ batch effect)", "Prediction"] = "Treatment\n& Severity"

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
ax[0].set_yscale("log")
ax[0].set_ylim(10**-5, 10**-1.5)
ax[1].set_ylim(0.5, 1.1)
ax[2].set_ylim(0.5, 1.1)
fig.savefig("results_simulated/prediction/scores_is_wo_batcheffect.png", bbox_inches = "tight")

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
ax[0].set_yscale("log")
ax[0].set_ylim(10**-5, 10**-1.5)
ax[1].set_ylim(0.5, 1.1)
ax[2].set_ylim(0.5, 1.1)

fig.savefig("results_simulated/prediction/scores_is_w_batcheffect.png", bbox_inches = "tight")

# In[]
scores1 = scores_oos.loc[(scores_oos["Prediction"] == "Condition 1\n(w/o batch effect)") | (scores_oos["Prediction"] == "Condition 2\n(w/o batch effect)") | (scores_oos["Prediction"] == "Condition 1&2\n(w/o batch effect)"),:]
scores1.loc[scores1["Prediction"] == "Condition 1\n(w/o batch effect)", "Prediction"] = "Treatment"
scores1.loc[scores1["Prediction"] == "Condition 2\n(w/o batch effect)", "Prediction"] = "Severity"
scores1.loc[scores1["Prediction"] == "Condition 1&2\n(w/o batch effect)", "Prediction"] = "Treatment\n& Severity"

scores2 = scores_oos.loc[(scores_oos["Prediction"] == "Condition 1\n(w/ batch effect)") | (scores_oos["Prediction"] == "Condition 2\n(w/ batch effect)") | (scores_oos["Prediction"] == "Condition 1&2\n(w/ batch effect)"),:]
scores2.loc[scores2["Prediction"] == "Condition 1\n(w/ batch effect)", "Prediction"] = "Treatment"
scores2.loc[scores2["Prediction"] == "Condition 2\n(w/ batch effect)", "Prediction"] = "Severity"
scores2.loc[scores2["Prediction"] == "Condition 1&2\n(w/ batch effect)", "Prediction"] = "Treatment\n& Severity"

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
ax[0].set_yscale("log")
ax[0].set_ylim(10**-5, 10**-1.5)
ax[1].set_ylim(0.5, 1.1)
ax[2].set_ylim(0.5, 1.1)
fig.savefig("results_simulated/prediction/scores_oos_wo_batcheffect.png", bbox_inches = "tight")

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
ax[0].set_yscale("log")
ax[0].set_ylim(10**-5, 10**-1.5)
ax[1].set_ylim(0.5, 1.1)
ax[2].set_ylim(0.5, 1.1)
fig.savefig("results_simulated/prediction/scores_oos_w_batcheffect.png", bbox_inches = "tight")

# In[]
# import seaborn as sns
# plt.rcParams["font.size"] = 15
# fig = plt.figure(figsize = (34,5))
# ax = fig.subplots(nrows = 1, ncols = 6)
# sns.boxplot(data = scores_is, x = "Prediction", hue = "Method", y = "MSE", ax = ax[0])
# sns.boxplot(data = scores_is, x = "Prediction", hue = "Method", y = "Pearson", ax = ax[1])
# sns.boxplot(data = scores_is, x = "Prediction", hue = "Method", y = "R2", ax = ax[2])

# graph = sns.boxplot(data = scores_is, x = "Prediction", hue = "Method", y = "MSE (ratio)", ax = ax[3])
# graph.axhline(1, ls = "--")
# graph = sns.boxplot(data = scores_is, x = "Prediction", hue = "Method", y = "Pearson (ratio)", ax = ax[4])
# graph.axhline(1, ls = "--")
# graph = sns.boxplot(data = scores_is, x = "Prediction", hue = "Method", y = "R2 (ratio)", ax = ax[5])
# graph.axhline(1, ls = "--")

# _ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45)
# _ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45)
# _ = ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 45)
# _ = ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation = 45)
# _ = ax[4].set_xticklabels(ax[4].get_xticklabels(), rotation = 45)
# _ = ax[5].set_xticklabels(ax[5].get_xticklabels(), rotation = 45)

# ax[0].get_legend().remove()
# ax[1].get_legend().remove()
# ax[2].get_legend().remove()
# ax[3].get_legend().remove()
# ax[4].get_legend().remove()
# ax[5].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6)

# ax[0].set_xlabel(None)
# ax[1].set_xlabel(None)
# ax[2].set_xlabel(None)
# ax[3].set_xlabel(None)
# ax[0].set_yscale("log")
# ax[3].set_yscale("log")
# ax[4].set_xlabel(None)
# ax[5].set_xlabel(None)
# ax[0].set_ylim(10e-7, 10e-2)
# ax[1].set_ylim(0.80, 1.03)
# ax[2].set_ylim(0.37, 1.03)
# fig.tight_layout()
# # ax[0].ticklabel_format(axis = "y", style = "scientific", scilimits = (0,0))
# fig.savefig("results_simulated/prediction/scores_is.png", bbox_inches = "tight")

# fig = plt.figure(figsize = (34,5))
# ax = fig.subplots(nrows = 1, ncols = 6)
# sns.boxplot(data = scores_oos, x = "Prediction", hue = "Method", y = "MSE", ax = ax[0])
# sns.boxplot(data = scores_oos, x = "Prediction", hue = "Method", y = "Pearson", ax = ax[1])
# sns.boxplot(data = scores_oos, x = "Prediction", hue = "Method", y = "R2", ax = ax[2])

# graph = sns.boxplot(data = scores_oos, x = "Prediction", hue = "Method", y = "MSE (ratio)", ax = ax[3])
# graph.axhline(1, ls = "--")
# graph = sns.boxplot(data = scores_oos, x = "Prediction", hue = "Method", y = "Pearson (ratio)", ax = ax[4])
# graph.axhline(1, ls = "--")
# graph = sns.boxplot(data = scores_oos, x = "Prediction", hue = "Method", y = "R2 (ratio)", ax = ax[5])
# graph.axhline(1, ls = "--")


# _ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45)
# _ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45)
# _ = ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 45)
# _ = ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation = 45)
# _ = ax[4].set_xticklabels(ax[4].get_xticklabels(), rotation = 45)
# _ = ax[5].set_xticklabels(ax[5].get_xticklabels(), rotation = 45)

# ax[0].get_legend().remove()
# ax[1].get_legend().remove()
# ax[2].get_legend().remove()
# ax[3].get_legend().remove()
# ax[4].get_legend().remove()
# ax[5].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6)

# ax[0].set_xlabel(None)
# ax[1].set_xlabel(None)
# ax[2].set_xlabel(None)
# ax[3].set_xlabel(None)
# ax[0].set_yscale("log")
# ax[3].set_yscale("log")
# ax[4].set_xlabel(None)
# ax[5].set_xlabel(None)
# ax[0].set_ylim(10e-7, 10e-2)
# ax[1].set_ylim(0.80, 1.03)
# ax[2].set_ylim(0.37, 1.03)
# fig.tight_layout()
# # ax[0].ticklabel_format(axis = "y", style = "scientific", scilimits = (0,0))
# fig.savefig("results_simulated/prediction/scores_oos.png", bbox_inches = "tight")

# %%
