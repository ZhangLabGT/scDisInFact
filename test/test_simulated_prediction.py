# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.bmk as bmk

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
counts_ctrl_healthy = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_severe = []
# cell types
label_annos = []

for batch_id in range(n_batches):
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
counts_train = []
metas_train = []
# counts_gt_train = []
# metas_gt_train = []
for batch_id in range(n_batches):
    ncells = counts_ctrl_severe[batch_id].shape[0]
    # generate permutation
    permute_idx = np.random.permutation(ncells)
    # since there are totally four combinations of conditions, separate the cells into four groups
    chunk_size = int(ncells/4)
    # original gt
    # counts_gt_train.append(np.concatenate([counts_ctrl_severe[batch_id][permute_idx[:chunk_size],:],
    #                                        counts_ctrl_severe[batch_id][permute_idx[chunk_size:(2*chunk_size)],:],
    #                                        counts_ctrl_severe[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:],
    #                                        counts_ctrl_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0))
    
    # meta_gt_train = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    # meta_gt_train["batch"] = np.array([batch_id] * ncells)
    # meta_gt_train["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["ctrl"] * (ncells - 3*chunk_size))
    # meta_gt_train["condition 2"] = np.array(["severe"] * chunk_size + ["severe"] * chunk_size + ["severe"] * chunk_size + ["severe"] * (ncells - 3*chunk_size))
    # meta_gt_train["annos"] = label_annos[batch_id][permute_idx]
    # metas_gt_train.append(meta_gt_train)

    counts_train.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:], 
                                       counts_ctrl_severe[batch_id][permute_idx[chunk_size:(2*chunk_size)],:], 
                                       counts_stim_healthy[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:], 
                                       counts_stim_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0))

    meta_train = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    meta_train["batch"] = np.array([batch_id] * ncells)
    meta_train["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["stim"] * chunk_size + ["stim"] * (ncells - 3*chunk_size))
    meta_train["condition 2"] = np.array(["healthy"] * chunk_size + ["severe"] * chunk_size + ["healthy"] * chunk_size + ["severe"] * (ncells - 3*chunk_size))
    meta_train["annos"] = label_annos[batch_id][permute_idx]
    metas_train.append(meta_train)

counts_train = np.concatenate(counts_train, axis = 0)
metas_train = pd.concat(metas_train, axis = 0)

counts_gt_train = counts_ctrl_severe[0]
metas_gt_train = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
metas_gt_train["batch"] = np.array([0] * counts_gt_train.shape[0])
metas_gt_train["condition 1"] = np.array(["ctrl"] * counts_gt_train.shape[0])
metas_gt_train["condition 2"] = np.array(["severe"] * counts_gt_train.shape[0])
metas_gt_train["annos"] = label_annos[0]
data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, metas_train, condition_key = ["condition 1", "condition 2"], batch_key = "batch")


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
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
Ks = [8, 2, 2]

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
# torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}.pth", map_location = device))

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
configs_input = [{"condition 1": "stim", "condition 2": "severe", "batch": 0, "type": "Condition 1\n(w/o batch effect)"},
                {"condition 1": "ctrl", "condition 2": "healthy", "batch": 0, "type": "Condition 2\n(w/o batch effect)"},
                {"condition 1": "stim", "condition 2": "healthy", "batch": 0, "type": "Condition 1&2\n(w/o batch effect)"},
                {"condition 1": "stim", "condition 2": "severe", "batch": 1, "type": "Condition 1\n(w/ batch effect)"},
                {"condition 1": "ctrl", "condition 2": "healthy", "batch": 1, "type": "Condition 2\n(w/ batch effect)"},
                {"condition 1": "stim", "condition 2": "healthy", "batch": 1, "type": "Condition 1&2\n(w/ batch effect)"}]

score_cluster_list = []
# score_list = []

for config in configs_input:
    print("input condition: " + str(config))

    # load input and gt count matrices
    idx = ((metas_train["condition 1"] == config["condition 1"]) & (metas_train["condition 2"] == config["condition 2"]) & (metas_train["batch"] == config["batch"])).values
    counts_input = counts_train[idx,:]
    meta_input = metas_train.loc[idx,:]
    counts_gt = counts_gt_train
    meta_gt = metas_gt_train

    # predict count
    counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                    batch_key = "batch", predict_conds = None, predict_batch = None)

    counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                        batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)


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

    score_cluster = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
    score_cluster["MSE"] = mses_scdisinfact
    score_cluster["MSE input"] = mses_input
    score_cluster["Pearson"] = pearsons_scdisinfact
    score_cluster["Pearson input"] = pearsons_input
    score_cluster["R2"] = r2_scdisinfact
    score_cluster["R2 input"] = r2_input
    score_cluster["Method"] = "scDisInFact"
    score_cluster["Prediction"] = config["type"]
    score_cluster_list.append(score_cluster)

    # cannot calculate cell level MSE across batches
    # mses_input = np.sum((counts_input - counts_gt_denoised) ** 2, axis = 1)
    # mses_scdisinfact = np.sum((counts_predict - counts_gt_denoised) ** 2, axis = 1)
    # # vector storing the pearson correlation for all cells
    # pearson_input = np.array([stats.pearsonr(counts_input[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
    # pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
    # # vector storing the R2 scores for all cells
    # r2_input = np.array([r2_score(y_pred = counts_input[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])
    # r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])

    # score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "AUPRC", "AUPRC (cell-level)", "Method", "Prediction"])
    # score["MSE"] = mses_scdisinfact
    # score["MSE input"] = mses_input
    # score["Pearson"] = pearsons_scdisinfact
    # score["Pearson input"] = pearson_input
    # score["R2"] = r2_scdisinfact
    # score["R2 input"] = r2_input
    # score["Method"] = "scDisInFact"
    # score["Prediction"] = config["type"]
    # score_list.append(score)

scores_cluster = pd.concat(score_cluster_list, axis = 0)
scores_cluster.to_csv(result_dir + "scores_cluster_full.csv")

# scores = pd.concat(score_list, axis = 0)
# scores.to_csv(result_dir + "scores_full.csv")

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# Out-of-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_ctrl_healthy = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_severe = []
# cell types
label_annos = []

for batch_id in range(n_batches):
    counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
    counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
    counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
    counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))  

# NOTE: select counts for each batch
np.random.seed(0)
counts_train = []
metas_train = []
# counts_gt_train = []
# metas_gt_train = []

for batch_id in range(n_batches):
    ncells = counts_ctrl_severe[batch_id].shape[0]
    # generate permutation
    permute_idx = np.random.permutation(ncells)
    # since there are totally four combinations of conditions, separate the cells into four groups
    chuck_size = int(ncells/4)

    # Training data: remove (ctrl, severe) for all batches
    counts_train.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                        counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                        counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
    meta_train = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    meta_train["batch"] = np.array([batch_id] * chuck_size + [batch_id] * chuck_size + [batch_id] * (ncells - 3*chuck_size))
    meta_train["condition 1"] = np.array(["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (ncells - 3*chuck_size))
    meta_train["condition 2"] = np.array(["healthy"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (ncells - 3*chuck_size))
    meta_train["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                            label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                            label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
    metas_train.append(meta_train)

    
    # # Ground truth dataset
    # counts_gt_train.append(np.concatenate([counts_ctrl_severe[batch_id][permute_idx[:chuck_size],:],
    #                                     counts_ctrl_severe[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:],
    #                                     counts_ctrl_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
    # meta_gt_train = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    # meta_gt_train["batch"] = np.array([batch_id] * chuck_size + [batch_id] * chuck_size + [batch_id] * (ncells - 3*chuck_size))
    # meta_gt_train["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["ctrl"] * (ncells - 3*chuck_size))
    # meta_gt_train["condition 2"] = np.array(["severe"] * chuck_size + ["severe"] * chuck_size + ["severe"] * (ncells - 3*chuck_size))
    # meta_gt_train["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
    #                                         label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
    #                                         label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
    # metas_gt_train.append(meta_gt_train)

counts_train = np.concatenate(counts_train, axis = 0)
metas_train = pd.concat(metas_train, axis = 0)
counts_gt_train = counts_ctrl_severe[0]
metas_gt_train = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
metas_gt_train["batch"] = np.array([0] * counts_gt_train.shape[0])
metas_gt_train["condition 1"] = np.array(["ctrl"] * counts_gt_train.shape[0])
metas_gt_train["condition 2"] = np.array(["severe"] * counts_gt_train.shape[0])
metas_gt_train["annos"] = label_annos[0]

data_dict = scdisinfact.create_scdisinfact_dataset(counts_train, metas_train, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

# train_joint is more efficient, but does not work as well compared to train
# model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
# _ = model.eval()
# torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_oos.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_oos.pth", map_location = device))


# In[]
configs_input = [{"condition 1": "stim", "condition 2": "severe", "batch": 0, "type": "Condition 1\n(w/o batch effect)"},
                {"condition 1": "ctrl", "condition 2": "healthy", "batch": 0, "type": "Condition 2\n(w/o batch effect)"},
                {"condition 1": "stim", "condition 2": "healthy", "batch": 0, "type": "Condition 1&2\n(w/o batch effect)"},
                {"condition 1": "stim", "condition 2": "severe", "batch": 1, "type": "Condition 1\n(w/ batch effect)"},
                {"condition 1": "ctrl", "condition 2": "healthy", "batch": 1, "type": "Condition 2\n(w/ batch effect)"},
                {"condition 1": "stim", "condition 2": "healthy", "batch": 1, "type": "Condition 1&2\n(w/ batch effect)"}]

score_cluster_list = []
score_list = []

for config in configs_input:
    print("input condition: " + str(config))

    idx = ((metas_train["condition 1"] == config["condition 1"]) & (metas_train["condition 2"] == config["condition 2"]) & (metas_train["batch"] == config["batch"])).values
    counts_input = counts_train[idx,:]
    meta_input = metas_train.loc[idx,:]
    counts_gt = counts_gt_train
    meta_gt = metas_gt_train

    counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                    batch_key = "batch", predict_conds = None, predict_batch = None)

    counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                        batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)

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

    score_cluster = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
    score_cluster["MSE"] = mses_scdisinfact
    score_cluster["MSE input"] = mses_input
    score_cluster["Pearson"] = pearsons_scdisinfact
    score_cluster["Pearson input"] = pearsons_input
    score_cluster["R2"] = r2_scdisinfact
    score_cluster["R2 input"] = r2_input
    score_cluster["Method"] = "scDisInFact"
    score_cluster["Prediction"] = config["type"]
    score_cluster_list.append(score_cluster)

    # cannot calculate cell level MSE
    # mses_input = np.sum((counts_input - counts_gt_denoised) ** 2, axis = 1)
    # mses_scdisinfact = np.sum((counts_predict - counts_gt_denoised) ** 2, axis = 1)
    # # vector storing the pearson correlation for all cells
    # pearson_input = np.array([stats.pearsonr(counts_input[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
    # pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
    # # vector storing the R2 scores for all cells
    # r2_input = np.array([r2_score(y_pred = counts_input[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])
    # r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])

    # score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "AUPRC", "AUPRC (cell-level)", "Method", "Prediction"])
    # score["MSE"] = mses_scdisinfact
    # score["MSE input"] = mses_input
    # score["Pearson"] = pearsons_scdisinfact
    # score["Pearson input"] = pearson_input
    # score["R2"] = r2_scdisinfact
    # score["R2 input"] = r2_input
    # score["Method"] = "scDisInFact"
    # score["Prediction"] = config["type"]
    # score_list.append(score)

scores_cluster = pd.concat(score_cluster_list, axis = 0)
scores_cluster.to_csv(result_dir + "scores_cluster_oos.csv")

# scores = pd.concat(score_list, axis = 0)
# scores.to_csv(result_dir + "scores_oos.csv")


# In[]
# -------------------------------------------------------------------------------------------------
#
# Test 1
#
# -------------------------------------------------------------------------------------------------
ngenes = 500
ncells_total = 10000 
sigma = 0.4
scores_all = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction", "training", "MSE (ratio)", "Pearson (ratio)", "R2 (ratio)"])
for n_diff_genes in [20, 50, 100]:
    for diff in [2, 4, 8]:
        scdisinfact_dir = f"./results_simulated/prediction/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
        scores_scdisinfact_is = pd.read_csv(scdisinfact_dir + "scores_cluster_full.csv", index_col = 0)
        scores_scdisinfact_oos = pd.read_csv(scdisinfact_dir + "scores_cluster_oos.csv", index_col = 0)
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
