# In[]
import sys, os
import time
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.bmk as bmk
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import r2_score 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [8, 2, 2]

lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4

# Read in the dataset
n_batches = 2

simulated_lists = [
  "2conds_base_10000_500_0.4_20_2",
  "2conds_base_10000_500_0.4_20_4",
  "2conds_base_10000_500_0.4_20_8",
  "2conds_base_10000_500_0.4_50_2",
  "2conds_base_10000_500_0.4_50_4",
  "2conds_base_10000_500_0.4_50_8",
  "2conds_base_10000_500_0.4_100_2",
  "2conds_base_10000_500_0.4_100_4",
  "2conds_base_10000_500_0.4_100_8",
 ]

# In[]
for dataset_dir in simulated_lists:
    print("# -------------------------------------------------------------------------------------------")
    print("#")
    print('# Dataset:' + dataset_dir)
    print("#")
    print("# -------------------------------------------------------------------------------------------")

    ngenes = eval(dataset_dir.split("_")[3])
    ndiff_genes = eval(dataset_dir.split("_")[5])
    ndiff = eval(dataset_dir.split("_")[6])
    data_dir = "../data/simulated/unif/" + dataset_dir + "/"
    result_dir = "./results_simulated/robustness_samplesize/" + dataset_dir + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    scores_clust = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "total_cells", "Prediction", "dataset"])
    scores = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "total_cells", "Prediction", "dataset"])
    auprc = pd.DataFrame(columns = ["ndiff_genes", "AUPRC", "AUPRC ratio", "total_cells", "ndiff"])

    for subsample in [1, 2, 5, 10]:


        print("# -------------------------------------------------------------------------------------------")
        print("#")
        print('# Downsampling rate:' + str(subsample ))
        print("#")
        print("# -------------------------------------------------------------------------------------------")

        counts_gt = []
        counts_ctrl_healthy = []
        counts_ctrl_severe = []
        counts_stim_healthy = []
        counts_stim_severe = []
        label_annos = []
        for batch_id in range(n_batches):
            counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T[::subsample,:])
            counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T[::subsample,:])
            counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T[::subsample,:])
            counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T[::subsample,:])
            counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T[::subsample,:])

            anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()[::subsample]
            # annotation labels
            label_annos.append(np.array([('cell type '+str(i)) for i in anno]))


        np.random.seed(0)
        counts_train = []
        meta_train = []
        counts_gt_full = []
        meta_gt_full = []
        for batch_id in range(n_batches):
            ncells = label_annos[batch_id].shape[0]
            # generate permutation
            permute_idx = np.random.permutation(ncells)
            # since there are totally four combinations of conditions, separate the cells into four groups
            chunk_size = int(ncells/4)

            # Training data: remove (ctrl, severe) for all batches
            counts_train.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:], 
                                                counts_stim_healthy[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:], 
                                                counts_stim_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0))
            meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * (ncells - 3*chunk_size))
            meta["condition 1"] = np.array(["ctrl"] * chunk_size + ["stim"] * chunk_size + ["stim"] * (ncells - 3*chunk_size))
            meta["condition 2"] = np.array(["healthy"] * chunk_size + ["healthy"] * chunk_size + ["severe"] * (ncells - 3*chunk_size))
            meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chunk_size]], 
                                            label_annos[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)]], 
                                            label_annos[batch_id][permute_idx[(3*chunk_size):]]], axis = 0)
            meta_train.append(meta)


            # Ground truth dataset
            counts_gt_full.append(np.concatenate([counts_ctrl_severe[batch_id][permute_idx[:chunk_size],:],
                                                counts_ctrl_severe[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:],
                                                counts_ctrl_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0))
            meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * (ncells - 3*chunk_size))
            meta["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["ctrl"] * (ncells - 3*chunk_size))
            meta["condition 2"] = np.array(["severe"] * chunk_size + ["severe"] * chunk_size + ["severe"] * (ncells - 3*chunk_size))
            meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chunk_size]], 
                                            label_annos[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)]], 
                                            label_annos[batch_id][permute_idx[(3*chunk_size):]]], axis = 0)
            meta_gt_full.append(meta)

        # full training dataset, input data are selected from the training dataset
        counts_train = np.concatenate(counts_train, axis = 0)
        meta_train = pd.concat(meta_train, axis = 0)
        # full ground truth dataset, prediction data are compared with the corresponding ground truth data
        counts_gt_full = np.concatenate(counts_gt_full, axis = 0)
        meta_gt_full = pd.concat(meta_gt_full, axis = 0)
        # create training dataset
        data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_train, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

        model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                        reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

        model.train()
        losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
        _ = model.eval()
        torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{subsample}.pth")
        model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{subsample}.pth", map_location = device))

        print("# -------------------------------------------------------------------------------------------")
        print("#")
        print('# Perturbation Prediction')
        print("#")
        print("# -------------------------------------------------------------------------------------------")

        # Prediction: Select input and predict conditions/batches
        configs_input = [{"condition 1": "stim", "condition 2": "severe", "batch": 0},
                        {"condition 1": "ctrl", "condition 2": "healthy", "batch": 0},
                        {"condition 1": "stim", "condition 2": "healthy", "batch": 0},
                        {"condition 1": "stim", "condition 2": "severe", "batch": 1},
                        {"condition 1": "ctrl", "condition 2": "healthy", "batch": 1},
                        {"condition 1": "stim", "condition 2": "healthy", "batch": 1}]

        print(f"Perturbation prediction of dataset: {dataset_dir}, predict: ctrl, severe, batch 0")
        for config in configs_input:
            print("input condition: " + str(config))

            # load input and gt count matrices
            idx = ((meta_train["condition 1"] == config["condition 1"]) & (meta_train["condition 2"] == config["condition 2"]) & (meta_train["batch"] == config["batch"])).values
            # input and ground truth, cells are matched
            counts_input = counts_train[idx, :]
            meta_input = meta_train.loc[idx, :]
            counts_gt = counts_gt_full[idx, :]
            meta_gt = meta_gt_full.loc[idx, :]

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

            scores_clust_sub = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "total_cells", "Prediction", "dataset"])
            scores_clust_sub["MSE"] = mses_scdisinfact
            scores_clust_sub["MSE input"] = mses_input
            scores_clust_sub["Pearson"] = pearsons_scdisinfact
            scores_clust_sub["Pearson input"] = pearsons_input
            scores_clust_sub["R2"] = r2_scdisinfact
            scores_clust_sub["R2 input"] = r2_input
            scores_clust_sub["total_cells"] = int(10000/subsample)
            scores_clust_sub["Prediction"] = config["condition 1"] + "_" + config["condition 2"] + "_" + str(config["batch"])
            scores_clust_sub["dataset"] = dataset_dir

            scores_clust = pd.concat([scores_clust, scores_clust_sub], axis = 0)

            # 2. 1-1 match, calculate cell-level score. Higher resolution when match exists
            mses_input = np.sum((counts_input - counts_gt_denoised) ** 2, axis = 1)
            mses_scdisinfact = np.sum((counts_predict - counts_gt_denoised) ** 2, axis = 1)
            # vector storing the pearson correlation for all cells
            pearson_input = np.array([stats.pearsonr(counts_input[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
            pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
            # vector storing the R2 scores for all cells
            r2_input = np.array([r2_score(y_pred = counts_input[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])
            r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])

            score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "total_cells", "Prediction", "dataset"])
            score["MSE"] = mses_scdisinfact
            score["MSE input"] = mses_input
            score["Pearson"] = pearsons_scdisinfact
            score["Pearson input"] = pearson_input
            score["R2"] = r2_scdisinfact
            score["R2 input"] = r2_input
            score["total_cells"] = int(10000/subsample)
            score["Prediction"] = config["condition 1"] + "_" + config["condition 2"] + "_" + str(config["batch"])
            score["dataset"] = dataset_dir
            scores = pd.concat([scores, score], axis = 0)


        print("# -------------------------------------------------------------------------------------------")
        print("#")
        print('# CKG')
        print("#")
        print("# -------------------------------------------------------------------------------------------")


        print("# -------------------------------------------------------------------------------------------")
        print("#")
        print('# 1st unshared encoder: ctrl and stim')
        print("#")
        print("# -------------------------------------------------------------------------------------------")

        model_params = torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{subsample}.pth")

        inf = np.array(model_params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        # ground truth
        gt = np.zeros((1, ngenes))
        gt[:,ndiff_genes:(2*ndiff_genes)] = 1
        gt = gt.squeeze()

        auprc = pd.concat([auprc,
                                pd.DataFrame.from_dict({
                                        "ndiff_genes": [ndiff_genes],
                                        "AUPRC": [bmk.compute_auprc(inf, gt)],
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "total_cells": int(10000/subsample) ,
                                        "ndiff": [ndiff]
                                        })], axis = 0, ignore_index = True)

        print("# -------------------------------------------------------------------------------------------")
        print("#")
        print('# 2nd unshared encoder: severe and healthy')
        print("#")
        print("# -------------------------------------------------------------------------------------------")

        inf = np.array(model_params["Enc_ds.1.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        # ground truth
        gt = np.zeros((1, ngenes))
        gt[:,:ndiff_genes] = 1
        gt = gt.squeeze()

        auprc = pd.concat([auprc,
                                pd.DataFrame.from_dict({
                                        "ndiff_genes": [ndiff_genes],
                                        "AUPRC": [bmk.compute_auprc(inf, gt)],
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "total_cells": int(10000/subsample),
                                        "ndiff": [ndiff]
                                        })], axis = 0, ignore_index = True)


    scores_clust.to_csv(result_dir + f"scores_clust.csv")
    scores.to_csv(result_dir + f"scores.csv")
    auprc.to_csv(result_dir + f"auprc.csv")


# In[]
ngenes = 500
ncells_total = 10000 
sigma = 0.4
scores_all = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "total_cells", "Prediction", "MSE (ratio)", "Pearson (ratio)", "R2 (ratio)", "cells_per_sample"])
auprc_all = pd.DataFrame(columns = ["ndiff_genes", "AUPRC", "AUPRC ratio", "total_cells", "ndiff", "cells_per_sample"])

for n_diff_genes in [20, 50, 100]:
    for diff in [2, 4, 8]:
        results_dir = f"./results_simulated/robustness_samplesize/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
        scores_clust = pd.read_csv(results_dir + "scores_clust.csv", index_col = 0)
        scores = pd.read_csv(results_dir + "scores.csv", index_col = 0)
        auprc = pd.read_csv(results_dir + "auprc.csv", index_col = 0)
        
        scores["MSE (ratio)"] = scores["MSE"].values/scores["MSE input"]
        scores["Pearson (ratio)"] = scores["Pearson"].values/scores["Pearson input"]
        scores["R2 (ratio)"] = scores["R2"].values/scores["R2 input"]
        scores["cells_per_sample"] = ((scores["total_cells"].values)/8).astype("int").astype("str")
        auprc["cells_per_sample"] = ((auprc["total_cells"].values)/8).astype("int").astype("str")
        
        scores_all = pd.concat([scores_all, scores], axis = 0)
        auprc_all = pd.concat([auprc_all, auprc],  axis = 0)


scores_all = scores_all[scores_all["Prediction"].isin(["ctrl_healthy_0", "stim_healthy_0", "stim_severe_0"])]

# In[]
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (14,5))
# ax = fig.subplots(nrows = 1, ncols = 3)
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='MSE', hue='total_cells', data=scores_all, ax = ax, ci = None)
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), title = "Number of cells")
for i in ax.containers:
    ax.bar_label(i, fmt='%.1e')    
# ax.set_ylim(0, 0.01)
# ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity"])
fig.savefig("./results_simulated/robustness_samplesize/prediction_MSE.png", bbox_inches = "tight")

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='R2', hue='total_cells', data=scores_all, ax = ax, ci = None)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')    
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), title = "Number of cells")
# ax.set_ylim(0.60, 1)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')    
# ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity"])
fig.tight_layout()
fig.savefig("./results_simulated/robustness_samplesize/prediction_R2.png", bbox_inches = "tight")

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot()
ax = sns.barplot(x = "ndiff", hue = 'total_cells', y ='AUPRC', data=auprc_all, ax = ax, ci = None)
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), title = "Number of cells")
# ax.set_ylim(0.50, 1.02)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')    
fig.tight_layout()
fig.savefig("./results_simulated/robustness_samplesize/CKGs_AUPRC.png", bbox_inches = "tight")


# %%
