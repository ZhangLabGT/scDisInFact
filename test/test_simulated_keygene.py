# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.bmk as bmk
from scipy.stats import pearsonr, spearmanr, kendalltau

from umap import UMAP
import time
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import seaborn as sns


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
# kl term explode when nepochs = 70
nepochs = 100
interval = 10
lr = 5e-4
n_batches = 6

data_dir = f"../data/simulated/unif/"

# In[] by number of ndiff_genes
plt.rcParams["font.size"] = 20

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

auprc_dict = pd.DataFrame(columns = ["dataset", "ndiff_genes", "AUPRC", "AUPRC ratio", "method", "ndiff"])
for dataset_dir in simulated_lists:
    ngenes = eval(dataset_dir.split("_")[3])
    ndiff_genes = eval(dataset_dir.split("_")[5])
    ndiff = eval(dataset_dir.split("_")[6])
    result_dir = './results_simulated/disentangle/unif/' + dataset_dir + "/"
    model_params = torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")

    # --------------------------------------------------------------------------------------------
    #
    # 1st unshared encoder: ctrl and stim
    #
    # --------------------------------------------------------------------------------------------
    inf = np.array(model_params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
    # ground truth 
    gt = np.zeros((1, ngenes))
    gt[:,ndiff_genes:(2*ndiff_genes)] = 1
    gt = gt.squeeze()
    # gt[:,:ndiff_genes] = 1

    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                    "ndiff_genes": [ndiff_genes], 
                                    "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                    "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                    "AUROC": [bmk.compute_auroc(inf, gt)],
                                    "Eprec": [bmk.compute_earlyprec(inf, gt)],
                                    "Pearson": [pearsonr(inf, gt)[0]],
                                    "method": ["scDisInFact"],
                                    "ndiff": [ndiff]
                                    })], axis = 0, ignore_index = True)
    
    # scinsight
    result_scinsight = './results_simulated/scinsight/'+dataset_dir + "/scinsight_ctrl_stim/"
    # scores for condition 1
    H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t").values
    # scores for condition 2
    H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t").values

    # variance of scores
    H = np.concatenate([H1, H2], axis = 0).T
    H_mean = np.mean(H, axis = 1, keepdims = True)
    H_var = np.mean((H - H_mean) ** 2, axis = 1)
    # normalize
    H_var = H_var/np.max(H_var)
    H_var = H_var.squeeze()

    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                    "ndiff_genes": [ndiff_genes], 
                                    "AUPRC": [bmk.compute_auprc(H_var, gt)],
                                    "AUPRC ratio": [bmk.compute_auprc(H_var, gt)/(ndiff_genes/ngenes)],
                                    "AUROC": [bmk.compute_auroc(H_var, gt)],
                                    "Eprec": [bmk.compute_earlyprec(H_var, gt)],
                                    "Pearson": [pearsonr(H_var, gt)[0]],
                                    "method": ["scINSIGHT"],
                                    "ndiff": [ndiff]
                                    })], axis = 0, ignore_index = True)
    
    # wilcoxon baseline
    counts = np.loadtxt(data_dir + dataset_dir + "/scinsight/counts.txt")
    meta = pd.read_csv(data_dir + dataset_dir + "/scinsight/meta.csv", index_col = 0)
    counts = np.log1p(counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6) * 100)
    counts_ctrl = counts[(meta["condition 1"] == "ctrl"),:]
    counts_stim = counts[(meta["condition 1"] == "stim"), :]

    pvals = bmk.wilcoxon_rank_sum(counts_ctrl, counts_stim, fdr = True)
    assert pvals.shape[0] == 500
    assert np.min(pvals) >= 0
    # scale to 0-1
    score_wilcoxon = 1 - pvals/np.max(pvals)
    score_wilcoxon  = score_wilcoxon.squeeze()

    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                    "ndiff_genes": [ndiff_genes], 
                                    "AUPRC": [bmk.compute_auprc(score_wilcoxon, gt)],
                                    "AUPRC ratio": [bmk.compute_auprc(score_wilcoxon, gt)/(ndiff_genes/ngenes)],
                                    "AUROC": [bmk.compute_auroc(score_wilcoxon, gt)],
                                    "Eprec": [bmk.compute_earlyprec(score_wilcoxon, gt)],
                                    "Pearson": [pearsonr(score_wilcoxon, gt)[0]],
                                    "method": ["Wilcoxon"],
                                    "ndiff": [ndiff]
                                    })], axis = 0, ignore_index = True)



    # --------------------------------------------------------------------------------------------
    #
    # 2nd unshared encoder: severe and healthy
    #
    # --------------------------------------------------------------------------------------------
    inf = np.array(model_params["Enc_ds.1.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
    # ground truth 
    gt = np.zeros((1, ngenes))
    gt[:,:ndiff_genes] = 1
    gt = gt.squeeze()

    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                    "ndiff_genes": [ndiff_genes], 
                                    "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                    "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                    "AUROC": [bmk.compute_auroc(inf, gt)],
                                    "Eprec": [bmk.compute_earlyprec(inf, gt)],
                                    "Pearson": [pearsonr(inf, gt)[0]],
                                    "method": ["scDisInFact"],
                                    "ndiff": [ndiff]
                                    })], axis = 0, ignore_index = True)
    
    # scinsight
    result_scinsight = './results_simulated/scinsight/'+dataset_dir + "/scinsight_healthy_severe/"
    # scores for condition 1
    H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t").values
    # scores for condition 2
    H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t").values

    # variance of scores
    H = np.concatenate([H1, H2], axis = 0).T
    H_mean = np.mean(H, axis = 1, keepdims = True)
    H_var = np.mean((H - H_mean) ** 2, axis = 1)
    # normalize
    H_var = H_var/np.max(H_var)
    H_var = H_var.squeeze()

    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                    "ndiff_genes": [ndiff_genes], 
                                    "AUPRC": [bmk.compute_auprc(H_var, gt)],
                                    "AUPRC ratio": [bmk.compute_auprc(H_var, gt)/(ndiff_genes/ngenes)],
                                    "AUROC": [bmk.compute_auroc(H_var, gt)],
                                    "Eprec": [bmk.compute_earlyprec(H_var, gt)],
                                    "Pearson": [pearsonr(H_var, gt)[0]],
                                    "method": ["scINSIGHT"],
                                    "ndiff": [ndiff]
                                    })], axis = 0, ignore_index = True)

    # wilcoxon baseline
    counts = np.loadtxt(data_dir + dataset_dir + "/scinsight/counts.txt")
    meta = pd.read_csv(data_dir + dataset_dir + "/scinsight/meta.csv", index_col = 0)
    counts = counts/np.sum(counts, axis = 1, keepdims = True) * 100
    counts = np.log1p(counts)
    counts_healthy = counts[meta["condition 2"] == "healthy",:]
    counts_severe = counts[meta["condition 2"] == "severe", :]

    pvals = bmk.wilcoxon_rank_sum(counts_healthy, counts_severe, fdr = True)
    assert pvals.shape[0] == 500
    assert np.min(pvals) >= 0
    # scale to 0-1
    score_wilcoxon = 1 - pvals/np.max(pvals)
    score_wilcoxon = score_wilcoxon.squeeze()

    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                    "ndiff_genes": [ndiff_genes], 
                                    "AUPRC": [bmk.compute_auprc(score_wilcoxon, gt)],
                                    "AUPRC ratio": [bmk.compute_auprc(score_wilcoxon, gt)/(ndiff_genes/ngenes)],
                                    "AUROC": [bmk.compute_auroc(score_wilcoxon, gt)],
                                    "Eprec": [bmk.compute_earlyprec(score_wilcoxon, gt)],
                                    "Pearson": [pearsonr(score_wilcoxon, gt)[0]],
                                    "method": ["Wilcoxon"],
                                    "ndiff": [ndiff]
                                    })], axis = 0, ignore_index = True)


# # In[]
# fig = plt.figure(figsize = (7,5))
# ax = fig.add_subplot()
# sns.barplot(x='ndiff_genes', y='AUPRC', hue='method', data=auprc_dict, ax = ax, errwidth=0) # 
# ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
# ax.set_xlabel("Number of perturbed genes")
# for i in ax.containers:
#     ax.bar_label(i, fmt='%.2f')

# fig.savefig("./results_simulated/disentangle/AUPRC (ndiff_genes).png", bbox_inches = "tight")


# In[]
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
ax = sns.barplot(x='ndiff', y='AUPRC', hue='method', data=auprc_dict, ax = ax, capsize = 0.1)
sns.stripplot(data = auprc_dict, x = "ndiff", y = "AUPRC", hue = "method", ax = ax, color = "black", dodge = True) 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Perturbation parameters")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)    


fig.savefig("./results_simulated/disentangle/AUPRC (ndiffs).png", bbox_inches = "tight")

# In[]
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
ax = sns.barplot(x='ndiff', y='AUROC', hue='method', data=auprc_dict, ax = ax, capsize = 0.1)
sns.stripplot(data = auprc_dict, x = "ndiff", y = "AUROC", hue = "method", ax = ax, color = "black", dodge = True) 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Perturbation parameters")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)   

fig.savefig("./results_simulated/disentangle/AUROC (ndiffs).png", bbox_inches = "tight")

# In[]
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
ax = sns.barplot(x='ndiff', y='Eprec', hue='method', data=auprc_dict, ax = ax, capsize = 0.1)
sns.stripplot(data = auprc_dict, x = "ndiff", y = "Eprec", hue = "method", ax = ax, color = "black", dodge = True) 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Perturbation parameters")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -70)  

fig.savefig("./results_simulated/disentangle/Eprec (ndiffs).png", bbox_inches = "tight")

# In[]
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
ax = sns.barplot(x='ndiff', y='Pearson', hue='method', data=auprc_dict, ax = ax, capsize = 0.1)
sns.stripplot(data = auprc_dict, x = "ndiff", y = "Pearson", hue = "method", ax = ax, color = "black", dodge = True) 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Perturbation parameters")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -60)  

fig.savefig("./results_simulated/disentangle/Pearson (ndiffs).png", bbox_inches = "tight")

# %%
