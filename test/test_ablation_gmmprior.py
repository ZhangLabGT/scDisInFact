# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model_gmm as scdisinfact
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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 

def rank_marker(counts_predict, counts_input_denoised, counts_input, counts_gt):
    """
    Description:
    -------------
        Measuring the perturbed gene detection accuracy using AUPRC. 
        The cells in input count matrices are all matched. 
    Parameters:
    -------------
        counts_predict: 
            The denoised count matrix under the predicted condition
        counts_input_denoised:
            The (scdisinfact) denoised count matrix under the input condition
        counts_input:
            The raw count matrix under the input condition
        counts_gt:
            The raw count matrix under the ground truth condition
    
    Return:
    -------------
        AUPRC: the accuracy of the detected gene expression change 
    """
    # counts_predict and counts_input_denoised are already normalized (decoder output)
    pred_diff = np.mean(counts_predict - counts_input_denoised, axis = 0)
    # gt_diff measures the change of gene expression, 
    # where nonzero means the corresponding gene is perturbed, 
    # and zero means the corresponding gene is not perturbed.
    # NOTE: gt_diff not accurate if normalized, before normalization
    gt_diff = np.mean(counts_gt - counts_input, axis = 0)
    AUPRC = bmk.compute_auprc(pred_diff, gt_diff)
    return AUPRC

print("# -------------------------------------------------------------------------------------------")
print("#")
print("# Ablation test on Contrastive loss")
print("#")
print("# -------------------------------------------------------------------------------------------")

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
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Out-of-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")


ablation = "tc"
if ablation == "contr":
    result_dir = "./results_simulated/ablation/contrastive_gmmprior/"
elif ablation == "gl":
    result_dir = "./results_simulated/ablation/group_lasso_gmmprior/"
elif ablation == "tc":
    result_dir = "./results_simulated/ablation/total_correlation_gmmprior/"

for dataset_dir in simulated_lists:
    data_dir = f"../data/simulated/unif/{dataset_dir}/"

    if not os.path.exists(result_dir + f"{dataset_dir}_oos/"):
        os.makedirs(result_dir + f"{dataset_dir}_oos/")

    # Load dataset
    counts_ctrl_healthy = []
    counts_ctrl_severe = []
    counts_stim_healthy = []
    counts_stim_severe = []
    label_annos = []
    for batch_id in range(n_batches):
        counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
        counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
        counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
        counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

        anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
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
        chuck_size = int(ncells/4)

        # Training data: remove (ctrl, severe) for all batches
        counts_train.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                            counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                            counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
        meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta["batch"] = np.array([batch_id] * chuck_size + [batch_id] * chuck_size + [batch_id] * (ncells - 3*chuck_size))
        meta["condition 1"] = np.array(["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (ncells - 3*chuck_size))
        meta["condition 2"] = np.array(["healthy"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (ncells - 3*chuck_size))
        meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                             label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                             label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
        meta_train.append(meta)

        
        # Ground truth dataset
        counts_gt_full.append(np.concatenate([counts_ctrl_severe[batch_id][permute_idx[:chuck_size],:],
                                         counts_ctrl_severe[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:],
                                         counts_ctrl_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
        meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta["batch"] = np.array([batch_id] * chuck_size + [batch_id] * chuck_size + [batch_id] * (ncells - 3*chuck_size))
        meta["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["ctrl"] * (ncells - 3*chuck_size))
        meta["condition 2"] = np.array(["severe"] * chuck_size + ["severe"] * chuck_size + ["severe"] * (ncells - 3*chuck_size))
        meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                        label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                        label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
        meta_gt_full.append(meta)

    # full training dataset, input data are selected from the training dataset
    counts_train = np.concatenate(counts_train, axis = 0)
    meta_train = pd.concat(meta_train, axis = 0)
    # full ground truth dataset, prediction data are compared with the corresponding ground truth data
    counts_gt_full = np.concatenate(counts_gt_full, axis = 0)
    meta_gt_full = pd.concat(meta_gt_full, axis = 0)
    # create training dataset
    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_train, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

    # Train the model
    # default parameters
    reg_mmd_comm = 1e-4
    reg_mmd_diff = 1e-4
    reg_tc = 0.5
    reg_class = 1
    reg_kl = 1e-5
    Ks = [8, 4, 4]
    batch_size = 64
    nepochs = 50
    interval = 10
    lr = 5e-4
    reg_gl = 0.01
    reg_contr = 0.01

    for reg in [0, 0.01, 0.1, 1]:
        if ablation == "gl":
            reg_gl = reg
        elif ablation == "contr":
            reg_contr = reg
        elif ablation == "tc":
            reg_tc = reg

        # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
        lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
        print(lambs)

        model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                        reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

        model.train()
        losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = reg_contr)
        _ = model.eval()

        torch.save(model.state_dict(), result_dir + f"{dataset_dir}_oos/" + f"scdisinfact_{Ks}_{lambs}.pth")


    # Prediction: Select input and predict conditions/batches
    configs_input = [{"condition 1": "stim", "condition 2": "severe", "batch": 0},
                    {"condition 1": "ctrl", "condition 2": "healthy", "batch": 0},
                    {"condition 1": "stim", "condition 2": "healthy", "batch": 0},
                    {"condition 1": "stim", "condition 2": "severe", "batch": 1},
                    {"condition 1": "ctrl", "condition 2": "healthy", "batch": 1},
                    {"condition 1": "stim", "condition 2": "healthy", "batch": 1}]


    score_list = []
    score_cluster_list = []
    for config in configs_input:
        print("input condition: " + str(config))
        for reg in [0, 0.01, 0.1, 1]:
            # Load input and ground truth data
            idx = ((meta_train["condition 1"] == config["condition 1"]) & (meta_train["condition 2"] == config["condition 2"]) & (meta_train["batch"] == config["batch"])).values
            # input and ground truth, cells are matched
            counts_input = counts_train[idx, :]
            meta_input = meta_train.loc[idx, :]
            counts_gt = counts_gt_full[idx, :]
            meta_gt = meta_gt_full.loc[idx, :]

            # load params
            if ablation == "gl":
                reg_gl = reg
            elif ablation == "contr":
                reg_contr = reg
            elif ablation == "tc":
                reg_tc = reg
            # load model
            lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
            model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                            reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                            reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)
            model.load_state_dict(torch.load(result_dir + f"{dataset_dir}_oos/" + f"scdisinfact_{Ks}_{lambs}.pth", map_location = device))
            # predict count
            counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                            batch_key = "batch", predict_conds = None, predict_batch = None)
            
            counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                            batch_key = "batch", predict_conds = None, predict_batch = None)

            counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                                batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)

            # NOTE: perturbed gene prediction accuracy, measured with AUPRC
            # should be calculated before normalization
            AUPRC = rank_marker(counts_predict = counts_predict, counts_input_denoised = counts_input_denoised, counts_input = counts_input, counts_gt = counts_gt)

            # normalize the count
            # Is normalization better? Make sure the libsize of predict and gt are the same for each cell (or they will not be on the same scale)
            # In addition, the prediction output is decoder mu, when calculate NB loss, decoder mu is multiplied with libsize and then compare with true count
            # which means that the predict count ignore the libsize effect.
            # Amount the matrices below, counts_gt and counts_input has libsize effect (they have to be normalized), 
            # counts_gt_denoise and counts_predict are all decoder output with no libsize effect (doesn't matter normalize or not).
            # Normalization does not introduce error.
            counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
            counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
            counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
            counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

            # 1. no 1-1 match, check cell-type level scores, but low resolution
            unique_celltypes = np.unique(meta_gt["annos"].values)
            mean_inputs = []
            mean_predicts = []
            mean_gts = []
            mean_gts_denoised = []
            for celltype in unique_celltypes:
                mean_input = np.mean(counts_input[meta_input["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_predict = np.mean(counts_predict[meta_input["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_gt = np.mean(counts_gt[meta_gt["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_gt_denoised = np.mean(counts_gt_denoised[meta_gt["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_inputs.append(mean_input)
                mean_predicts.append(mean_predict)
                mean_gts.append(mean_gt)
                mean_gts_denoised.append(mean_gt_denoised)

            mean_inputs = np.array(mean_inputs)
            mean_predicts = np.array(mean_predicts)
            mean_gts = np.array(mean_gts)
            mean_gts_denoised = np.array(mean_gts_denoised)

            # vector storing cell-type-specific normalized MSE for all clusters
            mses_input_cluster = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
            mses_scdisinfact_cluster = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
            # vector storing cell-type-specific pearson correlation
            pearsons_input_cluster = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
            pearsons_scdisinfact_cluster = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
            # vector storing cell-type-specific R2 score
            r2_input_cluster = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
            r2_scdisinfact_cluster = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

            score_cluster = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
            score_cluster["MSE"] = mses_scdisinfact_cluster
            score_cluster["MSE input"] = mses_input_cluster
            score_cluster["Pearson"] = pearsons_scdisinfact_cluster
            score_cluster["Pearson input"] = pearsons_input_cluster
            score_cluster["R2"] = r2_scdisinfact_cluster
            score_cluster["R2 input"] = r2_input_cluster
            score_cluster["Method"] = f"reg: {reg}"
            score_cluster["Prediction"] = config["condition 1"] + "_" + config["condition 2"] + "_" + str(config["batch"])
            score_cluster_list.append(score_cluster)

            # 2. 1-1 match, calculate cell-level score. Higher resolution when match exists
            # vector storing the normalized MSE for all cells
            mses_input = np.sum((counts_input - counts_gt_denoised) ** 2, axis = 1)
            mses_scdisinfact = np.sum((counts_predict - counts_gt_denoised) ** 2, axis = 1)
            # vector storing the pearson correlation for all cells
            pearson_input = np.array([stats.pearsonr(counts_input[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
            pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
            # vector storing the R2 scores for all cells
            r2_input = np.array([r2_score(y_pred = counts_input[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])
            r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])

            score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "AUPRC", "Method", "Prediction"])
            score["MSE"] = mses_scdisinfact
            score["MSE input"] = mses_input
            score["Pearson"] = pearsons_scdisinfact
            score["Pearson input"] = pearson_input
            score["R2"] = r2_scdisinfact
            score["R2 input"] = r2_input
            score["AUPRC"] = AUPRC
            score["Method"] = f"reg: {reg}"
            score["Prediction"] = config["condition 1"] + "_" + config["condition 2"] + "_" + str(config["batch"])
            score_list.append(score)

    scores_cluster = pd.concat(score_cluster_list, axis = 0)
    scores_cluster.to_csv(result_dir + f"{dataset_dir}_oos/" + "prediction_scores_cluster.csv")

    scores = pd.concat(score_list, axis = 0)
    scores.to_csv(result_dir + f"{dataset_dir}_oos/" + "prediction_scores.csv")

print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 2. CKG detection")
print("#")
print("# -------------------------------------------------------------------------------------------")
auprc_dict = pd.DataFrame(columns = ["dataset", "condition", "ndiff_genes", "AUPRC", "AUPRC ratio", "method", "ndiff"])

for dataset_dir in simulated_lists:
    ngenes = eval(dataset_dir.split("_")[3])
    ndiff_genes = eval(dataset_dir.split("_")[5])
    ndiff = eval(dataset_dir.split("_")[6])

    # no group lasso loss
    for reg in [0, 0.01, 0.1, 1]:
        # load params
        if ablation == "gl":
            reg_gl = reg
        elif ablation == "contr":
            reg_contr = reg
        elif ablation == "tc":
            reg_tc = reg

        # load model
        lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]

        params = torch.load(result_dir + f"{dataset_dir}_oos/scdisinfact_{Ks}_{lambs}.pth", map_location = device)

        gt = np.zeros((1, ngenes))
        gt[:,ndiff_genes:(2*ndiff_genes)] = 1
        inf = np.array(params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        auprc_dict = pd.concat([auprc_dict, pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                        "ndiff_genes": [ndiff_genes], 
                                        "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "Method": [f"reg: {reg}"],
                                        "ndiff": [ndiff],
                                        "condition": ["ctrl & stim"]
                                        })], axis = 0, ignore_index = True)

        gt = np.zeros((1, ngenes))
        gt[:,:ndiff_genes] = 1
        inf = np.array(params["Enc_ds.1.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        auprc_dict = pd.concat([auprc_dict, pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                        "ndiff_genes": [ndiff_genes], 
                                        "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "Method": [f"reg: {reg}"],
                                        "ndiff": [ndiff],
                                        "condition": ["severe & healthy"]
                                        })], axis = 0, ignore_index = True)

auprc_dict.to_csv(result_dir + "/CKG_scores.txt", sep = "\t")

# In[]
print("# ----------------------------------------------------------------------------------")
print("#")
print("# Evaluation")
print("#")
print("# ----------------------------------------------------------------------------------")

plt.rcParams["font.size"] = 15
scores_prediction = []

for dataset_dir in simulated_lists:
    scores_prediction.append(pd.read_csv(result_dir + f"{dataset_dir}_oos/prediction_scores.csv", index_col = 0))
scores_prediction = pd.concat(scores_prediction, axis = 0)
auprc_dict = pd.read_csv(result_dir + f"CKG_scores.txt", index_col = 0, sep = "\t")
scores_prediction["MSE ratio"] = scores_prediction["MSE"].values/scores_prediction["MSE input"].values
scores_prediction["R2 ratio"] = scores_prediction["R2"].values/scores_prediction["R2 input"].values

fig = plt.figure(figsize = (12,5))
# ax = fig.subplots(nrows = 1, ncols = 3)
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='MSE ratio', hue='Method', data=scores_prediction, ax = ax, errwidth=0.1) 
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
# ax.set_ylim(0, 0.01)
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
fig.savefig(result_dir + "prediction_MSE_ratio.png", bbox_inches = "tight")

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='AUPRC', hue='Method', data=scores_prediction, ax = ax, errwidth=0.1) 
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))

ax.set_ylim(0.0, 1.2)
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
fig.savefig(result_dir + "prediction_AUPRC.png", bbox_inches = "tight")

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='R2 ratio', hue='Method', data=scores_prediction, ax = ax, errwidth=0.1)
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
# ax.set_ylim(0.60, 1)
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
fig.tight_layout()
fig.savefig(result_dir + "prediction_R2_ratio.png", bbox_inches = "tight")

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot()
ax = sns.barplot(x='ndiff', y='AUPRC', hue = "Method", data=auprc_dict, ax = ax, errwidth=0.1)
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
# ax.set_ylim(0.60, 1)
fig.tight_layout()
fig.savefig(result_dir + "CKGs_AUPRC.png", bbox_inches = "tight")

# In[]


# %%
