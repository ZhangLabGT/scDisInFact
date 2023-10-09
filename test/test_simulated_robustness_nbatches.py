# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.loss_function as loss_func
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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 

simulated_lists = [
  "simulated_robustness_10000_4_2_1_4_20",
  "simulated_robustness_20000_8_2_1_4_20",
]
result_dir = "results_simulated/robustness_nbatches/"
# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# Out-of-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")

score_list = []
for dataset_dir in simulated_lists:
    data_dir = f"../data/simulated/{dataset_dir}/"
    n_batches = eval(dataset_dir.split("_")[3])
    ncondlabels = eval(dataset_dir.split("_")[4])
    ncondtypes = eval(dataset_dir.split("_")[5])

    if not os.path.exists(result_dir + f"{dataset_dir}/"):
        os.makedirs(result_dir + f"{dataset_dir}/")
    
    label_annos = []
    counts_gt = {}
    counts_train = []
    meta_train = []
    for batch_id in range(n_batches):
        anno = pd.read_csv(data_dir + f'cell_label_b{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
        label_annos.append(np.array([('cell type '+str(i)) for i in anno]))   
        ncells = label_annos[batch_id].shape[0]
        permute_idx = np.random.permutation(ncells)

        chunk_size = int(ncells/2)
        counts_c1 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1.txt', sep = "\t", header = None).values.T
        counts_c2 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2.txt', sep = "\t", header = None).values.T
        if batch_id == 0:
            counts_gt[batch_id] = {}
            counts_gt[batch_id]["c1"] = counts_c1[permute_idx[:chunk_size],:]
            counts_gt[batch_id]["c2"] = counts_c2[permute_idx[:chunk_size],:]
            # training data
            counts_train.append(counts_c1[permute_idx[:chunk_size],:])
            # training meta data    
            meta = pd.DataFrame(columns = ["batch", "condition 1", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size)
            meta["condition 1"] = np.array([1] * chunk_size)
            meta["annos"] = label_annos[batch_id][permute_idx[:chunk_size]]
            meta_train.append(meta)
        else:
            counts_gt[batch_id] = {}
            counts_gt[batch_id]["c1"] = counts_c1[permute_idx[:(2*chunk_size)],:]
            counts_gt[batch_id]["c2"] = counts_c2[permute_idx[:(2*chunk_size)],:]
            # training data
            counts_train.append(np.concatenate([counts_c1[permute_idx[:chunk_size],:], counts_c2[permute_idx[chunk_size:(2*chunk_size)],:]], axis = 0))
            # training meta data    
            meta = pd.DataFrame(columns = ["batch", "condition 1", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size + [batch_id] * chunk_size)
            meta["condition 1"] = np.array([1] * chunk_size + [2] * chunk_size)
            meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chunk_size]], label_annos[batch_id][permute_idx[chunk_size:(2*chunk_size)]]], axis = 0)
            meta_train.append(meta)

    counts_train = np.concatenate(counts_train, axis = 0)
    meta_train = pd.concat(meta_train, axis = 0)
    # create training dataset
    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_train, condition_key = ["condition 1"], batch_key = "batch")
    # corresponding gt
    for cond in counts_gt[0].keys():
        counts_gt[cond] = np.concatenate([counts_gt[batch_id][cond] for batch_id in range(n_batches)], axis = 0)

    # Train the model
    # default parameters
    reg_mmd_comm = 1e-4
    reg_mmd_diff = 1e-4
    reg_kl_comm = 1e-5
    reg_kl_diff = 1e-2
    reg_class = 1
    reg_gl = 1

    Ks = [8, 2]
    batch_size = 64
    nepochs = 50
    interval = 10
    lr = 5e-4
    lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
    model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                    reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
    model.train()
    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
    _ = model.eval()
    torch.save(model.state_dict(), result_dir + f"{dataset_dir}/" + f"scdisinfact_{Ks}_{lambs}.pth")
    # model.load_state_dict(torch.load(result_dir + f"{dataset_dir}/" + f"scdisinfact_{Ks}_{lambs}.pth", map_location = device))

    config_input = [1]
    # input: condition 1 of batch 1, perdict: last condition (ncondlabels) of batch 1
    idx = (meta_train["condition 1"] == config_input[0]) & (meta_train["batch"] == 0)
    counts_input = counts_train[idx, :]
    meta_input = meta_train.loc[idx, :]
    counts_predict_gt = counts_gt["c2"][idx,:]
    meta_gt = meta_input.copy()
    meta_gt["condition 1"] = 2
        
    # predict count
    counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1"], 
                                                 batch_key = "batch", predict_conds = None, predict_batch = None)
    counts_predict_gt_denoised = model.predict_counts(input_counts = counts_predict_gt, meta_cells = meta_gt, condition_keys = ["condition 1"], 
                                                     batch_key = "batch", predict_conds = None, predict_batch = None)
    counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1"], 
                                          batch_key = "batch", predict_conds = [2], predict_batch = 0)

    counts_predict_gt = counts_predict_gt/(np.sum(counts_predict_gt, axis = 1, keepdims = True) + 1e-6)
    counts_predict_gt_denoised = counts_predict_gt_denoised/(np.sum(counts_predict_gt_denoised, axis = 1, keepdims = True) + 1e-6)
    counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
    counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)
    counts_input_denoised = counts_input_denoised/(np.sum(counts_input_denoised, axis = 1, keepdims = True) + 1e-6)

    mses_input = np.sum((counts_input_denoised - counts_predict_gt_denoised) ** 2, axis = 1)
    mses_scdisinfact = np.sum((counts_predict - counts_predict_gt_denoised) ** 2, axis = 1)
    # vector storing the pearson correlation for all cells
    pearson_input = np.array([stats.pearsonr(counts_input_denoised[i,:], counts_predict_gt_denoised[i,:])[0] for i in range(counts_predict_gt_denoised.shape[0])])
    pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_predict_gt_denoised[i,:])[0] for i in range(counts_predict_gt_denoised.shape[0])])
    # vector storing the R2 scores for all cells
    r2_input = np.array([r2_score(y_pred = counts_input_denoised[i,:], y_true = counts_predict_gt_denoised[i,:]) for i in range(counts_predict_gt_denoised.shape[0])])
    r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_predict_gt_denoised[i,:]) for i in range(counts_predict_gt_denoised.shape[0])])

    score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Prediction", "Dataset"])
    score["MSE"] = mses_scdisinfact
    score["MSE input"] = mses_input
    score["Pearson"] = pearsons_scdisinfact
    score["Pearson input"] = pearson_input
    score["R2"] = r2_scdisinfact
    score["R2 input"] = r2_input
    score["Prediction"] = f"{n_batches} batches"
    score["Dataset"] = dataset_dir
    score_list.append(score)

scores = pd.concat(score_list, axis = 0)
scores.to_csv(result_dir + "prediction_scores.csv")

# In[]
auprc_dict = pd.DataFrame(columns = ["dataset", "condition", "ndiff_genes", "AUPRC", "AUPRC ratio", "ndiff"])

for dataset_dir in simulated_lists:
    data_dir = f"../data/simulated/{dataset_dir}/"
    ncells = eval(dataset_dir.split("_")[2])
    n_batches = eval(dataset_dir.split("_")[3])
    ncondlabels = eval(dataset_dir.split("_")[4])
    ncondtypes = eval(dataset_dir.split("_")[5])
    ndiff = eval(dataset_dir.split("_")[6])
    ndiff_genes = eval(dataset_dir.split("_")[7])
    ngenes = 500
    print(f"CKG detection: {dataset_dir}")

    # load model
    reg_mmd_comm = 1e-4
    reg_mmd_diff = 1e-4
    reg_kl_comm = 1e-5
    reg_kl_diff = 1e-2
    reg_class = 1
    reg_gl = 1
    Ks = [8, 2]
    lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

    params = torch.load(result_dir + f"{dataset_dir}/scdisinfact_{Ks}_{lambs}.pth", map_location = device)

    for cond in range(ncondtypes):
        gt = np.zeros(ngenes)
        gt[cond*ndiff_genes:((cond+1)*ndiff_genes)] = 1
        inf = np.array(params[f"Enc_ds.{cond}.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        auprc_dict = pd.concat([auprc_dict, 
                                pd.DataFrame.from_dict({"Dataset": [dataset_dir], 
                                        "ndiff_genes": [ndiff_genes], 
                                        "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "ndiff": [ndiff],
                                        "condition": [cond],
                                        "Prediction": [f"{n_batches} batches"]
                                        })], axis = 0, ignore_index = True)

auprc_dict.to_csv(result_dir + "CKG_scores.txt", sep = "\t")

# In[]
auprc_dict = pd.read_csv(result_dir + "CKG_scores.txt", sep = "\t", index_col = 0)
pred_scores = pd.read_csv(result_dir + "prediction_scores.csv", index_col = 0)
# pred_scores = pred_scores.loc[pred_scores["Dataset"].isin(["simulated_robustness_10000_2_3_1_8_20", "simulated_robustness_10000_2_4_1_8_20", "simulated_robustness_10000_2_5_1_8_20"]),:]
# auprc_dict = auprc_dict.loc[auprc_dict["Dataset"].isin(["simulated_robustness_10000_2_3_1_8_20", "simulated_robustness_10000_2_4_1_8_20", "simulated_robustness_10000_2_5_1_8_20"]),:]
pred_scores["MSE (ratio)"] = pred_scores["MSE"].values/pred_scores["MSE input"].values

plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (18,5))
# ax = fig.subplots(nrows = 1, ncols = 3)
ax = fig.subplots(nrows = 1, ncols = 3)
ax[0] = sns.barplot(x='Prediction', y='MSE', data=pred_scores, ax = ax[0], ci = None)
for i in ax[0].containers:
    ax[0].bar_label(i, fmt='%.1e')    


ax[1] = sns.barplot(x='Prediction', y='R2', data=pred_scores, ax = ax[1], ci = None)
for i in ax[1].containers:
    ax[1].bar_label(i, fmt='%.2f')    
fig.tight_layout()

ax[2] = sns.barplot(x = "Prediction", y ='AUPRC', data=auprc_dict, ax = ax[2], ci = None)
for i in ax[2].containers:
    ax[2].bar_label(i, fmt='%.2f')    
fig.tight_layout()
fig.savefig(result_dir + "scores_robustness.png", bbox_inches = "tight")


# %%
