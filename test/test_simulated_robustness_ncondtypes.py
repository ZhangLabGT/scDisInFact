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

n_batches = 2
simulated_lists = [
  "simulated_robustness_10000_2_2_2_4_20",
  "simulated_robustness_10000_2_2_3_4_20",
  "simulated_robustness_10000_2_2_4_4_20",
]
result_dir = "results_simulated/robustness_ncondtypes/"
# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Out-of-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")

score_list = []
for dataset_dir in simulated_lists:
    data_dir = f"../data/simulated/{dataset_dir}/"
    ncells = eval(dataset_dir.split("_")[2])
    nbatches = eval(dataset_dir.split("_")[3])
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
        if ncondtypes == 2:
            chunk_size = int(ncells/4)
            counts_c11 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c11.txt', sep = "\t", header = None).values.T
            counts_c12 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c12.txt', sep = "\t", header = None).values.T
            counts_c21 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c21.txt', sep = "\t", header = None).values.T
            counts_c22 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c22.txt', sep = "\t", header = None).values.T
            counts_gt[batch_id] = {}
            counts_gt[batch_id]["c11"] = counts_c11[permute_idx[:(3*chunk_size)],:]
            counts_gt[batch_id]["c12"] = counts_c12[permute_idx[:(3*chunk_size)],:]
            counts_gt[batch_id]["c21"] = counts_c21[permute_idx[:(3*chunk_size)],:]
            counts_gt[batch_id]["c22"] = counts_c22[permute_idx[:(3*chunk_size)],:]

            counts_train.append(np.concatenate([
                counts_c11[permute_idx[:chunk_size],:],
                counts_c12[permute_idx[chunk_size:(2*chunk_size)],:],
                counts_c21[permute_idx[(2*chunk_size):(3*chunk_size)],:]], axis = 0))
            
            meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size)
            meta["condition 1"] = np.array([1] * chunk_size + [1] * chunk_size + [2] * chunk_size)
            meta["condition 2"] = np.array([1] * chunk_size + [2] * chunk_size + [1] * chunk_size)
            meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chunk_size]], 
                                             label_annos[batch_id][permute_idx[(chunk_size):(2*chunk_size)]], 
                                             label_annos[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)]]], axis = 0)
            meta_train.append(meta)

        elif ncondtypes == 3:
            chunk_size = int(ncells/8)
            counts_c111 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c111.txt', sep = "\t", header = None).values.T
            counts_c112 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c112.txt', sep = "\t", header = None).values.T
            counts_c121 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c121.txt', sep = "\t", header = None).values.T
            counts_c211 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c211.txt', sep = "\t", header = None).values.T
            counts_c122 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c122.txt', sep = "\t", header = None).values.T
            counts_c212 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c212.txt', sep = "\t", header = None).values.T
            counts_c221 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c221.txt', sep = "\t", header = None).values.T
            counts_c222 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c222.txt', sep = "\t", header = None).values.T
            counts_gt[batch_id] = {}
            counts_gt[batch_id]["c111"] = counts_c111[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c112"] = counts_c112[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c121"] = counts_c121[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c211"] = counts_c211[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c122"] = counts_c122[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c212"] = counts_c212[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c221"] = counts_c221[permute_idx[:(7*chunk_size)],:]
            counts_gt[batch_id]["c222"] = counts_c222[permute_idx[:(7*chunk_size)],:]

            counts_train.append(np.concatenate([
                counts_c111[permute_idx[:chunk_size],:],
                counts_c112[permute_idx[chunk_size:(2*chunk_size)],:],
                counts_c121[permute_idx[(2*chunk_size):(3*chunk_size)],:],
                counts_c211[permute_idx[(3*chunk_size):(4*chunk_size)],:],
                counts_c122[permute_idx[(4*chunk_size):(5*chunk_size)],:],
                counts_c212[permute_idx[(5*chunk_size):(6*chunk_size)],:],
                counts_c221[permute_idx[(6*chunk_size):(7*chunk_size)],:]], axis = 0))

            meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "condition 3", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size
                                        + [batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size 
                                        + [batch_id] * chunk_size)
            meta["condition 1"] = np.array([1] * chunk_size + [1] * chunk_size + [1] * chunk_size 
                                           + [2] * chunk_size + [1] * chunk_size + [2] * chunk_size 
                                           + [2] * chunk_size)
            meta["condition 2"] = np.array([1] * chunk_size + [1] * chunk_size + [2] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [2] * chunk_size)
            meta["condition 3"] = np.array([1] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [2] * chunk_size
                                           + [1] * chunk_size)
            
            meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chunk_size]], 
                                            label_annos[batch_id][permute_idx[(chunk_size):(2*chunk_size)]], 
                                            label_annos[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)]], 
                                            label_annos[batch_id][permute_idx[(3*chunk_size):(4*chunk_size)]],
                                            label_annos[batch_id][permute_idx[(4*chunk_size):(5*chunk_size)]],
                                            label_annos[batch_id][permute_idx[(5*chunk_size):(6*chunk_size)]],
                                            label_annos[batch_id][permute_idx[(6*chunk_size):(7*chunk_size)]]], axis = 0)
            meta_train.append(meta)

        elif ncondtypes == 4:
            chunk_size = int(ncells/16)
            counts_c1111 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1111.txt', sep = "\t", header = None).values.T
            counts_c1112 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1112.txt', sep = "\t", header = None).values.T
            counts_c1121 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1121.txt', sep = "\t", header = None).values.T
            counts_c1211 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1211.txt', sep = "\t", header = None).values.T
            counts_c1122 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1122.txt', sep = "\t", header = None).values.T
            counts_c1212 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1212.txt', sep = "\t", header = None).values.T
            counts_c1221 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1221.txt', sep = "\t", header = None).values.T
            counts_c1222 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c1222.txt', sep = "\t", header = None).values.T
            counts_c2111 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2111.txt', sep = "\t", header = None).values.T
            counts_c2112 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2112.txt', sep = "\t", header = None).values.T
            counts_c2121 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2121.txt', sep = "\t", header = None).values.T
            counts_c2211 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2211.txt', sep = "\t", header = None).values.T
            counts_c2122 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2122.txt', sep = "\t", header = None).values.T
            counts_c2212 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2212.txt', sep = "\t", header = None).values.T
            counts_c2221 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2221.txt', sep = "\t", header = None).values.T
            counts_c2222 = pd.read_csv(data_dir + f'GxB{batch_id + 1}_c2222.txt', sep = "\t", header = None).values.T
            
            counts_gt[batch_id] = {}
            counts_gt[batch_id]["c1111"] = counts_c1111[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1112"] = counts_c1112[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1121"] = counts_c1121[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1211"] = counts_c1211[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1122"] = counts_c1122[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1212"] = counts_c1212[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1221"] = counts_c1221[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c1222"] = counts_c1222[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2111"] = counts_c2111[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2112"] = counts_c2112[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2121"] = counts_c2121[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2211"] = counts_c2211[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2122"] = counts_c2122[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2212"] = counts_c2212[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2221"] = counts_c2221[permute_idx[:(15*chunk_size)],:]
            counts_gt[batch_id]["c2222"] = counts_c2222[permute_idx[:(15*chunk_size)],:]

            counts_train.append(np.concatenate([
                counts_c1111[permute_idx[:chunk_size],:],
                counts_c1112[permute_idx[chunk_size:(2*chunk_size)],:],
                counts_c1121[permute_idx[(2*chunk_size):(3*chunk_size)],:],
                counts_c1211[permute_idx[(3*chunk_size):(4*chunk_size)],:],
                counts_c1122[permute_idx[(4*chunk_size):(5*chunk_size)],:],
                counts_c1212[permute_idx[(5*chunk_size):(6*chunk_size)],:],
                counts_c1221[permute_idx[(6*chunk_size):(7*chunk_size)],:],
                counts_c1222[permute_idx[(7*chunk_size):(8*chunk_size)],:], 
                counts_c2111[permute_idx[(8*chunk_size):(9*chunk_size)],:], 
                counts_c2112[permute_idx[(9*chunk_size):(10*chunk_size)],:], 
                counts_c2121[permute_idx[(10*chunk_size):(11*chunk_size)],:], 
                counts_c2211[permute_idx[(11*chunk_size):(12*chunk_size)],:], 
                counts_c2122[permute_idx[(12*chunk_size):(13*chunk_size)],:], 
                counts_c2212[permute_idx[(13*chunk_size):(14*chunk_size)],:], 
                counts_c2221[permute_idx[(14*chunk_size):(15*chunk_size)],:]], axis = 0))

            meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "condition 3", "condition 4", "annos"])
            meta["batch"] = np.array([batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size
                                     + [batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size 
                                     + [batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size
                                     + [batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size
                                     + [batch_id] * chunk_size + [batch_id] * chunk_size + [batch_id] * chunk_size)
            meta["condition 1"] = np.array([1] * chunk_size + [1] * chunk_size + [1] * chunk_size
                                           + [1] * chunk_size + [1] * chunk_size + [1] * chunk_size  
                                           + [1] * chunk_size + [1] * chunk_size + [2] * chunk_size 
                                           + [2] * chunk_size + [2] * chunk_size + [2] * chunk_size 
                                           + [2] * chunk_size + [2] * chunk_size + [2] * chunk_size)
            meta["condition 2"] = np.array([1] * chunk_size + [1] * chunk_size + [1] * chunk_size
                                           + [2] * chunk_size + [1] * chunk_size + [2] * chunk_size
                                           + [2] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [1] * chunk_size + [1] * chunk_size + [2] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [2] * chunk_size)
            meta["condition 3"] = np.array([1] * chunk_size + [1] * chunk_size + [2] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [2] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [2] * chunk_size + [1] * chunk_size + [2] * chunk_size)
            meta["condition 4"] = np.array([1] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [2] * chunk_size
                                           + [1] * chunk_size + [2] * chunk_size + [1] * chunk_size
                                           + [2] * chunk_size + [1] * chunk_size + [1] * chunk_size
                                           + [2] * chunk_size + [2] * chunk_size + [1] * chunk_size)
            meta["annos"] = label_annos[batch_id][permute_idx[:(15*chunk_size)]]
            meta_train.append(meta)

    counts_train = np.concatenate(counts_train, axis = 0)
    meta_train = pd.concat(meta_train, axis = 0)
    # create training dataset
    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_train, condition_key = [f"condition {x + 1}" for x in range(ncondtypes)], batch_key = "batch")
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

    Ks = [8] + [2] * ncondtypes
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

    config_input = [1] * ncondtypes
    if ncondtypes == 2:
        idx = (meta_train["condition 1"] == config_input[0]) & (meta_train["condition 2"] == config_input[1]) & (meta_train["batch"] == 0)
        counts_input = counts_train[idx, :]
        meta_input = meta_train.loc[idx, :]
        counts_predict_gt = counts_gt["c22"][idx,:]
        meta_gt = meta_input.copy()
        meta_gt["condition 1"] = 2
        meta_gt["condition 2"] = 2

    elif ncondtypes == 3:
        idx = (meta_train["condition 1"] == config_input[0]) & (meta_train["condition 2"] == config_input[1]) & (meta_train["condition 3"] == config_input[2]) & (meta_train["batch"] == 0)
        counts_input = counts_train[idx, :]
        meta_input = meta_train.loc[idx, :]
        counts_predict_gt = counts_gt["c222"][idx,:]
        meta_gt = meta_input.copy()
        meta_gt["condition 1"] = 2
        meta_gt["condition 2"] = 2
        meta_gt["condition 3"] = 2

    elif ncondtypes == 4:
        idx = (meta_train["condition 1"] == config_input[0]) & (meta_train["condition 2"] == config_input[1]) & (meta_train["condition 3"] == config_input[2]) & (meta_train["condition 4"] == config_input[3]) & (meta_train["batch"] == 0)
        counts_input = counts_train[idx, :]
        meta_input = meta_train.loc[idx, :]
        counts_predict_gt = counts_gt["c2222"][idx,:]
        meta_gt = meta_input.copy()
        meta_gt["condition 1"] = 2
        meta_gt["condition 2"] = 2
        meta_gt["condition 3"] = 2
        meta_gt["condition 4"] = 2
        
    # predict count
    counts_predict_gt_denoised = model.predict_counts(input_counts = counts_predict_gt, meta_cells = meta_gt, condition_keys = [f"condition {x+1}" for x in range(ncondtypes)], 
                                                     batch_key = "batch", predict_conds = None, predict_batch = None)
    counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = [f"condition {x+1}" for x in range(ncondtypes)], 
                                    batch_key = "batch", predict_conds = [2] * ncondtypes, predict_batch = 0)

    counts_predict_gt = counts_predict_gt/(np.sum(counts_predict_gt, axis = 1, keepdims = True) + 1e-6)
    counts_predict_gt_denoised = counts_predict_gt_denoised/(np.sum(counts_predict_gt_denoised, axis = 1, keepdims = True) + 1e-6)
    counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
    counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

    mses_input = np.sum((counts_input - counts_predict_gt_denoised) ** 2, axis = 1)
    mses_scdisinfact = np.sum((counts_predict - counts_predict_gt_denoised) ** 2, axis = 1)
    # vector storing the pearson correlation for all cells
    pearson_input = np.array([stats.pearsonr(counts_input[i,:], counts_predict_gt_denoised[i,:])[0] for i in range(counts_predict_gt_denoised.shape[0])])
    pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_predict_gt_denoised[i,:])[0] for i in range(counts_predict_gt_denoised.shape[0])])
    # vector storing the R2 scores for all cells
    r2_input = np.array([r2_score(y_pred = counts_input[i,:], y_true = counts_predict_gt_denoised[i,:]) for i in range(counts_predict_gt_denoised.shape[0])])
    r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_predict_gt_denoised[i,:]) for i in range(counts_predict_gt_denoised.shape[0])])

    score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Prediction", "Dataset"])
    score["MSE"] = mses_scdisinfact
    score["MSE input"] = mses_input
    score["Pearson"] = pearsons_scdisinfact
    score["Pearson input"] = pearson_input
    score["R2"] = r2_scdisinfact
    score["R2 input"] = r2_input
    score["Prediction"] = "c" + "2"*ncondtypes
    score["Dataset"] = dataset_dir
    score_list.append(score)

scores = pd.concat(score_list, axis = 0)
scores.to_csv(result_dir + "prediction_scores.csv")

# In[]
auprc_dict = pd.DataFrame(columns = ["dataset", "condition", "ndiff_genes", "AUPRC", "AUPRC ratio", "ndiff"])

for dataset_dir in simulated_lists:
    data_dir = f"../data/simulated/{dataset_dir}/"
    ncells = eval(dataset_dir.split("_")[2])
    nbatches = eval(dataset_dir.split("_")[3])
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
    Ks = [8] + [2] * ncondtypes
    lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

    params = torch.load(result_dir + f"{dataset_dir}/scdisinfact_{Ks}_{lambs}.pth", map_location = device)

    for cond in range(ncondtypes):
        gt = np.zeros(ngenes)
        gt[cond*ndiff_genes:((cond+1)*ndiff_genes)] = 1
        inf = np.array(params[f"Enc_ds.{cond}.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        auprc_dict = pd.concat([auprc_dict, 
                                pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                        "ndiff_genes": [ndiff_genes], 
                                        "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "ndiff": [ndiff],
                                        "condition": [cond]
                                        })], axis = 0, ignore_index = True)

auprc_dict.to_csv(result_dir + "CKG_scores.txt", sep = "\t")

# In[]
auprc_dict = pd.read_csv(result_dir + "CKG_scores.txt", sep = "\t", index_col = 0)
pred_scores = pd.read_csv(result_dir + "prediction_scores.csv", index_col = 0)
pred_scores["MSE (ratio)"] = pred_scores["MSE"].values/pred_scores["MSE input"].values

plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (20,5))
# ax = fig.subplots(nrows = 1, ncols = 3)
ax = fig.subplots(nrows = 1, ncols = 3)
ax[0] = sns.barplot(x='Prediction', y='MSE', data=pred_scores, ax = ax[0], ci = None)
for i in ax[0].containers:
    ax[0].bar_label(i, fmt='%.1e')    
ax[0].set_xticklabels(["2 cond-types", "3 cond-types", "4 cond-types"])
fig.savefig(result_dir + "prediction_MSE.png", bbox_inches = "tight")


ax[1] = sns.barplot(x='Prediction', y='R2', data=pred_scores, ax = ax[1], ci = None)
for i in ax[1].containers:
    ax[1].bar_label(i, fmt='%.2f')    
ax[1].set_xticklabels(["2 cond-types", "3 cond-types", "4 cond-types"])
fig.tight_layout()
fig.savefig(result_dir + "prediction_R2.png", bbox_inches = "tight")

ax[2] = sns.barplot(x = "dataset", y ='AUPRC', data=auprc_dict, ax = ax[2], ci = None)
for i in ax[2].containers:
    ax[2].bar_label(i, fmt='%.2f')    
ax[2].set_xticklabels(["2 cond-types", "3 cond-types", "4 cond-types"])
fig.tight_layout()
fig.savefig(result_dir + "CKGs_AUPRC.png", bbox_inches = "tight")


# %%
