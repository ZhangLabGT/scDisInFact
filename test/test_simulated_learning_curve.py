import sys, os
import time
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read in the dataset
sigma = 0.4
n_diff_genes = 100
diff = 8
ngenes = 500
ncells_total = 10000
n_batches = 2

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [8, 2, 2]

batch_size = 64
nepochs = 200
interval = 5
lr = 5e-4


Ks = [8, 4, 4]

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


for dataset_dir in simulated_lists:
    print("# -------------------------------------------------------------------------------------------")
    print("#")
    print('# Dataset:' + dataset_dir)
    print("#")
    print("# -------------------------------------------------------------------------------------------")

    data_dir = "../data/simulated/unif/" + dataset_dir + "/"
    result_dir = "./results_simulated/learning/" + dataset_dir + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    counts_gt = []
    counts_ctrl_healthy = []
    counts_ctrl_severe = []
    counts_stim_healthy = []
    counts_stim_severe = []
    label_annos = []

    for batch_id in range(n_batches):
        counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T[:,:])
        counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T[:,:])
        counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T[:,:])
        counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T[:,:])
        counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T[:,:])

        anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()[:]
        # annotation labels
        label_annos.append(np.array([('cell type '+str(i)) for i in anno]))

    np.random.seed(0)
    counts = []
    meta_cells = []

    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally four combinations of conditions, separate the cells into four groups
        chunk_size = int(counts_gt[batch_id].shape[0]/4)
        counts.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                    counts_ctrl_severe[batch_id][permute_idx[chunk_size:(2*chunk_size)],:],
                                    counts_stim_healthy[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:],
                                    counts_stim_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0))
        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"], dtype=object)
        meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["stim"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chunk_size + ["severe"] * chunk_size + ["healthy"] * chunk_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["annos"] = label_annos[batch_id][permute_idx]
        meta_cells.append(meta_cell)

    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

    model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                    reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

    model.train()
    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")

    loss_df = pd.DataFrame({
             "epochs": pd.Series(range(0,nepochs+1,interval)),
             "loss recon" : losses[1],
             "loss kl" : (sum(i) for i in zip(losses[2], losses[3])),
             "loss mmd" : (sum(i) for i in zip(losses[4], losses[5])), 
             "loss classi" : losses[6], 
             "loss gl" : losses[7],
         })

    loss_df.to_csv(result_dir + "loss_scdisinfact.csv")

    epochs = loss_df["epochs"].values.squeeze()

    plt.rcParams["font.size"] = 20

    fig = plt.figure(figsize = (6, 4))
    ax = fig.add_subplot()
    ax.plot(epochs, loss_df["loss recon"].values.squeeze(), "r-*", label = "Reconstruction")
    ax.plot(epochs, loss_df["loss kl"].values.squeeze(), "b-*", label = "KL")
    ax.plot(epochs, loss_df["loss mmd"].values.squeeze(), "c-*", label = "MMD")
    ax.plot(epochs, loss_df["loss classi"].values.squeeze(), "g-*", label = "Classifcation")
    ax.plot(epochs, loss_df["loss gl"].values.squeeze(), "m-*", label = "GL")
    ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("loss")
    fig.savefig(result_dir + "learning.png", bbox_inches = "tight")


