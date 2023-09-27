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

import scipy.sparse as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read in the dataset
data_dir = f"../data/GBM_treatment/Fig4/processed/"
result_dir = f"./results_GBM_treatment/learning/Fig4_patient/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [8, 2]

batch_size = 64
nepochs = 200
interval = 5
lr = 5e-4

genes = np.loadtxt(data_dir + "genes.txt", dtype = object)
# orig.ident: patient id _ timepoint (should be batches), 
# Patient: patient id, 
# Timepoint: timepoint of sampling, 
# Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cell = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
meta_cell_seurat = pd.read_csv(data_dir + "meta_cells_seurat.csv", sep = "\t", index_col = 0)
meta_cell["mstatus"] = meta_cell_seurat["mstatus"].values.squeeze()
meta_cell.loc[(meta_cell["mstatus"] != "Myeloid") & ((meta_cell["mstatus"] != "Oligodendrocytes") & (meta_cell["mstatus"] != "tumor")), "mstatus"] = "Other"
counts = sp.load_npz(data_dir + "counts_rna.npz")

data_dict_train = scdisinfact.create_scdisinfact_dataset(counts, meta_cell, condition_key = ["treatment"], batch_key = "patient_id", batch_cond_key = "sample_id")

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


