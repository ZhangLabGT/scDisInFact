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
data_dir = f"../data/covid_integrated/"
result_dir = f"results_covid/learning/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


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

GxC1 = sp.load_npz(data_dir + "GxC1.npz")
GxC2 = sp.load_npz(data_dir + "GxC2.npz")
GxC3 = sp.load_npz(data_dir + "GxC3.npz")
# be careful with the ordering
meta_c1 = pd.read_csv(data_dir + "meta_arunachalam_2020.txt", sep = "\t", index_col = 0)
meta_c2 = pd.read_csv(data_dir + "meta_lee_2020.txt", sep = "\t", index_col = 0)
meta_c3 = pd.read_csv(data_dir + "meta_wilk_2020.txt", sep = "\t", index_col = 0)

meta = pd.concat([meta_c1, meta_c2, meta_c3], axis = 0)
genes = pd.read_csv(data_dir + "genes_shared.txt", index_col = 0).values.squeeze()
# process age
age = meta.age.values.squeeze().astype(object)
age[meta["age"] < 40] = "40-"
age[(meta["age"] >= 40)&(meta["age"] < 65)] = "40-65"
age[meta["age"] >= 65] = "65+"
meta["age"] = age

meta.loc[meta["disease_severity"] == "moderate", "disease_severity"] = "COVID"
meta.loc[meta["disease_severity"] == "severe", "disease_severity"] = "COVID"

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_array, meta_cells_array, 
                                                   condition_key = ["disease_severity", "age"], 
                                                   batch_key = "dataset")

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


