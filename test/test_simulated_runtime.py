#!/usr/bin/env python
# coding: utf-8

import sys, os
import time
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact

import warnings
warnings.filterwarnings('ignore')

#from anndata import AnnData
#import scgen

#import scanpy as sc
#import scPreGAN

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read in the dataset
sigma = 0.4
n_diff_genes = 50
diff = 4
ngenes = 500
ncells_total = 10000
n_batches = 2
data_dir = f"../data/simulated/unif/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
result_dir = f"./results_simulated/runtime/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_contr = 0.01
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2

Ks = [8, 2, 2]

batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4

runtime_scdisinfact = pd.DataFrame(columns = ["total_cells", "runtime"], dtype=object)

for subsample in [1, 2, 4, 5, 10]:
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

    start_time = time.time()
    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")    
    model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                    reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

    model.train()
    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")

    end_time = time.time()
    time_total = end_time - start_time

    runtime_scdisinfact = pd.concat([runtime_scdisinfact, pd.DataFrame({
             "total_cells": int(10000/subsample),
             "runtime": time_total
         }, index=[0])], ignore_index = True)


runtime_scdisinfact.to_csv(result_dir + "runtime_scdisinfact.csv")

'''
runtime_scgen = pd.DataFrame(columns = ["total_cells", "runtime"], dtype=object)

for subsample in [1, 2, 4, 5, 10]:
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
    counts = []
    meta_cell_batch = []
    meta_cell_condition = []
    meta_cell_annos = []

    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally two combinations of conditions, separate the cells into two groups
        chunk_size = int(counts_gt[batch_id].shape[0]/2)
        counts.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                           counts_stim_healthy[batch_id][permute_idx[chunk_size:],:]], axis = 0))
        meta_cell_batch.append(np.array([batch_id] * counts_gt[batch_id].shape[0]))
        meta_cell_condition.append(np.array(["ctrl"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - chunk_size)))
        meta_cell_annos.append(label_annos[batch_id][permute_idx])

    counts_train = np.concatenate(counts, axis = 0)
    meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
    meta_obs_train["condition"] = np.concatenate(meta_cell_condition, axis = 0)
    meta_obs_train["cell_type"] = np.concatenate(meta_cell_annos, axis = 0)
    meta_obs_train["batch_id"] = np.concatenate(meta_cell_batch, axis = 0)
      
    adata_train = AnnData(X = counts_train, obs = meta_obs_train)
    
    start_time = time.time()

    scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(adata_train)
    model.train(
         max_epochs=100,
         batch_size=32,
         early_stopping=True,
         early_stopping_patience=25
    )
    
    end_time = time.time()
    time_stage1 = end_time - start_time
    
    np.random.seed(0)
    counts = []
    meta_cell_batch = []
    meta_cell_condition = []
    meta_cell_annos = []

    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally two combinations of conditions, separate the cells into two groups
        chunk_size = int(counts_gt[batch_id].shape[0]/2)
        counts.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                           counts_ctrl_severe[batch_id][permute_idx[chunk_size:],:]], axis = 0))
        meta_cell_batch.append(np.array([batch_id] * counts_gt[batch_id].shape[0]))
        meta_cell_condition.append(np.array(["healthy"] * chunk_size + ["severe"] * (counts_gt[batch_id].shape[0] - chunk_size)))
        meta_cell_annos.append(label_annos[batch_id][permute_idx])
        
    counts_train = np.concatenate(counts, axis = 0)
    meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
    meta_obs_train["condition"] = np.concatenate(meta_cell_condition, axis = 0)
    meta_obs_train["cell_type"] = np.concatenate(meta_cell_annos, axis = 0)
    meta_obs_train["batch_id"] = np.concatenate(meta_cell_batch, axis = 0)
        
    adata_train = AnnData(X = counts_train, obs = meta_obs_train)
    
    start_time = time.time()

    scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(adata_train)
    model.train(
         max_epochs=100,
         batch_size=32,
         early_stopping=True,
         early_stopping_patience=25
    )
    
    end_time = time.time()
    time_stage2 = end_time - start_time
    
    time_total = time_stage1 + time_stage2
    
    runtime_scgen = pd.concat([runtime_scgen, pd.DataFrame({
             "total_cells": int(10000/subsample),
             "runtime": time_total
         }, index=[0])], ignore_index = True)

    
runtime_scgen.to_csv(result_dir + "runtime_scgen.csv")

runtime_scpregan = pd.DataFrame(columns = ["total_cells", "runtime"], dtype=object)

for subsample in [1, 2, 4, 5, 10]:
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

    counts = []
    meta_cell_batch = []
    meta_cell_condition = []
    meta_cell_annos = []

    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally two combinations of conditions, separate the cells into two groups
        chunk_size = int(counts_gt[batch_id].shape[0]/2)
        counts.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                           counts_stim_healthy[batch_id][permute_idx[chunk_size:],:]], axis = 0))
        meta_cell_batch.append(np.array([batch_id] * counts_gt[batch_id].shape[0]))
        meta_cell_condition.append(np.array(["ctrl"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - chunk_size)))
        meta_cell_annos.append(label_annos[batch_id][permute_idx])
    
    counts_train = np.concatenate(counts, axis = 0)
    meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
    meta_obs_train["condition"] = np.concatenate(meta_cell_condition, axis = 0)
    meta_obs_train["cell_type"] = np.concatenate(meta_cell_annos, axis = 0)
    meta_obs_train["batch_id"] = np.concatenate(meta_cell_batch, axis = 0)

    adata_train = AnnData(X = counts_train, obs = meta_obs_train)
    sc.pp.normalize_per_cell(adata_train)
    adata_train.write_h5ad(result_dir + "adata_train.h5ad")
        
    start_time = time.time()

    model = scPreGAN.Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)
    train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                    condition_key="condition",
                    condition={"case": "ctrl", "control": "stim"},
                    cell_type_key="cell_type",
                    out_of_sample_prediction=False,
                    prediction_cell_type=None
                    )

    model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)
    
    end_time = time.time()
    time_stage1 = end_time - start_time
    
    np.random.seed(0)
    counts = []
    meta_cell_batch = []
    meta_cell_condition = []
    meta_cell_annos = []

    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally two combinations of conditions, separate the cells into two groups
        chunk_size = int(counts_gt[batch_id].shape[0]/2)
        counts.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                           counts_ctrl_severe[batch_id][permute_idx[chunk_size:],:]], axis = 0))
        meta_cell_batch.append(np.array([batch_id] * counts_gt[batch_id].shape[0]))
        meta_cell_condition.append(np.array(["healthy"] * chunk_size + ["severe"] * (counts_gt[batch_id].shape[0] - chunk_size)))
        meta_cell_annos.append(label_annos[batch_id][permute_idx])
    
    counts_train = np.concatenate(counts, axis = 0)
    meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
    meta_obs_train["condition"] = np.concatenate(meta_cell_condition, axis = 0)
    meta_obs_train["cell_type"] = np.concatenate(meta_cell_annos, axis = 0)
    meta_obs_train["batch_id"] = np.concatenate(meta_cell_batch, axis = 0)
        
    adata_train = AnnData(X = counts_train, obs = meta_obs_train)
    adata_train.write_h5ad(result_dir + "adata_train.h5ad")
    
    start_time = time.time()

    model = scPreGAN.Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)
    train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                    condition_key="condition",
                    condition={"case": "healthy", "control": "severe"},
                    cell_type_key="cell_type",
                    out_of_sample_prediction=False,
                    prediction_cell_type=None
                    )
    model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)
    
    end_time = time.time()
    time_stage2 = end_time - start_time
    
    time_total = time_stage1 + time_stage2

    runtime_scpregan = pd.concat([runtime_scpregan, pd.DataFrame({
             "total_cells": int(10000/subsample),
             "runtime": time_total
         }, index=[0])], ignore_index = True)

    
runtime_scpregan.to_csv(result_dir + "runtime_scpregan.csv")

'''
runtime_scdisinfact = pd.read_csv(result_dir + "runtime_scdisinfact.csv", index_col = 0)
runtime_scgen = pd.read_csv(result_dir + "runtime_scgen.csv", index_col = 0)
runtime_scpregan = pd.read_csv(result_dir + "runtime_scpregan.csv", index_col = 0)
runtime_scinsight = pd.read_csv(result_dir + "runtime_scinsight.csv", index_col = 0)
runtime_scinsight.loc[1:2,"times.total"] = runtime_scinsight.loc[1:2,"times.total"].values * 60 * 60
runtime_scinsight.loc[3:,"times.total"] = runtime_scinsight.loc[3:,"times.total"].values * 60

ncells = runtime_scdisinfact["total_cells"].values.squeeze()

plt.rcParams["font.size"] = 20

fig = plt.figure(figsize = (6, 4))
ax = fig.add_subplot()
ax.plot(ncells, runtime_scdisinfact["runtime"].values.squeeze(), "r-*", label = "scDisInFact")
ax.plot(ncells, runtime_scgen["runtime"].values.squeeze(), "b-*", label = "scGEN")
ax.plot(ncells, runtime_scpregan["runtime"].values.squeeze(), "c-*", label = "scPreGAN")
ax.plot(ncells, runtime_scinsight["times.total"].values.squeeze(), "g-*", label = "scINSIGHT")
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Number of cells")
ax.set_ylabel("Runtime (sec)")
fig.savefig(result_dir + "runtime.png", bbox_inches = "tight")


