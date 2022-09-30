# In[]
import scanpy as sc
import scgen
import sys
sys.path.append("../src")
import pandas as pd
import scanpy as sc
import numpy as np
import os
from anndata import AnnData
import time
import torch
import scdisinfact
import matplotlib.pyplot as plt
from scPreGAN import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[] Read in the dataset
sigma = 0.2
n_diff_genes = 20
diff = 2
ngenes = 500
ncells_total = 10000 
n_batches = 6
data_dir = f"../data/simulated/imputation_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

# In[]
result_dir = "./simulated/runtime/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# runtime_scgen = pd.DataFrame(columns = ["total_cells", "runtime"])
# for subsample in [1, 2, 4, 5, 10]:
#     # read in the count matrix and meta data, then conduct sub-sampling
#     counts_ctrls = []
#     counts_stims1 = []
#     counts_stims2 = []
#     label_annos = []
#     label_batches = []
#     counts_gt = []
#     label_ctrls = []
#     label_stims1 = []
#     label_stims2 = []
#     np.random.seed(0)
#     for batch_id in range(n_batches):
#         counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T[::subsample,:])
#         counts_ctrls.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T[::subsample,:])
#         counts_stims1.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T[::subsample,:])
#         counts_stims2.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T[::subsample,:])
#         anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()[::subsample]
#         # annotation labels
#         label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
#         # batch labels
#         label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
#         label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
#         label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
#         label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))
        
#     # 1st train with ctrl in batches 1 & 2, stim1 in batches 3 & 4
#     # training adata
#     label_conditions_train = label_ctrls[0:2] + label_stims1[2:4]
#     label_batches_train = label_batches[0:4]
#     label_annos_train = label_annos[0:4]
#     counts_train = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:4], axis = 0)
#     meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
#     meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
#     meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
#     meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
#     adata_train = AnnData(X = counts_train, obs = meta_obs_train)

#     # testing adata
#     label_conditions_test = label_stims1[2:4]
#     label_batches_test = label_batches[2:4]
#     label_annos_test = label_annos[2:4]
#     counts_test = np.concatenate(counts_stims1[2:4], axis = 0)
#     meta_obs_test = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
#     meta_obs_test["condition"] = np.concatenate(label_conditions_test, axis = 0)
#     meta_obs_test["cell_type"] = np.concatenate(label_annos_test, axis = 0)
#     meta_obs_test["batch_id"] = np.concatenate(label_batches_test, axis = 0)
#     adata_test = AnnData(X = counts_test, obs = meta_obs_test)

#     start_time = time.time()
#     scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
#     model = scgen.SCGEN(adata_train)
#     model.train(
#         max_epochs=100,
#         batch_size=32,
#         early_stopping=True,
#         early_stopping_patience=25
#     )
#     end_time = time.time()
#     time_stage1 = end_time - start_time


#     # 2nd train with ctrl in batches 1 & 2, stim2 in batches 5 & 6
#     # training adata
#     label_conditions_train = label_ctrls[0:2] + label_stims2[4:]
#     label_batches_train = label_batches[0:2] + label_batches[4:]
#     label_annos_train = label_annos[0:2] + label_batches[4:]
#     counts_train = np.concatenate(counts_ctrls[0:2] + counts_stims2[4:], axis = 0)
#     meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type"])
#     meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
#     meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
#     meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
#     adata_train = AnnData(X = counts_train, obs = meta_obs_train)

#     # testing adata
#     label_conditions_test = label_stims2[4:]
#     label_batches_test = label_batches[4:]
#     label_annos_test = label_batches[4:]
#     counts_test = np.concatenate(counts_stims2[4:], axis = 0)
#     meta_obs_test = pd.DataFrame(columns = ["condition", "cell_type"])
#     meta_obs_test["condition"] = np.concatenate(label_conditions_test, axis = 0)
#     meta_obs_test["cell_type"] = np.concatenate(label_annos_test, axis = 0)
#     meta_obs_test["batch_id"] = np.concatenate(label_batches_test, axis = 0)
#     adata_test = AnnData(X = counts_test, obs = meta_obs_test)

#     start_time = time.time()
#     scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
#     model = scgen.SCGEN(adata_train)
#     model.train(
#         max_epochs=100,
#         batch_size=32,
#         early_stopping=True,
#         early_stopping_patience=25
#     )

#     end_time = time.time()
#     time_stage2 = end_time - start_time

#     time_total = time_stage1 + time_stage2

#     runtime_scgen = runtime_scgen.append({
#         "total_cells": int(10000/subsample),
#         "runtime": time_total
#     }, ignore_index = True)

# runtime_scgen.to_csv(result_dir + "runtime_scgen.csv")

# In[]
# runtime_scdisinfact = pd.DataFrame(columns = ["total_cells", "runtime"])

# for subsample in [1, 2, 4, 5, 10]:
#     # read in the count matrix and meta data, then conduct sub-sampling
#     counts_ctrls = []
#     counts_stims1 = []
#     counts_stims2 = []
#     label_annos = []
#     label_batches = []
#     counts_gt = []
#     label_ctrls = []
#     label_stims1 = []
#     label_stims2 = []
#     np.random.seed(0)
#     for batch_id in range(n_batches):
#         counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T[::subsample,:])
#         counts_ctrls.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T[::subsample,:])
#         counts_stims1.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T[::subsample,:])
#         counts_stims2.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T[::subsample,:])
#         anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()[::subsample]
#         # annotation labels
#         label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
#         # batch labels
#         label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
#         label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
#         label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
#         label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))
    
#     # generate the datasets
#     label_conditions = label_ctrls[0:2] + label_stims1[2:4] + label_stims2[4:]
#     counts = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:4] + counts_stims2[4:], axis = 0)
#     condition_ids, condition_names = pd.factorize(np.concatenate(label_conditions, axis = 0))
#     batch_ids, batch_names = pd.factorize(np.concatenate(label_batches, axis = 0))
#     anno_ids, anno_names = pd.factorize(np.concatenate(label_annos, axis = 0))

#     datasets = []
#     for batch_id, batch_name in enumerate(batch_names):
#             datasets.append(scdisinfact.dataset(counts = counts[batch_ids == batch_id,:], 
#                                                 anno = anno_ids[batch_ids == batch_id], 
#                                                 diff_labels = [condition_ids[batch_ids == batch_id]], 
#                                                 batch_id = batch_ids[batch_ids == batch_id]))

#     start_time = time.time()
#     reg_mmd_comm = 1e-4
#     reg_mmd_diff = 1e-4
#     reg_gl = 0.1
#     reg_tc = 0.5
#     reg_class = 1
#     reg_kl = 1e-6
#     # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
#     lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
#     Ks = [8, 4]
#     nepochs = 100
#     interval = 10

#     # training model
#     model = scdisinfact.scdisinfact(datasets = datasets, Ks = Ks, batch_size = 64, interval = interval, lr = 5e-4, 
#                                     reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
#                                     reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

#     # train_joint is more efficient, but does not work as well compared to train
#     model.train()
#     losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)
#     _ = model.eval()

#     end_time = time.time()
#     time_total = end_time - start_time

#     runtime_scdisinfact = runtime_scdisinfact.append({
#         "total_cells": int(10000/subsample),
#         "runtime": time_total
#     }, ignore_index = True)

# runtime_scdisinfact.to_csv(result_dir + "runtime_scdisinfact.csv")

# In[]
runtime_scpregan = pd.DataFrame(columns = ["total_cells", "runtime"])
for subsample in [1, 2, 4, 5, 10]:
    # read in the count matrix and meta data, then conduct sub-sampling
    counts_ctrls = []
    counts_stims1 = []
    counts_stims2 = []
    label_annos = []
    label_batches = []
    counts_gt = []
    label_ctrls = []
    label_stims1 = []
    label_stims2 = []
    np.random.seed(0)
    for batch_id in range(n_batches):
        counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T[::subsample,:])
        counts_ctrls.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T[::subsample,:])
        counts_stims1.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T[::subsample,:])
        counts_stims2.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T[::subsample,:])
        anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()[::subsample]
        # annotation labels
        label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
        # batch labels
        label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
        label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
        label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
        label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))
        
    # 1st train with ctrl in batches 1 & 2, stim1 in batches 3 & 4
    # training adata
    label_conditions_train = label_ctrls[0:2] + label_stims1[2:4]
    label_batches_train = label_batches[0:4]
    label_annos_train = label_annos[0:4]
    counts_train = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:4], axis = 0)
    meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
    meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
    meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
    meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
    adata_train = AnnData(X = counts_train, obs = meta_obs_train)
    sc.pp.normalize_per_cell(adata_train)
    adata_train.write_h5ad(result_dir + "adata_train.h5ad")

    # testing adata
    adata_test = adata_train[adata_train.obs["condition"] == "stim1",:].copy()

    start_time = time.time()
    model = Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)

    train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                    condition_key="condition",
                    condition={"case": "ctrl", "control": "stim1"},
                    cell_type_key="cell_type",
                    out_of_sample_prediction=False,
                    prediction_cell_type=None
                    )

    # training
    model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)

    end_time = time.time()
    time_stage1 = end_time - start_time

    # prediction, non-negative because of the relu at the output
    pred1 = model.predict(control_adata=adata_test, cell_type_key="cell_type", condition_key="condition")
    pred1.obs = adata_test.obs

    # 2nd train with ctrl in batches 1 & 2, stim2 in batches 5 & 6
    # training adata
    label_conditions_train = label_ctrls[0:2] + label_stims2[4:]
    label_batches_train = label_batches[0:2] + label_batches[4:]
    label_annos_train = label_annos[0:2] + label_batches[4:]
    counts_train = np.concatenate(counts_ctrls[0:2] + counts_stims2[4:], axis = 0)
    meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type"])
    meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
    meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
    meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
    adata_train = AnnData(X = counts_train, obs = meta_obs_train)

    sc.pp.normalize_per_cell(adata_train)
    adata_train.write_h5ad(result_dir + "adata_train.h5ad")

    # testing adata
    adata_test = adata_train[adata_train.obs["condition"] == "stim2",:].copy()

    start_time = time.time()
    model = Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)

    train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                    condition_key="condition",
                    condition={"case": "ctrl", "control": "stim2"},
                    cell_type_key="cell_type",
                    out_of_sample_prediction=False,
                    prediction_cell_type=None
                    )

    # training
    model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)

    end_time = time.time()
    time_stage2 = end_time - start_time

    # prediction
    pred2 = model.predict(control_adata=adata_test, cell_type_key="cell_type", condition_key="condition")
    pred2.obs = adata_test.obs

    time_total = time_stage1 + time_stage2

    runtime_scpregan = runtime_scpregan.append({
        "total_cells": int(10000/subsample),
        "runtime": time_total
    }, ignore_index = True)

runtime_scpregan.to_csv(result_dir + "runtime_scpregan.csv")
# In[]
runtime_scdisinfact = pd.read_csv("./simulated/runtime/runtime_scdisinfact.csv", index_col = 0)
runtime_scgen = pd.read_csv("./simulated/runtime/runtime_scgen.csv", index_col = 0)
runtime_scpregan = pd.read_csv("./simulated/runtime/runtime_scpregan.csv", index_col = 0)
runtime_scinsight = pd.read_csv("./simulated/runtime/runtime_scinsight.csv", index_col = 0)
# mins to secs
runtime_scinsight["times.total"] = runtime_scinsight["times.total"].values * 60
# runtime_scinsight_nosearch = pd.read_csv("./simulated/runtime/runtime_scinsight_nosearch.csv", index_col = 0)
# # mins to secs
# runtime_scinsight_nosearch["times.total"] = runtime_scinsight_nosearch["times.total"].values * 60

ncells = runtime_scdisinfact["total_cells"].values.squeeze()

plt.rcParams["font.size"] = 20

fig = plt.figure(figsize = (6, 4))
ax = fig.add_subplot()
ax.plot(ncells, runtime_scdisinfact["runtime"].values.squeeze(), "r-*", label = "scDisInFact")
ax.plot(ncells, runtime_scgen["runtime"].values.squeeze(), "b-*", label = "scGEN")
ax.plot(ncells, runtime_scpregan["runtime"].values.squeeze(), "c-*", label = "scPreGAN")
ax.plot(ncells, runtime_scinsight["times.total"].values.squeeze(), "g-*", label = "scINSIGHT")
# ax.plot(ncells, runtime_scinsight_nosearch["runtime"].values.squeeze(), "r*", label = "scINSIGHT (no K search)")
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Number of cells")
ax.set_ylabel("Runtime (sec)")
fig.savefig("./simulated/runtime/runtime.png", bbox_inches = "tight")

# %%
