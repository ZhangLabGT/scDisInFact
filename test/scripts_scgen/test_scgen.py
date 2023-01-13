# In[]
import scanpy as sc
import scgen
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import os
from anndata import AnnData

# In[] Read in the dataset
# sigma = 0.2
# n_diff_genes = 20
# diff = 2
# ngenes = 500
# ncells_total = 10000 
# n_batches = 6
# data_dir = f"../data/simulated/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

n_batches = 6
data_dir = f"../data/simulated/" + sys.argv[1] + "/"

# TODO: randomly remove some celltypes?
counts_ctrls = []
counts_stims1 = []
counts_stims2 = []
# cell types
label_annos = []
# batch labels
label_batches = []
counts_gt = []
label_ctrls = []
label_stims1 = []
label_stims2 = []
np.random.seed(0)
for batch_id in range(n_batches):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts_ctrls.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T)
    counts_stims1.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T)
    counts_stims2.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T)
    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
    # batch labels
    label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
    label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
    label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
    label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))
    

# In[]
#------------------------------------------------------------------------------------------------------------------------------------------
#
# 1st testing scenario: 
# batches 1 & 2 are control, batches 3 & 4 are stimulation 1, batches 5 & 6 are stimulation 2. 
# Impute the control for batches 3, 4, 5, 6. 
#
# Since scGEN is only able to take 2 conditions, ran first with batches 1, 2, 3, 4, 
# and impute the control for batches 3 and 4. Ran then with batches
# 1, 2, 5, 6, and impute the control for batches 5, 6.
#
#------------------------------------------------------------------------------------------------------------------------------------------
# result_dir = f"./simulated/prediction/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/scGEN_scenario1/"
result_dir = f"./simulated/prediction/" + sys.argv[1] + "/scGEN_scenario1/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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

# preprocessing, normalize the count
sc.pp.normalize_per_cell(adata_train)

# testing adata
adata_test = adata_train[adata_train.obs["condition"] == "stim1",:].copy()

scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(adata_train)
model.train(
    max_epochs=100,
    batch_size=32,
    early_stopping=True,
    early_stopping_patience=25
)
# model.save(result_dir + "scGEN1", overwrite=True)

# In[]
# obtain latent space representation
latent_X = model.get_latent_representation()
latent_adata1 = sc.AnnData(X=latent_X, obs=adata_train.obs.copy())

# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred1, delta1 = model.predict(ctrl_key = "stim1", stim_key = "ctrl", adata_to_predict = adata_test)

# In[]
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

# preprocessing, normalize the count
sc.pp.normalize_per_cell(adata_train)

# testing adata
adata_test = adata_train[adata_train.obs["condition"] == "stim2",:].copy()

scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(adata_train)
model.train(
    max_epochs=100,
    batch_size=32,
    early_stopping=True,
    early_stopping_patience=25
)
# model.save(result_dir + "scGEN2", overwrite=True)

# In[]
# obtain latent space representation
latent_X = model.get_latent_representation()
latent_adata2 = sc.AnnData(X=latent_X, obs=adata_train.obs.copy())

# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred2, delta2 = model.predict(ctrl_key = "stim2", stim_key = "ctrl", adata_to_predict = adata_test)

# In[]
pred = pred1.concatenate(pred2)
for batch_id in [2,3,4,5]:
  pred_batch = pred[pred.obs["batch_id"] == "batch " + str(batch_id), :]
  X = pred_batch.X.T
  print(X.shape)
  np.savetxt(result_dir + f"GxC{batch_id + 1}_ctrl_impute.txt", X = X)

# In[]
'''
#------------------------------------------------------------------------------------------------------------------------------------------
#
# 2nd testing scenario: 
# 1. batch 1 with control and stimulation 1, batches 2, 3, 4, 5, 6 only have control. 
# Impute the stimulation 1 for batches 2, 3, 4, 5, 6. 
#
# 2. batch 1 with control and stimulation 2, batches 2, 3, 4, 5, 6 only have control.
# Impute the stimulation 2 for batches 2, 3, 4, 5, 6.
#------------------------------------------------------------------------------------------------------------------------------------------
# result_dir = f"./simulated/prediction/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/scGEN_scenario2/"
result_dir = f"./simulated/prediction/" + sys.argv[1] + "/scGEN_scenario2/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# 1. batch 1 with control and stimulation 1, batches 2, 3, 4, 5, 6 only have control. 
# training adata
label_conditions_train = [label_ctrls[0], label_stims1[0]]
label_batches_train = [label_batches[0], label_batches[0]]
label_annos_train = [label_annos[0], label_annos[0]]
counts_train = np.concatenate([counts_ctrls[0], counts_stims1[0]], axis = 0)
meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
adata_train = AnnData(X = counts_train, obs = meta_obs_train)

# testing adata
label_conditions_test = label_ctrls[1:]
label_batches_test = label_batches[1:]
label_annos_test = label_annos[1:]
counts_test = np.concatenate(counts_ctrls[1:], axis = 0)
meta_obs_test = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
meta_obs_test["condition"] = np.concatenate(label_conditions_test, axis = 0)
meta_obs_test["cell_type"] = np.concatenate(label_annos_test, axis = 0)
meta_obs_test["batch_id"] = np.concatenate(label_batches_test, axis = 0)
adata_test = AnnData(X = counts_test, obs = meta_obs_test)

scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(adata_train)
model.train(
    max_epochs=100,
    batch_size=32,
    early_stopping=True,
    early_stopping_patience=25
)
# model.save(result_dir + "scGEN1", overwrite=True)

# obtain latent space representation
latent_X = model.get_latent_representation()
latent_adata1 = sc.AnnData(X=latent_X, obs=adata_train.obs.copy())

# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred1, delta1 = model.predict(ctrl_key = "ctrl", stim_key = "stim1", adata_to_predict = adata_test)

for batch_id in [1,2,3,4,5]:
  pred_batch = pred1[pred1.obs["batch_id"] == "batch " + str(batch_id), :]
  X = pred_batch.X.T
  print(X.shape)
  np.savetxt(result_dir + f"GxC{batch_id + 1}_stim1_impute.txt", X = X)


# In[]
# 2. batch 1 with control and stimulation 2, batches 2, 3, 4, 5, 6 only have control. 
# training adata
label_conditions_train = [label_ctrls[0], label_stims2[0]]
label_batches_train = [label_batches[0], label_batches[0]]
label_annos_train = [label_annos[0], label_annos[0]]
counts_train = np.concatenate([counts_ctrls[0], counts_stims2[0]], axis = 0)
meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
adata_train = AnnData(X = counts_train, obs = meta_obs_train)

# testing adata
label_conditions_test = label_ctrls[1:]
label_batches_test = label_batches[1:]
label_annos_test = label_annos[1:]
counts_test = np.concatenate(counts_ctrls[1:], axis = 0)
meta_obs_test = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
meta_obs_test["condition"] = np.concatenate(label_conditions_test, axis = 0)
meta_obs_test["cell_type"] = np.concatenate(label_annos_test, axis = 0)
meta_obs_test["batch_id"] = np.concatenate(label_batches_test, axis = 0)
adata_test = AnnData(X = counts_test, obs = meta_obs_test)

scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(adata_train)
model.train(
    max_epochs=100,
    batch_size=32,
    early_stopping=True,
    early_stopping_patience=25
)
# model.save(result_dir + "scGEN1", overwrite=True)

# obtain latent space representation
latent_X = model.get_latent_representation()
latent_adata1 = sc.AnnData(X=latent_X, obs=adata_train.obs.copy())

# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred2, delta2 = model.predict(ctrl_key = "ctrl", stim_key = "stim2", adata_to_predict = adata_test)

for batch_id in [1,2,3,4,5]:
  pred_batch = pred2[pred2.obs["batch_id"] == "batch " + str(batch_id), :]
  X = pred_batch.X.T
  print(X.shape)
  np.savetxt(result_dir + f"GxC{batch_id + 1}_stim2_impute.txt", X = X)

'''
# %%
