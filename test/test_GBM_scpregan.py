# In[]
import scanpy as sc
import scgen
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import os
from anndata import AnnData
from scPreGAN import *
import torch
import scipy.sparse as sp

torch.cuda.set_device(2)

# In[]
data_dir = "../data/GBM_treatment/Fig4/processed/"
result_dir = "GBM_treatment/Fig4_minibatch8/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# NOTE: orig.ident: patient id _ timepoint (should be batches), Patient: patient id, Timepoint: timepoint of sampling, Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cell = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
meta_cell_seurat = pd.read_csv(data_dir + "meta_cells_seurat.csv", sep = "\t", index_col = 0)
meta_cell["mstatus"] = meta_cell_seurat["mstatus"].values.squeeze()
counts = sp.load_npz(data_dir + "counts_rna.npz")

# condition
treatment_id, treatments = pd.factorize(meta_cell["treatment"].values.squeeze())
# one patient has multiple batches
patient_ids, patient_names = pd.factorize(meta_cell["patient_id"].values.squeeze())
# batches, use samples as batches
sample_ids, sample_names = pd.factorize(meta_cell["sample_id"].values.squeeze())

counts_array = []
meta_cells_array = []
for sample_id, sample_name in enumerate(sample_names):
    counts_array.append(counts[sample_ids == sample_id, :].toarray())
    meta_cells_array.append(meta_cell.iloc[sample_ids == sample_id, :])



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

# 1st train with ctrl in batches 1 & 2, stim1 in batches 3 & 4
# training adata
label_conditions_train = []
label_batches_train = []
label_annos_train = []
for sample_id, sample_name in enumerate(sample_names):
    label_conditions_train.append(meta_cells_array[sample_id]["treatment"].values.squeeze())
    label_batches_train.append(meta_cells_array[sample_id]["sample_id"].values.squeeze())
    label_annos_train.append(meta_cells_array[sample_id]["mstatus"].values.squeeze())

counts_train = np.concatenate(counts_array, axis = 0)
meta_obs_train = pd.DataFrame(columns = ["condition", "cell_type", "batch_id"])
meta_obs_train["condition"] = np.concatenate(label_conditions_train, axis = 0)
meta_obs_train["cell_type"] = np.concatenate(label_annos_train, axis = 0)
meta_obs_train["batch_id"] = np.concatenate(label_batches_train, axis = 0)
adata_train = AnnData(X = counts_train, obs = meta_obs_train)

# preprocessing, normalize the count
sc.pp.normalize_per_cell(adata_train)
# # log-transform
# sc.pp.log1p(adata_train)
# # standardize to unit variance and 0 mean
# sc.pp.scale(adata_train)
adata_train.write_h5ad(result_dir + "adata_train.h5ad")

# testing adata
adata_test = adata_train[adata_train.obs["condition"] == "0.2 uM panobinostat",:].copy()
    
model = Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)

train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                condition_key="condition",
                condition={"case": "vehicle (DMSO)", "control": "0.2 uM panobinostat"},
                cell_type_key="cell_type",
                out_of_sample_prediction=False,
                prediction_cell_type=None
                )

# training
model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)

# prediction, non-negative because of the relu at the output
pred = model.predict(control_adata=adata_test, cell_type_key="cell_type", condition_key="condition")
pred.obs = adata_test.obs

pred.write_h5ad(result_dir + "scPreGAN/pred.h5ad")

# complete the matrix
counts = adata_train.X
counts[adata_train.obs["condition"] == "0.2 uM panobinostat", :] = pred.X
sp.save_npz(result_dir + "scPreGAN/counts_scpregan.npz", sp.csr_matrix(counts))
# %%
