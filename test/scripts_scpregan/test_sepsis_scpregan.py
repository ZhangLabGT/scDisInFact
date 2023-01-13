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
data_dir = "../data/sepsis/sepsis_batch_raw/split_batches/"
result_dir = "sepsis_raw/secondary/batch_batches/scPreGAN/"
genes = np.loadtxt(data_dir + "genes_raw.txt", dtype = object)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# read in the dataset
counts_array = []
meta_cell_array = []
for batch_id in range(1, 36):
    # not batches correspond to id 25 and 30
    if batch_id not in [25, 30]:
        # one patient can be in multiple batches, for both primary and secondary
        meta_cell = pd.read_csv(data_dir + f"meta_Batch_{batch_id}.csv", index_col = 0)
        # one processing batch have multiple conditions.        
        # Cell_Type stores the major cell type, Cell_State stores the detailed cell type, 
        # Cohort stores the condition of disease, and biosample_id stores the tissue type, 
        # Patient for the patient ID
        meta_cell_array.append(meta_cell)
        counts = sp.load_npz(data_dir + f"counts_Batch_{batch_id}.npz")
        counts_array.append(counts.toarray())

counts = np.concatenate(counts_array, axis = 0)
adata = AnnData(X = counts)
adata.obs = pd.concat(meta_cell_array, axis = 0)
adata.var.index = genes

adata_primary = adata[(adata.obs["Cohort"] == "Leuk-UTI")|(adata.obs["Cohort"] == "Int-URO")|(adata.obs["Cohort"] == "URO"), :]
adata_secondary = adata[(adata.obs["Cohort"] == "Bac-SEP")|(adata.obs["Cohort"] == "ICU-NoSEP")|(adata.obs["Cohort"] == "ICU-SEP"), :]
adata_pbmc = adata_primary[adata_primary.obs["biosample_id"] == "CD45",:]
adata_leuk_uti = adata[adata.obs["Cohort"] == "Leuk-UTI", :]
adata_int_uro = adata[adata.obs["Cohort"] == "Int-URO", :]
adata_uro = adata[adata.obs["Cohort"] == "URO", :]
adata_control = adata[adata.obs["Cohort"] == "Control", :]
adata_bac_sep = adata[adata.obs["Cohort"] == "Bac-SEP", :]
adata_icu_nosep = adata[adata.obs["Cohort"] == "ICU-NoSEP", :]
adata_icu_sep = adata[adata.obs["Cohort"] == "ICU-SEP", :]

adata = adata_secondary
# preprocessing, normalize the count
sc.pp.normalize_per_cell(adata)

# In[]
# Cohort: conditions
# Batches: batch id
# Cell_Type/Cell_State: cell type

# testing adata: ICU-SEP
adata_test = adata[adata.obs["Cohort"] == "ICU-SEP",:].copy()
adata_train = adata[(adata.obs["Cohort"] == "ICU-SEP")|(adata.obs["Cohort"] == "ICU-NoSEP"),:].copy()
adata_train.write_h5ad(result_dir + "adata_train.h5ad")

model = Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)
train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                condition_key="Cohort",
                condition={"case": "ICU-NoSEP", "control": "ICU-SEP"},
                cell_type_key="Cell_Type",
                out_of_sample_prediction=False,
                prediction_cell_type=None
                )
# training
model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)
# prediction, non-negative because of the relu at the output
pred = model.predict(control_adata=adata_test, cell_type_key="Cell_Type", condition_key="Cohort")
# complete the matrix
counts = adata.X.copy()
counts[adata.obs["Cohort"] == "ICU-SEP", :] = pred.X

# testing adata: Bac-SEP
adata_test = adata[adata.obs["Cohort"] == "Bac-SEP",:].copy()
adata_train = adata[(adata.obs["Cohort"] == "Bac-SEP")|(adata.obs["Cohort"] == "ICU-NoSEP"),:].copy()
adata_train.write_h5ad(result_dir + "adata_train.h5ad")
model = Model(n_features=adata_train.shape[1], z_dim=16, use_cuda=True)
train_data = model.load_anndata(data_path=result_dir + "adata_train.h5ad",
                condition_key="Cohort",
                condition={"case": "ICU-NoSEP", "control": "Bac-SEP"},
                cell_type_key="Cell_Type",
                out_of_sample_prediction=False,
                prediction_cell_type=None
                )
model.train(train_data=train_data, model_path = result_dir + "model", log_path = result_dir + "logger", niter=20000)
pred = model.predict(control_adata=adata_test, cell_type_key="Cell_Type", condition_key="Cohort")
counts[adata.obs["Cohort"] == "Bac-SEP", :] = pred.X

sp.save_npz(result_dir + "counts_scpregan.npz", sp.csr_matrix(counts))
# %%
