# In[]
import scanpy as sc
import scgen
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import os
from anndata import AnnData
import scipy.sparse as sp
import torch

# In[]
data_dir = "../data/sepsis/sepsis_batch_raw/split_batches/"
result_dir = "sepsis_raw/secondary/batch_batches/scGEN/"
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
sc.pp.log1p(adata)


# In[]
# Cohort: conditions
# Batches: batch id
# Cell_Type/Cell_State: cell type

# testing adata: ICU-SEP
adata_test = adata[adata.obs["Cohort"] == "ICU-SEP",:].copy()
adata_train = adata[(adata.obs["Cohort"] == "ICU-SEP")|(adata.obs["Cohort"] == "ICU-NoSEP"),:].copy()
scgen.SCGEN.setup_anndata(adata_train, batch_key="Cohort", labels_key="Cell_Type")
model = scgen.SCGEN(adata_train)
model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)
# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred, delta = model.predict(ctrl_key = "ICU-SEP", stim_key = "ICU-NoSEP", adata_to_predict = adata_test)
# complete the matrix
counts = adata.X
counts[adata.obs["Cohort"] == "ICU-SEP", :] = pred.X

# testing adata: Bac-SEP
adata_test = adata[adata.obs["Cohort"] == "Bac-SEP",:].copy()
adata_train = adata[(adata.obs["Cohort"] == "Bac-SEP")|(adata.obs["Cohort"] == "ICU-NoSEP"),:].copy()
scgen.SCGEN.setup_anndata(adata_train, batch_key="Cohort", labels_key="Cell_Type")
model = scgen.SCGEN(adata_train)
model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)
pred, delta = model.predict(ctrl_key = "Bac-SEP", stim_key = "ICU-NoSEP", adata_to_predict = adata_test)
counts[adata.obs["Cohort"] == "Bac-SEP", :] = pred.X

sp.save_npz(result_dir + "counts_scgen.npz", sp.csr_matrix(counts))

# %%
