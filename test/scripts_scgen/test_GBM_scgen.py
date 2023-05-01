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
data_dir = "../../data/GBM_treatment/Fig4/processed/"
result_dir = "../results_GBM_treatment/Fig4_patient/scGEN/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

genes = np.loadtxt(data_dir + "genes.txt", dtype = np.object)
# NOTE: orig.ident: patient id _ timepoint (should be batches), Patient: patient id, Timepoint: timepoint of sampling, Pseudotime_name/Pseudotime: severity of the disease (should be used as condition)
meta_cells = pd.read_csv(data_dir + "meta_cells.csv", sep = "\t", index_col = 0)
meta_cells_seurat = pd.read_csv(data_dir + "meta_cells_seurat.csv", sep = "\t", index_col = 0)
meta_cells["mstatus"] = meta_cells_seurat["mstatus"].values.squeeze()
counts = sp.load_npz(data_dir + "counts_rna.npz")


# In[]
# create training and testing dataset
test_idx = (meta_cells["sample_id"] == "PW034-705").values.squeeze()
train_idx = ~test_idx
# input matrix, considering two cases: 1. vehicle (DMSO) in the same batch; 2. vehicle (DMSO) in a different batch
# the first case
input_idx = train_idx & ((meta_cells["treatment"] == "vehicle (DMSO)") & (meta_cells["patient_id"] == "PW034")).values.squeeze()

counts_train = counts[train_idx,:]
meta_train = meta_cells.loc[train_idx,:]
counts_gt = counts[test_idx,:]
meta_gt = meta_cells.loc[test_idx,:]
counts_input = counts[input_idx,:]
meta_input = meta_cells.loc[input_idx,:]

adata_train = AnnData(X = counts_train, obs = meta_train)
sc.pp.normalize_per_cell(adata_train, counts_per_cell_after = 100)

adata_gt = AnnData(X = counts_gt, obs = meta_gt)
sc.pp.normalize_per_cell(adata_gt, counts_per_cell_after = 100)

adata_input = AnnData(X = counts_input, obs = meta_input)
sc.pp.normalize_per_cell(adata_input, counts_per_cell_after = 100)

# In[]
scgen.SCGEN.setup_anndata(adata_train, batch_key="treatment", labels_key="mstatus")
model = scgen.SCGEN(adata_train)
model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)

# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred, delta = model.predict(ctrl_key = "vehicle (DMSO)", stim_key = "0.2 uM panobinostat", adata_to_predict = adata_input)
pred.write_h5ad(result_dir + "samebatch/pred.h5ad")

counts_pred = pred.X
meta_pred = pred.obs
counts_pred = counts_pred * (counts_pred > 0)
sp.save_npz(result_dir + "samebatch/counts_scgen.npz", matrix = sp.csr_matrix(counts_pred))
meta_pred.to_csv(result_dir + "samebatch/meta_scgen.csv")


# In[]
# create training and testing dataset
test_idx = (meta_cells["sample_id"] == "PW034-705").values.squeeze()
train_idx = ~test_idx
# the second case
input_idx = train_idx & ((meta_cells["treatment"] == "vehicle (DMSO)")  & (meta_cells["patient_id"] == "PW030")).values.squeeze()

counts_train = counts[train_idx,:]
meta_train = meta_cells.loc[train_idx,:]
counts_gt = counts[test_idx,:]
meta_gt = meta_cells.loc[test_idx,:]
counts_input = counts[input_idx,:]
meta_input = meta_cells.loc[input_idx,:]

adata_train = AnnData(X = counts_train, obs = meta_train)
sc.pp.normalize_per_cell(adata_train, counts_per_cell_after = 100)

adata_gt = AnnData(X = counts_gt, obs = meta_gt)
sc.pp.normalize_per_cell(adata_gt, counts_per_cell_after = 100)

adata_input = AnnData(X = counts_input, obs = meta_input)
sc.pp.normalize_per_cell(adata_input, counts_per_cell_after = 100)

# In[]
scgen.SCGEN.setup_anndata(adata_train, batch_key="treatment", labels_key="mstatus")
model = scgen.SCGEN(adata_train)
model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)

# adata_to_predict must be under the ctrl_key, then the corresponding stim_key expression is generated
pred, delta = model.predict(ctrl_key = "vehicle (DMSO)", stim_key = "0.2 uM panobinostat", adata_to_predict = adata_input)
pred.write_h5ad(result_dir + "diffbatch/pred.h5ad")

counts_pred = pred.X
meta_pred = pred.obs
counts_pred = counts_pred * (counts_pred > 0)
sp.save_npz(result_dir + "diffbatch/counts_scgen.npz", matrix = sp.csr_matrix(counts_pred))
meta_pred.to_csv(result_dir + "diffbatch/meta_scgen.csv")

# %%
