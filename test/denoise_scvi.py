# In[]
import numpy as np
import matplotlib.pyplot as plt
import scvi
import anndata
import scipy.sparse as sp

# In[]
# DCA does not work
data_dir = "GBM_treatment/Fig4_patient/prediction/PW034-705-obs.h5ad"
adata = anndata.read_h5ad(data_dir)
scvi.model.SCVI.setup_anndata(adata, batch_key="patient_id")
vae = scvi.model.SCVI(adata, gene_likelihood = "nb")
vae.train()
adata.obsm["X_scVI"] = vae.get_latent_representation()
adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

sp.save_npz("GBM_treatment/Fig4_patient/prediction/PW034-705-scvi.npz", sp.csr_matrix(adata.obsm["X_normalized_scVI"].values))