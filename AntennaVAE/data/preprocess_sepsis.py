# In[]
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import os, sys
sys.path.append('../src')
from scipy import sparse
import random
import utils
random.seed(0)
np.random.seed(0)

# In[] Read in data
# meta = pd.read_table(r'./scp_gex_matrix/raw_data/scp_meta.txt',index_col=0)
# meta = meta.iloc[1:]
# meta = meta.reset_index()
# meta.index = meta["NAME"].values
# # adata gene by cell matrix, the csv file have row and column names, which is separated by comma, of the shape (122557, 22858)
# counts = sc.read_csv(r'./scp_gex_matrix/raw_data/scp_gex_matrix.csv')
# counts = counts.T

# meta = meta.loc[counts.obs.index.values, ["Cell_Type", "Cell_State", "Cohort", "Sort", "Patient"]]
# counts.obs = meta
# counts.write(r'./scp_gex_matrix/raw_data/sep_gex_raw.h5ad')
counts_raw = ad.read_h5ad(r'./scp_gex_matrix/raw_data/sep_gex_raw.h5ad')
counts = counts_raw.copy()
# filter genes
sc.pp.filter_genes(counts, min_cells = 1000)
# normalize
sc.pp.normalize_per_cell(counts)
sc.pp.log1p(counts)
sc.pp.highly_variable_genes(counts, n_top_genes = 2000)
counts = counts[:, counts.var["highly_variable"]]
counts_raw = counts_raw[:, counts.var.index.values]

sc.pp.neighbors(counts, n_neighbors=15, n_pcs=40)
sc.tl.umap(counts)

save = None
sc.pl.umap(counts, color = ['Cell_Type', 'Cohort', 'Patient'], save = save)

# In[] Separate batches
# Culculate indexes for each cohort 
meta = counts.obs
patients = meta['Cohort'].unique()
cohort_barcodes = {}
for i in patients:
    cohort_barcodes[i] = (meta[meta['Cohort'] == i].index)


info_dict = {}
for cohort, barcodes in cohort_barcodes.items():
    this_counts = counts[barcodes,:]
    info_dict[cohort] = [x for x in np.sort(this_counts.obs["Patient"].unique())]
    for id, patient in enumerate(np.sort(this_counts.obs["Patient"].unique())):
        this_batch_counts = this_counts[this_counts.obs["Patient"] == patient]
        # make sure the same patient have the same processing batches
        print(np.unique(np.array([x.split("-")[1] for x in this_batch_counts.obs.index.values])))

        if not os.path.exists(r'scp_gex_matrix/processed_sepsis_2000/{}'.format(cohort)):
            os.makedirs(r'scp_gex_matrix/processed_sepsis_2000/{}'.format(cohort))
        sparse.save_npz(r'scp_gex_matrix/processed_sepsis_2000/{}/mtx_{}_batch_{}.npz'.format(cohort, cohort, patient), this_batch_counts.X)
        this_batch_counts.obs.to_csv(r'./data/scp_gex_matrix/processed_sepsis_2000/{}/meta_{}_batch{}.csv'.format(cohort, cohort, patient),index=False)
        print(cohort,'batch',patient, 'Finished!')
    print(cohort,'finished!!!!!!')

for cohort, barcodes in cohort_barcodes.items():
    processing_batches = [x.split("-")[1] for x in barcodes]
    


# %%
