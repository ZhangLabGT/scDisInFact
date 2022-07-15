# In[]
from random import random
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import scdisinfact
import loss_function as loss_func
import utils
import bmk

import anndata as ad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

from umap import UMAP
import seaborn
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')


# In[] Read in the dataset
# version 1, all cells
data_dir = "../data/covid/batch_processed/"
result_dir = "covid/"
batches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]#, 17]

counts_array = []
meta_cells_array = []
datasets_array = []
for batch_id in batches:
    # TODO: read genes and barcodes
    counts_array.append(sparse.load_npz(data_dir + f"raw_filtered_batch{batch_id}.npz").todense())
    # NOTE: COLUMNS: cellName, sampleID, celltype, majorType, PatientID, Batches, City, Age, Sex, Sample type, CoVID-19 severity, Sample time, 
    # Sampling day (Days after symptom onset), SARS-CoV-2, Single cell sequencing platform, BCR single cell sequencing, CR single cell sequencing, 
    # Outcome, Comorbidities, COVID-19-related medication and anti-microbials Leukocytes [G/L], Neutrophils [G/L], Lymphocytes [G/L], Unpublished
    # IMPORTENT condition: Age, Sex, SARS-CoV-2 (positive or not), Comorbidities, COVID-19-related medication and anti-microbials
    meta_cells_array.append(pd.read_csv(data_dir + f"meta_filtered_batch{batch_id}.csv", index_col = 0))
    # datasets_array.append(scdisinfact.dataset(counts = counts_array[-1], anno = meta_cells_array[-1]["Cluster number"].values.squeeze(), diff_labels = [response], batch_id = batch_ids[batch_ids == batch_id]))
    

# batch_ids, batch_names = pd.factorize(meta_cells["characteristics: patinet ID (Pre=baseline; Post= on treatment)"].values.squeeze())
# response_ids, response_names = pd.factorize(meta_cells["characteristics: response"].values.squeeze())


# %%
