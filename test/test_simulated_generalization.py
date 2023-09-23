# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.bmk as bmk

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 

loss_df = pd.DataFrame(columns = ["loss recon (train)", "loss recon (test)", "loss kl (train)", "loss kl (test)", \
    "loss mmd (train)", "loss mmd (test)", "loss classi (train)", "loss classi (test)", "test"])

# In[]
sigma = 0.4
n_diff_genes = 100
diff = 8
ngenes = 500
ncells_total = 10000 
n_batches = 2
data_dir = f"../data/simulated/unif/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
result_dir = f"./results_simulated/generalization/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# data_dir = f"../data/simulated/" + sys.argv[1] + "/"
# # lsa performs the best
# result_dir = f"./results_simulated/generalization/" + sys.argv[1] + "/"
# n_diff_genes = eval(sys.argv[1].split("_")[4])
# n_batches = 2
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

# TODO: randomly remove some celltypes?
counts_gt = []
counts_ctrl_healthy = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_severe = []
# cell types
label_annos = []

for batch_id in range(n_batches):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
    counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
    counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
    counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))    

# In[]
# NOTE: select counts for each batch
np.random.seed(0)
counts_gt_test = []
counts_test = []
meta_cells = []
for batch_id in range(n_batches):
    # generate permutation
    permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
    # since there are totally four combinations of conditions, separate the cells into four groups
    chuck_size = int(counts_gt[batch_id].shape[0]/4)
    counts_gt_test.append(counts_gt[batch_id][permute_idx,:])

    counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                       counts_ctrl_severe[batch_id][permute_idx[chuck_size:(2*chuck_size)],:], 
                                       counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                       counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))

    
    meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
    meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
    meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["severe"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
    meta_cell["annos"] = label_annos[batch_id][permute_idx]
    meta_cells.append(meta_cell)

data_dict_full = scdisinfact.create_scdisinfact_dataset(counts_test, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")



# In[] training the model
# TODO: track the time usage and memory usage
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
Ks = [8, 2, 2]
batch_size = 64
nepochs = 50
interval = 10
lr = 5e-4


# In[]

print("#-------------------------------------------------------")
print("#")
print("# Test generalization -- in sample")
print("#")
print("#-------------------------------------------------------")
if not os.path.exists(result_dir + "generalization_is/"):
    os.makedirs(result_dir + "generalization_is/")


# train test split
np.random.seed(0)
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    
    permute_idx = np.random.permutation(np.arange(len(dataset)))
    train_idx = permute_idx[:int(0.9 * len(dataset))]
    test_idx = permute_idx[int(0.9 * len(dataset)):]
    
    dataset_train = scdisinfact.scdisinfact_dataset(counts = dataset[train_idx]["counts"], 
                                                    counts_norm = dataset[train_idx]["counts_norm"],
                                                    size_factor = dataset[train_idx]["size_factor"],
                                                    diff_labels = dataset[train_idx]["diff_labels"],
                                                    batch_id = dataset[train_idx]["batch_id"],
                                                    mmd_batch_id = dataset[train_idx]["mmd_batch_id"]
                                                    )

    dataset_test = scdisinfact.scdisinfact_dataset(counts = dataset[test_idx]["counts"], 
                                                   counts_norm = dataset[test_idx]["counts_norm"],
                                                   size_factor = dataset[test_idx]["size_factor"],
                                                   diff_labels = dataset[test_idx]["diff_labels"],
                                                   batch_id = dataset[test_idx]["batch_id"],
                                                   mmd_batch_id = dataset[test_idx]["mmd_batch_id"]
                                                   )
    
    datasets_array_train.append(dataset_train)
    datasets_array_test.append(dataset_test)
    meta_cells_array_train.append(meta_cells.iloc[train_idx,:])
    meta_cells_array_test.append(meta_cells.iloc[test_idx,:])

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict_full["matching_dict"], "scaler": data_dict_full["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict_full["matching_dict"], "scaler": data_dict_full["scaler"]}

# In[]

model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
_ = model.eval()
torch.save(model.state_dict(), result_dir + f"generalization_is/scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"generalization_is/scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

# In[] Plot results
LOSS_RECON_TRAIN = 0
LOSS_KL_COMM_TRAIN = 0
LOSS_KL_DIFF_TRAIN = 0
LOSS_MMD_COMM_TRAIN = 0
LOSS_MMD_DIFF_TRAIN = 0
LOSS_CLASS_TRAIN = 0
LOSS_GL_D_TRAIN = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]
    z_cs_train = []
    z_ds_train = []
    zs_train = []

    # load count data
    for x in datasets_array_train:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm.to(model.device), batch_ids = batch_id.to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id.to(model.device))

    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    
    z_ds_train.append([x.cpu().detach().numpy() for x in z_d])
    z_cs_train.append(z_c.cpu().detach().numpy())
    zs_train.append(np.concatenate([z_cs_train[-1]] + z_ds_train[-1], axis = 1))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor.to(model.device), \
        count = counts.to(model.device), batch_id = mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TRAIN += losses[0].item()
    LOSS_KL_COMM_TRAIN += losses[1].item()
    LOSS_KL_DIFF_TRAIN += losses[2].item()
    LOSS_MMD_COMM_TRAIN += losses[3].item()
    LOSS_MMD_DIFF_TRAIN += losses[4].item()
    LOSS_CLASS_TRAIN += losses[5].item()
    LOSS_GL_D_TRAIN += losses[6].item()

print("Train:")
print(f"LOSS RECON: {LOSS_RECON_TRAIN}")
print(f"LOSS KL (COMM): {LOSS_KL_COMM_TRAIN}")
print(f"LOSS KL (DIFF): {LOSS_KL_DIFF_TRAIN}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TRAIN}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TRAIN}")
print(f"LOSS CLASS: {LOSS_CLASS_TRAIN}")
print(f"LOSS GL D: {LOSS_GL_D_TRAIN}")



# In[] NOTE: Check, even when training together, the loss is very different, potential bug
# NOTE: could be the issue of standard scaler in the preprocessing step between train and test
LOSS_RECON_TEST = 0
LOSS_KL_COMM_TEST = 0
LOSS_KL_DIFF_TEST = 0
LOSS_MMD_COMM_TEST = 0
LOSS_MMD_DIFF_TEST = 0
LOSS_CLASS_TEST = 0
LOSS_GL_D_TEST = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]
    z_cs_test = []
    z_ds_test = []
    zs_test = []

    # load count data
    for x in datasets_array_test:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm.to(model.device), batch_ids = batch_id.to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id.to(model.device))
    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    
    z_ds_test.append([x.cpu().detach().numpy() for x in z_d])
    z_cs_test.append(z_c.cpu().detach().numpy())
    zs_test.append(np.concatenate([z_cs_test[-1]] + z_ds_test[-1], axis = 1))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor.to(model.device), \
        count = counts.to(model.device), batch_id = mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TEST += losses[0].item()
    LOSS_KL_COMM_TEST += losses[1].item()
    LOSS_KL_DIFF_TEST += losses[2].item()
    LOSS_MMD_COMM_TEST += losses[3].item()
    LOSS_MMD_DIFF_TEST += losses[4].item()
    LOSS_CLASS_TEST += losses[5].item()
    LOSS_GL_D_TEST += losses[6].item()

print("\nTEST:")
print(f"LOSS RECON: {LOSS_RECON_TEST}")
print(f"LOSS KL (COMM): {LOSS_KL_COMM_TEST}")
print(f"LOSS KL (DIFF): {LOSS_KL_DIFF_TEST}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TEST}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TEST}")
print(f"LOSS CLASS: {LOSS_CLASS_TEST}")
print(f"LOSS GL D: {LOSS_GL_D_TEST}")


loss_df = pd.concat([loss_df, pd.DataFrame.from_dict({"loss recon (train)": [LOSS_RECON_TRAIN], "loss recon (test)": [LOSS_RECON_TEST], \
    "loss kl (train)": [LOSS_KL_COMM_TRAIN + LOSS_KL_DIFF_TRAIN], "loss kl (test)": [LOSS_KL_COMM_TEST + LOSS_KL_DIFF_TEST], "loss mmd (train)": [LOSS_MMD_COMM_TRAIN + LOSS_MMD_DIFF_TRAIN], \
        "loss mmd (test)": [LOSS_MMD_COMM_TEST + LOSS_MMD_DIFF_TEST], "loss classi (train)": [LOSS_CLASS_TRAIN], "loss classi (test)": [LOSS_CLASS_TEST],\
          "test": ["in sample"], "dataset": [f"2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}"]})], 
            ignore_index = True)
# In[]
umap_op = UMAP(min_dist = 0.1, random_state = 0)
pca_op = PCA(n_components = 2)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + z_cs_test))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + z_ds_test], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"

for x in meta_cells_array_test:
    x["mode"] = "test"

comment = f"generalization_is/figs_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

meta_cells = pd.concat(meta_cells_array_train + meta_cells_array_test, axis = 0, ignore_index = True)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["annos"].values.squeeze(), batches = meta_cells["batch"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"annos_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells["condition 1"].values.squeeze(), batches = meta_cells["batch"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"condition1_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells["condition 2"].values.squeeze(), batches = meta_cells["batch"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"condition2_joint.png" if result_dir else None, markerscale = 6, s = 5)


# In[]
print("#-------------------------------------------------------")
print("#")
print("# Test generalization -- out of sample")
print("#")
print("#-------------------------------------------------------")
if not os.path.exists(result_dir + "generalization_oos/"):
    os.makedirs(result_dir + "generalization_oos/")

# construct training and testing data
np.random.seed(0)
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta_cells in zip(data_dict_full["datasets"], data_dict_full["meta_cells"]):
    test_idx = ((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values
    train_idx = ~test_idx
    # include more missing matrix, if include all below, severe cannot make fair prediction
    # train_idx = train_idx & ~((meta_cells["condition 1"] == "ctrl") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 1)).values
    # train_idx = train_idx & ~((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "healthy") & (meta_cells["batch"] == 1)).values
    # train_idx = train_idx & ~((meta_cells["condition 1"] == "stim") & (meta_cells["condition 2"] == "severe") & (meta_cells["batch"] == 0)).values


    if np.sum(train_idx) > 0:
        dataset_train = scdisinfact.scdisinfact_dataset(counts = dataset[train_idx]["counts"], 
                                                        counts_norm = dataset[train_idx]["counts_norm"],
                                                        size_factor = dataset[train_idx]["size_factor"],
                                                        diff_labels = dataset[train_idx]["diff_labels"],
                                                        batch_id = dataset[train_idx]["batch_id"],
                                                        mmd_batch_id = dataset[train_idx]["mmd_batch_id"]
                                                        )
        datasets_array_train.append(dataset_train)
        meta_cells_array_train.append(meta_cells.iloc[train_idx,:])
    
    if np.sum(test_idx) > 0:
        dataset_test = scdisinfact.scdisinfact_dataset(counts = dataset[test_idx]["counts"], 
                                                    counts_norm = dataset[test_idx]["counts_norm"],
                                                    size_factor = dataset[test_idx]["size_factor"],
                                                    diff_labels = dataset[test_idx]["diff_labels"],
                                                    batch_id = dataset[test_idx]["batch_id"],
                                                    mmd_batch_id = dataset[test_idx]["mmd_batch_id"]
                                                    )
        datasets_array_test.append(dataset_test)
        meta_cells_array_test.append(meta_cells.iloc[test_idx,:])

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict_full["matching_dict"], "scaler": data_dict_full["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict_full["matching_dict"], "scaler": data_dict_full["scaler"]}

# In[]
model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
_ = model.eval()
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))


# In[] Plot results
LOSS_RECON_TRAIN = 0
LOSS_KL_COMM_TRAIN = 0
LOSS_KL_DIFF_TRAIN = 0
LOSS_MMD_COMM_TRAIN = 0
LOSS_MMD_DIFF_TRAIN = 0
LOSS_CLASS_TRAIN = 0
LOSS_GL_D_TRAIN = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]
    z_cs_train = []
    z_ds_train = []
    zs_train = []

    # load count data
    for x in datasets_array_train:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm.to(model.device), batch_ids = batch_id.to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id.to(model.device))

    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    
    z_ds_train.append([x.cpu().detach().numpy() for x in z_d])
    z_cs_train.append(z_c.cpu().detach().numpy())
    zs_train.append(np.concatenate([z_cs_train[-1]] + z_ds_train[-1], axis = 1))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor.to(model.device), \
        count = counts.to(model.device), batch_id = mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TRAIN += losses[0].item()
    LOSS_KL_COMM_TRAIN += losses[1].item()
    LOSS_KL_DIFF_TRAIN += losses[2].item()
    LOSS_MMD_COMM_TRAIN += losses[3].item()
    LOSS_MMD_DIFF_TRAIN += losses[4].item()
    LOSS_CLASS_TRAIN += losses[5].item()
    LOSS_GL_D_TRAIN += losses[6].item()

print("Train:")
print(f"LOSS RECON: {LOSS_RECON_TRAIN}")
print(f"LOSS KL (COMM): {LOSS_KL_COMM_TRAIN}")
print(f"LOSS KL (DIFF): {LOSS_KL_DIFF_TRAIN}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TRAIN}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TRAIN}")
print(f"LOSS CLASS: {LOSS_CLASS_TRAIN}")
print(f"LOSS GL D: {LOSS_GL_D_TRAIN}")



# In[] NOTE: Check, even when training together, the loss is very different, potential bug
# NOTE: could be the issue of standard scaler in the preprocessing step between train and test
LOSS_RECON_TEST = 0
LOSS_KL_COMM_TEST = 0
LOSS_KL_DIFF_TEST = 0
LOSS_MMD_COMM_TEST = 0
LOSS_MMD_DIFF_TEST = 0
LOSS_CLASS_TEST = 0
LOSS_GL_D_TEST = 0

np.random.seed(0)
with torch.no_grad():
    # loop through the data batches correspond to diffferent data matrices
    counts_norm = []
    batch_id = []
    mmd_batch_id = []
    size_factor = []
    counts = []
    diff_labels = [[] for x in range(model.n_diff_factors)]
    z_cs_test = []
    z_ds_test = []
    zs_test = []

    # load count data
    for x in datasets_array_test:
        counts_norm.append(x.counts_norm)
        batch_id.append(x.batch_id[:, None])
        mmd_batch_id.append(x.mmd_batch_id)
        size_factor.append(x.size_factor)
        counts.append(x.counts)       
        for diff_factor in range(model.n_diff_factors):
            diff_labels[diff_factor].append(x.diff_labels[diff_factor])

    counts_norm = torch.cat(counts_norm, dim = 0)
    batch_id = torch.cat(batch_id, dim = 0)
    mmd_batch_id = torch.cat(mmd_batch_id, dim = 0)
    size_factor = torch.cat(size_factor, dim = 0)
    counts = torch.cat(counts, dim = 0)
    for diff_factor in range(model.n_diff_factors):
        diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

    # pass through the encoders
    dict_inf = model.inference(counts = counts_norm.to(model.device), batch_ids = batch_id.to(model.device), print_stat = True)
    # pass through the decoder
    dict_gen = model.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id.to(model.device))
    z_c = dict_inf["mu_c"]
    z_d = dict_inf["mu_d"]
    z = torch.cat([z_c] + z_d, dim = 1)
    mu = dict_gen["mu"]    
    z_ds_test.append([x.cpu().detach().numpy() for x in z_d])
    z_cs_test.append(z_c.cpu().detach().numpy())
    zs_test.append(np.concatenate([z_cs_test[-1]] + z_ds_test[-1], axis = 1))

    # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
    losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = size_factor.to(model.device), \
        count = counts.to(model.device), batch_id = mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in diff_labels], recon_loss = "NB")

    LOSS_RECON_TEST += losses[0].item()
    LOSS_KL_COMM_TEST += losses[1].item()
    LOSS_KL_DIFF_TEST += losses[2].item()
    LOSS_MMD_COMM_TEST += losses[3].item()
    LOSS_MMD_DIFF_TEST += losses[4].item()
    LOSS_CLASS_TEST += losses[5].item()
    LOSS_GL_D_TEST += losses[6].item()

print("\nTEST:")
print(f"LOSS RECON: {LOSS_RECON_TEST}")
print(f"LOSS KL (COMM): {LOSS_KL_COMM_TEST}")
print(f"LOSS KL (DIFF): {LOSS_KL_DIFF_TEST}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM_TEST}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF_TEST}")
print(f"LOSS CLASS: {LOSS_CLASS_TEST}")
print(f"LOSS GL D: {LOSS_GL_D_TEST}")

loss_df = pd.concat([loss_df, pd.DataFrame.from_dict({"loss recon (train)": [LOSS_RECON_TRAIN], "loss recon (test)": [LOSS_RECON_TEST], \
    "loss kl (train)": [LOSS_KL_COMM_TRAIN + LOSS_KL_DIFF_TRAIN], "loss kl (test)": [LOSS_KL_COMM_TEST + LOSS_KL_DIFF_TEST], "loss mmd (train)": [LOSS_MMD_COMM_TRAIN + LOSS_MMD_DIFF_TRAIN], \
        "loss mmd (test)": [LOSS_MMD_COMM_TEST + LOSS_MMD_DIFF_TEST], "loss classi (train)": [LOSS_CLASS_TRAIN], "loss classi (test)": [LOSS_CLASS_TEST],\
          "test": ["out of sample"], "dataset": [f"2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}"]})], 
            ignore_index = True)
# In[]
loss_df.to_csv(result_dir + "loss_df.csv")

umap_op = UMAP(min_dist = 0.1, random_state = 0)
pca_op = PCA(n_components = 2)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train + z_cs_test))
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + z_ds_test], axis = 0)))
for x in meta_cells_array_train:
    x["mode"] = "train"

for x in meta_cells_array_test:
    x["mode"] = "test"

comment = f"generalization_oos/figs_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

meta_cells = pd.concat(meta_cells_array_train + meta_cells_array_test, axis = 0, ignore_index = True)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["annos"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"annos_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[0], annos = meta_cells["condition 1"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"condition1_joint.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umap[1], annos = meta_cells["condition 2"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"condition2_joint.png" if result_dir else None, markerscale = 6, s = 5)


# In[]
loss_is = pd.DataFrame(columns = ["loss recon", "loss kl", "loss mmd", "loss class", "test"])
loss_oos = pd.DataFrame(columns = ["loss recon", "loss kl", "loss mmd", "loss class", "test"])

# totally 9 datasets
for ndiff_genes in [20, 50, 100]:
    for diff in [2, 4, 8]:
        loss = pd.read_csv(f"results_simulated/generalization/2conds_base_10000_500_0.4_{ndiff_genes}_{diff}/loss_df.csv", index_col = 0)
        print(loss)
        # loss.to_csv(f"results_simulated/generalization/2conds_base_10000_500_0.4_{ndiff_genes}_{diff}/loss_df.csv")
        loss_is = pd.concat([loss_is, pd.DataFrame.from_dict({"loss recon": [loss.loc[0, "loss recon (train)"], loss.loc[0, "loss recon (test)"]],
                                                              "loss kl": [loss.loc[0, "loss kl (train)"], loss.loc[0, "loss kl (test)"]],
                                                              "loss mmd":[loss.loc[0, "loss mmd (train)"], loss.loc[0, "loss mmd (test)"]],
                                                              "loss class":[loss.loc[0, "loss classi (train)"], loss.loc[0, "loss classi (test)"]],
                                                              "test": ["Train", "Test"]
                                                              })], axis = 0, ignore_index = True)

        loss_oos = pd.concat([loss_oos, pd.DataFrame.from_dict({"loss recon": [loss.loc[1, "loss recon (train)"], loss.loc[1, "loss recon (test)"]],
                                                              "loss kl": [loss.loc[1, "loss kl (train)"], loss.loc[1, "loss kl (test)"]],
                                                              "loss mmd":[loss.loc[1, "loss mmd (train)"], loss.loc[1, "loss mmd (test)"]],
                                                              "loss class":[loss.loc[1, "loss classi (train)"], loss.loc[1, "loss classi (test)"]],
                                                              "test": ["Train", "Test"]
                                                              })], axis = 0, ignore_index = True)


import seaborn as sns
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize = (12,5))
ax = fig.subplots(nrows = 1, ncols = 2)
loss = pd.DataFrame()
loss["value"] = np.concatenate([loss_is.loc[:, "loss recon"].values, loss_is.loc[:, "loss kl"].values, loss_is.loc[:, "loss mmd"].values, loss_is.loc[:, "loss class"].values])
loss["test"] = np.concatenate([loss_is.loc[:, "test"].values, loss_is.loc[:, "test"].values, loss_is.loc[:, "test"].values, loss_is.loc[:, "test"].values])
loss["loss"] = ["loss recon"] * loss_is.shape[0] + ["loss kl"] * loss_is.shape[0] + ["loss mmd"] * loss_is.shape[0] + ["loss class"] * loss_is.shape[0]

sns.barplot(loss, hue = "test", x = "loss", y = "value", ax = ax[0], capsize = 0.1)
handles, labels = ax[0].get_legend_handles_labels()
sns.stripplot(data = loss, hue = "test", x = "loss", y = "value", ax = ax[0], color = "black", dodge = True) 
print(loss)

loss = pd.DataFrame()
loss["value"] = np.concatenate([loss_oos.loc[:, "loss recon"].values, loss_is.loc[:, "loss kl"].values, loss_is.loc[:, "loss mmd"].values, loss_is.loc[:, "loss class"].values])
loss["test"] = np.concatenate([loss_oos.loc[:, "test"].values, loss_is.loc[:, "test"].values, loss_is.loc[:, "test"].values, loss_is.loc[:, "test"].values])
loss["loss"] = ["loss recon"] * loss_oos.shape[0] + ["loss kl"] * loss_is.shape[0] + ["loss mmd"] * loss_is.shape[0] + ["loss class"] * loss_is.shape[0]

print(loss)
sns.barplot(loss, hue = "test", x = "loss", y = "value", ax = ax[1], capsize = 0.1)
sns.stripplot(data = loss, hue = "test", x = "loss", y = "value", ax = ax[1], color = "black", dodge = True) 

ax[0].get_legend().remove()
ax[1].get_legend().remove()

ax[0].set_yscale("log")
ax[1].set_yscale("log")
l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon = False)

ax[0].set_xlabel(None)
ax[1].set_xlabel(None)
ax[0].set_ylabel("Loss")
ax[1].set_ylabel("Loss")

ax[0].set_xticklabels(["$\ell_{recon}$", "$\ell_{kl}$", "$\ell_{mmd}$", "$\ell_{class}$"])
ax[1].set_xticklabels(["$\ell_{recon}$", "$\ell_{kl}$", "$\ell_{mmd}$", "$\ell_{class}$"])
fig.tight_layout()

# TODO: make certain
ax[0].set_title("In sample")
ax[1].set_title("Out of sample")
fig.savefig("results_simulated/generalization/scores.png", bbox_inches = "tight")


# %%

# %%
