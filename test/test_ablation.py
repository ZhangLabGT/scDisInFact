# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import scdisinfact
import utils
import bmk
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score 


# In[]
# sigma = 0.2
# n_diff_genes = 100
# diff = 8
# ngenes = 500
# ncells_total = 10000 
# n_batches = 6
# data_dir = f"../data/simulated/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
# result_dir = f"./ablation_test/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

data_dir = f"../data/simulated/" + sys.argv[1] + "/"
# lsa performs the best
result_dir = f"./ablation_test/" + sys.argv[1] + "/"
n_diff_genes = eval(sys.argv[1].split("_")[4])
n_batches = 6
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

counts_ctrls = []
counts_stims1 = []
counts_stims2 = []
# cell types
annos = []
# batch labels
batches = []
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
    annos.append(np.array([('cell type '+str(i)) for i in anno]))
    # batch labels
    batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
    label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
    label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
    label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))

# training test split
counts_ctrls_train = []
counts_ctrls_test = []
counts_stims1_train = []
counts_stims1_test = []
counts_stims2_train = []
counts_stims2_test = []
counts_gt_train = []
counts_gt_test = []

annos_train = []
annos_test = []
batches_train = []
batches_test = []
label_ctrls_train = []
label_ctrls_test = []
label_stims1_train = []
label_stims1_test = []
label_stims2_train = []
label_stims2_test = []

np.random.seed(0)
for batch_id in range(n_batches):
    batchsize = counts_ctrls[batch_id].shape[0]
    train_idx = np.array([False] * batchsize)
    train_idx[np.random.choice(batchsize, int(0.9 * batchsize), replace = False)] = True

    counts_ctrls_train.append(counts_ctrls[batch_id][train_idx, :])
    counts_ctrls_test.append(counts_ctrls[batch_id][~train_idx, :])
    counts_stims1_train.append(counts_stims1[batch_id][train_idx, :])
    counts_stims1_test.append(counts_stims1[batch_id][~train_idx, :])
    counts_stims2_train.append(counts_stims2[batch_id][train_idx, :])
    counts_stims2_test.append(counts_stims2[batch_id][~train_idx, :])
    counts_gt_train.append(counts_gt[batch_id][train_idx, :])
    counts_gt_test.append(counts_gt[batch_id][~train_idx, :])
        
    annos_train.append(annos[batch_id][train_idx])
    annos_test.append(annos[batch_id][~train_idx])
    batches_train.append(batches[batch_id][train_idx])
    batches_test.append(batches[batch_id][~train_idx])
    label_ctrls_train.append(label_ctrls[batch_id][train_idx])
    label_stims1_train.append(label_stims1[batch_id][train_idx])
    label_stims2_train.append(label_stims2[batch_id][train_idx])
    label_ctrls_test.append(label_ctrls[batch_id][~train_idx])
    label_stims1_test.append(label_stims1[batch_id][~train_idx])
    label_stims2_test.append(label_stims2[batch_id][~train_idx])

# In[]
# Train with ctrl in batches 1, 2, 4, stim1 in batches 1, 3, 5, stim2 in batches 2, 3, 6
counts = []
label_conditions = []
label_batches = []
label_annos = []
np.random.seed(0)
# BATCH 1: randomly allocate 50% of cells of batch 1 into ctrl, and the remaining into stim 1.
idx = np.array([False] * len(label_ctrls_train[0]))
idx[np.random.choice(len(label_ctrls_train[0]), size = int(0.5 * len(label_ctrls_train[0])), replace = False)] = True
label_batches.append(batches_train[0][idx])
label_conditions.append(label_ctrls_train[0][idx])
label_annos.append(annos_train[0][idx])
counts.append(counts_ctrls_train[0][idx])

label_batches.append(batches_train[0][~idx])
label_conditions.append(label_stims1_train[0][~idx])
label_annos.append(annos_train[0][~idx])
counts.append(counts_stims1_train[0][~idx])

# BATCH 2: randomly allocate 50% of cells of batch 2 into ctrl, and the remaining into stim 2.
idx = np.array([False] * len(label_ctrls_train[1]))
idx[np.random.choice(len(label_ctrls_train[1]), size = int(0.5 * len(label_ctrls_train[1])), replace = False)] = True
label_batches.append(batches_train[1][idx])
label_conditions.append(label_ctrls_train[1][idx])
label_annos.append(annos_train[1][idx])
counts.append(counts_ctrls_train[1][idx])

label_batches.append(batches_train[1][~idx])
label_conditions.append(label_stims2_train[1][~idx])
label_annos.append(annos_train[1][~idx])
counts.append(counts_stims2_train[1][~idx])

# BATCH 3: randomly allocate 50% of cells of batch 3 into stim 1, and the remaining into stim 2.
idx = np.array([False] * len(label_stims1_train[2]))
idx[np.random.choice(len(label_stims1_train[2]), size = int(0.5 * len(label_stims1_train[2])), replace = False)] = True
label_batches.append(batches_train[2][idx])
label_conditions.append(label_stims1_train[2][idx])
label_annos.append(annos_train[2][idx])
counts.append(counts_stims1_train[2][idx])

label_batches.append(batches_train[2][~idx])
label_conditions.append(label_stims2_train[2][~idx])
label_annos.append(annos_train[2][~idx])
counts.append(counts_stims2_train[2][~idx])

# BATCH 4: 1 condition (ctrl)
# label_conditions.append(label_ctrls_train[3])
# label_batches.append(batches_train[3])
# label_annos.append(annos_train[3])
# counts.append(counts_ctrls_train[3])

# 2 conditions (ctrl and stim2)
idx = np.array([False] * len(label_ctrls_train[3]))
idx[np.random.choice(len(label_ctrls_train[3]), size = int(0.5 * len(label_ctrls_train[3])), replace = False)] = True
label_batches.append(batches_train[3][idx])
label_conditions.append(label_ctrls_train[3][idx])
label_annos.append(annos_train[3][idx])
counts.append(counts_ctrls_train[3][idx])

label_batches.append(batches_train[3][~idx])
label_conditions.append(label_stims2_train[3][~idx])
label_annos.append(annos_train[3][~idx])
counts.append(counts_stims2_train[3][~idx])

# BATCH 5: 1 condition (stim1)
# label_conditions.append(label_stims1_train[4])
# label_batches.append(batches_train[4])
# label_annos.append(annos_train[4])
# counts.append(counts_stims1_train[4])

# 2 conditions (stim1 and stim2)
idx = np.array([False] * len(label_stims1_train[4]))
idx[np.random.choice(len(label_stims1_train[4]), size = int(0.5 * len(label_stims1_train[4])), replace = False)] = True
label_batches.append(batches_train[4][idx])
label_conditions.append(label_stims1_train[4][idx])
label_annos.append(annos_train[4][idx])
counts.append(counts_stims1_train[4][idx])

label_batches.append(batches_train[4][~idx])
label_conditions.append(label_stims2_train[4][~idx])
label_annos.append(annos_train[4][~idx])
counts.append(counts_stims2_train[4][~idx])


# BATCH 6: 1 condition (stim2)
# label_conditions.append(label_stims2_train[5])
# label_batches.append(batches_train[5])
# label_annos.append(annos_train[5])
# counts.append(counts_stims1_train[5])

# 2 conditions (ctrl and stim1)
idx = np.array([False] * len(label_ctrls_train[5]))
idx[np.random.choice(len(label_ctrls_train[5]), size = int(0.5 * len(label_ctrls_train[5])), replace = False)] = True
label_batches.append(batches_train[5][idx])
label_conditions.append(label_ctrls_train[5][idx])
label_annos.append(annos_train[5][idx])
counts.append(counts_ctrls_train[5][idx])

label_batches.append(batches_train[5][~idx])
label_conditions.append(label_stims1_train[5][~idx])
label_annos.append(annos_train[5][~idx])
counts.append(counts_stims1_train[5][~idx])

# create training dataset
counts_train = np.concatenate(counts, axis = 0)
meta_cells_train = pd.DataFrame(columns = ["condition", "batch", "anno"])
meta_cells_train["condition"] = np.concatenate(label_conditions, axis = 0)
meta_cells_train["batch"] = np.concatenate(label_batches, axis = 0)
meta_cells_train["anno"] = np.concatenate(label_annos, axis = 0)
meta_genes = pd.DataFrame(columns = ["genes"])
meta_genes["genes"] = np.array(["gene_" + str(x) for x in range(counts_train.shape[1])])

datasets_array_train, meta_cells_array_train, matching_dict_train = scdisinfact.create_scdisinfact_dataset(counts = counts_train, meta_cells = meta_cells_train, meta_genes = meta_genes, condition_key = ["condition"], batch_key = "batch")

# create testing dataset
counts_test = np.concatenate(counts_ctrls_test + counts_stims1_test + counts_stims2_test, axis = 0)
counts_gt_test = np.concatenate(counts_gt_test + counts_gt_test + counts_gt_test, axis = 0)
meta_cells_test = pd.DataFrame(columns = ["condition", "batch", "anno"])
meta_cells_test["condition"] = np.concatenate(label_ctrls_test + label_stims1_test + label_stims2_test, axis = 0)
meta_cells_test["batch"] = np.concatenate(batches_test + batches_test + batches_test, axis = 0)
meta_cells_test["anno"] = np.concatenate(annos_test + annos_test + annos_test, axis = 0)
datasets_array_test, meta_cells_array_test, matching_dict_test = scdisinfact.create_scdisinfact_dataset(counts = counts_test, meta_cells = meta_cells_test, meta_genes = meta_genes, condition_key = ["condition"], batch_key = "batch")
datasets_array_test_gt, meta_cells_array_test_gt, matching_dict_test_gt = scdisinfact.create_scdisinfact_dataset(counts = counts_gt_test, meta_cells = meta_cells_test, meta_genes = meta_genes, condition_key = ["condition"], batch_key = "batch")


# In[] training the model
# TODO: track the time usage and memory usage
import importlib 
importlib.reload(scdisinfact)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

start_time = time.time()
# reg_mmd_comm = 1e-4
# reg_mmd_diff = 1e-2
# reg_gl = 1
# reg_tc = 0.5
# reg_class = 1
# reg_kl = 1e-5
# # Ablation test, should check the performance when reg_class, reg_contr are [0, 0.1], [1, 0], [1, 0.01]
# reg_contr = 0.1
# # mmd, cross_entropy, total correlation, group_lasso, kl divergence, contrastive loss
# lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
# Ks = [8, 4]
# # too many epoches affect the generalization ability
# nepochs = 100
# interval = 10

reg_mmd_comm = eval(sys.argv[2])
reg_mmd_diff = eval(sys.argv[3])
reg_gl = eval(sys.argv[4])
reg_tc = eval(sys.argv[5])
reg_class = eval(sys.argv[6])
reg_kl = eval(sys.argv[7])
reg_contr = eval(sys.argv[8])
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
interval = 10
Ks = [8, 4]
nepochs = 100
interval = 10

print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_array_train, Ks = Ks, batch_size = 64, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))

losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = reg_contr)

torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth", map_location = device))
end_time = time.time()
print("time cost: {:.2f}".format(end_time - start_time))



# In[] Plot results
z_cs_train = []
z_ds_train = []
zs_train = []
# one forward pass
with torch.no_grad():
    for batch_id, dataset in enumerate(datasets_array_train):
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]
        z_cs_train.append(z_c.cpu().detach().numpy())
        z_ds_train.append([x.cpu().detach().numpy() for x in z_d])   
        zs_train.append(z.cpu().detach().numpy())

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_train, axis = 0))

z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train], axis = 0)))

comment = f"figures_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

meta_cells = pd.concat(meta_cells_array_train, axis =0, ignore_index = True)
# check same batch different conditions
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["anno"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes_train.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umap, batches = meta_cells["batch"].values.squeeze(), mode = "batches", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches_train.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["condition"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_condition_train.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["anno"].values.squeeze(), batches = meta_cells["batch_cond"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,30), save = result_dir + comment+"common_celltypes_separate_train.png" if result_dir else None , markerscale = 6, s = 5)

for diff_factor in range(model.n_diff_factors):
    utils.plot_latent(zs = z_ds_umap[diff_factor], annos = meta_cells["anno"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_celltypes_train.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umap[diff_factor], batches = meta_cells["batch"].values.squeeze(), mode = "batches", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_batch_train.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umap[diff_factor], annos = meta_cells["condition"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_condition_train.png" if result_dir else None, markerscale = 6, s = 5)

# In[]
LOSS_RECON = 0
LOSS_KL = 0
LOSS_MMD_COMM = 0
LOSS_MMD_DIFF = 0
LOSS_CLASS = 0
LOSS_CONTR = 0
LOSS_TC = 0
LOSS_GL_D = 0

z_cs_test = []
z_ds_test = []
zs_test = []
# one forward pass
with torch.no_grad():
    for batch_id, dataset in enumerate(datasets_array_test):
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        dict_inf["z_d"] = dict_inf["mu_d"]
        dict_inf["z_c"] = dict_inf["mu_c"]
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]
        z_cs_test.append(z_c.cpu().detach().numpy())
        zs_test.append(z.cpu().detach().numpy())
        z_ds_test.append([x.cpu().detach().numpy() for x in z_d])   

        # calculate loss, the mmd_batch_id of test is not the same as train, but doesn't affect the result (only affect the mmd loss)
        losses = model.loss(dict_inf = dict_inf, dict_gen = dict_gen, size_factor = dataset.size_factor.to(model.device), \
            count = dataset.counts.to(model.device), batch_id = dataset.mmd_batch_id.to(model.device), diff_labels = [x.to(model.device) for x in dataset.diff_labels], recon_loss = "NB")
        LOSS_RECON += losses[0].item()
        LOSS_KL += losses[1].item()
        LOSS_MMD_COMM += losses[2].item()
        LOSS_MMD_DIFF += losses[3].item()
        LOSS_CLASS += losses[4].item()
        LOSS_CONTR += losses[5].item()
        LOSS_TC += losses[6].item()
        LOSS_GL_D += losses[7].item()
        

print(f"LOSS RECON: {LOSS_RECON}")
print(f"LOSS KL: {LOSS_KL}")
print(f"LOSS MMD (COMM): {LOSS_MMD_COMM}")
print(f"LOSS MMD (DIFF): {LOSS_MMD_DIFF}")
print(f"LOSS CLASS: {LOSS_CLASS}")
print(f"LOSS CONTR: {LOSS_CONTR}")
print(f"LOSS TC: {LOSS_TC}")
print(f"LOSS GL D: {LOSS_GL_D}")


# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs_test, axis = 0))

z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_test], axis = 0)))


comment = f"figures_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

meta_cells = pd.concat(meta_cells_array_test, axis =0, ignore_index = True)
# check same batch different conditions
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["anno"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes_test.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umap, batches = meta_cells["batch"].values.squeeze(), mode = "batches", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches_test.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["condition"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_condition_test.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umap, annos = meta_cells["anno"].values.squeeze(), batches = meta_cells["batch_cond"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,30), save = result_dir + comment+"common_celltypes_separate_test.png" if result_dir else None , markerscale = 6, s = 5)

for diff_factor in range(model.n_diff_factors):
    utils.plot_latent(zs = z_ds_umap[diff_factor], annos = meta_cells["anno"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_celltypes_test.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umap[diff_factor], batches = meta_cells["batch"].values.squeeze(), mode = "batches", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_batch_test.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umap[diff_factor], annos = meta_cells["condition"].values.squeeze(), mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_condition_test.png" if result_dir else None, markerscale = 6, s = 5)

# In[]
z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds_train + z_ds_test], axis = 0)))

for x in meta_cells_array_train:
    x["mode"] = "train"
for x in meta_cells_array_test:
    x["mode"] = "test"
meta_cells = pd.concat(meta_cells_array_train + meta_cells_array_test, axis = 0, ignore_index = True)

for diff_factor in range(model.n_diff_factors):
    utils.plot_latent(zs = z_ds_umap[diff_factor], annos = meta_cells["condition"].values.squeeze(), batches = meta_cells["mode"].values.squeeze(), mode = "separate", axis_label = "UMAP", figsize = (10,12), save = result_dir + comment+f"diff{diff_factor}_condition_joint.png" if result_dir else None, markerscale = 6, s = 5)

# In[]
#------------------------------------------------------------------------------------------------------------------------------------------
#
# Benchmark, common space, check removal of batch effect (=condition effect), keep cluster information
#
#------------------------------------------------------------------------------------------------------------------------------------------
meta_cells = pd.concat(meta_cells_array_test, axis =0, ignore_index = True)

# removal of batch and condition effect
# 1. scdisinfact
n_neighbors = 30
gc_cluster_scdisinfact = bmk.graph_connectivity(X = np.concatenate(z_cs_test, axis = 0), groups = meta_cells["anno"].values.squeeze(), k = n_neighbors)
print('GC cluster (scDisInFact): {:.3f}'.format(gc_cluster_scdisinfact))
silhouette_batch_scdisinfact = bmk.silhouette_batch(X = np.concatenate(z_cs_test, axis = 0), batch_gt = meta_cells["batch"].values.squeeze(), group_gt = meta_cells["anno"].values.squeeze(), verbose = False)
print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))

# NMI and ARI measure the separation of cell types
# 1. scdisinfact
nmi_cluster_scdisinfact = []
ari_cluster_scdisinfact = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs_test, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_cluster_scdisinfact.append(bmk.nmi(group1 = meta_cells["anno"].values.squeeze(), group2 = leiden_labels_clusters))
    ari_cluster_scdisinfact.append(bmk.ari(group1 = meta_cells["anno"].values.squeeze(), group2 = leiden_labels_clusters))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_cluster_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_cluster_scdisinfact)))


#------------------------------------------------------------------------------------------------------------------------------------------
#
# condition-specific space, check removal of batch effect, removal of cell type effect, keep condition information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# removal of batch effect, removal of cell type effect
gc_condition_scdisinfact = bmk.graph_connectivity(X = np.concatenate([x[0] for x in z_ds_test], axis = 0), groups = meta_cells["condition"].values.squeeze(), k = n_neighbors)
print('GC condition (scDisInFact): {:.3f}'.format(gc_condition_scdisinfact))
silhouette_condition_scdisinfact = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds_test], axis = 0), batch_gt = meta_cells["batch"].values.squeeze(), group_gt = meta_cells["condition"].values.squeeze(), verbose = False)
print('Silhouette condition, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact))
silhouette_condition_scdisinfact2 = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds_test], axis = 0), batch_gt = meta_cells["anno"].values.squeeze(), group_gt = meta_cells["condition"].values.squeeze(), verbose = False)
print('Silhouette condition, removal of cell type effeect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact2))

# keep of condition information
nmi_condition_scdisinfact = []
ari_condition_scdisinfact = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds_test], axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_condition_scdisinfact.append(bmk.nmi(group1 = meta_cells["condition"].values.squeeze(), group2 = leiden_labels_conditions))
    ari_condition_scdisinfact.append(bmk.ari(group1 = meta_cells["condition"].values.squeeze(), group2 = leiden_labels_conditions))

print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact)))


scores_scdisinfact = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "GC (common)", \
    "GC (condition)", "Silhouette batch (common)", "Silhouette batch (condition & celltype)", "Silhouette batch (condition & batches)", \
        "LOSS_RECON", "LOSS_KL", "LOSS_MMD_COMM", "LOSS_MMD_DIFF", "LOSS_CLASS", "LOSS_CONTR", "LOSS_TC", "MSE", "MSE (diff)", "Pearson", "R2", "R2 (diff)"])
scores_scdisinfact["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact)])
scores_scdisinfact["ARI (common)"] = np.array([max(ari_cluster_scdisinfact)])
scores_scdisinfact["GC (common)"] = np.array([gc_cluster_scdisinfact])
scores_scdisinfact["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact])

scores_scdisinfact["NMI (condition)"] = np.array([max(nmi_condition_scdisinfact)])
scores_scdisinfact["ARI (condition)"] = np.array([max(ari_condition_scdisinfact)])
scores_scdisinfact["GC (condition)"] = np.array([gc_condition_scdisinfact])
scores_scdisinfact["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scdisinfact])
scores_scdisinfact["Silhouette batch (condition & celltype)"] = np.array([silhouette_condition_scdisinfact2])

scores_scdisinfact["LOSS_RECON"] = np.array([LOSS_RECON])
scores_scdisinfact["LOSS_KL"] = np.array([LOSS_KL])
scores_scdisinfact["LOSS_MMD_COMM"] = np.array([LOSS_MMD_COMM])
scores_scdisinfact["LOSS_MMD_DIFF"] = np.array([LOSS_MMD_DIFF])
scores_scdisinfact["LOSS_CLASS"] = np.array([LOSS_CLASS])
scores_scdisinfact["LOSS_CONTR"] = np.array([LOSS_CONTR])
scores_scdisinfact["LOSS_TC"] = np.array([LOSS_TC])
scores_scdisinfact["methods"] = np.array(["scDisInFact"])

# In[] Check prediction accuracy
pred_conds = [np.where(matching_dict_train["cond_names"][0] == "ctrl")[0][0]]
X_scdisinfact_impute = []

# make sure that the pred_conds 
for _, dataset in enumerate(datasets_array_test):
    X = model.predict_counts(predict_dataset = dataset, predict_conds = pred_conds, predict_batch = np.unique(dataset.batch_id)[0])
    X_scdisinfact_impute.append(X.detach().cpu().numpy())

X_scdisinfact_impute = np.concatenate(X_scdisinfact_impute)

# -------------------------------------------------------------------------------
#
# cell-specific scores with ground truth
#
# -------------------------------------------------------------------------------
mses_scdisinfact = []
mses_diff_scdisinfact = []
pearsons_scdisinfact = []
r2_scdisinfact = []
r2_diff_scdisinfact = []

X_gt = []
for dataset_gt in datasets_array_test_gt:
    X_gt.append(dataset_gt.counts.numpy())
X_gt = np.concatenate(X_gt)

# normalize the counts
X_scdisinfact_impute_norm = X_scdisinfact_impute/(np.sum(X_scdisinfact_impute, axis = 1, keepdims = True) + 1e-6)
X_gt_norm = X_gt/(np.sum(X_gt, axis = 1, keepdims = True) + 1e-6)

# cell-specific normalized MSE
mses_scdisinfact = np.mean(np.sum((X_scdisinfact_impute_norm - X_gt_norm) ** 2, axis = 1))
# cell-specific normalized MSE on diff genes
mses_diff_scdisinfact = np.mean(np.sum((X_scdisinfact_impute_norm[:,:n_diff_genes] - X_gt_norm[:,:n_diff_genes]) ** 2, axis = 1))
# cell-specific pearson correlation
pearsons_scdisinfact = np.mean([stats.pearsonr(X_scdisinfact_impute_norm[i,:], X_gt_norm[i,:])[0] for i in range(X_gt_norm.shape[0])])
# cell-specific R2 score
r2_scdisinfact = np.mean([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scdisinfact_impute_norm, X_gt_norm)])
# cell-specific R2 diff score
r2_diff_scdisinfact = np.mean([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scdisinfact_impute_norm[:,:n_diff_genes], X_gt_norm[:,:n_diff_genes])])


scores_scdisinfact["MSE"] = np.array([mses_scdisinfact])
scores_scdisinfact["MSE (diff)"] = np.array([mses_diff_scdisinfact])
scores_scdisinfact["Pearson"] = np.array([pearsons_scdisinfact])
scores_scdisinfact["R2"] = np.array([r2_scdisinfact])
scores_scdisinfact["R2 (diff)"] = np.array([r2_diff_scdisinfact])

scores_scdisinfact.to_csv(result_dir + comment + f"scores.csv")


# In[]
scores = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "GC (common)", \
    "GC (condition)", "Silhouette batch (common)", "Silhouette batch (condition & celltype)", "Silhouette batch (condition & batches)", \
        "LOSS_RECON", "LOSS_KL", "LOSS_MMD_COMM", "LOSS_MMD_DIFF", "LOSS_CLASS", "LOSS_CONTR", "LOSS_TC", "MSE", "MSE (diff)", "Pearson", "R2", "R2 (diff)"])

for ndiff in [20, 50, 100]:
    for sigma in [0.2, 0.3, 0.4]:
        for diff in [2]:
            score = pd.read_csv(f"ablation_test/1condition_10000_500_{sigma}_{ndiff}_{diff}/figures_[8, 4]_[0.001, 0.001, 0, 1, 0.5, 1e-05, 0.1]/scores.csv", index_col = 0)
            score["methods"] = "[0.0001, 0.0001, 0, 1, 0.5, 1e-06, 0.1]"
            score["methods"] = "only contrastive"
            scores = pd.concat([scores, score], axis = 0, ignore_index = True)

            score = pd.read_csv(f"ablation_test/1condition_10000_500_{sigma}_100_{diff}/figures_[8, 4]_[0.001, 0.001, 1, 1, 0.5, 1e-05, 0.0]/scores.csv", index_col = 0)
            score["methods"] = "[0.0001, 0.0001, 1, 1, 0.5, 1e-06, 0.0]"
            score["methods"] = "only classifier"
            scores = pd.concat([scores, score], axis = 0, ignore_index = True)

            score = pd.read_csv(f"ablation_test/1condition_10000_500_{sigma}_100_{diff}/figures_[8, 4]_[0.001, 0.001, 1, 1, 0.5, 1e-05, 0.01]/scores.csv", index_col = 0)
            score["methods"] = "[0.0001, 0.0001, 1, 1, 0.5, 1e-06, 0.01]"
            score["methods"] = "both"
            scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    # for sigma in [0.2]:
    #     for diff in [4, 8]:
    #         score = pd.read_csv(f"ablation_test/1condition_10000_500_{sigma}_{ndiff}_{diff}/figures_[8, 4]_[0.001, 0.001, 0, 1, 0.5, 1e-05, 0.1]/scores.csv", index_col = 0)
    #         score["methods"] = "[0.0001, 0.0001, 0, 1, 0.5, 1e-06, 0.1]"
    #         score["methods"] = "only contrastive"
    #         scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    #         score = pd.read_csv(f"ablation_test/1condition_10000_500_{sigma}_{ndiff}_{diff}/figures_[8, 4]_[0.001, 0.001, 1, 1, 0.5, 1e-05, 0.0]/scores.csv", index_col = 0)
    #         score["methods"] = "[0.0001, 0.0001, 1, 1, 0.5, 1e-06, 0.0]"
    #         score["methods"] = "only classifier"
    #         scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    #         score = pd.read_csv(f"ablation_test/1condition_10000_500_{sigma}_{ndiff}_{diff}/figures_[8, 4]_[0.001, 0.001, 1, 1, 0.5, 1e-05, 0.01]/scores.csv", index_col = 0)
    #         score["methods"] = "[0.0001, 0.0001, 1, 1, 0.5, 1e-06, 0.01]"
    #         score["methods"] = "both"
    #         scores = pd.concat([scores, score], axis = 0, ignore_index = True)

# violin plot
import seaborn as sns
fig = plt.figure(figsize = (25,5))
ax = fig.subplots(nrows = 1, ncols =6)
sns.violinplot(data = scores, x = "methods", y = "ARI (condition)", ax = ax[0])
sns.violinplot(data = scores, x = "methods", y = "MSE", ax = ax[1])
sns.violinplot(data = scores, x = "methods", y = "R2", ax = ax[2])
sns.violinplot(data = scores, x = "methods", y = "MSE (diff)", ax = ax[3])
sns.violinplot(data = scores, x = "methods", y = "R2 (diff)", ax = ax[4])
sns.violinplot(data = scores, x = "methods", y = "Pearson", ax = ax[5])
fig.tight_layout()
scores = scores.groupby("methods").mean()
print(scores)
# The result shows that using only contrastive loss performs the worst, using both performs the best.

# %%
