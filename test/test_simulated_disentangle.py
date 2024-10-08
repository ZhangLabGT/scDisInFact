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
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 

# In[]
sigma = 0.4
n_diff_genes = 100
diff = 8
ngenes = 500
ncells_total = 10000 
n_batches = 2
data_dir = f"../data/simulated/unif/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
result_dir = f"./results_simulated/disentangle/unif/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: randomly remove some celltypes?
counts_gt = []
counts_ctrl_healthy = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_severe = []
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
counts_test = []
meta_cells = []
for batch_id in range(n_batches):
    # generate permutation
    permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
    # since there are totally four combinations of conditions, separate the cells into four groups
    chuck_size = int(counts_gt[batch_id].shape[0]/4)

    # counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
    #                                     counts_ctrl_severe[batch_id][permute_idx[chuck_size:(2*chuck_size)],:], 
    #                                     counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
    #                                     counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))

    # meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
    # meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
    # meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
    # meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["severe"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
    # meta_cell["annos"] = label_annos[batch_id][permute_idx]
    # meta_cells.append(meta_cell)

    # remove some matrix (or batch effect will not affect the wilcoxon test)
    if batch_id == 0:
        counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                            # counts_ctrl_severe[batch_id][permute_idx[chuck_size:(2*chuck_size)],:], 
                                            counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                            counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))

        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta_cell["batch"] = np.array([batch_id] * counts_test[-1].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["annos"] = np.concatenate([label_annos[batch_id][permute_idx][:chuck_size], label_annos[batch_id][permute_idx][(2*chuck_size):(3*chuck_size)], label_annos[batch_id][permute_idx][(3*chuck_size):]], axis = 0)
        meta_cells.append(meta_cell)

    else:
        counts_test.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                            counts_ctrl_severe[batch_id][permute_idx[chuck_size:(2*chuck_size)],:], 
                                            # counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                            counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
    

    
        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta_cell["batch"] = np.array([batch_id] * counts_test[-1].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chuck_size + ["severe"] * chuck_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chuck_size))
        meta_cell["annos"] = np.concatenate([label_annos[batch_id][permute_idx][:chuck_size], label_annos[batch_id][permute_idx][chuck_size:(2*chuck_size)], label_annos[batch_id][permute_idx][(3*chuck_size):]], axis = 0)
        meta_cells.append(meta_cell)

data_dict = scdisinfact.create_scdisinfact_dataset(counts_test, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

counts_scinsight = []
meta_scinsight = []
for dataset, meta_cell in zip(data_dict["datasets"], data_dict["meta_cells"]):
    counts_scinsight.append(dataset.counts.numpy())
    meta_scinsight.append(meta_cell)

meta_scinsight = pd.concat(meta_scinsight, axis = 0)
counts_scinsight = np.concatenate(counts_scinsight, axis = 0)
# os.mkdir(data_dir + "scinsight")
# np.savetxt(data_dir + "scinsight/counts.txt", counts_scinsight)
# meta_scinsight.to_csv(data_dir + "scinsight/meta.csv")

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
nepochs = 100
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
_ = model.eval()
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
_ = model.eval()


# In[] Plot results
# one forward pass
z_cs = []
z_ds = []

for dataset in data_dict["datasets"]:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        z_cs.append(z_c.cpu().detach().numpy())
        z_ds.append([x.cpu().detach().numpy() for x in z_d])

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
pca_op = PCA(n_components = 2)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))


comment = f'results_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

batch_annos = np.concatenate([x["batch"].values.squeeze() for x in data_dict["meta_cells"]])
batch_annos = np.where(batch_annos == 0, "batch 1", "batch 2")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "UMAP", figsize = (10,5), save = (result_dir + comment+"common_dims_annos.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "separate", axis_label = "UMAP", figsize = (10,10), save = (result_dir + comment+"common_dims_annos_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "batches", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "UMAP", figsize = (7,5), save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + comment+"diff_dims1_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "batches", axis_label = "PCA", figsize = (7,5), save = (result_dir + comment+"diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + comment+"diff_dims2_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + comment+"diff_dims2_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "batches", axis_label = "PCA", figsize = (7,5), save = (result_dir + comment+"diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[] 
#-----------------------------------------------------------------------------------------------------------------------------------------
#
# scInsight
#
#-----------------------------------------------------------------------------------------------------------------------------------------
result_scinsight = f"./results_simulated/scinsight/2conds_base_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/scinsight_cartesian/"

W2 = pd.read_csv(result_scinsight + "W2.txt", sep = "\t")
H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t")
H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t")
H3 = pd.read_csv(result_scinsight + "H_3.txt", sep = "\t")
H4 = pd.read_csv(result_scinsight + "H_4.txt", sep = "\t")
W11 = pd.read_csv(result_scinsight + "W11.txt", sep = "\t")
W12 = pd.read_csv(result_scinsight + "W12.txt", sep = "\t")
W13 = pd.read_csv(result_scinsight + "W13.txt", sep = "\t")
W14 = pd.read_csv(result_scinsight + "W14.txt", sep = "\t")
W15 = pd.read_csv(result_scinsight + "W15.txt", sep = "\t")
W16 = pd.read_csv(result_scinsight + "W16.txt", sep = "\t")


#  [ctrl, healthy, batch 0], [ctrl, healthy, batch 1], [stim, healthy, batch0],
#  [ctrl, severe, batch 1], [stim, severe, batch 0], [stim, severe, batch 1]
x_cond = [W11.values@H1.values, W12.values@H1.values, W13.values@H2.values, \
          W14.values@H3.values, W15.values@H4.values, W16.values@H4.values]
x_cond = np.concatenate(x_cond, axis = 0)

umap_op = UMAP(min_dist = 0.1, random_state = 0)
w2_umap = umap_op.fit_transform(W2.values)

x_cond_umap = umap_op.fit_transform(x_cond)

label_annos = np.concatenate([meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "annos"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 1), "annos"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "annos"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "annos"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 0), "annos"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "annos"].values.squeeze(),
                            ], axis = 0)

label_batch = np.concatenate([meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "batch"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 1), "batch"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "batch"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "batch"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 0), "batch"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "batch"].values.squeeze(),
                            ], axis = 0)

label_condition1 = np.concatenate([meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "condition 1"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 1), "condition 1"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "condition 1"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "condition 1"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 0), "condition 1"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "condition 1"].values.squeeze()
                              ], axis = 0)

label_condition2 = np.concatenate([meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "condition 2"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 1), "condition 2"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "healthy") & (meta_scinsight["batch"] == 0), "condition 2"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "ctrl") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "condition 2"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 0), "condition 2"].values.squeeze(),
                              meta_scinsight.loc[(meta_scinsight["condition 1"] == "stim") & (meta_scinsight["condition 2"] == "severe") & (meta_scinsight["batch"] == 1), "condition 2"].values.squeeze()
                              ], axis = 0)

label_condition_comb = np.array([x + "_" + y for x,y in zip(label_condition1, label_condition2)])


utils.plot_latent(zs = w2_umap, annos = label_annos, mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_celltypes.png" if result_scinsight else None , markerscale = 6, s = 5)
utils.plot_latent(zs = w2_umap, annos = label_annos, batches = label_batch, mode = "separate", axis_label = "UMAP", figsize = (10,10), save = result_scinsight + "common_celltypes_sep.png" if result_scinsight else None , markerscale = 6, s = 5)
utils.plot_latent(zs = w2_umap, annos = label_batch, mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_batches.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = w2_umap, annos = label_condition_comb, mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_condition.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_cond_umap, annos = label_annos, mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff_celltypes.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_cond_umap, annos = label_batch, mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff_batch.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_cond_umap, annos = label_condition_comb, mode = "annos", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff_condition.png" if result_scinsight else None, markerscale = 6, s = 5)


# In[]
#------------------------------------------------------------------------------------------------------------------------------------------
#
# Benchmark, common space, check removal of batch effect (=condition effect), keep cluster information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# removal of batch and condition effect
# 1. scdisinfact
silhouette_batch_scdisinfact = bmk.silhouette_batch(X = np.concatenate(z_cs, axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict["meta_cells"]]), \
                                                    group_gt = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), verbose = False)
print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))
# 2. scinsight
silhouette_batch_scinsight = bmk.silhouette_batch(X = W2.values, batch_gt = label_batch, group_gt = label_annos, verbose = False)
print('Silhouette batch (scInsight): {:.3f}'.format(silhouette_batch_scinsight))


# NMI and ARI measure the separation of cell types
# 1. scdisinfact
nmi_cluster_scdisinfact = []
ari_cluster_scdisinfact = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    print(np.unique(leiden_labels_clusters).shape[0])
    nmi_cluster_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_clusters))
    ari_cluster_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_clusters))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_cluster_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_cluster_scdisinfact)))

# 2. scinsight
nmi_cluster_scinsight = []
ari_cluster_scinsight = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = W2.values, knn_indices = None, knn_dists = None, resolution = resolution)
    print(np.unique(leiden_labels_clusters).shape[0])
    nmi_cluster_scinsight.append(bmk.nmi(group1 = label_annos, group2 = leiden_labels_clusters))
    ari_cluster_scinsight.append(bmk.ari(group1 = label_annos, group2 = leiden_labels_clusters))
print('NMI (scInsight): {:.3f}'.format(max(nmi_cluster_scinsight)))
print('ARI (scInsight): {:.3f}'.format(max(ari_cluster_scinsight)))


#------------------------------------------------------------------------------------------------------------------------------------------
#
# condition-specific space, check removal of batch effect, keep condition information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# remove batch effect
silhouette_condition_scdisinfact1 = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict["meta_cells"]]), group_gt = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict["meta_cells"]]), verbose = False)
print('Silhouette condition 1, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact1))
silhouette_condition_scdisinfact2 = bmk.silhouette_batch(X = np.concatenate([x[1] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict["meta_cells"]]), group_gt = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict["meta_cells"]]), verbose = False)
print('Silhouette condition 2, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact2))

silhouette_condition_scinsight = bmk.silhouette_batch(X = x_cond, batch_gt = label_batch, group_gt = label_condition_comb, verbose = False)
print('Silhouette condition, removal of batch effect (scInsight): {:.3f}'.format(silhouette_condition_scinsight))

# keep condition information
nmi_condition_scdisinfact1 = []
ari_condition_scdisinfact1 = []
for resolution in range(-3, 1, 1):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
    print(np.unique(leiden_labels_conditions).shape[0])
    nmi_condition_scdisinfact1.append(bmk.nmi(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_conditions))
    ari_condition_scdisinfact1.append(bmk.ari(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_conditions))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact1)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact1)))

nmi_condition_scdisinfact2 = []
ari_condition_scdisinfact2 = []
for resolution in range(-3, 1, 1):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[1] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
    print(np.unique(leiden_labels_conditions).shape[0])
    nmi_condition_scdisinfact2.append(bmk.nmi(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_conditions))
    ari_condition_scdisinfact2.append(bmk.ari(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict["meta_cells"]]), group2 = leiden_labels_conditions))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact2)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact2)))


nmi_condition_scinsight = []
ari_condition_scinsight = []
for resolution in range(-3, 1, 1):
    leiden_labels_conditions = bmk.leiden_cluster(X = x_cond, knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
    print(np.unique(leiden_labels_conditions).shape[0])
    nmi_condition_scinsight.append(bmk.nmi(group1 = label_condition_comb, group2 = leiden_labels_conditions))
    ari_condition_scinsight.append(bmk.ari(group1 = label_condition_comb, group2 = leiden_labels_conditions))
print('NMI (scInsight): {:.3f}'.format(max(nmi_condition_scinsight)))
print('ARI (scInsight): {:.3f}'.format(max(ari_condition_scinsight)))


scores_scdisinfact = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "Silhouette batch (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (condition & batches)"])
scores_scdisinfact["methods"] = np.array(["scDisInFact", "scDisInFact"])
scores_scdisinfact["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact), max(nmi_cluster_scdisinfact)])
scores_scdisinfact["ARI (common)"] = np.array([max(ari_cluster_scdisinfact), max(ari_cluster_scdisinfact)])
scores_scdisinfact["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact, silhouette_batch_scdisinfact])
scores_scdisinfact["NMI (condition)"] = np.array([max(nmi_condition_scdisinfact1), max(nmi_condition_scdisinfact2)])
scores_scdisinfact["ARI (condition)"] = np.array([max(ari_condition_scdisinfact1), max(ari_condition_scdisinfact2)])
scores_scdisinfact["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scdisinfact1, silhouette_condition_scdisinfact2])


scores_scinsight = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "Silhouette batch (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (condition & batches)"])
scores_scinsight["methods"] = np.array(["scInsight"])
scores_scinsight["NMI (common)"] = np.array([max(nmi_cluster_scinsight)])
scores_scinsight["ARI (common)"] = np.array([max(ari_cluster_scinsight)])
scores_scinsight["Silhouette batch (common)"] = np.array([silhouette_batch_scinsight])
scores_scinsight["NMI (condition)"] = np.array([max(nmi_condition_scinsight)])
scores_scinsight["ARI (condition)"] = np.array([max(ari_condition_scinsight)])
scores_scinsight["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scinsight])

scores = pd.concat([scores_scdisinfact, scores_scinsight], axis = 0)
scores.to_csv(result_dir + "score_cartesian.csv")



# In[] 
# NOTE: Plot barplot of the scores if we have baseline
if True:
    plt.rcParams["font.size"] = 20
    def show_values_on_bars(axs):
        def _show_on_single_plot(ax):        
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = '{:.4f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    scores_all = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (common)", "Silhouette batch (condition & batches)"])
    result_dir = "./results_simulated/disentangle/unif/"
    for dataset in ["0.4_20_2", "0.4_50_2", "0.4_100_2", "0.4_20_4", "0.4_50_4", "0.4_100_4", "0.4_20_8", "0.4_50_8", "0.4_100_8"]:
        # first condition
        score = pd.read_csv(result_dir + "2conds_base_10000_500_" + dataset + "/score_cartesian.csv", index_col = 0)
                
        # NMI
        nmi_common_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "NMI (common)"].values)
        nmi_condition_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "NMI (condition)"].values)
        nmi_common_scinsight = np.max(score.loc[score["methods"] == "scInsight", "NMI (common)"].values)
        nmi_condition_scinsight = np.max(score.loc[score["methods"] == "scInsight", "NMI (condition)"].values)

        # ARI
        ari_common_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "ARI (common)"].values)
        ari_condition_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "ARI (condition)"].values)
        ari_common_scinsight = np.max(score.loc[score["methods"] == "scInsight", "ARI (common)"].values)
        ari_condition_scinsight = np.max(score.loc[score["methods"] == "scInsight", "ARI (condition)"].values)

        # silhouette
        sil_common_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "Silhouette batch (common)"].values)
        sil_common_scinsight = np.max(score.loc[score["methods"] == "scInsight", "Silhouette batch (common)"].values)
        sil_condition_batches_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "Silhouette batch (condition & batches)"].values)
        sil_condition_batches_scinsight = np.max(score.loc[score["methods"] == "scInsight", "Silhouette batch (condition & batches)"].values)

        scores_all = pd.concat([scores_all,
                                pd.DataFrame.from_dict({
                                    "methods": ["scDisInFact"],
                                    "NMI (common)": [nmi_common_scdisinfact], 
                                    "ARI (common)": [ari_common_scdisinfact], 
                                    "NMI (condition)": [nmi_condition_scdisinfact], 
                                    "ARI (condition)": [ari_condition_scdisinfact], 
                                    "Silhouette batch (common)": [sil_common_scdisinfact], 
                                    "Silhouette batch (condition & batches)": [sil_condition_batches_scdisinfact]
                                })], axis = 0, ignore_index = True)
        
        scores_all = pd.concat([scores_all,
                                pd.DataFrame.from_dict({
                                    "methods": ["scINSIGHT"],
                                    "NMI (common)": [nmi_common_scinsight], 
                                    "ARI (common)": [ari_common_scinsight], 
                                    "NMI (condition)": [nmi_condition_scinsight], 
                                    "ARI (condition)": [ari_condition_scinsight], 
                                    "Silhouette batch (common)": [sil_common_scinsight], 
                                    "Silhouette batch (condition & batches)": [sil_condition_batches_scinsight]
                                })], axis = 0, ignore_index = True)

# In[]
if True:
    fig = plt.figure(figsize = (15,5))
    ax = fig.subplots(nrows = 1, ncols = 3)
    sns.boxplot(data = scores_all, x = "methods", y = "NMI (common)", ax = ax[0])
    ax[0].set_ylabel("NMI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("NMI (shared)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all, x = "methods", y = "ARI (common)", ax = ax[1])
    ax[1].set_ylabel("ARI")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("ARI (shared)")
    ax[1].set_xlabel(None)


    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (common)", ax = ax[2])
    ax[2].set_ylabel("Silhouette batch")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[2].set_title("Silhouette batch\n(shared)")
    ax[2].set_ylim(0.8, 1)
    ax[2].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot_common.png", bbox_inches = "tight")

    fig = plt.figure(figsize = (15,5))
    ax = fig.subplots(nrows = 1, ncols = 3)
    sns.boxplot(data = scores_all, x = "methods", y = "NMI (condition)", ax = ax[0])
    ax[0].set_ylabel("NMI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("NMI (condition)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all, x = "methods", y = "ARI (condition)", ax = ax[1])
    ax[1].set_ylabel("ARI")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("ARI (condition)")
    ax[1].set_xlabel(None)


    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2])
    ax[2].set_ylabel("ASW-batch")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[2].set_title("Silhouette batch\n(condition & batches)")
    ax[2].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot_condition.png", bbox_inches = "tight")

# In[]
if True:
    fig = plt.figure(figsize = (15,5))
    ax = fig.subplots(nrows = 1, ncols = 3)

    sns.boxplot(data = scores_all, x = "methods", y = "ARI (common)", ax = ax[0])
    sns.stripplot(data = scores_all, x = "methods", y = "ARI (common)", ax = ax[0], color = "black", dodge = True) 
    ax[0].set_ylabel("ARI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("ARI\n(shared)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (common)", ax = ax[1])
    sns.stripplot(data = scores_all, x = "methods", y = "Silhouette batch (common)", ax = ax[1], color = "black", dodge = True) 
    ax[1].set_ylabel("Silhouette batch")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("Silhouette batch\n(shared)")
    ax[1].set_ylim(0.8, 1)
    ax[1].set_xlabel(None)

    # plt.tight_layout()
    # fig.savefig(result_dir + "boxplot_common.png", bbox_inches = "tight")

    # fig = plt.figure(figsize = (10,5))
    # ax = fig.subplots(nrows = 1, ncols = 2)

    # sns.boxplot(data = scores_all, x = "methods", y = "ARI (condition)", ax = ax[0])
    # sns.stripplot(data = scores_all, x = "methods", y = "ARI (condition)", ax = ax[0], color = "black", dodge = True) 
    # ax[0].set_ylabel("ARI")
    # ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax[0].set_title("ARI\n(condition)")
    # # ax[0].set_ylim(0, 0.5)
    # ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2])
    sns.stripplot(data = scores_all, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2], color = "black", dodge = True) 
    ax[2].set_ylabel("ASW-batch")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[2].set_title("Silhouette batch\n(condition)")
    # ax[2].set_ylim(0.5, 1)
    ax[2].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot.png", bbox_inches = "tight")


# %%




