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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 
# In[]
sigma = 0.4
n_diff_genes = 20
diff = 2
ngenes = 500
ncells_total = 10000 
n_batches = 2
data_dir = f"../data/simulated/simulated_rare_celltype/"
result_dir = f"./results_simulated/rare_celltypes/simulated_rare_celltype/"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

counts_gt = []
counts_ctrl = []
counts_stim = []
label_annos_ctrl = []
label_annos_stim = []

for batch_id in range(n_batches):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts_ctrl.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T)
    counts_stim.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim.txt', sep = "\t", header = None).values.T)

    anno_ctrl = pd.read_csv(data_dir + f'cell_label_ctrl{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    anno_stim = pd.read_csv(data_dir + f'cell_label_stim{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()

    # annotation labels
    label_annos_ctrl.append(np.array([('cell type '+str(i)) for i in anno_ctrl]))
    label_annos_stim.append(np.array([('cell type '+str(i)) for i in anno_stim]))    

# In[]
np.random.seed(0)
counts_test = []
meta_cells = []
for batch_id in range(n_batches):
    # generate permutation
    permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
    # since there are totally four combinations of conditions, separate the cells into four groups
    chuck_size = int(counts_gt[batch_id].shape[0]/2)

    counts_test.append(np.concatenate([counts_ctrl[batch_id][permute_idx[:chuck_size],:], 
                                        counts_stim[batch_id][permute_idx[chuck_size:],:]], axis = 0))

    meta_cell = pd.DataFrame(columns = ["batch", "condition", "annos"])
    meta_cell["batch"] = np.array([batch_id] * counts_test[-1].shape[0])
    meta_cell["condition"] = np.array(["ctrl"] * chuck_size + ["stim"] * (counts_gt[batch_id].shape[0] - chuck_size))
    meta_cell["annos"] = np.concatenate([label_annos_ctrl[batch_id][permute_idx][:chuck_size], label_annos_stim[batch_id][permute_idx][chuck_size:]], axis = 0)
    meta_cells.append(meta_cell)

counts_test = np.concatenate(counts_test, axis = 0)
meta_cells = pd.concat(meta_cells, axis = 0)

data_dict = scdisinfact.create_scdisinfact_dataset(counts_test, meta_cells, condition_key = ["condition"], batch_key = "batch")

# In[]
# Visualize the subsampled data
# log-norm
counts_test_norm = np.log1p(counts_test/np.sum(counts_test, axis = 1, keepdims = True) * 100)
x_umap = UMAP(n_components = 2, min_dist = 0.4).fit_transform(counts_test_norm)
meta_cells["sample"] = meta_cells[["condition", "batch"]].apply(lambda row: '_'.join(row.to_numpy().astype(str)), axis=1)
utils.plot_latent(x_umap, annos = meta_cells["annos"].values.squeeze(), batches = meta_cells["sample"].values.squeeze(), \
                  mode = "separate", figsize = (10, 25), save = None, markerscale = 6)
utils.plot_latent(x_umap, annos = meta_cells["annos"].values.squeeze(), batches = meta_cells["sample"].values.squeeze(), \
                  mode = "separate", figsize = (10, 25), save = None, markerscale = 6)



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
Ks = [8, 2]
batch_size = 64
nepochs = 100
interval = 10
lr = 5e-4

model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
# model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
# _ = model.eval()
# torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
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
z_cs = np.concatenate(z_cs, axis = 0)
z_cs_umap = umap_op.fit_transform(z_cs)
z_ds_umap = []
z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))


comment = f'results_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]])
batch_annos = np.concatenate([x["batch"].values.squeeze() for x in data_dict["meta_cells"]])
batch_annos = np.where(batch_annos == 0, "batch 1", "batch 2")
condition = np.concatenate([x["condition"] for x in data_dict["meta_cells"]])
batch_cond = np.array([batch + "-" + cond for batch, cond in zip(batch_annos, condition)])

utils.plot_latent(zs = z_cs_umap, annos = annos, batches = batch_annos, mode = "annos", axis_label = "UMAP", \
                  figsize = (10,5), save = (result_dir + comment+"common_dims_annos.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = annos, batches = batch_cond, mode = "separate", axis_label = "UMAP", \
                  figsize = (10,20), save = (result_dir + comment+"common_dims_annos_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + comment+"diff_dims_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition"].values.squeeze() for x in data_dict["meta_cells"]]), batches = batch_annos, \
    mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + comment+"diff_dims_cond.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]
# running leiden clustering algorithm
# leiden_labels_clusters = bmk.leiden_cluster(X = z_cs, knn_indices = None, knn_dists = None, resolution = 2.0)
# np.savetxt("results_simulated/rare_celltypes/simulated_rare_celltype/leiden_cluster_id.txt", leiden_labels_clusters, fmt = "%d")

leiden_labels_clusters = np.loadtxt("results_simulated/rare_celltypes/simulated_rare_celltype/leiden_cluster_id.txt").astype(int)
utils.plot_latent(zs = z_cs_umap, annos = leiden_labels_clusters, batches = batch_cond, mode = "separate", axis_label = "UMAP", \
                  figsize = (10,20), save = result_dir + comment+"common_dims_leiden_separate.png" , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

# In[]
# the rare cell type correspond to cluster # 16
# leiden_labels_clusters
counts = []
for dataset in data_dict["datasets"]:
    with torch.no_grad():
       counts.append(dataset.counts_norm.numpy()) 
counts = np.concatenate(counts, axis = 0)
pvals = bmk.wilcoxon_rank_sum(counts[leiden_labels_clusters == 16,:], counts[leiden_labels_clusters != 16,:], fdr = True)

# calculate AUPRC
pvals_gt = np.ones(500)
pvals_gt[:50] = 0

AUPRC = bmk.compute_auprc(1 - pvals, 1 - pvals_gt)
AUROC = bmk.compute_auroc(1 - pvals, 1 - pvals_gt)
Eprec = bmk.compute_earlyprec(1 - pvals, 1 - pvals_gt)
print(AUPRC)
print(AUROC)
print(Eprec)
scores = pd.DataFrame(columns = ["Score", "Name"])
scores["Score"] = np.array([AUPRC, AUROC, Eprec])
scores["Name"] = np.array(["AUPRC", "AUROC", "Eprec"])

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot()
ax = sns.barplot(scores, x = "Name", y = "Score", ax = ax)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = 0)  
ax.set_xlabel(None)
fig.savefig("results_simulated/rare_celltypes/simulated_rare_celltype/boxplot.png", bbox_inches = "tight")

# %%
