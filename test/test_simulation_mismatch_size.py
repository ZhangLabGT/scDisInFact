# In[]
import sys, os
import time
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.bmk as bmk

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import scipy.stats as stats
from umap import UMAP
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset_dir = "1condition_2"

n_batches = 2

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [8, 2]

lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

batch_size = 64
nepochs = 100
interval = 10
lr = 5e-4


# In[]

print("# -------------------------------------------------------------------------------------------")
print("#")
print('# Dataset' + dataset_dir)
print("#")
print("# -------------------------------------------------------------------------------------------")

data_dir = "../data/simulated/" + dataset_dir + "/"
result_dir = "./results_simulated/mismatch_size/" + dataset_dir + "/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
scores_all = []

counts_gt = []
counts_ctrl = []
counts_stim = []
label_annos = []
for batch_id in range(n_batches):
    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    
    # sub-sampling
    idx1 = np.where(anno == 1)[0]
    idx2 = np.where(anno == 2)[0][::2]
    idx3 = np.where(anno == 3)[0][::5]
    idx4 = np.where(anno == 4)[0][::7]
    idx5 = np.where(anno == 5)[0][::10]
    idx = np.concatenate([idx1, idx2, idx3, idx4, idx5])
    print(idx.shape)
    anno = anno[idx]
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T[idx,:])
    counts_ctrl.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T[idx,:])
    counts_stim.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T[idx,:])
    
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))

# In[]
np.random.seed(0)
counts = []
meta_cells = []
for batch_id in range(n_batches):
    # generate permutation
    permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
    # since there are totally four combinations of conditions, separate the cells into four groups
    chunk_size = int(counts_gt[batch_id].shape[0]/2)
    count = np.concatenate([counts_ctrl[batch_id][permute_idx[:chunk_size],:],
                            counts_stim[batch_id][permute_idx[chunk_size:],:]], axis = 0)

    meta_cell = pd.DataFrame(columns = ["batch", "condition", "annos"])
    meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
    meta_cell["condition"] = np.array(["ctrl"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - chunk_size))
    meta_cell["annos"] = label_annos[batch_id][permute_idx]
        
    meta_cells.append(meta_cell)
    counts.append(count)

counts = np.concatenate(counts, axis = 0)
meta_cells = pd.concat(meta_cells, axis = 0)

clusts, clust_counts = np.unique(meta_cells["annos"].values.squeeze(), return_counts = True)
clusts = [x for x in clusts[np.argsort(clust_counts)[::-1]]]
clust_counts = [x for x in np.sort(clust_counts)[::-1]]
clusts.append(None)
clust_counts.append(None)

for clust, clust_count in zip(clusts, clust_counts):    
    if clust is None:
        data_dict_full = scdisinfact.create_scdisinfact_dataset(counts, meta_cells, condition_key  = ["condition"], batch_key = "batch")
        model = scdisinfact.scdisinfact(data_dict = data_dict_full, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                        reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
        model.train()
        losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
        _ = model.eval()
        torch.save(model.state_dict(), result_dir + f"scdisinfact_full.pth")
        model.load_state_dict(torch.load(result_dir + f"scdisinfact_full.pth", map_location = device))

    else:
        # remove cell type according to size
        # 1. mismatch across conditions
        # remove_idx = ((meta_cells["annos"] == clust) & (meta_cells["condition"] == "ctrl")).values.squeeze()
        # 2. remove across batches
        remove_idx = ((meta_cells["annos"] == clust) & (meta_cells["batch"] == 0)).values.squeeze()
        meta_cells_new = meta_cells.loc[~remove_idx,:]
        counts_new = counts[~remove_idx,:]
        # print(np.unique(meta_cells_new.loc[meta_cells_new["batch"] == 0, "annos"].values.squeeze(), return_counts = True))
        # print(np.unique(meta_cells_new.loc[meta_cells_new["batch"] == 1, "annos"].values.squeeze(), return_counts = True))    

        # train model
        data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_new, meta_cells_new, condition_key = ["condition"], batch_key = "batch")
        model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                        reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
        model.train()
        losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
        _ = model.eval()

        torch.save(model.state_dict(), result_dir + f"mismatch_cond_{clust_count}.pth")
        model.load_state_dict(torch.load(result_dir + f"mismatch_cond_{clust_count}.pth", map_location = device))

    z_cs = []
    z_ds = []
    for dataset in data_dict_train["datasets"]:
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

    batch_annos = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]])
    batch_annos = np.where(batch_annos == 0, "batch 1", "batch 2")

    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "separate", axis_label = "UMAP", figsize = (10,10), save = (result_dir + f"shared_annos_sep_{clust_count}.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + f"unshared_annos_{clust_count}.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

    silhouette_batch_scdisinfact = bmk.silhouette_batch(X = np.concatenate(z_cs, axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
                                                        group_gt = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), verbose = False)
    print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))
    # NMI and ARI measure the separation of cell types
    nmi_cluster_scdisinfact = []
    ari_cluster_scdisinfact = []
    for resolution in np.arange(0.1, 10, 0.5):
        leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
        nmi_cluster_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_clusters))
        ari_cluster_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_clusters))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_cluster_scdisinfact)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_cluster_scdisinfact)))


    silhouette_condition_scdisinfact = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
                                                             group_gt = np.concatenate([x["condition"].values.squeeze() for x in data_dict_train["meta_cells"]]), verbose = False)
    print('Silhouette condition, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact))
    nmi_condition_scdisinfact = []
    ari_condition_scdisinfact = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        nmi_condition_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["condition"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["condition"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact)))

    scores_scdisinfact = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "Silhouette batch (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (condition & batches)"])
    scores_scdisinfact["methods"] = np.array([f"rm_{clust_count}"])
    scores_scdisinfact["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact)])
    scores_scdisinfact["ARI (common)"] = np.array([max(ari_cluster_scdisinfact)])
    scores_scdisinfact["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact])
    scores_scdisinfact["NMI (condition)"] = np.array([max(nmi_condition_scdisinfact)])
    scores_scdisinfact["ARI (condition)"] = np.array([max(ari_condition_scdisinfact)])
    scores_scdisinfact["Silhouette batch (condition & batches)"] = np.max(np.array([silhouette_condition_scdisinfact]))    
    scores_all.append(scores_scdisinfact)

scores_all = pd.concat(scores_all, axis = 0)
scores_all.to_csv(result_dir + "score_all.csv")

# In[]
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

scores_scdisinfact = []
for dataset_dir in ["1condition_0", "1condition_1", "1condition_2"]:
    result_dir = "./results_simulated/mismatch_size/" + dataset_dir + "/"
    scores_all = pd.read_csv(result_dir + "score_all.csv", index_col = 0)
    scores_scdisinfact.append(scores_all)


scores_scdisinfact = pd.concat(scores_scdisinfact, axis = 0)
scores_scdisinfact["methods"] = [eval(x.split("_")[1]) if x is not None else 0 for x in scores_scdisinfact["methods"]]
scores_scdisinfact.index = np.arange(scores_scdisinfact.shape[0])

nmi_common = scores_scdisinfact[["methods", "NMI (common)"]]
ari_common = scores_scdisinfact[["methods", "ARI (common)"]]
asw_common = scores_scdisinfact[["methods", "Silhouette batch (common)"]]
nmi_condition = scores_scdisinfact[["methods", "NMI (condition)"]]
ari_condition = scores_scdisinfact[["methods", "ARI (condition)"]]
asw_condition = scores_scdisinfact[["methods", "Silhouette batch (condition & batches)"]]

fig = plt.figure(figsize = (30,5))
ax = fig.subplots(nrows = 1, ncols = 3)

sns.lineplot(data = scores_scdisinfact, x = "methods", y = "NMI (common)", ax = ax[0], ci=None, marker = "o")
ax[0].set_ylabel("NMI")
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0].set_title("NMI (shared)")
ax[0].set_ylim(0.5, 1.1)
ax[0].set_xlabel("Mismatched cluster size")

sns.lineplot(data = scores_scdisinfact, x = "methods", y = "ARI (common)", ax = ax[1], ci=None, marker = "o")
ax[1].set_ylabel("ARI")
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].set_title("ARI (shared)")
ax[1].set_ylim(0.5, 1.1)
ax[1].set_xlabel("Mismatched cluster size")

sns.lineplot(data = scores_scdisinfact, x = "methods", y = "Silhouette batch (common)", ax = ax[2], ci=None, marker = "o")
ax[2].set_ylabel("ASW-batch")
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].set_title("ASW-batch\n(shared)")
ax[2].set_ylim(0.5, 1.1)
ax[2].set_xlabel("Mismatched cluster size")

plt.tight_layout()
fig.savefig("./results_simulated/mismatch_size/plot_common.png", bbox_inches = "tight")

fig = plt.figure(figsize = (30,5))
ax = fig.subplots(nrows = 1, ncols = 3)
sns.lineplot(data = scores_scdisinfact, x = "methods", y = "NMI (condition)", ax = ax[0], ci=None, marker = "o")
ax[0].set_ylabel("NMI")
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0].set_title("NMI (condition)")
ax[0].set_ylim(0.5, 1.1)
ax[0].set_xlabel("Mismatched cluster size")

sns.lineplot(data = scores_scdisinfact, x = "methods", y = "ARI (condition)", ax = ax[1], ci=None, marker = "o")
ax[1].set_ylabel("ARI")
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].set_title("ARI (condition)")
ax[1].set_ylim(0.5, 1.1)
ax[1].set_xlabel("Mismatched cluster size")

sns.lineplot(data = scores_scdisinfact, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2], ci=None, marker = "o")
ax[2].set_ylabel("ASW-batch")
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].set_title("ASW-batch\n(condition & batches)")
ax[2].set_ylim(0.5, 1.1)
ax[2].set_xlabel("Mismatched cluster size")

plt.tight_layout()
fig.savefig("./results_simulated/mismatch_size/plot_condition.png", bbox_inches = "tight")

# In[]
if True:
    fig = plt.figure(figsize = (12,5))
    ax = fig.subplots(nrows = 1, ncols = 2)

    sns.boxplot(data = scores_all, x = "methods", y = "ARI (common)", ax = ax[0])
    sns.stripplot(data = scores_all, x = "methods", y = "ARI (common)", ax = ax[0], color = "black", dodge = True) 
    ax[0].set_ylabel("ARI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("ARI\n(shared)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (common)", ax = ax[1])
    sns.stripplot(data = scores_all, x = "methods", y = "Silhouette batch (common)", ax = ax[1], color = "black", dodge = True) 
    ax[1].set_ylabel("ASW-batch")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("ASW-batch\n(shared)")
    ax[1].set_ylim(0.8, 1)
    ax[1].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot.png", bbox_inches = "tight")



# %%
