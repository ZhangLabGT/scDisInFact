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


# In[]
# sigma = 0.2
# n_diff_genes = 50
# diff = 8
# ngenes = 500
# ncells_total = 10000 
# n_batches = 6
# data_dir = f"../data/simulated_new/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
# result_dir = f"./simulated/prediction/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

data_dir = f"../data/simulated/" + sys.argv[1] + "/"
# lsa performs the best
result_dir = f"./simulated/prediction/" + sys.argv[1] + "/"
n_diff_genes = eval(sys.argv[1].split("_")[4])
n_batches = 6
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: randomly remove some celltypes?
counts_ctrls = []
counts_stims1 = []
counts_stims2 = []
# cell types
label_annos = []
# batch labels
label_batches = []
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
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
    # batch labels
    label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
    label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
    label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
    label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))

# In[]
# Train with ctrl in batches 1 & 2, stim1 in batches 3 & 4, stim2 in batches 5 & 6
label_conditions = label_ctrls[0:2] + label_stims1[2:4] + label_stims2[4:]
# sub
# label_conditions = label_ctrls[0:2] + label_stims1[2:]
# sub2
# label_conditions = label_stims1[0:2] + label_stims2[2:]

condition_ids, condition_names = pd.factorize(np.concatenate(label_conditions, axis = 0))
batch_ids, batch_names = pd.factorize(np.concatenate(label_batches, axis = 0))
anno_ids, anno_names = pd.factorize(np.concatenate(label_annos, axis = 0))

counts = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:4] + counts_stims2[4:], axis = 0)
# sub
# counts = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:], axis = 0)
# sub2
# counts = np.concatenate(counts_stims1[0:2] + counts_stims2[2:], axis = 0)

datasets = []
for batch_id, batch_name in enumerate(batch_names):
        datasets.append(scdisinfact.dataset(counts = counts[batch_ids == batch_id,:], 
                                            anno = anno_ids[batch_ids == batch_id], 
                                            diff_labels = [condition_ids[batch_ids == batch_id]], 
                                            batch_id = batch_ids[batch_ids == batch_id]))


# In[] training the model
# TODO: track the time usage and memory usage
import importlib 
importlib.reload(scdisinfact)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

start_time = time.time()
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 0.1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]
nepochs = 50
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets, Ks = Ks, batch_size = 64, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))

# train_joint is more efficient, but does not work as well compared to train
# model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)
# _ = model.eval()
# torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")

model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth", map_location = device))
end_time = time.time()
print("time cost: {:.2f}".format(end_time - start_time))



# In[] Plot results
z_cs = []
z_ds = []
zs = []
# one forward pass
with torch.no_grad():
    for batch_id, dataset in enumerate(datasets):
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(z.cpu().detach().numpy())
        z_ds.append([x.cpu().detach().numpy() for x in z_d])   

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))

z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds], axis = 0)))

zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[] * model.n_diff_factors]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]

        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])
        for diff_factor in range(model.n_diff_factors):
            z_ds_umaps[diff_factor].append(z_ds_umap[diff_factor][start_pointer:end_pointer,:])


    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]

        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])
        for diff_factor in range(model.n_diff_factors):
            z_ds_umaps[diff_factor].append(z_ds_umap[diff_factor][start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]

        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])
        for diff_factor in range(model.n_diff_factors):
            z_ds_umaps[diff_factor].append(z_ds_umap[diff_factor][start_pointer:end_pointer,:])

comment = f"fig_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"common_celltypes_sep.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_condition.png" if result_dir else None, markerscale = 6, s = 5)

for diff_factor in range(model.n_diff_factors):
    utils.plot_latent(zs = z_ds_umaps[diff_factor], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umaps[diff_factor], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_batch.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umaps[diff_factor], annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_condition.png" if result_dir else None, markerscale = 6, s = 5)

# In[] 
#-----------------------------------------------------------------------------------------------------------------------------------------
#
# scInsight
#
#-----------------------------------------------------------------------------------------------------------------------------------------
# read in the result of scInsight
result_scinsight = result_dir + "scinsight_2/"
W2 = pd.read_csv(result_scinsight + "W2.txt", sep = "\t")
H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t")
H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t")
H3 = pd.read_csv(result_scinsight + "H_3.txt", sep = "\t")
W11 = pd.read_csv(result_scinsight + "W11.txt", sep = "\t")
W12 = pd.read_csv(result_scinsight + "W12.txt", sep = "\t")
W13 = pd.read_csv(result_scinsight + "W13.txt", sep = "\t")
W14 = pd.read_csv(result_scinsight + "W14.txt", sep = "\t")
W15 = pd.read_csv(result_scinsight + "W15.txt", sep = "\t")
W16 = pd.read_csv(result_scinsight + "W16.txt", sep = "\t")

x_cond = [W11.values@H1.values, W12.values@H1.values, W13.values@H2.values, W14.values@H2.values, W15.values@H3.values, W16.values@H3.values]
x_cond = np.concatenate(x_cond, axis = 0)

umap_op = UMAP(min_dist = 0.1, random_state = 0)
w2_umap = umap_op.fit_transform(W2.values)

# x_cond = PCA(n_components = 30).fit_transform(x_cond)
# x_cond_umap = x_cond[:,:2]
x_cond_umap = umap_op.fit_transform(x_cond)

w2_umaps = []
x_cond_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        w2_umaps.append(w2_umap[start_pointer:end_pointer,:])
        x_cond_umaps.append(x_cond_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        w2_umaps.append(w2_umap[start_pointer:,:])
        x_cond_umaps.append(x_cond_umap[start_pointer:,:])
        
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]

        w2_umaps.append(w2_umap[start_pointer:end_pointer,:])
        x_cond_umaps.append(x_cond_umap[start_pointer:end_pointer,:])

utils.plot_latent(zs = w2_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_celltypes.png" if result_scinsight else None , markerscale = 6, s = 5)
utils.plot_latent(zs = w2_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_scinsight + "common_celltypes_sep.png" if result_scinsight else None , markerscale = 6, s = 5)
utils.plot_latent(zs = w2_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_batches.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = w2_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + "common_condition.png" if result_scinsight else None, markerscale = 6, s = 5)
diff_factor = 0
utils.plot_latent(zs = x_cond_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff{diff_factor}_celltypes.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_cond_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff{diff_factor}_batch.png" if result_scinsight else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_cond_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_scinsight + f"diff{diff_factor}_condition.png" if result_scinsight else None, markerscale = 6, s = 5)

# In[]
#------------------------------------------------------------------------------------------------------------------------------------------
#
# Benchmark, common space, check removal of batch effect (=condition effect), keep cluster information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# removal of batch and condition effect
# 1. scdisinfact
n_neighbors = 30
gc_cluster_scdisinfact = bmk.graph_connectivity(X = np.concatenate(z_cs, axis = 0), groups = np.concatenate(label_annos, axis = 0), k = n_neighbors)
print('GC cluster (scDisInFact): {:.3f}'.format(gc_cluster_scdisinfact))
silhouette_batch_scdisinfact = bmk.silhouette_batch(X = np.concatenate(z_cs, axis = 0), batch_gt = np.concatenate(label_batches, axis = 0), group_gt = np.concatenate(label_annos, axis = 0), verbose = False)
print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))
# 2. scinsight
gc_cluster_scinsight = bmk.graph_connectivity(X = W2.values, groups = np.concatenate(label_annos, axis = 0), k = n_neighbors)
print('GC cluster (scInsight): {:.3f}'.format(gc_cluster_scinsight))
silhouette_batch_scinsight = bmk.silhouette_batch(X = W2.values, batch_gt = np.concatenate(label_batches, axis = 0), group_gt = np.concatenate(label_annos, axis = 0), verbose = False)
print('Silhouette batch (scInsight): {:.3f}'.format(silhouette_batch_scinsight))




# NMI and ARI measure the separation of cell types
# 1. scdisinfact
nmi_cluster_scdisinfact = []
ari_cluster_scdisinfact = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_cluster_scdisinfact.append(bmk.nmi(group1 = np.concatenate(label_annos), group2 = leiden_labels_clusters))
    ari_cluster_scdisinfact.append(bmk.ari(group1 = np.concatenate(label_annos), group2 = leiden_labels_clusters))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_cluster_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_cluster_scdisinfact)))

# 2. scinsight
nmi_cluster_scinsight = []
ari_cluster_scinsight = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_clusters = bmk.leiden_cluster(X = W2.values, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_cluster_scinsight.append(bmk.nmi(group1 = np.concatenate(label_annos), group2 = leiden_labels_clusters))
    ari_cluster_scinsight.append(bmk.ari(group1 = np.concatenate(label_annos), group2 = leiden_labels_clusters))
print('NMI (scInsight): {:.3f}'.format(max(nmi_cluster_scinsight)))
print('ARI (scInsight): {:.3f}'.format(max(ari_cluster_scinsight)))


#------------------------------------------------------------------------------------------------------------------------------------------
#
# condition-specific space, check removal of batch effect, removal of cell type effect, keep condition information
#
#------------------------------------------------------------------------------------------------------------------------------------------
# removal of batch effect, removal of cell type effect
gc_condition_scdisinfact = bmk.graph_connectivity(X = np.concatenate([x[0] for x in z_ds], axis = 0), groups = np.concatenate(label_conditions, axis = 0), k = n_neighbors)
print('GC condition (scDisInFact): {:.3f}'.format(gc_condition_scdisinfact))
silhouette_condition_scdisinfact = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate(label_batches, axis = 0), group_gt = np.concatenate(label_conditions, axis = 0), verbose = False)
print('Silhouette condition, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact))
silhouette_condition_scdisinfact2 = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate(label_annos, axis = 0), group_gt = np.concatenate(label_conditions, axis = 0), verbose = False)
print('Silhouette condition, removal of cell type effeect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact2))

gc_condition_scinsight = bmk.graph_connectivity(X = x_cond, groups = np.concatenate(label_conditions, axis = 0), k = n_neighbors)
print('GC condition (scInsight): {:.3f}'.format(gc_condition_scinsight))
silhouette_condition_scinsight = bmk.silhouette_batch(X = x_cond, batch_gt = np.concatenate(label_batches, axis = 0), group_gt = np.concatenate(label_conditions, axis = 0), verbose = False)
print('Silhouette condition, removal of batch effect (scInsight): {:.3f}'.format(silhouette_condition_scinsight))
silhouette_condition_scinsight2 = bmk.silhouette_batch(X = x_cond, batch_gt = np.concatenate(label_annos, axis = 0), group_gt = np.concatenate(label_conditions, axis = 0), verbose = False)
print('Silhouette condition, removal of cell type effect (scInsight): {:.3f}'.format(silhouette_condition_scinsight2))

# keep of condition information
nmi_condition_scdisinfact = []
ari_condition_scdisinfact = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_condition_scdisinfact.append(bmk.nmi(group1 = np.concatenate(label_conditions), group2 = leiden_labels_conditions))
    ari_condition_scdisinfact.append(bmk.ari(group1 = np.concatenate(label_conditions), group2 = leiden_labels_conditions))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact)))

nmi_condition_scinsight = []
ari_condition_scinsight = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate(x_cond, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_condition_scinsight.append(bmk.nmi(group1 = np.concatenate(label_conditions), group2 = leiden_labels_conditions))
    ari_condition_scinsight.append(bmk.ari(group1 = np.concatenate(label_conditions), group2 = leiden_labels_conditions))
print('NMI (scInsight): {:.3f}'.format(max(nmi_condition_scinsight)))
print('ARI (scInsight): {:.3f}'.format(max(ari_condition_scinsight)))


scores_scdisinfact = pd.DataFrame(columns = ["methods", "resolution", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "GC (common)", "GC (condition)", "Silhouette batch (common)", "Silhouette batch (condition & celltype)", "Silhouette batch (condition & batches)"])
scores_scdisinfact["NMI (common)"] = np.array(nmi_cluster_scdisinfact)
scores_scdisinfact["ARI (common)"] = np.array(ari_cluster_scdisinfact)
scores_scdisinfact["GC (common)"] = np.array([gc_cluster_scdisinfact] * len(nmi_cluster_scdisinfact))
scores_scdisinfact["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact] * len(nmi_cluster_scdisinfact))

scores_scdisinfact["NMI (condition)"] = np.array(nmi_cluster_scdisinfact)
scores_scdisinfact["ARI (condition)"] = np.array(ari_cluster_scdisinfact)
scores_scdisinfact["GC (condition)"] = np.array([gc_condition_scdisinfact] * len(nmi_cluster_scdisinfact))
scores_scdisinfact["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scdisinfact] * len(nmi_cluster_scdisinfact))
scores_scdisinfact["Silhouette batch (condition & celltype)"] = np.array([silhouette_condition_scdisinfact2] * len(nmi_cluster_scdisinfact))

scores_scdisinfact["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)])
scores_scdisinfact["methods"] = np.array(["scDisInFact"] * len(ari_cluster_scdisinfact))

scores_scinsight = pd.DataFrame(columns = ["methods", "resolution", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "GC (common)", "GC (condition)"])
scores_scinsight["NMI (common)"] = np.array(nmi_cluster_scinsight)
scores_scinsight["ARI (common)"] = np.array(ari_cluster_scinsight)
scores_scinsight["GC (common)"] = np.array([gc_cluster_scinsight] * len(nmi_cluster_scinsight))
scores_scinsight["Silhouette batch (common)"] = np.array([silhouette_batch_scinsight] * len(nmi_cluster_scinsight))

scores_scinsight["NMI (condition)"] = np.array(nmi_condition_scinsight)
scores_scinsight["ARI (condition)"] = np.array(ari_condition_scinsight)
scores_scinsight["GC (condition)"] = np.array([gc_condition_scinsight] * len(nmi_condition_scinsight))
scores_scinsight["Silhouette batch (condition & batches)"] = np.array([silhouette_condition_scinsight] * len(nmi_condition_scinsight))
scores_scinsight["Silhouette batch (condition & celltype)"] = np.array([silhouette_condition_scinsight2] * len(nmi_condition_scinsight))

scores_scinsight["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)])
scores_scinsight["methods"] = np.array(["scInsight"] * len(ari_cluster_scinsight))

scores = pd.concat([scores_scdisinfact, scores_scinsight], axis = 0)
scores.to_csv(result_dir + "latent_score_2.csv")

# In[] 
# NOTE: Plot barplot of the scores if we have baseline
from matplotlib.ticker import FormatStrFormatter
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

    scores_all = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "GC (common)", "GC (condition)", "Silhouette batch (common)", "Silhouette batch (condition & celltype)", "Silhouette batch (condition & batches)"])
    result_dir = "./simulated/prediction/"
    for dataset in ["0.2_20_2", "0.2_50_2", "0.2_100_2", "0.2_20_4", "0.2_50_4", "0.2_100_4", "0.2_20_8", "0.2_50_8", "0.2_100_8"]:
    # for dataset in ["0.2_20_2", "0.2_50_2", "0.2_100_2", "0.3_20_2", "0.3_50_2", "0.3_100_2", "0.4_20_2", "0.4_50_2", "0.4_100_2"]:
        score = pd.read_csv(result_dir + "1condition_10000_500_" + dataset + "/latent_score_2.csv", index_col = 0)
        
        # GC score
        gc_common_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "GC (common)"].values)
        gc_condition_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "GC (condition)"].values)
        gc_common_scinsight = np.max(score.loc[score["methods"] == "scInsight", "GC (common)"].values)
        gc_condition_scinsight = np.max(score.loc[score["methods"] == "scInsight", "GC (condition)"].values)
        
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
        sil_condition_celltype_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "Silhouette batch (condition & celltype)"].values)
        sil_condition_celltype_scinsight = np.max(score.loc[score["methods"] == "scInsight", "Silhouette batch (condition & celltype)"].values)
        sil_condition_batches_scdisinfact = np.max(score.loc[score["methods"] == "scDisInFact", "Silhouette batch (condition & batches)"].values)
        sil_condition_batches_scinsight = np.max(score.loc[score["methods"] == "scInsight", "Silhouette batch (condition & batches)"].values)

        scores_all = scores_all.append({
            "methods": "scDisInFact",
            "NMI (common)": nmi_common_scdisinfact, 
            "ARI (common)": ari_common_scdisinfact, 
            "NMI (condition)": nmi_condition_scdisinfact, 
            "ARI (condition)": ari_condition_scdisinfact, 
            "GC (common)": gc_common_scdisinfact, 
            "GC (condition)": gc_condition_scdisinfact, 
            "Silhouette batch (common)": sil_common_scdisinfact, 
            "Silhouette batch (condition & celltype)": sil_condition_celltype_scdisinfact, 
            "Silhouette batch (condition & batches)": sil_condition_batches_scdisinfact
        }, ignore_index = True)
        
        scores_all = scores_all.append({
            "methods": "scINSIGHT",
            "NMI (common)": nmi_common_scinsight, 
            "ARI (common)": ari_common_scinsight, 
            "NMI (condition)": nmi_condition_scinsight, 
            "ARI (condition)": ari_condition_scinsight, 
            "GC (common)": gc_common_scinsight, 
            "GC (condition)": gc_condition_scinsight, 
            "Silhouette batch (common)": sil_common_scinsight, 
            "Silhouette batch (condition & celltype)": sil_condition_celltype_scinsight, 
            "Silhouette batch (condition & batches)": sil_condition_batches_scinsight
        }, ignore_index = True)

    fig = plt.figure(figsize = (20,5))
    ax = fig.subplots(nrows = 1, ncols = 4)
    sns.boxplot(data = scores_all, x = "methods", y = "NMI (common)", ax = ax[0])
    ax[0].set_ylabel("NMI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "ARI (common)", ax = ax[1])
    ax[1].set_ylabel("ARI")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "GC (common)", ax = ax[2])
    ax[2].set_ylabel("GC")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (common)", ax = ax[3])
    ax[3].set_ylabel("Silhouette batch")
    ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot_common.png", bbox_inches = "tight")

    fig = plt.figure(figsize = (25,5))
    ax = fig.subplots(nrows = 1, ncols = 5)
    sns.boxplot(data = scores_all, x = "methods", y = "NMI (condition)", ax = ax[0])
    ax[0].set_ylabel("NMI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "ARI (condition)", ax = ax[1])
    ax[1].set_ylabel("ARI")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "GC (condition)", ax = ax[2])
    ax[2].set_ylabel("GC")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (condition & celltype)", ax = ax[3])
    ax[3].set_ylabel("ASW-batch")
    ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.boxplot(data = scores_all, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[4])
    ax[4].set_ylabel("ASW-batch")
    ax[4].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    fig.savefig(result_dir + "boxplot_condition.png", bbox_inches = "tight")


# %%




