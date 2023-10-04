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

simulated_lists = [
  "2conds_base_10000_500_0.4_20_2",
  "2conds_base_10000_500_0.4_20_4",
  "2conds_base_10000_500_0.4_20_8",
  "2conds_base_10000_500_0.4_50_2",
  "2conds_base_10000_500_0.4_50_4",
  "2conds_base_10000_500_0.4_50_8",
  "2conds_base_10000_500_0.4_100_2",
  "2conds_base_10000_500_0.4_100_4",
  "2conds_base_10000_500_0.4_100_8",
 ]

n_batches = 2

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [8, 2, 2]

lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

batch_size = 64
nepochs = 100
interval = 10
lr = 5e-4


for dataset_dir in simulated_lists:

    print("# -------------------------------------------------------------------------------------------")
    print("#")
    print('# Dataset' + dataset_dir)
    print("#")
    print("# -------------------------------------------------------------------------------------------")

    data_dir = "../data/simulated/unif/" + dataset_dir + "/"
    result_dir = "./results_simulated/partial_latent_cond/" + dataset_dir + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

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

    np.random.seed(0)
    counts = []
    meta_cells = []
    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally four combinations of conditions, separate the cells into four groups
        chunk_size = int(counts_gt[batch_id].shape[0]/4)
        counts.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                           counts_ctrl_severe[batch_id][permute_idx[chunk_size:(2*chunk_size)],:],
                                           counts_stim_healthy[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:],
                                           counts_stim_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0))
        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"], dtype=object)
        meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["stim"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chunk_size + ["severe"] * chunk_size + ["healthy"] * chunk_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["annos"] = label_annos[batch_id][permute_idx]
        meta_cells.append(meta_cell)

    data_dict_full = scdisinfact.create_scdisinfact_dataset(counts, meta_cells,
                                                       condition_key  = ["condition 1", "condition 2"],
                                                       batch_key = "batch")
    
    model = scdisinfact.scdisinfact(data_dict = data_dict_full, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                    reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

    model.train()
    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
    _ = model.eval()
    torch.save(model.state_dict(), result_dir + f"scdisinfact_full_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
    model.load_state_dict(torch.load(result_dir + f"scdisinfact_full_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

    z_cs = []
    z_ds = []

    for dataset in data_dict_full["datasets"]:
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


    silhouette_batch_scdisinfact = bmk.silhouette_batch(X = np.concatenate(z_cs, axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), \
                                                        group_gt = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), verbose = False)
    print('Silhouette batch (scDisInFact): {:.3f}'.format(silhouette_batch_scdisinfact))

    # NMI and ARI measure the separation of cell types
    nmi_cluster_scdisinfact = []
    ari_cluster_scdisinfact = []
    for resolution in np.arange(0.1, 10, 0.5):
        leiden_labels_clusters = bmk.leiden_cluster(X = np.concatenate(z_cs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
        nmi_cluster_scdisinfact.append(bmk.nmi(group1 = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), group2 = leiden_labels_clusters))
        ari_cluster_scdisinfact.append(bmk.ari(group1 = np.concatenate([x["annos"].values.squeeze() for x in data_dict_full["meta_cells"]]), group2 = leiden_labels_clusters))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_cluster_scdisinfact)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_cluster_scdisinfact)))

    silhouette_condition_scdisinfact1 = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), group_gt = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_full["meta_cells"]]), verbose = False)
    print('Silhouette condition 1, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact1))
    silhouette_condition_scdisinfact2 = bmk.silhouette_batch(X = np.concatenate([x[1] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_full["meta_cells"]]), group_gt = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_full["meta_cells"]]), verbose = False)
    print('Silhouette condition 2, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact2))


    # keep condition information
    nmi_condition_scdisinfact1 = []
    ari_condition_scdisinfact1 = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        print(np.unique(leiden_labels_conditions).shape[0])
        nmi_condition_scdisinfact1.append(bmk.nmi(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_full["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact1.append(bmk.ari(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_full["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact1)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact1)))

    nmi_condition_scdisinfact2 = []
    ari_condition_scdisinfact2 = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[1] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        nmi_condition_scdisinfact2.append(bmk.nmi(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_full["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact2.append(bmk.ari(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_full["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact2)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact2)))


    scores_scdisinfact = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "Silhouette batch (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (condition & batches)"])
    scores_scdisinfact["methods"] = np.array(["All CT"])
    scores_scdisinfact["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact)])
    scores_scdisinfact["ARI (common)"] = np.array([max(ari_cluster_scdisinfact)])
    scores_scdisinfact["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact])
    scores_scdisinfact["NMI (condition)"] = np.max(np.array([max(nmi_condition_scdisinfact1), max(nmi_condition_scdisinfact2)]))
    scores_scdisinfact["ARI (condition)"] = np.max(np.array([max(ari_condition_scdisinfact1), max(ari_condition_scdisinfact2)]))
    scores_scdisinfact["Silhouette batch (condition & batches)"] = np.max(np.array([silhouette_condition_scdisinfact1, silhouette_condition_scdisinfact2]))
        
    np.random.seed(0)
    counts = []
    meta_cells = []
    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally four combinations of conditions, separate the cells into four groups
        chunk_size = int(counts_gt[batch_id].shape[0]/4)
        count = np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                        counts_ctrl_severe[batch_id][permute_idx[chunk_size:(2*chunk_size)],:],
                                        counts_stim_healthy[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:],
                                        counts_stim_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0)


        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["stim"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chunk_size + ["severe"] * chunk_size + ["healthy"] * chunk_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["annos"] = label_annos[batch_id][permute_idx]
        
        remove_idx = meta_cell.loc[(meta_cell["annos"] == "cell type 16") & (meta_cell["condition 2"] == "healthy")].index
        meta_cell.drop(remove_idx, inplace = True)
        meta_cell.reset_index(drop=True)
        new_count = np.delete(count, remove_idx, axis = 0)
            
        meta_cells.append(meta_cell)
        counts.append(new_count)

    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

    model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                    reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

    model.train()
    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
    _ = model.eval()
    torch.save(model.state_dict(), result_dir + f"scdisinfact_1CT_rem_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
    model.load_state_dict(torch.load(result_dir + f"scdisinfact_1CT_rem_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

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
    z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))

    batch_annos = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]])
    batch_annos = np.where(batch_annos == 0, "batch 1", "batch 2")
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "UMAP", figsize = (10,5), save = (result_dir + "1_cell_type_removed_" + "common_dims_annos.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "separate", axis_label = "UMAP", figsize = (10,10), save = (result_dir + "1_cell_type_removed_" + "common_dims_annos_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "batches", axis_label = "UMAP", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "UMAP", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "UMAP", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims1_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims1_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "batches", axis_label = "PCA", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims2_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims2_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "batches", axis_label = "PCA", figsize = (7,5), save = (result_dir + "1_cell_type_removed_" + "diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
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

    silhouette_condition_scdisinfact1 = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]]), group_gt = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), verbose = False)
    print('Silhouette condition 1, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact1))
    silhouette_condition_scdisinfact2 = bmk.silhouette_batch(X = np.concatenate([x[1] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]]), group_gt = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), verbose = False)
    print('Silhouette condition 2, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact2))


    # keep condition information
    nmi_condition_scdisinfact1 = []
    ari_condition_scdisinfact1 = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        nmi_condition_scdisinfact1.append(bmk.nmi(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact1.append(bmk.ari(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact1)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact1)))

    nmi_condition_scdisinfact2 = []
    ari_condition_scdisinfact2 = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[1] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        nmi_condition_scdisinfact2.append(bmk.nmi(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact2.append(bmk.ari(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact2)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact2)))


    scores_scdisinfact1 = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "Silhouette batch (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (condition & batches)"])
    scores_scdisinfact1["methods"] = np.array(["1 CT rem"])
    scores_scdisinfact1["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact)])
    scores_scdisinfact1["ARI (common)"] = np.array([max(ari_cluster_scdisinfact)])
    scores_scdisinfact1["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact])
    scores_scdisinfact1["NMI (condition)"] = np.max(np.array([max(nmi_condition_scdisinfact1), max(nmi_condition_scdisinfact2)]))
    scores_scdisinfact1["ARI (condition)"] = np.max(np.array([max(ari_condition_scdisinfact1), max(ari_condition_scdisinfact2)]))
    scores_scdisinfact1["Silhouette batch (condition & batches)"] = np.max(np.array([silhouette_condition_scdisinfact1, silhouette_condition_scdisinfact2]))
    
    np.random.seed(0)
    counts = []
    meta_cells = []
    for batch_id in range(n_batches):
        # generate permutation
        permute_idx = np.random.permutation(counts_gt[batch_id].shape[0])
        # since there are totally four combinations of conditions, separate the cells into four groups
        chunk_size = int(counts_gt[batch_id].shape[0]/4)
        count = np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chunk_size],:],
                                        counts_ctrl_severe[batch_id][permute_idx[chunk_size:(2*chunk_size)],:],
                                        counts_stim_healthy[batch_id][permute_idx[(2*chunk_size):(3*chunk_size)],:],
                                        counts_stim_severe[batch_id][permute_idx[(3*chunk_size):],:]], axis = 0)


        meta_cell = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta_cell["batch"] = np.array([batch_id] * counts_gt[batch_id].shape[0])
        meta_cell["condition 1"] = np.array(["ctrl"] * chunk_size + ["ctrl"] * chunk_size + ["stim"] * chunk_size + ["stim"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["condition 2"] = np.array(["healthy"] * chunk_size + ["severe"] * chunk_size + ["healthy"] * chunk_size + ["severe"] * (counts_gt[batch_id].shape[0] - 3*chunk_size))
        meta_cell["annos"] = label_annos[batch_id][permute_idx]

        remove_idx = meta_cell.loc[((meta_cell["annos"] == "cell type 16") | (meta_cell["annos"] == "cell type 10")) \
                                       & (meta_cell["condition 2"] == "healthy")].index
        meta_cell.drop(remove_idx, inplace = True)
        meta_cell.reset_index(drop=True)
        new_count = np.delete(count, remove_idx, axis = 0)
        
        meta_cells.append(meta_cell)
        counts.append(new_count)

    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts, meta_cells, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

    model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr,
                                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl,
                                    reg_class = reg_class, reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

    model.train()
    losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
    _ = model.eval()
    torch.save(model.state_dict(), result_dir + f"scdisinfact_2CT_rem_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
    model.load_state_dict(torch.load(result_dir + f"scdisinfact_2CT_rem_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))

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
    z_ds_umap.append(pca_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))


    batch_annos = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]])
    batch_annos = np.where(batch_annos == 0, "batch 1", "batch 2")
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "UMAP", figsize = (10,5), save = (result_dir + "2_cell_types_removed_" + "common_dims_annos.png") if result_dir else None, markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "separate", axis_label = "UMAP", figsize = (10,10), save = (result_dir + "2_cell_types_removed_" + "common_dims_annos_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha     = 0.5, label_inplace = False, text_size = "small")
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
    mode = "batches", axis_label = "UMAP", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "UMAP", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "UMAP", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims1_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims1_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "batches", axis_label = "PCA", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (10,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims2_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims2_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "annos", axis_label = "PCA", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
    utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = batch_annos, \
        mode = "batches", axis_label = "PCA", figsize = (7,5), save = (result_dir + "2_cell_types_removed_" + "diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

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

    silhouette_condition_scdisinfact1 = bmk.silhouette_batch(X = np.concatenate([x[0] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]]), group_gt = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), verbose = False)
    print('Silhouette condition 1, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact1))
    silhouette_condition_scdisinfact2 = bmk.silhouette_batch(X = np.concatenate([x[1] for x in z_ds], axis = 0), batch_gt = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]]), group_gt = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), verbose = False)
    print('Silhouette condition 2, removal of batch effect (scDisInFact): {:.3f}'.format(silhouette_condition_scdisinfact2))


    # keep condition information
    nmi_condition_scdisinfact1 = []
    ari_condition_scdisinfact1 = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[0] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        nmi_condition_scdisinfact1.append(bmk.nmi(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact1.append(bmk.ari(group1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact1)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact1)))

    nmi_condition_scdisinfact2 = []
    ari_condition_scdisinfact2 = []
    for resolution in range(-3, 1, 1):
        leiden_labels_conditions = bmk.leiden_cluster(X = np.concatenate([x[1] for x in z_ds], axis = 0), knn_indices = None, knn_dists = None, resolution = 10 ** resolution)
        nmi_condition_scdisinfact2.append(bmk.nmi(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
        ari_condition_scdisinfact2.append(bmk.ari(group1 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]]), group2 = leiden_labels_conditions))
    print('NMI (scDisInFact): {:.3f}'.format(max(nmi_condition_scdisinfact2)))
    print('ARI (scDisInFact): {:.3f}'.format(max(ari_condition_scdisinfact2)))


    scores_scdisinfact2 = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "Silhouette batch (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (condition & batches)"])
    scores_scdisinfact2["methods"] = np.array(["2 CT rem"])
    scores_scdisinfact2["NMI (common)"] = np.array([max(nmi_cluster_scdisinfact)])
    scores_scdisinfact2["ARI (common)"] = np.array([max(ari_cluster_scdisinfact)])
    scores_scdisinfact2["Silhouette batch (common)"] = np.array([silhouette_batch_scdisinfact])
    scores_scdisinfact2["NMI (condition)"] = np.max(np.array([max(nmi_condition_scdisinfact1), max(nmi_condition_scdisinfact2)]))
    scores_scdisinfact2["ARI (condition)"] = np.max(np.array([max(ari_condition_scdisinfact1), max(ari_condition_scdisinfact2)]))
    scores_scdisinfact2["Silhouette batch (condition & batches)"] = np.max(np.array([silhouette_condition_scdisinfact1, silhouette_condition_scdisinfact2]))
    
    scores_all = pd.concat([scores_scdisinfact, scores_scdisinfact1, scores_scdisinfact2], axis = 0, ignore_index = True)
    scores_all.to_csv(result_dir + "score_all.csv")


if True:
    scores_all_datasets = pd.DataFrame(columns = ["methods", "NMI (common)", "ARI (common)", "NMI (condition)", "ARI (condition)", "Silhouette batch (common)", "Silhouette batch (condition & batches)"])

    for dataset_dir in simulated_lists:
        result_dir = "./results_simulated/partial_latent_cond/"
        scores_all = pd.read_csv(result_dir + dataset_dir + "/score_all.csv", index_col = 0)
        scores_all_datasets = pd.concat([scores_all_datasets, scores_all], axis = 0, ignore_index = True)


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

    fig = plt.figure(figsize = (15,5))
    ax = fig.subplots(nrows = 1, ncols = 3)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "NMI (common)", ax = ax[0])
    ax[0].set_ylabel("NMI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("NMI (shared)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "ARI (common)", ax = ax[1])
    ax[1].set_ylabel("ARI")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("ARI (shared)")
    ax[1].set_xlabel(None)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "Silhouette batch (common)", ax = ax[2])
    ax[2].set_ylabel("Silhouette batch")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[2].set_title("Silhouette batch\n(shared)")
    ax[2].set_ylim(0.8, 1)
    ax[2].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot_common.png", bbox_inches = "tight")

    fig = plt.figure(figsize = (15,5))
    ax = fig.subplots(nrows = 1, ncols = 3)
    sns.boxplot(data = scores_all_datasets, x = "methods", y = "NMI (condition)", ax = ax[0])
    ax[0].set_ylabel("NMI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("NMI (condition)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "ARI (condition)", ax = ax[1])
    ax[1].set_ylabel("ARI")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("ARI (condition)")
    ax[1].set_xlabel(None)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "Silhouette batch (condition & batches)", ax = ax[2])
    ax[2].set_ylabel("ASW-batch")
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[2].set_title("Silhouette batch\n(condition & batches)")
    ax[2].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot_condition.png", bbox_inches = "tight")

if True:
    fig = plt.figure(figsize = (15,5))
    ax = fig.subplots(nrows = 1, ncols = 2)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "ARI (common)", ax = ax[0])
    sns.stripplot(data = scores_all_datasets, x = "methods", y = "ARI (common)", ax = ax[0], color = "black", dodge = True) 
    ax[0].set_ylabel("ARI")
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title("ARI\n(shared)")
    ax[0].set_xlabel(None)

    sns.boxplot(data = scores_all_datasets, x = "methods", y = "Silhouette batch (common)", ax = ax[1])
    sns.stripplot(data = scores_all_datasets, x = "methods", y = "Silhouette batch (common)", ax = ax[1], color = "black", dodge = True) 
    ax[1].set_ylabel("Silhouette batch")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("Silhouette batch\n(shared)")
    ax[1].set_ylim(0.8, 1)
    ax[1].set_xlabel(None)

    plt.tight_layout()
    fig.savefig(result_dir + "boxplot.png", bbox_inches = "tight")


