# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd

sys.path.append("..")
import scDisInFact.model as scdisinfact
import scDisInFact.utils as utils
import scDisInFact.loss_function as loss_func
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

def rank_marker(counts_predict, counts_input_denoised, counts_input, counts_gt):
    """
    Description:
    -------------
        Measuring the perturbed gene detection accuracy using AUPRC. 
        The cells in input count matrices are all matched. 
    Parameters:
    -------------
        counts_predict: 
            The denoised count matrix under the predicted condition
        counts_input_denoised:
            The (scdisinfact) denoised count matrix under the input condition
        counts_input:
            The raw count matrix under the input condition
        counts_gt:
            The raw count matrix under the ground truth condition
    
    Return:
    -------------
        AUPRC: the accuracy of the detected gene expression change 
    """
    # counts_predict and counts_input_denoised are already normalized (decoder output)
    pred_diff = np.mean(counts_predict - counts_input_denoised, axis = 0)
    # gt_diff measures the change of gene expression, 
    # where nonzero means the corresponding gene is perturbed, 
    # and zero means the corresponding gene is not perturbed.
    # NOTE: gt_diff not accurate if normalized, before normalization
    gt_diff = np.mean(counts_gt - counts_input, axis = 0)
    AUPRC = bmk.compute_auprc(pred_diff, gt_diff)
    return AUPRC

def rank_marker_separate(counts_predict, counts_input_denoised, counts_input, counts_gt):
    """
    Description:
    -------------
        Measuring the perturbed gene detection accuracy using AUPRC. 
        The cells in input count matrices are all matched. 
    Parameters:
    -------------
        counts_predict: 
            The denoised count matrix under the predicted condition
        counts_input_denoised:
            The (scdisinfact) denoised count matrix under the input condition
        counts_input:
            The raw count matrix under the input condition
        counts_gt:
            The raw count matrix under the ground truth condition
    
    Return:
    -------------
        AUPRCs: the accuracy of the detected gene expression change 
    """
    ncells = counts_predict.shape[0]
    AUPRCs = []
    for cell_id in range(ncells):
        pred_diff = counts_predict[cell_id,:] - counts_input_denoised[cell_id,:]
        gt_diff = counts_gt[cell_id,:] - counts_input[cell_id,:]
        AUPRC = bmk.compute_auprc(pred_diff, gt_diff)
        AUPRCs.append(AUPRC)
    AUPRCs = np.array(AUPRCs)
    return AUPRCs


n_batches = 2
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

ablation = "kl"
if ablation == "gl":
    regs = [0, 0.01, 0.1, 1, 5]
    result_dir = "./results_simulated/ablation/group_lasso/"
elif ablation == "mmd":
    regs = [0, 1e-4, 1e-2, 1]
    result_dir = "./results_simulated/ablation/MMD/"
elif ablation == "class":
    regs = [0, 0.01, 0.1, 1, 5]
    result_dir = "./results_simulated/ablation/class/"
elif ablation == "latent_dims":
    regs = [[4,2,2], [8,2,2], [16,2,2], [8,4,4], [16,4,4]]
    result_dir = "./results_simulated/ablation/latent_dims/"
elif ablation == "kl":
    regs = [[1e-5, 1e-4], [1e-3, 1e-4], [1e-5, 1e-2], [1e-5, 1e-1], [1e-5, 1], [1e-3, 1e-2], [1e-3, 1e-1], [1e-3, 1]]
    result_dir = "./results_simulated/ablation/KL/"

# In[]
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Out-of-sample test")
print("#")
print("# -------------------------------------------------------------------------------------------")



for dataset_dir in simulated_lists:
    data_dir = f"../data/simulated/unif/{dataset_dir}/"

    if not os.path.exists(result_dir + f"{dataset_dir}_oos/"):
        os.makedirs(result_dir + f"{dataset_dir}_oos/")

    # Load dataset
    counts_ctrl_healthy = []
    counts_ctrl_severe = []
    counts_stim_healthy = []
    counts_stim_severe = []
    label_annos = []
    for batch_id in range(n_batches):
        counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
        counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
        counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
        counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

        anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
        # annotation labels
        label_annos.append(np.array([('cell type '+str(i)) for i in anno]))    

    np.random.seed(0)
    counts_train = []
    meta_train = []
    counts_gt_full = []
    meta_gt_full = []
    for batch_id in range(n_batches):
        ncells = label_annos[batch_id].shape[0]
        # generate permutation
        permute_idx = np.random.permutation(ncells)
        # since there are totally four combinations of conditions, separate the cells into four groups
        chuck_size = int(ncells/4)

        # Training data: remove (ctrl, severe) for all batches
        counts_train.append(np.concatenate([counts_ctrl_healthy[batch_id][permute_idx[:chuck_size],:], 
                                            counts_stim_healthy[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:], 
                                            counts_stim_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
        meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta["batch"] = np.array([batch_id] * chuck_size + [batch_id] * chuck_size + [batch_id] * (ncells - 3*chuck_size))
        meta["condition 1"] = np.array(["ctrl"] * chuck_size + ["stim"] * chuck_size + ["stim"] * (ncells - 3*chuck_size))
        meta["condition 2"] = np.array(["healthy"] * chuck_size + ["healthy"] * chuck_size + ["severe"] * (ncells - 3*chuck_size))
        meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                             label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                             label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
        meta_train.append(meta)

        
        # Ground truth dataset
        counts_gt_full.append(np.concatenate([counts_ctrl_severe[batch_id][permute_idx[:chuck_size],:],
                                         counts_ctrl_severe[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)],:],
                                         counts_ctrl_severe[batch_id][permute_idx[(3*chuck_size):],:]], axis = 0))
        meta = pd.DataFrame(columns = ["batch", "condition 1", "condition 2", "annos"])
        meta["batch"] = np.array([batch_id] * chuck_size + [batch_id] * chuck_size + [batch_id] * (ncells - 3*chuck_size))
        meta["condition 1"] = np.array(["ctrl"] * chuck_size + ["ctrl"] * chuck_size + ["ctrl"] * (ncells - 3*chuck_size))
        meta["condition 2"] = np.array(["severe"] * chuck_size + ["severe"] * chuck_size + ["severe"] * (ncells - 3*chuck_size))
        meta["annos"] = np.concatenate([label_annos[batch_id][permute_idx[:chuck_size]], 
                                        label_annos[batch_id][permute_idx[(2*chuck_size):(3*chuck_size)]], 
                                        label_annos[batch_id][permute_idx[(3*chuck_size):]]], axis = 0)
        meta_gt_full.append(meta)

    # full training dataset, input data are selected from the training dataset
    counts_train = np.concatenate(counts_train, axis = 0)
    meta_train = pd.concat(meta_train, axis = 0)
    # full ground truth dataset, prediction data are compared with the corresponding ground truth data
    counts_gt_full = np.concatenate(counts_gt_full, axis = 0)
    meta_gt_full = pd.concat(meta_gt_full, axis = 0)
    # create training dataset
    data_dict_train = scdisinfact.create_scdisinfact_dataset(counts_train, meta_train, condition_key = ["condition 1", "condition 2"], batch_key = "batch")

    # Train the model
    # default parameters
    reg_mmd_comm = 1e-4
    reg_mmd_diff = 1e-4
    reg_kl_comm = 1e-5
    reg_kl_diff = 1e-2
    reg_class = 1
    reg_gl = 1

    Ks = [8, 2, 2]
    batch_size = 64
    nepochs = 50
    interval = 10
    lr = 5e-4
    
    # training model, comment out if already trained
    for reg in regs:
        if ablation == "gl":
            reg_gl = reg
        elif ablation == "contr":
            reg_contr = reg
        elif ablation == "tc":
            reg_tc = reg
        elif ablation == "mmd":
            reg_mmd_comm = reg
            reg_mmd_diff = reg
        elif ablation == "class":
            reg_class = reg
        elif ablation == "latent_dims":
            Ks = reg
        elif ablation == "kl":
            reg_kl_comm, reg_kl_diff = reg

        # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
        lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
        print(lambs)
        model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                        reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
        # model.train()
        # losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
        # _ = model.eval()
        # torch.save(model.state_dict(), result_dir + f"{dataset_dir}_oos/" + f"scdisinfact_{Ks}_{lambs}.pth")
        model.load_state_dict(torch.load(result_dir + f"{dataset_dir}_oos/" + f"scdisinfact_{Ks}_{lambs}.pth", map_location = device))

    # Prediction: Select input and predict conditions/batches
    configs_input = [{"condition 1": "stim", "condition 2": "severe", "batch": 0},
                    {"condition 1": "ctrl", "condition 2": "healthy", "batch": 0},
                    {"condition 1": "stim", "condition 2": "healthy", "batch": 0},
                    {"condition 1": "stim", "condition 2": "severe", "batch": 1},
                    {"condition 1": "ctrl", "condition 2": "healthy", "batch": 1},
                    {"condition 1": "stim", "condition 2": "healthy", "batch": 1}]


    score_list = []
    score_cluster_list = []
    score_disentangle_list = [] 

    for reg in regs:
        if ablation == "gl":
            reg_gl = reg
        elif ablation == "contr":
            reg_contr = reg
        elif ablation == "tc":
            reg_tc = reg
        elif ablation == "mmd":
            reg_mmd_comm = reg
            reg_mmd_diff = reg
        elif ablation == "class":
            reg_class = reg
        elif ablation == "latent_dims":
            Ks = reg
        elif ablation == "kl":
            reg_kl_comm, reg_kl_diff = reg

        # load model
        lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
        print("Loading model...")
        print(f"Ks: {Ks}")
        print(f"Lambs: {lambs}")
        model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                                        reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
        model.load_state_dict(torch.load(result_dir + f"{dataset_dir}_oos/" + f"scdisinfact_{Ks}_{lambs}.pth", map_location = device))
        
        print("latent space (unshared-bio factors) disentanglement.")
        z_cs = []        
        z_ds = []
        for dataset in data_dict_train["datasets"]:
            with torch.no_grad():
                # pass through the encoders
                dict_inf = model.inference(counts = dataset.counts_norm.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = False)
                z_c = dict_inf["mu_c"]
                z_d = dict_inf["mu_d"]
                z_cs.append(z_c.cpu().detach())
                z_ds.append([x.cpu().detach() for x in z_d])
 
        z_cs = torch.concat(z_cs, dim = 0)
        z_ds_cond1 = torch.concat([z_d[0] for z_d in z_ds], dim = 0)
        z_ds_cond2 = torch.concat([z_d[1] for z_d in z_ds], dim = 0)
        # ctrl/stim
        condition1 = np.concatenate([x["condition 1"].values.squeeze() for x in data_dict_train["meta_cells"]])
        # healthy/severe
        condition2 = np.concatenate([x["condition 2"].values.squeeze() for x in data_dict_train["meta_cells"]])
        batch = np.concatenate([x["batch"].values.squeeze() for x in data_dict_train["meta_cells"]])
        annos = np.concatenate([x["annos"].values.squeeze() for x in data_dict_train["meta_cells"]])

        umap_op = UMAP(min_dist = 0.1, random_state = 0)
        pca_op = PCA(n_components = 2)
        z_cs_umap = umap_op.fit_transform(z_cs.numpy())
        z_ds_umap = []
        z_ds_umap.append(pca_op.fit_transform(z_ds_cond1.numpy()))
        z_ds_umap.append(pca_op.fit_transform(z_ds_cond2.numpy()))

        comment = f"{dataset_dir}_oos/results_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/"
        if not os.path.exists(result_dir + comment):
            os.makedirs(result_dir + comment)

        batch_annos = np.where(batch == 0, "batch 1", "batch 2")
        utils.plot_latent(zs = z_cs_umap, annos = annos, batches = batch_annos, mode = "annos", axis_label = "UMAP", figsize = (10,5), \
                          save = (result_dir + comment+"common_dims_annos.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
        utils.plot_latent(zs = z_cs_umap, annos = annos, batches = batch_annos, mode = "separate", axis_label = "UMAP", figsize = (10,10), \
            save = (result_dir + comment+"common_dims_annos_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")


        utils.plot_latent(zs = z_cs_umap, annos = annos, batches = batch_annos, mode = "batches", axis_label = "UMAP", figsize = (7,5), \
                          save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
        utils.plot_latent(zs = z_cs_umap, annos = annos, batches = batch_annos, mode = "annos", axis_label = "UMAP", figsize = (7,5), \
                          save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

        utils.plot_latent(zs = z_ds_umap[0], annos = annos, batches = batch_annos, mode = "annos", axis_label = "PCA", figsize = (10,5), \
                          save = (result_dir + comment+"diff_dims1_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
        utils.plot_latent(zs = z_ds_umap[0], annos = condition1, batches = batch_annos, mode = "annos", axis_label = "PCA", figsize = (7,5), \
                          save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
        utils.plot_latent(zs = z_ds_umap[0], annos = condition2, batches = batch_annos, mode = "annos", axis_label = "PCA", figsize = (7,5), \
                          save = (result_dir + comment+"diff_dims1_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

        utils.plot_latent(zs = z_ds_umap[1], annos = annos, batches = batch_annos, mode = "annos", axis_label = "PCA", figsize = (10,5), \
                          save = (result_dir + comment+"diff_dims2_annos.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
        utils.plot_latent(zs = z_ds_umap[1], annos = condition1, batches = batch_annos, mode = "annos", axis_label = "PCA", figsize = (7,5), \
                          save = (result_dir + comment+"diff_dims2_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
        utils.plot_latent(zs = z_ds_umap[1], annos = condition2, batches = batch_annos, mode = "annos", axis_label = "PCA", figsize = (7,5), \
                          save = (result_dir + comment+"diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

        # NOTE: measure independence across factors. (No need for TC) Already eliminated by MMD loss

        # independence of unshared factors
        idx_stim_healthy = ((condition1 == "stim") & (condition2 == "healthy"))
        idx_stim_severe = ((condition1 == "stim") & (condition2 == "severe"))
        mmd_conds = loss_func.maximum_mean_discrepancy(xs = torch.cat([z_ds_cond1[idx_stim_healthy,:], z_ds_cond1[idx_stim_severe,:]], dim = 0), 
                                                       batch_ids = torch.Tensor([0] * np.sum(idx_stim_healthy) + [1] * np.sum(idx_stim_severe)), device = torch.device("cpu"))
        asw_conds = bmk.silhouette_batch(X = torch.cat([z_ds_cond1[idx_stim_healthy,:], z_ds_cond1[idx_stim_severe,:]], dim = 0).numpy(), 
                                         batch_gt = np.array([0] * np.sum(idx_stim_healthy) + [1] * np.sum(idx_stim_severe)), 
                                         group_gt = np.array([0] * (np.sum(idx_stim_healthy) + np.sum(idx_stim_severe))), verbose = False)
        idx_stim_healthy = ((condition1 == "stim") & (condition2 == "healthy"))
        idx_ctrl_healthy = ((condition1 == "ctrl") & (condition2 == "healthy"))
        mmd_conds += loss_func.maximum_mean_discrepancy(xs = torch.cat([z_ds_cond2[idx_stim_healthy,:], z_ds_cond2[idx_ctrl_healthy,:]], dim = 0), 
                                                        batch_ids = torch.Tensor([0] * np.sum(idx_stim_healthy) + [1] * np.sum(idx_ctrl_healthy)), device = torch.device("cpu"))
        asw_conds += bmk.silhouette_batch(X = torch.cat([z_ds_cond1[idx_stim_healthy,:], z_ds_cond1[idx_ctrl_healthy,:]], dim = 0).numpy(), 
                                          batch_gt = np.array([0] * np.sum(idx_stim_healthy) + [1] * np.sum(idx_ctrl_healthy)), 
                                          group_gt = np.array([0] * (np.sum(idx_stim_healthy) + np.sum(idx_ctrl_healthy))), verbose = False)
        mmd_conds = mmd_conds * 0.5
        asw_conds = asw_conds * 0.5

        asw_comm_batch = bmk.silhouette_batch(X = z_cs.numpy(), batch_gt = batch, group_gt = annos, verbose = False)
        asw_comm_cond1 = bmk.silhouette_batch(X = z_cs.numpy(), batch_gt = condition1, group_gt = annos, verbose = False)
        asw_comm_cond2 = bmk.silhouette_batch(X = z_cs.numpy(), batch_gt = condition2, group_gt = annos, verbose = False)
        asw_cond1_batch = bmk.silhouette_batch(X = z_ds_cond1.numpy(), batch_gt = batch, group_gt = condition1, verbose = False)
        asw_cond2_batch = bmk.silhouette_batch(X = z_ds_cond2.numpy(), batch_gt = batch, group_gt = condition2, verbose = False)

        score_disentangle = pd.DataFrame(data = [[dataset_dir, mmd_conds.item(), asw_conds, asw_comm_batch, (asw_cond1_batch + asw_cond2_batch)/2, (asw_comm_cond1 + asw_comm_cond2)/2, f"reg: {reg}"]], 
                                         columns = ["dataset", "MMD (indep conds)", "ASW (indep conds)", "ASW (indep batch&comm)", "ASW (indep batch&cond)", "ASW (indep cond&comm)", "method"])
        score_disentangle_list.append(score_disentangle)

        print(f"Perturbation prediction of dataset: {dataset_dir}, predict: ctrl, severe, batch 0")
        for config in configs_input:
            print("input condition: " + str(config))
                        
            # load input and gt count matrices
            idx = ((meta_train["condition 1"] == config["condition 1"]) & (meta_train["condition 2"] == config["condition 2"]) & (meta_train["batch"] == config["batch"])).values
            # input and ground truth, cells are matched
            counts_input = counts_train[idx, :]
            meta_input = meta_train.loc[idx, :]
            counts_gt = counts_gt_full[idx, :]
            meta_gt = meta_gt_full.loc[idx, :]

            # predict count
            counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                            batch_key = "batch", predict_conds = None, predict_batch = None)
            
            counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["condition 1", "condition 2"], 
                                            batch_key = "batch", predict_conds = None, predict_batch = None)

            counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["condition 1", "condition 2"], 
                                                batch_key = "batch", predict_conds = ["ctrl", "severe"], predict_batch = 0)

            # NOTE: perturbed gene prediction accuracy, measured with AUPRC
            # should be calculated before normalization
            # AUPRC = rank_marker(counts_predict = counts_predict, counts_input_denoised = counts_input_denoised, counts_input = counts_input, counts_gt = counts_gt)
            # AUPRC_sep = rank_marker_separate(counts_predict = counts_predict, counts_input_denoised = counts_input_denoised, counts_input = counts_input, counts_gt = counts_gt)

            # normalize the count
            # Is normalization better? Make sure the libsize of predict and gt are the same for each cell (or they will not be on the same scale)
            # In addition, the prediction output is decoder mu, when calculate NB loss, decoder mu is multiplied with libsize and then compare with true count
            # which means that the predict count ignore the libsize effect.
            # Amount the matrices below, counts_gt and counts_input has libsize effect (they have to be normalized), 
            # counts_gt_denoise and counts_predict are all decoder output with no libsize effect (doesn't matter normalize or not).
            # Normalization does not introduce error.
            counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
            counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
            counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
            counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

            # 1. no 1-1 match, check cell-type level scores, but low resolution
            unique_celltypes = np.unique(meta_gt["annos"].values)
            mean_inputs = []
            mean_predicts = []
            mean_gts = []
            mean_gts_denoised = []
            for celltype in unique_celltypes:
                mean_input = np.mean(counts_input[meta_input["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_predict = np.mean(counts_predict[meta_input["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_gt = np.mean(counts_gt[meta_gt["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_gt_denoised = np.mean(counts_gt_denoised[meta_gt["annos"].values.squeeze() == celltype,:], axis = 0)
                mean_inputs.append(mean_input)
                mean_predicts.append(mean_predict)
                mean_gts.append(mean_gt)
                mean_gts_denoised.append(mean_gt_denoised)

            mean_inputs = np.array(mean_inputs)
            mean_predicts = np.array(mean_predicts)
            mean_gts = np.array(mean_gts)
            mean_gts_denoised = np.array(mean_gts_denoised)

            # vector storing cell-type-specific normalized MSE for all clusters
            mses_input_cluster = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
            mses_scdisinfact_cluster = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
            # vector storing cell-type-specific pearson correlation
            pearsons_input_cluster = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
            pearsons_scdisinfact_cluster = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
            # vector storing cell-type-specific R2 score
            r2_input_cluster = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
            r2_scdisinfact_cluster = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

            score_cluster = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
            score_cluster["MSE"] = mses_scdisinfact_cluster
            score_cluster["MSE input"] = mses_input_cluster
            score_cluster["Pearson"] = pearsons_scdisinfact_cluster
            score_cluster["Pearson input"] = pearsons_input_cluster
            score_cluster["R2"] = r2_scdisinfact_cluster
            score_cluster["R2 input"] = r2_input_cluster
            score_cluster["Method"] = f"reg: {reg}"
            score_cluster["Prediction"] = config["condition 1"] + "_" + config["condition 2"] + "_" + str(config["batch"])
            score_cluster_list.append(score_cluster)

            # 2. 1-1 match, calculate cell-level score. Higher resolution when match exists
            # vector storing the normalized MSE for all cells
            mses_input = np.sum((counts_input - counts_gt_denoised) ** 2, axis = 1)
            mses_scdisinfact = np.sum((counts_predict - counts_gt_denoised) ** 2, axis = 1)
            # vector storing the pearson correlation for all cells
            pearson_input = np.array([stats.pearsonr(counts_input[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
            pearsons_scdisinfact = np.array([stats.pearsonr(counts_predict[i,:], counts_gt_denoised[i,:])[0] for i in range(counts_gt_denoised.shape[0])])
            # vector storing the R2 scores for all cells
            r2_input = np.array([r2_score(y_pred = counts_input[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])
            r2_scdisinfact = np.array([r2_score(y_pred = counts_predict[i,:], y_true = counts_gt_denoised[i,:]) for i in range(counts_gt_denoised.shape[0])])

            score = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "AUPRC", "AUPRC (cell-level)", "Method", "Prediction"])
            score["MSE"] = mses_scdisinfact
            score["MSE input"] = mses_input
            score["Pearson"] = pearsons_scdisinfact
            score["Pearson input"] = pearson_input
            score["R2"] = r2_scdisinfact
            score["R2 input"] = r2_input
            # score["AUPRC"] = AUPRC
            # score["AUPRC (cell-level)"] = AUPRC_sep
            score["Method"] = f"reg: {reg}"
            score["Prediction"] = config["condition 1"] + "_" + config["condition 2"] + "_" + str(config["batch"])
            score_list.append(score)

    scores_cluster = pd.concat(score_cluster_list, axis = 0)
    scores_cluster.to_csv(result_dir + f"{dataset_dir}_oos/" + "prediction_scores_cluster.csv")

    scores = pd.concat(score_list, axis = 0)
    scores.to_csv(result_dir + f"{dataset_dir}_oos/" + "prediction_scores.csv")

    scores_disentangle = pd.concat(score_disentangle_list, axis = 0)
    scores_disentangle.to_csv(result_dir + f"{dataset_dir}_oos/" + "disentangle_scores.csv")


# In[]
auprc_dict = pd.DataFrame(columns = ["dataset", "condition", "ndiff_genes", "AUPRC", "AUPRC ratio", "method", "ndiff"])

for dataset_dir in simulated_lists:
    ngenes = eval(dataset_dir.split("_")[3])
    ndiff_genes = eval(dataset_dir.split("_")[5])
    ndiff = eval(dataset_dir.split("_")[6])
    print(f"CKG detection: {dataset_dir}")

    # no group lasso loss
    for reg in regs:
        if ablation == "gl":
            reg_gl = reg
        elif ablation == "contr":
            reg_contr = reg
        elif ablation == "tc":
            reg_tc = reg
        elif ablation == "mmd":
            reg_mmd_comm = reg
            reg_mmd_diff = reg
        elif ablation == "class":
            reg_class = reg
        elif ablation == "latent_dims":
            Ks = reg
        elif ablation == "kl":
            reg_kl_comm, reg_kl_diff = reg

        # load model
        lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]

        params = torch.load(result_dir + f"{dataset_dir}_oos/scdisinfact_{Ks}_{lambs}.pth", map_location = device)

        gt = np.zeros((1, ngenes))
        gt[:,ndiff_genes:(2*ndiff_genes)] = 1
        gt = gt.squeeze()
        inf = np.array(params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        auprc_dict = pd.concat([auprc_dict, pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                        "ndiff_genes": [ndiff_genes], 
                                        "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "Method": [f"reg: {reg}"],
                                        "ndiff": [ndiff],
                                        "condition": ["ctrl & stim"]
                                        })], axis = 0, ignore_index = True)

        gt = np.zeros((1, ngenes))
        gt[:,:ndiff_genes] = 1
        gt = gt.squeeze()
        inf = np.array(params["Enc_ds.1.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
        auprc_dict = pd.concat([auprc_dict, pd.DataFrame.from_dict({"dataset": [dataset_dir], 
                                        "ndiff_genes": [ndiff_genes], 
                                        "AUPRC": [bmk.compute_auprc(inf, gt)], 
                                        "AUPRC ratio": [bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)],
                                        "Method": [f"reg: {reg}"],
                                        "ndiff": [ndiff],
                                        "condition": ["severe & healthy"]
                                        })], axis = 0, ignore_index = True)

auprc_dict.to_csv(result_dir + "/CKG_scores.txt", sep = "\t")


# In[]
print("# ----------------------------------------------------------------------------------")
print("#")
print("# Evaluation")
print("#")
print("# ----------------------------------------------------------------------------------")

plt.rcParams["font.size"] = 15
scores_prediction = []
for dataset_dir in simulated_lists:
    scores_prediction.append(pd.read_csv(result_dir + f"{dataset_dir}_oos/prediction_scores.csv", index_col = 0))
scores_prediction = pd.concat(scores_prediction, axis = 0)
scores_prediction["MSE ratio"] = scores_prediction["MSE"].values/scores_prediction["MSE input"].values
scores_prediction["R2 ratio"] = scores_prediction["R2"].values/scores_prediction["R2 input"].values

auprc_dict = pd.read_csv(result_dir + f"CKG_scores.txt", index_col = 0, sep = "\t")
scores_disentangle = []
for dataset_dir in simulated_lists:
    scores_disentangle.append(pd.read_csv(result_dir + f"{dataset_dir}_oos/disentangle_scores.csv", index_col = 0))
scores_disentangle = pd.concat(scores_disentangle, axis = 0)
scores_disentangle.index = scores_disentangle["dataset"].values

scores_prediction = scores_prediction[scores_prediction["Prediction"].isin(["ctrl_healthy_0", "stim_healthy_0", "stim_severe_0"])]

if ablation == "kl":
    scores_prediction.loc[scores_prediction["Method"] == "reg: [1e-05, 0.0001]","Method"] = "reg: 0.0001"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [1e-05, 0.01]","Method"] = "reg: 0.01"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [1e-05, 0.1]","Method"] = "reg: 0.1"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [1e-05, 1]","Method"] = "reg: 1"
    scores_prediction = scores_prediction.loc[scores_prediction["Method"].isin(["reg: 1", "reg: 0.1", "reg: 0.01", "reg: 0.0001"])]

    auprc_dict.loc[auprc_dict["Method"] == "reg: [1e-05, 0.0001]","Method"] = "reg: 0.0001"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [1e-05, 0.01]","Method"] = "reg: 0.01"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [1e-05, 0.1]","Method"] = "reg: 0.1"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [1e-05, 1]","Method"] = "reg: 1"
    auprc_dict = auprc_dict.loc[auprc_dict["Method"].isin(["reg: 1", "reg: 0.1", "reg: 0.01", "reg: 0.0001"])]

elif ablation == "latent_dims":
    scores_prediction.loc[scores_prediction["Method"] == "reg: [4, 2, 2]","Method"] = "shared: 4; unshared: 2"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [8, 2, 2]","Method"] = "shared: 8; unshared: 2"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [16, 2, 2]","Method"] = "shared: 16; unshared: 2"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [8, 4, 4]","Method"] = "shared: 8; unshared: 4"
    scores_prediction.loc[scores_prediction["Method"] == "reg: [16, 4, 4]","Method"] = "shared: 16; unshared: 4"

    auprc_dict.loc[auprc_dict["Method"] == "reg: [4, 2, 2]","Method"] = "shared: 4; unshared: 2"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [8, 2, 2]","Method"] = "shared: 8; unshared: 2"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [16, 2, 2]","Method"] = "shared: 16; unshared: 2"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [8, 4, 4]","Method"] = "shared: 8; unshared: 4"
    auprc_dict.loc[auprc_dict["Method"] == "reg: [16, 4, 4]","Method"] = "shared: 16; unshared: 4"

fig = plt.figure(figsize = (45,5))
axs = fig.subplots(nrows = 1, ncols = 4)
ax = sns.barplot(x='method', y="ASW (indep conds)", data=scores_disentangle, ax = axs[0], capsize = 0.1)
sns.stripplot(data = scores_disentangle, x = "method", y = "ASW (indep conds)", ax = axs[0], color = "black", dodge = True) 
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)  
# ax.set_ylim(0.3, 0.6) 
ax.set_ylabel("ASW")
ax.set_xlabel(None)
ax.set_title("Indep (unshared factors)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[2:], labels[2:], bbox_to_anchor=(1, 1.02), loc='upper left')

ax = sns.barplot(x='method', y="ASW (indep batch&comm)", data=scores_disentangle, ax = axs[1], capsize = 0.1) 
sns.stripplot(data = scores_disentangle, x = "method", y = "ASW (indep conds)", ax = axs[1], color = "black", dodge = True) 
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)  
ax.set_ylim(0.6, 1.0) 
ax.set_ylabel("ASW")
ax.set_xlabel(None)
ax.set_title("Indep (batch & shared factor)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[2:], labels[2:], bbox_to_anchor=(1, 1.02), loc='upper left')

ax = sns.barplot(x='method', y="ASW (indep batch&cond)", data=scores_disentangle, ax = axs[2], capsize = 0.1) 
sns.stripplot(data = scores_disentangle, x = "method", y = "ASW (indep batch&cond)", ax = axs[2], color = "black", dodge = True) 
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)  
ax.set_ylim(0.6, 1.0) 
ax.set_ylabel("ASW")
ax.set_xlabel(None)
ax.set_title("Indep (batch & unshared factor)")
ax = sns.barplot(x='method', y="ASW (indep cond&comm)", data=scores_disentangle, ax = axs[3], capsize = 0.1)
sns.stripplot(data = scores_disentangle, x = "method", y = "ASW (indep cond&comm)", ax = axs[3], color = "black", dodge = True) 
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)   
ax.set_ylim(0.6, 1.0) 
ax.set_ylabel("ASW")
ax.set_xlabel(None)
ax.set_title("Indep (unshared & shared factor)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[2:], labels[2:], bbox_to_anchor=(1, 1.02), loc='upper left')

plt.tight_layout()
fig.savefig(result_dir + "scores_disentangle_hyperparams.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20,5))
# ax = fig.subplots(nrows = 1, ncols = 3)
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='MSE', hue='Method', data=scores_prediction, ax = ax, capsize = 0.1) 
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
for i in ax.containers:
    ax.bar_label(i, fmt='%.1e', padding = -30)    
ax.set_ylim(0, 0.01)
# ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity"])
fig.savefig(result_dir + "prediction_MSE_hyperparams.png", bbox_inches = "tight")

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot()
ax = sns.barplot(x='Prediction', y='R2', hue='Method', data=scores_prediction, ax = ax, capsize = 0.1)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)    
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
# ax.set_ylim(0.60, 1)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)    
# ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity", "Treatment\n& Batch", "Severity\n& Batch", "Treatment\n& Severity\n& Batch"])
ax.set_xticklabels(["Treatment", "Severity", "Treatment\n& Severity"])
fig.tight_layout()
fig.savefig(result_dir + "prediction_R2_hyperparams.png", bbox_inches = "tight")

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot()
ax = sns.barplot(x = "ndiff", hue = 'Method', y ='AUPRC', data=auprc_dict, ax = ax, capsize = 0.1)
sns.stripplot(data = auprc_dict, x = "ndiff", hue = "Method", y = "AUPRC", ax = ax, color = "black", dodge = True) 

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[int(len(handles)/2):], labels[int(len(handles)/2):], loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
# ax.set_ylim(0.50, 1.02)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f', padding = -100)    
fig.tight_layout()
fig.savefig(result_dir + "CKGs_AUPRC_hyperparams.png", bbox_inches = "tight")


# %%
