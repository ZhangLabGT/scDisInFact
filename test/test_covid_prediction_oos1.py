# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")

import scdisinfact
import utils
import bmk

import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score 
import scipy.stats as stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


#-------------------------------------------------------------------------------------------
#
# statsmodel
#
#-------------------------------------------------------------------------------------------
def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    '''
    pvalue correction for false discovery rate.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like, 1d
        Set of p-values of the individual tests.
    alpha : float, optional
        Family-wise error rate. Defaults to ``0.05``.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests). ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).
        Defaults to ``'indep'``.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to ``alpha * m/m_0`` where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Both methods exposed via this function (Benjamini/Hochberg, Benjamini/Yekutieli)
    are also available in the function ``multipletests``, as ``method="fdr_bh"`` and
    ``method="fdr_by"``, respectively.

    See also
    --------
    multipletests

    '''
    pvals = np.asarray(pvals)
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

def wilcoxon_rank_sum(counts_x, counts_y, fdr = False):
    # wilcoxon rank sum test
    ngenes = counts_x.shape[1]
    assert ngenes == counts_y.shape[1]
    pvals = []
    for gene_i in range(ngenes):
        _, pval = stats.ranksums(x = counts_x[:, gene_i].squeeze(), y = counts_y[:, gene_i].squeeze())
        pvals.append(pval)
    pvals = np.array(pvals)

    # fdr correction
    if fdr:
        _, pvals = fdrcorrection(pvals)
    return pvals

# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Load the dataset, treat each dataset as a batch, as the authors of each data claim that there were minor batch effect
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_dir = "../data/covid_integrated/"
result_dir = "results_covid/dropout/prediction_oos1/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

GxC1 = sparse.load_npz(data_dir + "GxC1.npz")
GxC2 = sparse.load_npz(data_dir + "GxC2.npz")
GxC3 = sparse.load_npz(data_dir + "GxC3.npz")
# be careful with the ordering
meta_c1 = pd.read_csv(data_dir + "meta_arunachalam_2020.txt", sep = "\t", index_col = 0)
meta_c2 = pd.read_csv(data_dir + "meta_lee_2020.txt", sep = "\t", index_col = 0)
meta_c3 = pd.read_csv(data_dir + "meta_wilk_2020.txt", sep = "\t", index_col = 0)

meta = pd.concat([meta_c1, meta_c2, meta_c3], axis = 0)
genes = pd.read_csv(data_dir + "genes_shared.txt", index_col = 0, sep = "\t").values.squeeze()
# process age
age = meta.age.values.squeeze().astype(object)
age[meta["age"] < 40] = "40-"
age[(meta["age"] >= 40)&(meta["age"] < 65)] = "40-65"
age[meta["age"] >= 65] = "65+"
meta["age"] = age

counts_array = [GxC1.T, GxC2.T, GxC3.T]
meta_cells_array = [meta[meta["dataset"] == "arunachalam_2020"], meta[meta["dataset"] == "lee_2020"], meta[meta["dataset"] == "wilk_2020"]]

# no mmd batches
data_dict = scdisinfact.create_scdisinfact_dataset(counts_array, meta_cells_array, 
                                                   condition_key = ["disease_severity", "age"], 
                                                   batch_key = "dataset")
                    
# construct training and testing data
np.random.seed(0)
datasets_array_train = []
datasets_array_test = []
meta_cells_array_train = []
meta_cells_array_test = []
for dataset, meta_cells in zip(data_dict["datasets"], data_dict["meta_cells"]):
    # NOTE: X52, moderate, 40-65, lee_2020
    test_idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] == "lee_2020")).values
    train_idx = ~test_idx

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

data_dict_train = {"datasets": datasets_array_train, "meta_cells": meta_cells_array_train, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}
data_dict_test = {"datasets": datasets_array_test, "meta_cells": meta_cells_array_test, "matching_dict": data_dict["matching_dict"], "scaler": data_dict["scaler"]}


# In[]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Train scDisInFact
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import importlib 
importlib.reload(scdisinfact)
#----------------------------------------------------------------------------
# # reference setting
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
# loss_kl explode, 1e-5 is too large
reg_kl = 1e-5
reg_contr = 0.01
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
Ks = [8, 4, 4]

batch_size = 64
# kl term explode when nepochs = 70
nepochs = 50
interval = 10
lr = 5e-4

#----------------------------------------------------------------------------

# # argument
# reg_mmd_comm = eval(sys.argv[1])
# reg_mmd_diff = eval(sys.argv[2])
# reg_gl = eval(sys.argv[3])
# reg_tc = eval(sys.argv[4])
# reg_class = eval(sys.argv[5])
# reg_kl = eval(sys.argv[6])
# reg_contr = eval(sys.argv[7])
# # mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
# lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl, reg_contr]
# nepochs = eval(sys.argv[8])
# lr = eval(sys.argv[9])
# batch_size = eval(sys.argv[10])
# interval = 10
# Ks = [8, 4, 4]

#----------------------------------------------------------------------------

print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(data_dict = data_dict_train, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = reg_contr)
torch.save(model.state_dict(), result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
model.load_state_dict(torch.load(result_dir + f"scdisinfact_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location = device))
_ = model.eval()
# In[] Plot results
z_cs = []
z_ds = []
zs = []

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
        z_ds.append([x.cpu().detach().numpy() for x in z_d])
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))


comment = f'figures_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}/'
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)


utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l1.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l2"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "joint", axis_label = "UMAP", figsize = (17,7), save = (result_dir + comment+"common_dims_celltypes_l2.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l3"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "joint", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_celltypes_l3.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "separate", axis_label = "UMAP", figsize = (10,21), save = (result_dir + comment+"common_dims_celltypes_l1_separate.png") if result_dir else None , markerscale = 9, s = 1, alpha = 0.5, label_inplace = False, text_size = "small")

utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"common_dims_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_cs_umap, annos = np.concatenate([x["age"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"common_dims_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims1_cond1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[0], annos = np.concatenate([x["disease_severity"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "batches", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims1_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)

utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_celltypes_l1.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["age"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (10,7), save = (result_dir + comment+"diff_dims2_cond2.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)
utils.plot_latent(zs = z_ds_umap[1], annos = np.concatenate([x["predicted.celltype.l1"].values.squeeze() for x in data_dict_train["meta_cells"]]), batches = np.concatenate([x["dataset"].values.squeeze() for x in data_dict_train["meta_cells"]]), \
    mode = "annos", axis_label = "UMAP", figsize = (12,7), save = (result_dir + comment+"diff_dims2_batches.png".format()) if result_dir else None, markerscale = 9, s = 1, alpha = 0.5)


# In[]
# NOTE: note scenario 1 and 2 can also be achieved with 1 scgen/scpregan model too. The training would be 
# scenario 1: traing with data from healthy 40-, healthy 40-65. Input moderate 40-, predict moderate 40-65 (make it easy, of data from lee_2020)
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 1. Test data of condition: moderate, 40-65, lee_2020. Input data of condition: moderate, 40-, lee_2020. Training data: except moderate, 40-65, lee_2020.")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-") & (meta_cells["dataset"] == "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)
mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores1 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores1["MSE"] = mses_scdisinfact
scores1["MSE input"] = mses_input
scores1["Pearson"] = pearsons_scdisinfact
scores1["Pearson input"] = pearsons_input
scores1["R2"] = r2_scdisinfact
scores1["R2 input"] = r2_input
scores1["Method"] = "scDisInFact"
scores1["Prediction"] = "unseen (moderate, 40-, lee_2020)"

# de gene analysis
counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                             batch_key = "dataset", predict_conds = None, predict_batch = None)
pvals = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_predict, fdr = True)
pvals_gt = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_gt_denoised, fdr = True)
pvals_scores1 = pd.DataFrame(columns = ["genes", "pvals (predict)", "pvals (gt)", "Prediction"])
pvals_scores1["genes"] = genes
pvals_scores1["pvals (predict)"] = pvals
pvals_scores1["pvals (gt)"] = pvals_gt
pvals_scores1["Prediction"] = "unseen (moderate, 40-, lee_2020)"

# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 2. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 40-65, lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
# scenario 2: traing with data from healthy 40-, moderate 40-. Input healthy 40-65, predict moderate 40-65 (make it easy, of data from lee_2020)
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] == "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores2 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores2["MSE"] = mses_scdisinfact
scores2["MSE input"] = mses_input
scores2["Pearson"] = pearsons_scdisinfact
scores2["Pearson input"] = pearsons_input
scores2["R2"] = r2_scdisinfact
scores2["R2 input"] = r2_input
scores2["Method"] = "scDisInFact"
scores2["Prediction"] = "unseen (healthy, 40-65, lee_2020)"

# de gene analysis
counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                             batch_key = "dataset", predict_conds = None, predict_batch = None)
pvals = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_predict, fdr = True)
pvals_gt = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_gt_denoised, fdr = True)
pvals_scores2 = pd.DataFrame(columns = ["genes", "pvals (predict)", "pvals (gt)", "Prediction"])
pvals_scores2["genes"] = genes
pvals_scores2["pvals (predict)"] = pvals
pvals_scores2["pvals (gt)"] = pvals_gt
pvals_scores2["Prediction"] = "unseen (healthy, 40-65, lee_2020)"

# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 3. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 65+, lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
# scenario 3: for scgen/scpregan, need a cascade of two model, one predict disease effect, another predict age effect. 
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "65+") & (meta_cells["dataset"] == "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)

print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores3 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores3["MSE"] = mses_scdisinfact
scores3["MSE input"] = mses_input
scores3["Pearson"] = pearsons_scdisinfact
scores3["Pearson input"] = pearsons_input
scores3["R2"] = r2_scdisinfact
scores3["R2 input"] = r2_input
scores3["Method"] = "scDisInFact"
scores3["Prediction"] = "unseen (healthy, 65+, lee_2020)"

# de gene analysis
counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                             batch_key = "dataset", predict_conds = None, predict_batch = None)
pvals = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_predict, fdr = True)
pvals_gt = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_gt_denoised, fdr = True)
pvals_scores3 = pd.DataFrame(columns = ["genes", "pvals (predict)", "pvals (gt)", "Prediction"])
pvals_scores3["genes"] = genes
pvals_scores3["pvals (predict)"] = pvals
pvals_scores3["pvals (gt)"] = pvals_gt
pvals_scores3["Prediction"] = "unseen (healthy, 65+, lee_2020)"

# In[]
# NOTE: include batch effect
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 4. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: moderate, 40-, !lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "moderate") & (meta_cells["age"] == "40-") & (meta_cells["dataset"] != "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)
mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores4 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores4["MSE"] = mses_scdisinfact
scores4["MSE input"] = mses_input
scores4["Pearson"] = pearsons_scdisinfact
scores4["Pearson input"] = pearsons_input
scores4["R2"] = r2_scdisinfact
scores4["R2 input"] = r2_input
scores4["Method"] = "scDisInFact"
scores4["Prediction"] = "unseen (moderate, 40-, !lee_2020)"


# de gene analysis
counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                             batch_key = "dataset", predict_conds = None, predict_batch = None)
pvals = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_predict, fdr = True)
pvals_gt = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_gt_denoised, fdr = True)
pvals_scores4 = pd.DataFrame(columns = ["genes", "pvals (predict)", "pvals (gt)", "Prediction"])
pvals_scores4["genes"] = genes
pvals_scores4["pvals (predict)"] = pvals
pvals_scores4["pvals (gt)"] = pvals_gt
pvals_scores4["Prediction"] = "unseen (moderate, 40-, !lee_2020)"
# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 5. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 40-65, !lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "40-65") & (meta_cells["dataset"] != "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)
mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores5 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores5["MSE"] = mses_scdisinfact
scores5["MSE input"] = mses_input
scores5["Pearson"] = pearsons_scdisinfact
scores5["Pearson input"] = pearsons_input
scores5["R2"] = r2_scdisinfact
scores5["R2 input"] = r2_input
scores5["Method"] = "scDisInFact"
scores5["Prediction"] = "unseen (healthy, 40-65, !lee_2020)"

# de gene analysis
counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                             batch_key = "dataset", predict_conds = None, predict_batch = None)
pvals = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_predict, fdr = True)
pvals_gt = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_gt_denoised, fdr = True)
pvals_scores5 = pd.DataFrame(columns = ["genes", "pvals (predict)", "pvals (gt)", "Prediction"])
pvals_scores5["genes"] = genes
pvals_scores5["pvals (predict)"] = pvals
pvals_scores5["pvals (gt)"] = pvals_gt
pvals_scores5["Prediction"] = "unseen (healthy, 40-65, !lee_2020)"

# In[] 
print("# -------------------------------------------------------------------------------------------")
print("#")
print("# 6. Test on the unseen combination: moderate, 40-65, lee_2020. Input: training data of condition: healthy, 65+, !lee_2020")
print("#")
print("# -------------------------------------------------------------------------------------------")
counts_input = []
meta_input = []
for dataset, meta_cells in zip(data_dict_train["datasets"], data_dict_train["meta_cells"]):
    idx = ((meta_cells["disease_severity"] == "healthy") & (meta_cells["age"] == "65+") & (meta_cells["dataset"] != "lee_2020")).values
    counts_input.append(dataset.counts[idx,:].numpy())
    meta_input.append(meta_cells.loc[idx,:])

counts_input = np.concatenate(counts_input, axis = 0)
meta_input = pd.concat(meta_input, axis = 0, ignore_index = True)
counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                      batch_key = "dataset", predict_conds = ["moderate", "40-65"], predict_batch = "lee_2020")
print("input:")
print([x for x in np.unique(meta_input["batch_cond"].values)])


counts_gt = []
meta_gt = []
for dataset, meta_cells in zip(data_dict_test["datasets"], data_dict_test["meta_cells"]):
    counts_gt.append(dataset.counts.numpy())
    meta_gt.append(meta_cells)

counts_gt = np.concatenate(counts_gt, axis = 0)
meta_gt = pd.concat(meta_gt, axis = 0, ignore_index = True)
counts_gt_denoised = model.predict_counts(input_counts = counts_gt, meta_cells = meta_gt, condition_keys = ["disease_severity", "age"], 
                                          batch_key = "dataset", predict_conds = None, predict_batch = None)
print("ground truth:")
print([x for x in np.unique(meta_gt["batch_cond"].values)])

# optional: normalize the count
counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)
counts_gt_denoised = counts_gt_denoised/(np.sum(counts_gt_denoised, axis = 1, keepdims = True) + 1e-6)
counts_predict = counts_predict/(np.sum(counts_predict, axis = 1, keepdims = True) + 1e-6)
counts_input = counts_input/(np.sum(counts_input, axis = 1, keepdims = True) + 1e-6)

# no 1-1 match, check cell-type level scores
unique_celltypes = np.unique(meta_gt["predicted.celltype.l1"].values)
mean_inputs = []
mean_predicts = []
mean_gts = []
mean_gts_denoised = []
for celltype in unique_celltypes:
    mean_input = np.mean(counts_input[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_predict = np.mean(counts_predict[np.where(meta_input["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt = np.mean(counts_gt[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    mean_gt_denoised = np.mean(counts_gt_denoised[np.where(meta_gt["predicted.celltype.l1"].values.squeeze() == celltype)[0],:], axis = 0)
    
    mean_inputs.append(mean_input)
    mean_predicts.append(mean_predict)
    mean_gts.append(mean_gt)
    mean_gts_denoised.append(mean_gt_denoised)

mean_inputs = np.array(mean_inputs)
mean_predicts = np.array(mean_predicts)
mean_gts = np.array(mean_gts)
mean_gts_denoised = np.array(mean_gts_denoised)

# cell-type-specific normalized MSE
mses_input = np.sum((mean_inputs - mean_gts_denoised) ** 2, axis = 1)
mses_scdisinfact = np.sum((mean_predicts - mean_gts_denoised) ** 2, axis = 1)
# cell-type-specific pearson correlation
pearsons_input = np.array([stats.pearsonr(mean_inputs[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
pearsons_scdisinfact = np.array([stats.pearsonr(mean_predicts[i,:], mean_gts_denoised[i,:])[0] for i in range(mean_gts_denoised.shape[0])])
# cell-type-specific R2 score
r2_input = np.array([r2_score(y_pred = mean_inputs[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])
r2_scdisinfact = np.array([r2_score(y_pred = mean_predicts[i,:], y_true = mean_gts_denoised[i,:]) for i in range(mean_gts_denoised.shape[0])])

scores6 = pd.DataFrame(columns = ["MSE", "Pearson", "R2", "MSE input", "Pearson input", "R2 input", "Method", "Prediction"])
scores6["MSE"] = mses_scdisinfact
scores6["MSE input"] = mses_input
scores6["Pearson"] = pearsons_scdisinfact
scores6["Pearson input"] = pearsons_input
scores6["R2"] = r2_scdisinfact
scores6["R2 input"] = r2_input
scores6["Method"] = "scDisInFact"
scores6["Prediction"] = "unseen (healthy, 65+, !lee_2020)"

# de gene analysis
counts_input_denoised = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = ["disease_severity", "age"], 
                                             batch_key = "dataset", predict_conds = None, predict_batch = None)
pvals = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_predict, fdr = True)
pvals_gt = wilcoxon_rank_sum(counts_x = counts_input_denoised, counts_y = counts_gt_denoised, fdr = True)
pvals_scores6 = pd.DataFrame(columns = ["genes", "pvals (predict)", "pvals (gt)", "Prediction"])
pvals_scores6["genes"] = genes
pvals_scores6["pvals (predict)"] = pvals
pvals_scores6["pvals (gt)"] = pvals_gt
pvals_scores6["Prediction"] = "unseen (healthy, 65+, !lee_2020)"
# In[]
scores = pd.concat([scores1, scores2, scores3, scores4, scores5, scores6], axis = 0)
scores.to_csv(result_dir + "scores.csv")

# de_pvals = pd.concat([pvals_scores1, pvals_scores2, pvals_scores3, pvals_scores4, pvals_scores5, pvals_scores6], axis = 0)
# de_pvals.to_csv(result_dir + "de_pvals.csv")



# %%
