# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import scdisinfact
import utils
from umap import UMAP
import time
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import seaborn as sns
import bmk

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.4f}'.format(p.get_height())
            ax.text(_x, _y + 0.1, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


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

reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]
nepochs = 50
interval = 10
n_batches = 6

data_dir = f"../data/simulated_new/"

# In[] by number of ndiff_genes
plt.rcParams["font.size"] = 20

simulated_lists = [
 'imputation_10000_500_0.2_20_2',
 'imputation_10000_500_0.3_20_2',
 'imputation_10000_500_0.4_20_2',
 'imputation_10000_500_0.2_50_2',
 'imputation_10000_500_0.3_50_2',
 'imputation_10000_500_0.4_50_2',
 'imputation_10000_500_0.2_100_2',
 'imputation_10000_500_0.3_100_2',
 'imputation_10000_500_0.4_100_2'  
 ]

auprc_dict = pd.DataFrame(columns = ["dataset", "ndiff_genes", "AUPRC", "AUPRC ratio", "method", "ndiff"])
for dataset_dir in simulated_lists:
    ngenes = eval(dataset_dir.split("_")[2])
    ndiff_genes = eval(dataset_dir.split("_")[4])
    ndiff = eval(dataset_dir.split("_")[5])
    gt = np.zeros((1, ngenes))
    gt[:,:ndiff_genes] = 1
    result_dir = './simulated/imputation_new/'+dataset_dir + "/"
    model_params = torch.load(result_dir + f"model_{Ks}_{lambs}.pth")
    inf = np.array(model_params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])

    auprc_dict = auprc_dict.append({"dataset": dataset_dir, 
                                    "ndiff_genes": ndiff_genes, 
                                    "AUPRC": bmk.compute_auprc(inf, gt), 
                                    "AUPRC ratio": bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes),
                                    "method": "scDisInFact",
                                    "ndiff": ndiff
                                    }, ignore_index = True)
    
    # scinsight
    result_scinsight = result_dir + "scinsight/"
    # scores for condition 1
    H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t").values
    # scores for condition 2
    H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t").values
    # scores for condition 3
    H3 = pd.read_csv(result_scinsight + "H_3.txt", sep = "\t").values

    # variance of scores
    H = np.concatenate([H1, H2, H3], axis = 0).T
    H_mean = np.mean(H, axis = 1, keepdims = True)
    H_var = np.mean((H - H_mean) ** 2, axis = 1)
    # normalize
    H_var = H_var/np.max(H_var)

    auprc_dict = auprc_dict.append({"dataset": dataset_dir, 
                                    "ndiff_genes": ndiff_genes, 
                                    "AUPRC": bmk.compute_auprc(H_var, gt),
                                    "AUPRC ratio": bmk.compute_auprc(H_var, gt)/(ndiff_genes/ngenes),
                                    "method": "scINSIGHT",
                                    "ndiff": ndiff
                                    }, ignore_index = True)    

    # wilcoxon baseline
    counts_ctrls = []
    counts_stims1 = []
    counts_stims2 = []
    for batch_id in range(n_batches):
        counts_ctrl = pd.read_csv(data_dir + dataset_dir + f'/GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T
        counts_stim1 = pd.read_csv(data_dir + dataset_dir + f'/GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T
        counts_stim2 = pd.read_csv(data_dir + dataset_dir + f'/GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T
        counts_ctrl = counts_ctrl/np.sum(counts_ctrl, axis = 1, keepdims = True) * 100
        counts_stim1 = counts_stim1/np.sum(counts_stim1, axis = 1, keepdims = True) * 100
        counts_stim2 = counts_stim2/np.sum(counts_stim2, axis = 1, keepdims = True) * 100
        counts_ctrl = np.log1p(counts_ctrl)
        counts_stim1 = np.log1p(counts_stim1)
        counts_stim2 = np.log1p(counts_stim2)

        counts_ctrls.append(counts_ctrl)
        counts_stims1.append(counts_stim1)
        counts_stims2.append(counts_stim2)
    # concatenated matrices
    counts_x = np.concatenate(counts_ctrls[:2], axis = 0)
    counts_y = np.concatenate(counts_stims1[2:4] + counts_stims2[4:], axis = 0)
    pvals = wilcoxon_rank_sum(counts_x, counts_y, fdr = True)
    assert pvals.shape[0] == 500
    assert np.min(pvals) >= 0
    # scale to 0-1
    score_wilcoxon = 1 - pvals/np.max(pvals)

    auprc_dict = auprc_dict.append({"dataset": dataset_dir, 
                                    "ndiff_genes": ndiff_genes, 
                                    "AUPRC": bmk.compute_auprc(score_wilcoxon, gt),
                                    "AUPRC ratio": bmk.compute_auprc(score_wilcoxon, gt)/(ndiff_genes/ngenes),
                                    "method": "Wilcoxon",
                                    "ndiff": ndiff
                                    }, ignore_index = True)




fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.barplot(x='ndiff_genes', y='AUPRC', hue='method', data=auprc_dict, ax = ax, errwidth=0) # 
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Number of perturbed genes")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')

fig.savefig("./simulated/imputation_new/AUPRC (ndiff_genes).png", bbox_inches = "tight")
    
# In[] by number of ndiff_genes
plt.rcParams["font.size"] = 20

simulated_lists = [
 'imputation_10000_500_0.2_20_2',
 'imputation_10000_500_0.3_20_2',
 'imputation_10000_500_0.4_20_2',
 'imputation_10000_500_0.2_20_4',
 'imputation_10000_500_0.3_20_4',
 'imputation_10000_500_0.4_20_4',
 'imputation_10000_500_0.2_20_8',
 'imputation_10000_500_0.3_20_8',
 'imputation_10000_500_0.4_20_8'  
 ]

auprc_dict = pd.DataFrame(columns = ["dataset", "ndiff_genes", "AUPRC", "AUPRC ratio", "method", "ndiff"])
for dataset_dir in simulated_lists:
    ngenes = eval(dataset_dir.split("_")[2])
    ndiff_genes = eval(dataset_dir.split("_")[4])
    ndiff = eval(dataset_dir.split("_")[5])
    gt = np.zeros((1, ngenes))
    gt[:,:ndiff_genes] = 1
    result_dir = './simulated/imputation_new/'+dataset_dir + "/"
    model_params = torch.load(result_dir + f"model_{Ks}_{lambs}.pth")
    inf = np.array(model_params["Enc_ds.0.fc.fc_layers.Layer 0.0.weight"].detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:ngenes])
    auprc_dict = auprc_dict.append({"dataset": dataset_dir, 
                                    "ndiff_genes": ndiff_genes, 
                                    "AUPRC": bmk.compute_auprc(inf, gt), 
                                    "AUPRC ratio": bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes),
                                    "method": "scDisInFact",
                                    "ndiff": ndiff
                                    }, ignore_index = True)    

    # scinsight
    result_scinsight = result_dir + "scinsight/"
    # scores for condition 1
    H1 = pd.read_csv(result_scinsight + "H_1.txt", sep = "\t").values
    # scores for condition 2
    H2 = pd.read_csv(result_scinsight + "H_2.txt", sep = "\t").values
    # scores for condition 3
    H3 = pd.read_csv(result_scinsight + "H_3.txt", sep = "\t").values

    # variance of scores
    H = np.concatenate([H1, H2, H3], axis = 0).T
    H_mean = np.mean(H, axis = 1, keepdims = True)
    H_var = np.mean((H - H_mean) ** 2, axis = 1)
    # normalize
    H_var = H_var/np.max(H_var)

    auprc_dict = auprc_dict.append({"dataset": dataset_dir, 
                                    "ndiff_genes": ndiff_genes, 
                                    "AUPRC": bmk.compute_auprc(H_var, gt),
                                    "AUPRC ratio": bmk.compute_auprc(H_var, gt)/(ndiff_genes/ngenes),
                                    "method": "scINSIGHT",
                                    "ndiff": ndiff
                                    }, ignore_index = True)    


    # wilcoxon baseline
    counts_ctrls = []
    counts_stims1 = []
    counts_stims2 = []
    for batch_id in range(n_batches):
        counts_ctrl = pd.read_csv(data_dir + dataset_dir + f'/GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T
        counts_stim1 = pd.read_csv(data_dir + dataset_dir + f'/GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T
        counts_stim2 = pd.read_csv(data_dir + dataset_dir + f'/GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T
        counts_ctrl = counts_ctrl/np.sum(counts_ctrl, axis = 1, keepdims = True) * 100
        counts_stim1 = counts_stim1/np.sum(counts_stim1, axis = 1, keepdims = True) * 100
        counts_stim2 = counts_stim2/np.sum(counts_stim2, axis = 1, keepdims = True) * 100
        counts_ctrl = np.log1p(counts_ctrl)
        counts_stim1 = np.log1p(counts_stim1)
        counts_stim2 = np.log1p(counts_stim2)
        
        counts_ctrls.append(counts_ctrl)
        counts_stims1.append(counts_stim1)
        counts_stims2.append(counts_stim2)
    # concatenated matrices
    counts_x = np.concatenate(counts_ctrls[:2], axis = 0)
    counts_y = np.concatenate(counts_stims1[2:4] + counts_stims2[4:], axis = 0)
    pvals = wilcoxon_rank_sum(counts_x, counts_y, fdr = True)
    assert pvals.shape[0] == 500
    assert np.min(pvals) >= 0
    # scale to 0-1
    score_wilcoxon = 1 - pvals/np.max(pvals)

    auprc_dict = auprc_dict.append({"dataset": dataset_dir, 
                                    "ndiff_genes": ndiff_genes, 
                                    "AUPRC": bmk.compute_auprc(score_wilcoxon, gt),
                                    "AUPRC ratio": bmk.compute_auprc(score_wilcoxon, gt)/(ndiff_genes/ngenes),
                                    "method": "Wilcoxon",
                                    "ndiff": ndiff
                                    }, ignore_index = True)



import seaborn as sns
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
ax = sns.barplot(x='ndiff', y='AUPRC', hue='method', data=auprc_dict, ax = ax, errwidth=0) # , errwidth=0
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_xlabel("Perturbation parameters")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')    

fig.savefig("./simulated/imputation_new/AUPRC (ndiffs).png", bbox_inches = "tight")


# %%
