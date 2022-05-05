import numpy as np
import pandas as pd 
import os
import subprocess

import scipy.special
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import silhouette_samples, silhouette_score
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import NMF
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay,roc_curve,auc,RocCurveDisplay, average_precision_score, roc_auc_score

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Benchmark the discovery of key genes
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def compute_auprc(G_inf, G_true):
    """
    Calculate AUPRC score
    """
    G_inf_abs = np.abs(G_inf)
    G_true_abs = np.abs(G_true)
    G_true_abs = (G_true_abs > 1e-6).astype(int)
    G_inf_abs = (G_inf_abs - np.min(G_inf_abs))/(np.max(G_inf_abs) - np.min(G_inf_abs) + 1e-12)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(G_inf_abs, G_true_abs)
    return AUPRC

def _compute_auc(estm_adj, gt_adj):
    """\
    Description:
    ------------
        calculate AUPRC and AUROC
    Parameters:
    ------------
        estm_adj: predict graph adjacency matrix
        gt_adj: ground truth graph adjacency matrix
        directed: the directed estimation or not
    Return:
    ------------
        prec: precision
        recall: recall
        fpr: false positive rate
        tpr: true positive rate
        AUPRC, AUROC
    """
    
    if np.max(estm_adj) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        fpr, tpr, thresholds = roc_curve(y_true=gt_adj.reshape(-1,), y_score=estm_adj.reshape(-1,), pos_label=1)
        
        if len(set(gt_adj.reshape(-1,))) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt_adj.reshape(-1,), probas_pred=estm_adj.reshape(-1,), pos_label=1)

        # the same
        # AUPRC = average_precision_score(gt_adj.reshape(-1,), estm_adj.reshape(-1,)) 
        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    


'''
def compute_earlyprec(estm_adj, gt_adj, directed = False, TFEdges = False):
    """\
    Description:
    ------------
        Calculate the early precision ratio. 
        Early precision: the fraction of true positives in the top-k edges. 
        Early precision ratio: the ratio of the early precision between estim and random estim.
        directed: the directed estimation or not
    Parameters:
    ------------
        estm_adj: estimated adjacency matrix
        gt_adj: ground truth adjacency matrix
        TFEdges: use transcription factor
    
    """
    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    if np.max(estm_norm_adj) == 0:
        return 0, 0
    else:
        # estm_adj = (estm_adj - np.min(estm_adj))/(np.max(estm_adj) - np.min(estm_adj))
        
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        # if directed == False:
        #     gt_adj = ((gt_adj + gt_adj.T) > 0).astype(np.int)
        # np.fill_diagonal(gt_adj, 0)
        # np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        trueEdgesDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        trueEdgesDF.Gene1 = np.array([str(x) for x in rows], dtype = np.object)
        trueEdgesDF.Gene2 = np.array([str(y) for y in cols], dtype = np.object)
        trueEdgesDF.EdgeWeight = 1

        rows, cols = np.where(estm_norm_adj != 0)
        predEdgeDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        predEdgeDF.Gene1 = np.array([str(x) for x in rows], dtype = np.object)
        predEdgeDF.Gene2 = np.array([str(y) for y in cols], dtype = np.object)
        predEdgeDF.EdgeWeight = np.array([estm_norm_adj[i,j] for i,j in zip(rows,cols)])

        # order according to ranks
        order = np.argsort(predEdgeDF.EdgeWeight.values.squeeze())[::-1]
        predEdgeDF = predEdgeDF.iloc[order,:]


        trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
        trueEdgesDF.drop_duplicates(keep = 'first', inplace=True)
        trueEdgesDF.reset_index(drop=True, inplace=True)


        predEdgeDF = predEdgeDF.loc[(predEdgeDF['Gene1'] != predEdgeDF['Gene2'])]
        predEdgeDF.drop_duplicates(keep = 'first', inplace=True)
        predEdgeDF.reset_index(drop=True, inplace=True)

        if TFEdges:
            # Consider only edges going out of TFs

            # Get a list of all possible TF to gene interactions 
            uniqueNodes = np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']])
            possibleEdges_TF = set(product(set(trueEdgesDF.Gene1),set(uniqueNodes)))

            # Get a list of all possible interactions 
            possibleEdges_noSelf = set(permutations(uniqueNodes, r = 2))

            # Find intersection of above lists to ignore self edges
            # TODO: is there a better way of doing this?
            possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)

            TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}

            trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdges = trueEdges[trueEdges.isin(TrueEdgeDict)]
            print("\nEdges considered ", len(trueEdges))
            numEdges = len(trueEdges)

            predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
            # limit the predicted edges to the genes that are in the ground truth
            predEdgeDF = predEdgeDF[predEdgeDF['Edges'].isin(TrueEdgeDict)]

        else:
            trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdges = set(trueEdges.values)
            numEdges = len(trueEdges)

        # check if ranked edges list is empty
        # if so, it is just set to an empty set
        if not predEdgeDF.shape[0] == 0:

            # we want to ensure that we do not include
            # edges without any edge weight
            # so check if the non-zero minimum is
            # greater than the edge weight of the top-kth
            # node, else use the non-zero minimum value.
            predEdgeDF.EdgeWeight = predEdgeDF.EdgeWeight.round(6)
            predEdgeDF.EdgeWeight = predEdgeDF.EdgeWeight.abs()

            # Use num True edges or the number of
            # edges in the dataframe, which ever is lower
            maxk = min(predEdgeDF.shape[0], numEdges)
            # find the maxkth edge weight
            edgeWeightTopk = predEdgeDF.iloc[maxk-1].EdgeWeight

            # find the smallest non-zero edge weight
            nonZeroMin = np.nanmin(predEdgeDF.EdgeWeight.replace(0, np.nan).values)

            # choose the largest one from nonZeroMin and edgeWeightTopk
            bestVal = max(nonZeroMin, edgeWeightTopk)

            # find all the edges with edge weight larger than the bestVal
            newDF = predEdgeDF.loc[(predEdgeDF['EdgeWeight'] >= bestVal)]
            # rankDict is a set that stores all significant edges
            rankDict = set(newDF['Gene1'] + "|" + newDF['Gene2'])
        else:
            # raise ValueError("No prediction")
            rankDict = []

        if len(rankDict) != 0:
            intersectionSet = rankDict.intersection(trueEdges)
            Eprec = len(intersectionSet)/(len(rankDict)+1e-12)
            Erec = len(intersectionSet)/(len(trueEdges)+1e-12)
        else:
            Eprec = 0
            Erec = 0

    return Eprec, Erec

def compute_eprec_abs(G_inf, G_true):
    # assert np.allclose(G_inf, G_inf.T, atol = 1e-7)
    # assert np.allclose(G_true, G_true.T, atol = 1e-7)
    G_inf_abs = np.abs(G_inf)
    G_true_abs = np.abs(G_true)
    G_true_abs = (G_true_abs > 1e-6).astype(int)
    G_inf_abs = (G_inf_abs - np.min(G_inf_abs))/(np.max(G_inf_abs) - np.min(G_inf_abs) + 1e-12)
    Eprec, Erec = compute_earlyprec(G_inf_abs, G_true_abs)
    return Eprec
'''

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g


def _compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    from umap.umap_ import fuzzy_simplicial_set
    from scipy.sparse import coo_matrix

    # place holder since we use precompute matrix
    X = coo_matrix(([], ([], [])), shape=(knn_indices.shape[0], 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    return connectivities.tocsr()

def leiden_cluster(
    X = None, 
    knn_indices = None,
    knn_dists = None,
    resolution = 30.0,
    random_state = 0,
    n_iterations: int = -1,
    k_neighs = 30,
    sigma = 1,
    affin = None,
    **partition_kwargs):

    from sklearn.neighbors import NearestNeighbors

    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
        )

    partition_kwargs = dict(partition_kwargs)
    
    if affin is None:
        if (knn_indices is None) or (knn_dists is None):
            # X is needed
            if X is None:
                raise ValueError("`X' and `knn_indices & knn_dists', at least one need to be provided.")

            neighbor = NearestNeighbors(n_neighbors = k_neighs)
            neighbor.fit(X)
            # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
            knn_dists, knn_indices = neighbor.kneighbors(X, n_neighbors = k_neighs, return_distance = True)

        affin = _compute_connectivities_umap(knn_indices = knn_indices, knn_dists = knn_dists, n_neighbors = k_neighs, set_op_mix_ratio=1.0, local_connectivity=1.0)
        affin = affin.todense()
        
    partition_type = leidenalg.RBConfigurationVertexPartition
    g = get_igraph_from_adjacency(affin, directed = True)

    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution

    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)

    return groups



#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Conservation of biological variation in single-cell data
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------


########################################################################################
#
# ARI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################

def ari(group1, group2, implementation=None):
    """ Adjusted Rand Index
    The function is symmetric, so group1 and group2 can be switched
    For single cell integration evaluation the scenario is:
        predicted cluster assignments vs. ground-truth (e.g. cell type) assignments
    :param adata: anndata object
    :param group1: string of column in adata.obs containing labels
    :param group2: string of column in adata.obs containing labels
    :params implementation: of set to 'sklearn', uses sklearns implementation,
        otherwise native implementation is taken
    """

    if len(group1) != len(group2):
        raise ValueError(
            f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})'
        )

    if implementation == 'sklearn':
        return adjusted_rand_score(group1, group2)

    def binom_sum(x, k=2):
        return scipy.special.binom(x, k).sum()

    n = len(group1)
    contingency = pd.crosstab(group1, group2)

    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))

    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)




########################################################################################
#
# NMI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################
def nmi(group1, group2, method="arithmetic", nmi_dir=None):
    """
    Wrapper for normalized mutual information NMI between two different cluster assignments
    :param adata: Anndata object
    :param group1: column name of `adata.obs`
    :param group2: column name of `adata.obs`
    :param method: NMI implementation
        'max': scikit method with `average_method='max'`
        'min': scikit method with `average_method='min'`
        'geometric': scikit method with `average_method='geometric'`
        'arithmetic': scikit method with `average_method='arithmetic'`
        'Lancichinetti': implementation by A. Lancichinetti 2009 et al. https://sites.google.com/site/andrealancichinetti/mutual
        'ONMI': implementation by Aaron F. McDaid et al. https://github.com/aaronmcdaid/Overlapping-NMI
    :param nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI' are specified as `method`.
        These packages need to be compiled as specified in the corresponding READMEs.
    :return:
        Normalized mutual information NMI value
    """
    
    if len(group1) != len(group2):
        raise ValueError(
            f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})'
        )

    # choose method
    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = normalized_mutual_info_score(group1, group2, average_method=method)
    elif method == "Lancichinetti":
        nmi_value = nmi_Lanc(group1, group2, nmi_dir=nmi_dir)
    elif method == "ONMI":
        nmi_value = onmi(group1, group2, nmi_dir=nmi_dir)
    else:
        raise ValueError(f"Method {method} not valid")

    return nmi_value


def onmi(group1, group2, nmi_dir=None, verbose=True):
    """
    Based on implementation https://github.com/aaronmcdaid/Overlapping-NMI
    publication: Aaron F. McDaid, Derek Greene, Neil Hurley 2011
    params:
        nmi_dir: directory of compiled C code
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from "
            "https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz"
        )

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "onmi", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)

    nmi_out = stdout.decode()
    if verbose:
        print(nmi_out)

    nmi_split = [x.strip().split('\t') for x in nmi_out.split('\n')]
    nmi_max = float(nmi_split[0][1])

    # remove temporary files
    os.remove(group1_file)
    os.remove(group2_file)

    return nmi_max


def nmi_Lanc(group1, group2, nmi_dir="external/mutual3/", verbose=True):
    """
    paper by A. Lancichinetti 2009
    https://sites.google.com/site/andrealancichinetti/mutual
    recommended by Malte
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz")

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "mutual", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)
    nmi_out = stdout.decode().strip()

    return float(nmi_out.split('\t')[1])


def write_tmp_labels(group_assignments, to_int=False, delim='\n'):
    """
    write the values of a specific obs column into a temporary file in text format
    needed for external C NMI implementations (onmi and nmi_Lanc functions), because they require files as input
    params:
        to_int: rename the unique column entries by integers in range(1,len(group_assignments)+1)
    """
    import tempfile

    if to_int:
        label_map = {}
        i = 1
        for label in set(group_assignments):
            label_map[label] = i
            i += 1
        labels = delim.join([str(label_map[name]) for name in group_assignments])
    else:
        labels = delim.join([str(name) for name in group_assignments])

    clusters = {label: [] for label in set(group_assignments)}
    for i, label in enumerate(group_assignments):
        clusters[label].append(str(i))

    output = '\n'.join([' '.join(c) for c in clusters.values()])
    output = str.encode(output)

    # write to file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(output)
        filename = f.name

    return filename




########################################################################################
#
# silhouette score from scIB(https://github.com/theislab/scib/tree/main/scib), silhouette
# score measures the differences between within cluster distances and between cluster distances
# for cell types, larger silhouette score means better result, and for batches smaller 
# silhouette score means better result. 
# 
# FUNCTIONS:
# -------------------------------------------------
# silhouette: calculate the cell type separation
# silhouette_batch: calculate the batch mixing
# -------------------------------------------------
#
########################################################################################

def silhouette(
        X,
        group_gt,
        metric='euclidean',
        scale=True
):
    """
    Wrapper for sklearn silhouette function values range from [-1, 1] with
        1 being an ideal fit
        0 indicating overlapping clusters and
        -1 indicating misclassified cells
    By default, the score is scaled between 0 and 1. This is controlled `scale=True`
    :param group_gt: cell labels
    :param X: embedding e.g. PCA
    :param scale: default True, scale between 0 (worst) and 1 (best)
    """
    asw = silhouette_score(
        X=X,
        labels=group_gt,
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


def silhouette_batch(
        X,
        batch_gt,
        group_gt,
        metric='euclidean',
        return_all=False,
        scale=True,
        verbose=True
):
    """
    Absolute silhouette score of batch labels subsetted for each group.
    :param batch_key: batches to be compared against
    :param group_key: group labels to be subsetted by e.g. cell type
    :param embed: name of column in adata.obsm
    :param metric: see sklearn silhouette score
    :param scale: if True, scale between 0 and 1
    :param return_all: if True, return all silhouette scores and label means
        default False: return average width silhouette (ASW)
    :param verbose:
    :return:
        average width silhouette ASW
        mean silhouette per group in pd.DataFrame
        Absolute silhouette scores per group label
    """

    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])

    for group in np.sort(np.unique(group_gt)):
        X_group = X[group_gt == group, :]
        batch_group = batch_gt[group_gt == group]
        n_batches = np.unique(batch_group).shape[0]

        if (n_batches == 1) or (n_batches == X_group.shape[0]):
            continue

        sil_per_group = silhouette_samples(
            X_group,
            batch_group,
            metric=metric
        )

        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]

        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]

        sil_all = sil_all.append(
            pd.DataFrame({
                'group': [group] * len(sil_per_group),
                'silhouette_score': sil_per_group
            })
        )

    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()
    asw = sil_means['silhouette_score'].mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')

    if return_all:
        return asw, sil_means, sil_all

    return asw




#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Batch effect removal per cell identity label (consider the batch mixing for each cell type)
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

########################################################################################
#
# graph connectivity score from scIB (https://github.com/theislab/scib/tree/main/scib), 
# It measures the mixing of the batches, and assesses whether the kNN graph 
# representation, ​G,​ of the integrated data directly connects all cells with 
# the same cell identity label. It first construct a KNN graph on the integrated latent 
# embedding, then it calculate that for cells of each cell type, how well the subgraph
# is connected. 
# 
# FUNCTIONS:
# -------------------------------------------------
# def graph_connectivity(adata, label_key):
#     """"
#     Quantify how connected the subgraph corresponding to each batch cluster is.
#     Calculate per label: #cells_in_largest_connected_component/#all_cells
#     Final score: Average over labels

#     :param adata: adata with computed neighborhood graph
#     :param label_key: name in adata.obs containing the cell identity labels
#     """
#     if 'neighbors' not in adata.uns:
#         raise KeyError(
#             'Please compute the neighborhood graph before running this function!'
#         )

#     clust_res = []

#     for label in adata.obs[label_key].cat.categories:
#         adata_sub = adata[adata.obs[label_key].isin([label])]
#         _, labels = connected_components(
#             adata_sub.obsp['connectivities'],
#             connection='strong'
#         )
#         tab = pd.value_counts(labels)
#         clust_res.append(tab.max() / sum(tab))

#     return np.mean(clust_res)
# -------------------------------------------------
########################################################################################

def graph_connectivity(X = None, G = None, groups = None, k = 10):
    """"
    Quantify how connected the subgraph corresponding to each batch cluster is.
    Calculate per label: #cells_in_largest_connected_component/#all_cells
    Final score: Average over labels

    :param adata: adata with computed neighborhood graph
    :param label_key: name in adata.obs containing the cell identity labels
    """
    clust_res = []
    if X is not None: 
        # calculate the adjacency matrix of the neighborhood graph
        G = kneighbors_graph(X, n_neighbors = k, mode='connectivity', include_self = False)
    elif G is None:
        raise ValueError("Either X or G should be provided")
    
    # make sure all cells have labels
    assert groups.shape[0] == G.shape[0]
    
    for group in np.sort(np.unique(groups)):
        G_sub = G[groups == group,:][:, groups == group]
        # print(G_sub.shape)
        _, labels = connected_components(csr_matrix(G_sub), connection='strong')
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Batch effect removal regardless cell identity label (consider the batch mixing in whole)
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
