import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay,roc_curve,auc,RocCurveDisplay, average_precision_score, roc_auc_score

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