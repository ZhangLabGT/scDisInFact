import anndata as ad
import scanpy as sc
import utils
import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from umap import UMAP

def ari(adata, annos, resolution=0.1, show_fig=False):
    embeddings = adata.X
    nbatches = len(annos)
    y_true = list(torch.cat(annos))
    umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.1, random_state = 0) 
    x_umap = umap_op.fit_transform(embeddings)
    # use leiden for clustering, setting resolution to ensure the cluster number matches true labels
    
#     data = ad.AnnData(np.concatenate(embeddings))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution)
    
    y_pre = adata.obs['leiden']
    k_pre = adata.obs['leiden'].unique().shape[0]
    nmi_score = NMI(y_true, y_pre)
    air_score = ARI(y_true, y_pre)
    print('NMI Score : ', nmi_score, '\n', 'ARI Score: ', air_score, '\n', 'Cluster Number:', k_pre)
    if show_fig:
        x_umaps = []
        y_pre_annos = []
        for batch in range(nbatches):
            if batch == 0:
                start_pointer = 0
                end_pointer = start_pointer + embeddings[batch].shape[0]
                x_umaps.append(x_umap[start_pointer:end_pointer,:])
                y_pre_annos.append(y_pre[start_pointer:end_pointer])
            elif batch == (nbatches - 1):
                start_pointer = start_pointer + embeddings[batch - 1].shape[0]
                x_umaps.append(x_umap[start_pointer:,:])
                y_pre_annos.append(y_pre[start_pointer:])
            else:
                start_pointer = start_pointer + embeddings[batch - 1].shape[0]
                end_pointer = start_pointer + embeddings[batch].shape[0]
                x_umaps.append(x_umap[start_pointer:end_pointer,:])
                y_pre_annos.append(y_pre[start_pointer:end_pointer])

        utils.plot_latent(x_umaps, annos = y_pre_annos, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)