import matplotlib.pyplot as plt
# from umap import UMAP
# from matplotlib import rcParams
import numpy as np
# from scipy import sparse
# import os
# import pandas as pd
# from adjustText import adjust_text

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

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


def plot_latent(zs, annos = None, batches = None, mode = "annos", save = None, figsize = (20,10), axis_label = "Latent", label_inplace = False, legend = True, **kwargs):
    """\
    Description
        Plot latent space
    Parameters
        z1
            the latent space of first data batch, of the shape (n_samples, n_dimensions)
        z2
            the latent space of the second data batch, of the shape (n_samples, n_dimensions)
        anno1
            the cluster annotation of the first data batch, of the  shape (n_samples,)
        anno2
            the cluster annotation of the second data batch, of the  shape (n_samples,)
        mode
            "joint": plot two latent spaces(from two batches) into one figure
            "separate" plot two latent spaces separately
        save
            file name for the figure
        figsize
            figure size
    """
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
        "markerscale": 1,
        "text_size": "large",
        "colormap": None
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize, dpi = 300, constrained_layout=True)
    if (mode == "annos") | (mode == "batches"):
        ax = fig.add_subplot()
        if mode == "annos":
            unique_cluster = np.unique(annos)
            if _kwargs["colormap"] is None:
                colormap = plt.cm.get_cmap("tab20b", len(unique_cluster))
            else:
                colormap = _kwargs["colormap"]
        else:
            unique_cluster = np.unique(batches)
            if _kwargs["colormap"] is None:
                colormap = plt.cm.get_cmap("tab20b", len(unique_cluster))
            else:
                colormap = _kwargs["colormap"]

        texts = []
        for i, cluster_type in enumerate(unique_cluster):
            if mode == "annos":
                index = np.where(annos == cluster_type)[0]
            else:
                index = np.where(batches == cluster_type)[0]
            z_clust = zs[index,:]
            ax.scatter(z_clust[:,0], z_clust[:,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
            # text on plot
            if label_inplace:
                texts.append(ax.text(np.median(z_clust[:,0]), np.median(z_clust[:,1]), color = "black", s = unique_cluster[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
        
        if legend:
            leg = ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(unique_cluster) // 15) + 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        # adjust position
        # if label_inplace:
        #     adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})

    elif mode == "separate":
        unique_batch = np.unique(batches)
        unique_cluster = np.unique(annos) 
        axs = fig.subplots(len(unique_batch),1)
        if _kwargs["colormap"] is None:
            colormap = plt.cm.get_cmap("tab20b", len(unique_cluster))
        else:
            colormap = _kwargs["colormap"]

        for i, batch in enumerate(unique_batch):
            zs_batch = zs[batches == batch]
            annos_batch = annos[batches == batch]
            z_clust = []
            texts = []
            for j, cluster_type in enumerate(unique_cluster):
                index = np.where(annos_batch == cluster_type)[0]
                if len(index) > 0:
                    axs[i].scatter(zs_batch[index,0], zs_batch[index,1], color = colormap(j), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
                    # text on plot
                    if label_inplace:
                        # if exist cells
                        if zs_batch[index,0].shape[0] > 0:
                            texts.append(axs[i].text(np.median(zs_batch[index,0]), np.median(zs_batch[index,1]), color = "black", s = cluster_type, fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
            
            if legend:
                leg = axs[i].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(unique_cluster) // 15) + 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)

                
            axs[i].set_title(batch, fontsize = 25)

            axs[i].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[i].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[i].set_ylabel(axis_label + " 2", fontsize = 19)

            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)  

            axs[i].set_xlim(np.min(zs[:,0]), np.max(zs[:,0]))
            axs[i].set_ylim(np.min(zs[:,1]), np.max(zs[:,1]))
            axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
            axs[i].yaxis.set_major_locator(plt.MaxNLocator(4))
            axs[i].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

            # if label_inplace:
            #     adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})        
            plt.tight_layout()
    if save:
        fig.savefig(save, bbox_inches = "tight")
