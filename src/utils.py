from pickle import TRUE
import matplotlib.pyplot as plt
import scanpy as sc
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import sparse
import os
import pandas as pd
from adjustText import adjust_text
"""
# This is the function to load batches and corresponding meta files,
def load_batches(batch_id_1, batch_id_2, meta_file_lst, mtx_file_lst, path_base):

    batch_file_1 = [i for i in mtx_file_lst if str(batch_id_1) in i][0]
    batch_file_2 = [i for i in mtx_file_lst if str(batch_id_2) in i][0]

    meta_file_1 = [i for i in meta_file_lst if str(batch_id_1) in i][0]
    meta_file_2 = [i for i in meta_file_lst if str(batch_id_2) in i][0]
    print('Loading batch file 1: {}\t with meta file 1:{}'.format(batch_file_1, meta_file_1))
    print('Loading batch file 1: {}\t with meta file 1:{}'.format(batch_file_2, meta_file_2))

    batch_mtx_1 = np.array(sparse.load_npz(os.path.join(path_base, batch_file_1)).todense()).T
    batch_mtx_2 = np.array(sparse.load_npz(os.path.join(path_base, batch_file_2)).todense()).T

    batch_meta_1 = pd.read_csv(os.path.join(path_base, meta_file_1))
    batch_meta_2 = pd.read_csv(os.path.join(path_base, meta_file_2))
    print('Batch 1 size:{}\tMeta 1 size:{}'.format(batch_mtx_1.shape, batch_meta_1.shape))
    print('Batch 2 size:{}\tMeta 2 size:{}'.format(batch_mtx_2.shape, batch_meta_2.shape))

    return batch_mtx_1, batch_meta_1, batch_mtx_2, batch_meta_2
# Data preprocessing
# Make sure the intput data is cell x gene format with gene on each column
def preproc_filter(data1, data2, min_cells):
    if not min_cells:
        assert ValueError
    cell_num1 = data1.shape[0]

    print("Original shape of Data1:{} \t Data2:{}".format(data1.shape, data2.shape))

    cmb_data = sc.concat([data1, data2])
    sc.pp.filter_genes(cmb_data, min_cells=min_cells)

    filtered_data1 = cmb_data[:cell_num1, :]
    filtered_data2 = cmb_data[cell_num1:, :]
    print('FIltered shape of Data1:{}, \t Data2: {}'.format(
        filtered_data1.shape, filtered_data2.shape))

    return filtered_data1, filtered_data2


def vis_latent_emb_simp(data_Loader_1, data_loader_2, encoder, device, labels, title):
    rcParams["font.size"] = 20

    umap_op = UMAP(n_components=2, min_dist=0.4, random_state=0)
    for data in data_Loader_1:
        z_data_1 = encoder(data["count"].to(device))

    for data in data_loader_2:
        z_data_2 = encoder(data["count"].to(device))

    z_umap_1 = umap_op.fit_transform(z_data_1.detach().cpu().numpy())
    z_umap_2 = umap_op.fit_transform(z_data_2.detach().cpu().numpy())

    # plot the figure
    n_clust = 2
    clust_color = plt.cm.get_cmap("Paired", n_clust)
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot()
    ax.scatter(z_umap_1[:,0], z_umap_1[:, 1], color = 'r', label = labels[0])
    ax.scatter(z_umap_2[:,0], z_umap_2[:, 1], color = 'b', label = labels[1])
    ax.legend()
    ax.title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

# This is the fuction to visilize the embedding results
def vis_embedding(test_loader_1, test_loader_2, encoder, 
                    device, title, batches, maps, preview=False):
    umap_op = UMAP(n_components=2, min_dist=0.4, random_state=0)

    for data in test_loader_1:
        z_data_1 = encoder(data["count"].to(device))
        cell_types_1 = data['anno']

    for data in test_loader_2:
        z_data_2 = encoder(data["count"].to(device))
        cell_types_2 = data['anno']

    z_umap_1 = umap_op.fit_transform(z_data_1.detach().cpu().numpy())
    z_umap_2 = umap_op.fit_transform(z_data_2.detach().cpu().numpy())
    # This is the encoded data
    z_umap = np.concatenate((z_umap_1 ,z_umap_2), axis=0)
    colors = np.concatenate((cell_types_1, cell_types_2), axis=0)
        
    for i in maps.items():
        colors = np.where((colors == i[1]), i[0], colors)

    fig = plt.figure(figsize = (20,16))
    ax1 = fig.add_subplot(2,2,1)
    # Plot cells colored by batch ID
    ax1.scatter(z_umap_1[:,0], z_umap_1[:, 1],c = 'b', label = batches[0])
    ax1.scatter(z_umap_2[:,0], z_umap_2[:, 1],c = 'r', label = batches[1])
    ax1.title.set_text(title + ' by batch')
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    ax2 = fig.add_subplot(2,2,2)
    # Plot cells colored by cell type
    cmap = plt.cm.get_cmap('Paired', len(maps.items()))
    ax2.scatter(z_umap[:,0], z_umap[:, 1],c = colors.astype(np.float))
    ax2.legend()

    scatter = ax2.scatter(z_umap[:, 0], z_umap[:, 1], c = colors.astype(np.int), cmap=cmap)
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)

    legend2 = ax2.legend(handles, maps.values(), loc="upper right", title="cell types")
    ax2.title.set_text(title + ' by cell type')
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    
    for data in test_loader_1:
        z_data_1_raw = data["count"]
        cell_types_1_raw = data['anno']

    for data in test_loader_2:
        z_data_2_raw = data["count"]
        cell_types_2_raw = data['anno']

    z_umap_1_raw = umap_op.fit_transform(z_data_1_raw.detach().cpu().numpy())
    z_umap_2_raw = umap_op.fit_transform(z_data_2_raw.detach().cpu().numpy())
    # This is the raw data
    z_umap_raw = np.concatenate((z_umap_1_raw ,z_umap_2_raw), axis=0)
    colors_raw = np.concatenate((cell_types_1_raw, cell_types_2_raw), axis=0)
        
    for i in maps.items():
        colors_raw = np.where((colors_raw == i[1]), i[0], colors_raw)

    ax3 = fig.add_subplot(2,2,3)
    # Plot cells colored by batch ID
    ax3.scatter(z_umap_1_raw[:,0], z_umap_1_raw[:, 1],c = 'b', label = batches[0])
    ax3.scatter(z_umap_2_raw[:,0], z_umap_2_raw[:, 1],c = 'r', label = batches[1])
    ax3.title.set_text("Raw data {} and {}".format(batches[0], batches[1]) + ' by batch')
    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")

    ax4 = fig.add_subplot(2,2,4)
    # Plot cells colored by cell type
    cmap = plt.cm.get_cmap('Paired', len(maps.items()))
    ax4.scatter(z_umap_raw[:,0], z_umap_raw[:, 1],c = colors_raw.astype(np.float))
    ax4.legend()

    scatter = ax4.scatter(z_umap_raw[:, 0], z_umap_raw[:, 1], c = colors_raw.astype(np.int), cmap=cmap)
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)

    legend2 = ax4.legend(handles, maps.values(), loc="upper right", title="cell types")
    ax4.title.set_text("Raw data {} and {}".format(batches[0], batches[1]) + ' by cell type')
    ax4.set_xlabel("UMAP 1")
    ax4.set_ylabel("UMAP 2")
    plt.tight_layout()

# PLOT
def plot_train(all_loss, type=None, batches=None):
    x_rg = range(len(all_loss['train_loss']))
    fig = plt.figure()
    if type == "sep":
        fig.add_subplot(1,1,1)
        plt.plot(x_rg, all_loss['train_loss'])
        plt.plot(x_rg, all_loss['test_loss'])
        plt.legend(['train_loss', 'test_loss'])
    elif type == 'cmb':
        ax1 = fig.add_subplot(2,2,1)

        ax1.plot(x_rg, all_loss['train_loss'])
        ax1.plot(x_rg, all_loss['test_loss'])
        ax1.legend(['train_loss', 'test_loss'])
        ax1.title.set_text('Combine Loss Fig')

        ax2 = fig.add_subplot(2,2,2)

        ax2.plot(x_rg, all_loss['train_loss'])
        ax2.title.set_text('Train_Loss Fig')
        
        test_loss_fig = fig.add_subplot(2,2,3)
        test_loss_fig.plot(x_rg, all_loss['test_loss'])
        test_loss_fig.title.set_text('Test_Loss Fig')
    elif type == None:
        ax = fig.add_subplot()
        
        ax.plot(x_rg, all_loss['train_loss'])
        ax.legend(['train_loss'])
        if batches == None:
            ax.title.set_text('Train Loss Fig')
        else:
            ax.title.set_text('Train Loss Fig:{} and {}'.format(batches[0], batches))
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
    plt.show()

"""
#######################################################################
# newly added functions
#######################################################################

def plot_latent(zs, annos = None, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", label_inplace = False, **kwargs):
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
        "colormap": "tab20b"
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("Paired", len(zs))
        ax = fig.add_subplot()
        
        for batch in range(len(zs)):
            ax.scatter(zs[batch][:,0], zs[batch][:,1], color = colormap(batch), label = "batch " + str(batch + 1), s = _kwargs["s"], alpha = _kwargs["alpha"])
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  

    elif mode == "joint":
        ax = fig.add_subplot()
        cluster_types = set()
        for batch in range(len(zs)):
            cluster_types = cluster_types.union(set([x for x in np.unique(annos[batch])]))
        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_types))
        cluster_types = sorted(list(cluster_types))
        
        texts = []
        for i, cluster_type in enumerate(cluster_types):
            z_clust = []
            for batch in range(len(zs)):
                index = np.where(annos[batch] == cluster_type)[0]
                z_clust.append(zs[batch][index,:])
            ax.scatter(np.concatenate(z_clust, axis = 0)[:,0], np.concatenate(z_clust, axis = 0)[:,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
            # text on plot
            if label_inplace:
                texts.append(ax.text(np.median(np.concatenate(z_clust, axis = 0)[:,0]), np.median(np.concatenate(z_clust, axis = 0)[:,1]), color = "black", s = cluster_types[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
        
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(cluster_types) // 15) + 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        # adjust position
        if label_inplace:
            adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})

    elif mode == "separate":
        axs = fig.subplots(len(zs),1)
        cluster_types = set()
        for batch in range(len(zs)):
            cluster_types = cluster_types.union(set([x for x in np.unique(annos[batch])]))
        cluster_types = sorted(list(cluster_types))
        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_types))


        for batch in range(len(zs)):
            z_clust = []
            texts = []
            for i, cluster_type in enumerate(cluster_types):
                index = np.where(annos[batch] == cluster_type)[0]
                axs[batch].scatter(zs[batch][index,0], zs[batch][index,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
                # text on plot
                if label_inplace:
                    # if exist cells
                    if zs[batch][index,0].shape[0] > 0:
                        texts.append(axs[batch].text(np.median(zs[batch][index,0]), np.median(zs[batch][index,1]), color = "black", s = cluster_types[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
            
            axs[batch].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(cluster_types) // 15) + 1, bbox_to_anchor=(0.94, 1), markerscale = _kwargs["markerscale"])
            axs[batch].set_title("batch " + str(batch + 1), fontsize = 25)

            axs[batch].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[batch].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[batch].set_ylabel(axis_label + " 2", fontsize = 19)

            axs[batch].spines['right'].set_visible(False)
            axs[batch].spines['top'].set_visible(False)  

            axs[batch].set_xlim(np.min(np.concatenate([x[:,0] for x in zs])), np.max(np.concatenate([x[:,0] for x in zs])))
            axs[batch].set_ylim(np.min(np.concatenate([x[:,1] for x in zs])), np.max(np.concatenate([x[:,1] for x in zs])))

            # if label_inplace:
            #     adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})        
    plt.tight_layout()
    if save:
        fig.savefig(save, bbox_inches = "tight")
