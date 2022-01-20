import matplotlib.pyplot as plt
import scanpy as sc
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import sparse
import os
import pandas as pd

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
    # TODO: will be discardedthis is only for AE_model.ipynb file, 
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
def vis_latent_emb(test_loader_1, test_loader_2, encoder, 
                    device, title, batches, maps=None):
    umap_op = UMAP(n_components=2, min_dist=0.4, random_state=0)
    for data in test_loader_1:
        z_data_1 = encoder(data["count"].to(device))
        cell_types_1 = data['anno']

    for data in test_loader_2:
        z_data_2 = encoder(data["count"].to(device))
        cell_types_2 = data['anno']

    z_umap_1 = umap_op.fit_transform(z_data_1.detach().cpu().numpy())
    z_umap_2 = umap_op.fit_transform(z_data_2.detach().cpu().numpy())

    colors_1 = np.array(cell_types_1)
    colors_2 = np.array(cell_types_2)
    for i in maps.items():
        colors_1 = np.where((colors_1 == i[1]), i[0], colors_1)
    for i in maps.items():
        colors_2 = np.where((colors_2 == i[1]), i[0], colors_2)

    fig = plt.figure(figsize = (20,7))
    ax1 = fig.add_subplot(1,2,1)
    # Plot cells colored by batch ID
    ax1.scatter(z_umap_1[:,0], z_umap_1[:, 1],c = 'b', label = batches[0])
    ax1.scatter(z_umap_2[:,0], z_umap_2[:, 1],c = 'r', label = batches[1])
    ax1.title.set_text(title + ' by batch')
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    
    ax2 = fig.add_subplot(1,2,2)
    # Plot cells colored by cell type
    ax2.scatter(z_umap_1[:,0], z_umap_1[:, 1],c = colors_1.astype(np.int), cmap = 'Spectral')
    ax2.scatter(z_umap_2[:,0], z_umap_2[:, 1],c = colors_2.astype(np.int) , cmap = 'Spectral')
    ax2.legend()
    scatter = ax2.scatter(z_umap_2[:, 0], z_umap_2[:, 1], c = colors_2.astype(np.int))
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)

    legend2 = ax2.legend(handles, maps.values(), loc="upper right", title="cell types")
    ax2.title.set_text(title + ' by cell type')
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
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

