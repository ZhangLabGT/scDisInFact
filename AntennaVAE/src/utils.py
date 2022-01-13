import matplotlib.pyplot as plt
import scanpy as sc
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
    print('FIltered shape of Data1:{}, \t Data2: {}'.format(filtered_data1.shape, filtered_data2.shape))

    return filtered_data1, filtered_data2

def vis_latent_emb(data_Loader_1, data_loader_2, encoder, device):
    rcParams["font.size"] = 20

    umap_op = UMAP(n_components = 2, min_dist = 0.4, random_state = 0)
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
    ax.scatter(z_umap_1[:,0], z_umap_1[:, 1], color = clust_color(0), label = "Day4")
    ax.scatter(z_umap_2[:,0], z_umap_2[:, 1], color = clust_color(1), label = "Day2")
    ax.legend()
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

# PLOT
def plot_train(all_loss, type=None):
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
        ax = fig.add_subplot(1,1,1)
        
        ax.plot(x_rg, all_loss['train_loss'])
        ax.legend(['train_loss'])
        ax.title.set_text('Train Loss Fig')
    plt.tight_layout()
    plt.show()

