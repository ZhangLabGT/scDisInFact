# In[]
import sys, os
import torch
import numpy as np 
import pandas as pd
sys.path.append("../src")
import scdisinfact
import utils
import bmk
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# In[]
data_dir = f"../data/scINSIGHT/"
# lsa performs the best
result_dir = f"./simulated/scINSIGHT/original/"
n_batches = 6
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: randomly remove some celltypes?
counts = []
# cell types
label_annos = []
# batch labels
label_batches = []
# conditions
label_conditions = []
np.random.seed(0)
for batch_id in range(n_batches):
    count = pd.read_csv(data_dir + f'GxC{batch_id + 1}.txt', sep = "\t", index_col = 0)
    # plt.hist(np.sum(count, axis = 0), bins = 20)
    genes = count.index.values.squeeze()    
    anno = np.array([x.split(".")[0] for x in count.columns.values])
    label_annos.append(anno)
    label_batches.append(np.array(['batch ' + str(batch_id)] * count.shape[1]))
    label_conditions.append(np.array(['T ' + str(int(batch_id/2))] * count.shape[1]))
    counts.append(count.values.T)

# In[]
condition_ids, condition_names = pd.factorize(np.concatenate(label_conditions, axis = 0))
batch_ids, batch_names = pd.factorize(np.concatenate(label_batches, axis = 0))
anno_ids, anno_names = pd.factorize(np.concatenate(label_annos, axis = 0))

counts = np.concatenate(counts, axis = 0)

datasets = []
for batch_id, batch_name in enumerate(batch_names):
    # NOTE: the data was preprocessed already
    datasets.append(scdisinfact.dataset(counts = counts[batch_ids == batch_id,:], 
                                        anno = anno_ids[batch_ids == batch_id], 
                                        diff_labels = [condition_ids[batch_ids == batch_id]], 
                                        batch_id = batch_ids[batch_ids == batch_id],
                                        normalize = False
                                        ))


# check the visualization before integration
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 

counts_norms = []
for batch in range(n_batches):
    counts_norms.append(datasets[batch].counts_norm)

x_umap = umap_op.fit_transform(np.concatenate(counts_norms, axis = 0))
# separate into batches
x_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts_norms[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + counts_norms[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + counts_norms[batch - 1].shape[0]
        end_pointer = start_pointer + counts_norms[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

save_file = None

utils.plot_latent(x_umaps, annos = label_annos, mode = "joint", save = save_file, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = label_conditions, mode = "joint", save = save_file, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[] training the model
# TODO: track the time usage and memory usage
import importlib 
importlib.reload(scdisinfact)
start_time = time.time()
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 0.1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]
nepochs = 50
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets, Ks = Ks, batch_size = 64, interval = interval, lr = 5e-4, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
# train_joint is more efficient, but does not work as well compared to train
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

end_time = time.time()
print("time cost: {:.2f}".format(end_time - start_time))

torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth"))

_ = model.eval()

# In[] Plot results
z_cs = []
z_ds = []
zs = []
# one forward pass
with torch.no_grad():
    for batch_id, dataset in enumerate(datasets):
        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(z.cpu().detach().numpy())
        z_ds.append([x.cpu().detach().numpy() for x in z_d])   

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))

z_ds_umap = []
for diff_factor in range(model.n_diff_factors):
    z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[diff_factor] for z_d in z_ds], axis = 0)))

zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[] * model.n_diff_factors]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]

        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])
        for diff_factor in range(model.n_diff_factors):
            z_ds_umaps[diff_factor].append(z_ds_umap[diff_factor][start_pointer:end_pointer,:])


    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]

        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])
        for diff_factor in range(model.n_diff_factors):
            z_ds_umaps[diff_factor].append(z_ds_umap[diff_factor][start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]

        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])
        for diff_factor in range(model.n_diff_factors):
            z_ds_umaps[diff_factor].append(z_ds_umap[diff_factor][start_pointer:end_pointer,:])

comment = f"fig_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"common_celltypes_sep.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_condition.png" if result_dir else None, markerscale = 6, s = 5)

for diff_factor in range(model.n_diff_factors):
    utils.plot_latent(zs = z_ds_umaps[diff_factor], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umaps[diff_factor], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_batch.png" if result_dir else None, markerscale = 6, s = 5)
    utils.plot_latent(zs = z_ds_umaps[diff_factor], annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+f"diff{diff_factor}_condition.png" if result_dir else None, markerscale = 6, s = 5)


# %%
