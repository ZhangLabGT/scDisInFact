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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# In[]
sigma = 0.4
n_diff_genes = 20
diff = 2
ngenes = 500
ncells_total = 10000 
n_batches = 6
data_dir = f"../data/simulated/imputation_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
result_dir = f"./simulated/imputation/contrastive_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: randomly remove some celltypes?
counts_ctrls = []
counts_stims1 = []
counts_stims2 = []
# cell types
label_annos = []
# batch labels
label_batches = []
counts_gt = []
label_ctrls = []
label_stims1 = []
label_stims2 = []
np.random.seed(0)
for batch_id in range(n_batches):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts_ctrls.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl.txt', sep = "\t", header = None).values.T)
    counts_stims1.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim1.txt', sep = "\t", header = None).values.T)
    counts_stims2.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim2.txt', sep = "\t", header = None).values.T)
    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
    # batch labels
    label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrls[-1].shape[0]))
    label_ctrls.append(np.array(["ctrl"] * counts_ctrls[-1].shape[0]))
    label_stims1.append(np.array(["stim1"] * counts_stims1[-1].shape[0]))
    label_stims2.append(np.array(["stim2"] * counts_stims2[-1].shape[0]))
    

# In[]
# Train with ctrl in batches 1 & 2, stim1 in batches 3 & 4, stim2 in batches 5 & 6
label_conditions = label_ctrls[0:2] + label_stims1[2:4] + label_stims2[4:]
# sub
# label_conditions = label_ctrls[0:2] + label_stims1[2:]
# sub2
# label_conditions = label_stims1[0:2] + label_stims2[2:]

condition_ids, condition_names = pd.factorize(np.concatenate(label_conditions, axis = 0))
batch_ids, batch_names = pd.factorize(np.concatenate(label_batches, axis = 0))
anno_ids, anno_names = pd.factorize(np.concatenate(label_annos, axis = 0))

counts = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:4] + counts_stims2[4:], axis = 0)
# sub
# counts = np.concatenate(counts_ctrls[0:2] + counts_stims1[2:], axis = 0)
# sub2
# counts = np.concatenate(counts_stims1[0:2] + counts_stims2[2:], axis = 0)

datasets = []
for batch_id, batch_name in enumerate(batch_names):
        datasets.append(scdisinfact.dataset(counts = counts[batch_ids == batch_id,:], 
                                            anno = anno_ids[batch_ids == batch_id], 
                                            diff_labels = [condition_ids[batch_ids == batch_id]], 
                                            batch_id = batch_ids[batch_ids == batch_id]))


# In[]
sigma = 0.4
n_diff_genes = 20
diff = 2
ngenes = 500
ncells_total = 10000 
n_batches = 6

# permute = True
permute = False

# data_dir = f"../data/simulated/two_cond/dataset_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
data_dir = f"../data/simulated/generalize_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

if permute:
    result_dir = f"./simulated/generalization/permute_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
else:
    result_dir = f"./simulated/generalization/dataset_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

counts = []
# cell types
label_annos = []
# batch labels
label_batches = []
counts_gt = []
label_cond1 = []
label_cond2 = []
np.random.seed(0)
for batch_id in range(6):
    # counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    counts.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}.txt', sep = "\t", header = None).values.T)
    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
    # batch labels
    label_batches.append(np.array(['batch ' + str(batch_id)] * counts[-1].shape[0]))
    
    if batch_id in [1, 2]:
        label_cond1.append(np.array(["ctrl"] * counts[-1].shape[0]))
    elif batch_id in [3,4]:
        label_cond1.append(np.array(["stim1"] * counts[-1].shape[0]))
    else:
        label_cond1.append(np.array(["stim2"] * counts[-1].shape[0]))

    if batch_id in [1,2,3]:
        label_cond2.append(np.array(["age_group1"] * counts[-1].shape[0])) 
    else:
        label_cond2.append(np.array(["age_group2"] * counts[-1].shape[0]))       


cond1_ids, cond1_names = pd.factorize(np.concatenate(label_cond1, axis = 0))
cond2_ids, cond2_names = pd.factorize(np.concatenate(label_cond2, axis = 0))
if permute: 
    permute_ids = np.random.permutation(cond1_ids.shape[0])
    cond1_ids = cond1_ids[permute_ids]
    cond2_ids = cond2_ids[permute_ids]

batch_ids, batch_names = pd.factorize(np.concatenate(label_batches, axis = 0))
anno_ids, anno_names = pd.factorize(np.concatenate(label_annos, axis = 0))
counts = np.concatenate(counts, axis = 0)

datasets_train = []
datasets_test = []
np.random.seed(0)
for batch_id, batch_name in enumerate(batch_names):
    count_batch = counts[batch_ids == batch_id,:]
    anno_batch = anno_ids[batch_ids == batch_id]
    diff_labels_batch = [cond1_ids[batch_ids == batch_id], cond2_ids[batch_ids == batch_id]]
    batch_ids_batch = batch_ids[batch_ids == batch_id]

    # generate random indices
    permute_idx = np.random.permutation(np.arange(count_batch.shape[0]))
    train_idx = permute_idx[:int(0.8 * count_batch.shape[0])]
    test_idx = permute_idx[int(0.8 * count_batch.shape[0]):]

    dataset_train = scdisinfact.dataset(counts = count_batch[train_idx,:], 
                                        anno = anno_batch[train_idx], 
                                        diff_labels = [diff_labels_batch[0][train_idx]], 
                                        batch_id = batch_ids_batch[train_idx])

    dataset_test = scdisinfact.dataset(counts = count_batch[test_idx,:], 
                                        anno = anno_batch[test_idx], 
                                        diff_labels = [diff_labels_batch[0][test_idx]], 
                                        batch_id = batch_ids_batch[test_idx])

    datasets_train.append(dataset_train)
    datasets_test.append(dataset_test)

datasets = datasets_train 

# In[] training the model
# TODO: track the time usage and memory usage
import importlib 
importlib.reload(scdisinfact)
start_time = time.time()
reg_mmd_comm = 5e-2
reg_mmd_diff = 1e-2
reg_gl = 1
reg_tc = 0.1
reg_class = 0.1
reg_kl = 1e-5
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4]
nepochs = 50
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets, Ks = Ks, batch_size = 64, interval = interval, lr = 1e-3, 
                                reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_tc = reg_tc, 
                                reg_kl = reg_kl, reg_class = reg_class, seed = 0, device = device)

print("GPU memory usage after constructing model: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
# train_joint is more efficient, but does not work as well compared to train
model.train()
losses = model.train_contr(nepochs = nepochs, recon_loss = "NB")
end_time = time.time()
print("time cost: {:.2f}".format(end_time - start_time))

torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth"))

model.eval()

# In[] Plot results
z_cs = []
z_ds = []
zs = []

for batch_id, dataset in enumerate(datasets):
    with torch.no_grad():
        z_c, logvar_c = model.Enc_c(torch.concat([dataset.counts_stand, dataset.batch_id[:,None]], dim = 1).to(model.device))
        z_ds.append([])
        for Enc_d in model.Enc_ds:
            z_d, logvar_d = Enc_d(torch.concat([dataset.counts_stand, dataset.batch_id[:,None]], dim = 1).to(model.device))
            z_ds[-1].append(z_d.cpu().detach().numpy())
        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))
        
        print(f"batch ID: {batch_id}")
        print("mean z_c")
        print(torch.mean(z_c))
        print("mean var z_c")
        print(torch.mean(logvar_c.mul(0.5).exp_()))
        print("mean z_d")
        print(torch.mean(z_d))
        print("mean var z_d")
        print(torch.mean(logvar_d.mul(0.5).exp_()))
    

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[]]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f"fig_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"common_celltypes_sep.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_condition.png" if result_dir else None, markerscale = 6, s = 5)

utils.plot_latent(zs = z_ds_umaps[0], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff_condition.png" if result_dir else None, markerscale = 6, s = 5)


# In[] Test imputation accuracy
# NOTE: impute control for batches 3, 4, 5, 6, only removed of conditions
# first calculate the mean z_ds for each condition

# control, batch 0 and 1
mean_ctrl = np.mean(np.concatenate([x[0] for x in z_ds[0:2]], axis = 0), axis = 0)[None,:]
# stim1, batch 2 and 3
mean_stim1 = np.mean(np.concatenate([x[0] for x in z_ds[2:4]], axis = 0), axis = 0)[None,:]
# stim2, batch 4 and 5
mean_stim2 = np.mean(np.concatenate([x[0] for x in z_ds[4:]], axis = 0), axis = 0)[None,:]

z_ds_ctrl = np.concatenate([x[0] for x in z_ds[0:2]], axis = 0)
# find the clostest
# mean_ctrl = z_ds_ctrl[[np.argmin(np.sum((z_ds_ctrl - mean_ctrl) ** 2, axis = 1))], :]

# NOTE: Impute, removed of both conditions and batch effect
zs = []
z_ds = []
mu_imputes = []

np.random.seed(0)
for batch_id, dataset in enumerate(datasets):
    # removed of batch effect
    ref_batch = 0.0
    with torch.no_grad():
        z_c, logvar_c = model.Enc_c(torch.concat([dataset.counts_stand, torch.tensor([ref_batch] * dataset.counts_stand.shape[0])[:,None]], dim = 1).to(model.device))
        if batch_id in [2,3,4,5]:
            # NOTE: removed of condition effect
            # manner one, use mean
            z_d = torch.tensor(mean_ctrl, device = model.device).expand(dataset.counts_stand.shape[0], Ks[1])
            # manner two, random sample, better than mean
            # idx = np.random.choice(z_ds_ctrl.shape[0], z_c.shape[0], replace = True)
            # z_d = torch.tensor(z_ds_ctrl[idx,:], device = model.device)
            
            z = torch.concat((z_c, z_d, torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None]), dim = 1)

        else:
            print(batch_id)
            z_ds.append([])
            for Enc_d in model.Enc_ds:
                z_d, logvar_d = Enc_d(torch.concat([dataset.counts_stand, torch.tensor([ref_batch] * dataset.counts_stand.shape[0])[:,None]], dim = 1).to(model.device))
                z_ds[-1].append(z_d)

            z = torch.concat([z_c] + z_ds[-1] + [torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None]], axis = 1)        

        mu_impute, _, _ = model.Dec(z)
        mu_impute = mu_impute.cpu().detach().numpy() 
        mu_imputes.append(mu_impute)
        zs.append(z.detach().cpu().numpy())

mu_impute = np.concatenate(mu_imputes, axis = 0)
# normalize and log transform
mu_impute = mu_impute/np.sum(mu_impute, axis = 1, keepdims = True) * 100
mu_impute = np.log1p(mu_impute)
# UMAP visualization
umap_op = UMAP(min_dist = 0.4, random_state = 0, n_neighbors = 30)
x_umap = umap_op.fit_transform(mu_impute)
z_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

x_umaps = []
z_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + mu_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        z_umaps.append(z_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + mu_imputes[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        z_umaps.append(z_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + mu_imputes[batch - 1].shape[0]
        end_pointer = start_pointer + mu_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        z_umaps.append(z_umap[start_pointer:end_pointer,:])

utils.plot_latent(zs = x_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_celltypes_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"impute_celltypes_sep_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_batches_ctrl.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_condition_ctrl.png" if result_dir else None, markerscale = 6, s = 5)

utils.plot_latent(zs = z_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"latent_celltypes_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"latent_celltypes_sep_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"latent_batches_ctrl.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"latent_condition_ctrl.png" if result_dir else None, markerscale = 6, s = 5)







    

# %%
