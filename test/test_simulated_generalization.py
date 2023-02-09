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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import r2_score 

# In[]
sigma = 0.4
n_diff_genes = 20
diff = 2
ngenes = 500
ncells_total = 10000 
n_batches = 6

permute = False

data_dir = f"../data/simulated/2conditions_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

if permute:
    result_dir = f"./simulated/generalization/permute_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
else:
    result_dir = f"./simulated/generalization/dataset_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# cell types
label_annos = []
# batch labels
label_batches = []

counts_gt = []
counts_ctrl_healthy = []
counts_ctrl_mild = []
counts_ctrl_severe = []
counts_stim_healthy = []
counts_stim_mild = []
counts_stim_severe = []

np.random.seed(0)
for batch_id in range(6):
    counts_gt.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_true.txt', sep = "\t", header = None).values.T)
    
    counts_ctrl_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_healthy.txt', sep = "\t", header = None).values.T)
    counts_ctrl_mild.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_mild.txt', sep = "\t", header = None).values.T)
    counts_ctrl_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_ctrl_severe.txt', sep = "\t", header = None).values.T)
    counts_stim_healthy.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_healthy.txt', sep = "\t", header = None).values.T)
    counts_stim_mild.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_mild.txt', sep = "\t", header = None).values.T)
    counts_stim_severe.append(pd.read_csv(data_dir + f'GxC{batch_id + 1}_stim_severe.txt', sep = "\t", header = None).values.T)

    anno = pd.read_csv(data_dir + f'cell_label{batch_id + 1}.txt', sep = "\t", index_col = 0).values.squeeze()
    # annotation labels
    label_annos.append(np.array([('cell type '+str(i)) for i in anno]))
    # batch labels
    label_batches.append(np.array(['batch ' + str(batch_id)] * counts_ctrl_healthy[-1].shape[0]))

# select the dataset
label_cond1 = ["healthy"] * counts_ctrl_healthy[0].shape[0] + ["mild"] * counts_ctrl_healthy[1].shape[0] + ["severe"] * counts_ctrl_healthy[2].shape[0] \
    + ["healthy"] * counts_ctrl_healthy[3].shape[0] + ["mild"] * counts_ctrl_healthy[4].shape[0] + ["severe"] * counts_ctrl_healthy[5].shape[0]

label_cond2 = ["ctrl"] * (counts_ctrl_healthy[0].shape[0] + counts_ctrl_healthy[1].shape[0] + counts_ctrl_healthy[2].shape[0]) \
    + ["stim"] * (counts_ctrl_healthy[3].shape[0] + counts_ctrl_healthy[4].shape[0] + counts_ctrl_healthy[5].shape[0])

counts = [counts_ctrl_healthy[0], counts_ctrl_mild[1], counts_ctrl_severe[2], counts_stim_healthy[3], counts_stim_mild[4], counts_stim_severe[5]]

# In[]
# Train with ctrl in batches 1 & 2, stim1 in batches 3 & 4, stim2 in batches 5 & 6
cond1_ids, cond1_names = pd.factorize(np.array(label_cond1))
cond2_ids, cond2_names = pd.factorize(np.array(label_cond2))
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
                                        diff_labels = [diff_labels_batch[0][train_idx], diff_labels_batch[1][train_idx]], 
                                        batch_id = batch_ids_batch[train_idx])

    dataset_test = scdisinfact.dataset(counts = count_batch[test_idx,:], 
                                        anno = anno_batch[test_idx], 
                                        diff_labels = [diff_labels_batch[0][test_idx], diff_labels_batch[1][test_idx]], 
                                        batch_id = batch_ids_batch[test_idx])

    datasets_train.append(dataset_train)
    datasets_test.append(dataset_test)
    


# In[] training the model
# TODO: track the time usage and memory usage
import importlib 
importlib.reload(scdisinfact)
start_time = time.time()
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_gl = 1
reg_tc = 0.5
reg_class = 1
reg_kl = 1e-6
# mmd, cross_entropy, total correlation, group_lasso, kl divergence, 
lambs = [reg_mmd_comm, reg_mmd_diff, reg_class, reg_gl, reg_tc, reg_kl]
Ks = [8, 4, 4]
nepochs = 50
interval = 10
print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))
model = scdisinfact.scdisinfact(datasets = datasets_train, Ks = Ks, batch_size = 64, interval = interval, lr = 1e-3, 
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

# In[] Plot train results
z_cs = []
z_ds = []
zs = []
d_preds = []
targets = []

loss_class = 0
loss_contr = 0
loss_recon = 0
ce_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
contr_loss = scdisinfact.loss_func.CircleLoss(m=0.25, gamma=80)

with torch.no_grad():
    for batch_id, dataset in enumerate(datasets_train):
        d_preds.append([])
        targets.append([])
        z_ds.append([])

        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]        
        for diff_factor in range(model.n_diff_factors):
            # check classification accuracy
            d_pred = model.classifiers[diff_factor](z_d[diff_factor])
            target = dataset.diff_labels[diff_factor].to(model.device)            

            z_ds[-1].append(z_d[diff_factor].cpu().detach().numpy())
            d_preds[-1].append(d_pred)
            targets[-1].append(target)

        mu, pi, theta = model.Dec(torch.concat([z_c] + [torch.tensor(x).to(model.device) for x in z_ds[-1]], dim = 1), dataset.batch_id[:,None].to(model.device))

        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

        loss_recon += scdisinfact.loss_func.NB(theta = theta, scale_factor = dataset.size_factor.to(model.device), device = model.device).loss(y_true = dataset.counts.to(model.device), y_pred = mu)

    for diff_factor in range(model.n_diff_factors):
        loss_class += ce_loss(input = torch.cat([d_pred[diff_factor] for d_pred in d_preds], dim = 0), target = torch.cat([target[diff_factor] for target in targets], dim = 0))
        loss_contr += contr_loss(torch.nn.functional.normalize(torch.cat([torch.tensor(z_d[diff_factor], device = model.device) for z_d in z_ds], dim = 0)), torch.cat([target[diff_factor] for target in targets], dim = 0))
    print("loss classification on train dataset: {:.5f}".format(loss_class.item()))
    print("loss contrastive on train dataset: {:.5f}".format(loss_contr.item()))
    print("loss likelihood on train dataset: {:.5f}".format(loss_recon.item()))  

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f"train_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = [np.take(anno_names, dataset.anno) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(anno_names, dataset.anno) for dataset in datasets_train], mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"common_celltypes_sep.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(batch_names, dataset.batch_id) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(cond1_names, dataset.diff_labels[0]) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_cond1.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(cond2_names, dataset.diff_labels[1]) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_cond2.png" if result_dir else None, markerscale = 6, s = 5)

utils.plot_latent(zs = z_ds_umaps[0], annos = [np.take(anno_names, dataset.anno) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [np.take(batch_names, dataset.batch_id) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [np.take(cond1_names, dataset.diff_labels[0]) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_cond1.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [np.take(anno_names, dataset.anno) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [np.take(batch_names, dataset.batch_id) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [np.take(cond2_names, dataset.diff_labels[1]) for dataset in datasets_train], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_cond2.png" if result_dir else None, markerscale = 6, s = 5)


# In[] Plot test results
z_cs = []
z_ds = []
zs = []
d_preds = []
targets = []

loss_class = 0
loss_contr = 0
loss_recon = 0
ce_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
contr_loss = scdisinfact.loss_func.CircleLoss(m=0.25, gamma=80)

with torch.no_grad():
    for batch_id, dataset in enumerate(datasets_test):
        d_preds.append([])
        targets.append([])
        z_ds.append([])

        # pass through the encoders
        dict_inf = model.inference(counts = dataset.counts_stand.to(model.device), batch_ids = dataset.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        z = torch.cat([z_c] + z_d, dim = 1)
        mu = dict_gen["mu"]    
        for diff_factor in range(model.n_diff_factors):
            # check classification accuracy
            d_pred = model.classifiers[diff_factor](z_d[diff_factor])
            target = dataset.diff_labels[diff_factor].to(model.device)            

            z_ds[-1].append(z_d[diff_factor].cpu().detach().numpy())
            d_preds[-1].append(d_pred)
            targets[-1].append(target)

        mu, pi, theta = model.Dec(torch.concat([z_c] + [torch.tensor(x).to(model.device) for x in z_ds[-1]], dim = 1), dataset.batch_id[:,None].to(model.device))

        z_cs.append(z_c.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))

        loss_recon += scdisinfact.loss_func.NB(theta = theta, scale_factor = dataset.size_factor.to(model.device), device = model.device).loss(y_true = dataset.counts.to(model.device), y_pred = mu)

    for diff_factor in range(model.n_diff_factors):
        loss_class += ce_loss(input = torch.cat([d_pred[diff_factor] for d_pred in d_preds], dim = 0), target = torch.cat([target[diff_factor] for target in targets], dim = 0))
        loss_contr += contr_loss(torch.nn.functional.normalize(torch.cat([torch.tensor(z_d[diff_factor], device = model.device) for z_d in z_ds], dim = 0)), torch.cat([target[diff_factor] for target in targets], dim = 0))
    print("loss classification on test dataset: {:.5f}".format(loss_class.item()))
    print("loss contrastive on test dataset: {:.5f}".format(loss_contr.item()))
    print("loss likelihood on test dataset: {:.5f}".format(loss_recon.item()))  

        
# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f"test_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = [np.take(anno_names, dataset.anno) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(anno_names, dataset.anno) for dataset in datasets_test], mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"common_celltypes_sep.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(batch_names, dataset.batch_id) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(cond1_names, dataset.diff_labels[0]) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_cond1.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = [np.take(cond2_names, dataset.diff_labels[1]) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_cond2.png" if result_dir else None, markerscale = 6, s = 5)

utils.plot_latent(zs = z_ds_umaps[0], annos = [np.take(anno_names, dataset.anno) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [np.take(batch_names, dataset.batch_id) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = [np.take(cond1_names, dataset.diff_labels[0]) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_cond1.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [np.take(anno_names, dataset.anno) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [np.take(batch_names, dataset.batch_id) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = [np.take(cond2_names, dataset.diff_labels[1]) for dataset in datasets_test], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_cond2.png" if result_dir else None, markerscale = 6, s = 5)

# In[] Plot joint results
z_cs = []
z_ds = []
zs = []

label_annos = []
label_batches = []
label_cond1 = []
label_cond2 = []
label_train_test = []

for batch_id, (dataset_train, dataset_test) in enumerate(zip(datasets_train, datasets_test)):
    with torch.no_grad():
        # pass through the encoders
        dict_inf = model.inference(counts = dataset_train.counts_stand.to(model.device), batch_ids = dataset_train.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset_train.batch_id[:,None].to(model.device))
        z_c_train = dict_inf["mu_c"]
        z_d_train = dict_inf["mu_d"]
        z = torch.cat([z_c_train] + z_d_train, dim = 1)
        mu = dict_gen["mu"]

        z_ds.append([x.cpu().detach().numpy() for x in z_d_train])
        z_cs.append(z_c_train.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))
        label_annos.append(np.take(anno_names, dataset_train.anno))
        label_batches.append(np.take(batch_names, dataset_train.batch_id))
        label_cond1.append(np.take(cond1_names, dataset_train.diff_labels[0]))
        label_cond2.append(np.take(cond2_names, dataset_train.diff_labels[1]))
        label_train_test.append(np.array(len(dataset_train) * ["train"]))

        # pass through the encoders
        dict_inf = model.inference(counts = dataset_test.counts_stand.to(model.device), batch_ids = dataset_test.batch_id[:,None].to(model.device), print_stat = True, eval_model = True)
        # pass through the decoder
        dict_gen = model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset_test.batch_id[:,None].to(model.device))
        z_c_test = dict_inf["mu_c"]
        z_d_test = dict_inf["mu_d"]
        z = torch.cat([z_c_test] + z_d_test, dim = 1)
        mu = dict_gen["mu"]
        z_ds.append([x.cpu().detach().numpy() for x in z_d_test])
        z_cs.append(z_c_test.cpu().detach().numpy())
        zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis = 1))
        label_annos.append(np.take(anno_names, dataset_test.anno))
        label_batches.append(np.take(batch_names, dataset_test.batch_id))
        label_cond1.append(np.take(cond1_names, dataset_test.diff_labels[0]))
        label_cond2.append(np.take(cond2_names, dataset_test.diff_labels[1]))
        label_train_test.append(np.array(len(dataset_test) * ["test"]))

# UMAP
umap_op = UMAP(min_dist = 0.1, random_state = 0)
z_cs_umap = umap_op.fit_transform(np.concatenate(z_cs, axis = 0))
z_ds_umap = []
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[0] for z_d in z_ds], axis = 0)))
z_ds_umap.append(umap_op.fit_transform(np.concatenate([z_d[1] for z_d in z_ds], axis = 0)))
zs_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

z_ds_umaps = [[], []]
z_cs_umaps = []
zs_umaps = []
for batch in range(n_batches * 2):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

    elif batch == (2 * n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:,:])
        zs_umaps.append(zs_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        z_ds_umaps[0].append(z_ds_umap[0][start_pointer:end_pointer,:])
        z_ds_umaps[1].append(z_ds_umap[1][start_pointer:end_pointer,:])
        z_cs_umaps.append(z_cs_umap[start_pointer:end_pointer,:])
        zs_umaps.append(zs_umap[start_pointer:end_pointer,:])

comment = f"joint_{Ks}_{lambs}/"
if not os.path.exists(result_dir + comment):
    os.makedirs(result_dir + comment)

utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_celltypes.png" if result_dir else None , markerscale = 6, s = 5)
# utils.plot_latent(zs = z_cs_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"common_celltypes_sep.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_batches.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_cond1, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_cond1.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_cond2, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_cond2.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_cs_umaps, annos = label_train_test, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"common_train_test.png" if result_dir else None, markerscale = 6, s = 3, alpha = 0.9)
utils.plot_latent(zs = [np.concatenate([z_cs_umaps[x] for x in range(0, len(z_cs_umaps), 2)], axis = 0), np.concatenate([z_cs_umaps[x] for x in range(1, len(z_cs_umaps), 2)], axis = 0)], 
                  annos = [np.concatenate([label_annos[x] for x in range(0, len(label_annos), 2)]), np.concatenate([label_annos[x] for x in range(1, len(label_annos), 2)])], 
                  mode = "separate", axis_label = "UMAP", figsize = (10,10), save = result_dir + comment+"common_celltype_sep.png" if result_dir else None, markerscale = 6, s = 3, alpha = 0.9)

utils.plot_latent(zs = z_ds_umaps[0], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_cond1, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_cond1.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[0], annos = label_train_test, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff1_train_test.png" if result_dir else None, markerscale = 6, s = 3, alpha = 0.9)
utils.plot_latent(zs = [np.concatenate([z_ds_umaps[0][x] for x in range(0, len(z_ds_umaps[0]), 2)], axis = 0), np.concatenate([z_ds_umaps[0][x] for x in range(1, len(z_ds_umaps[0]), 2)], axis = 0)], 
                  annos = [np.concatenate([label_cond1[x] for x in range(0, len(label_cond1), 2)]), np.concatenate([label_cond1[x] for x in range(1, len(label_cond1), 2)])], 
                  mode = "separate", axis_label = "UMAP", figsize = (10,10), save = result_dir + comment+"diff1_cond1_sep.png" if result_dir else None, markerscale = 6, s = 3, alpha = 0.9)

utils.plot_latent(zs = z_ds_umaps[1], annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_celltypes.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_batch.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = label_cond2, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_cond2.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = z_ds_umaps[1], annos = label_train_test, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"diff2_train_test.png" if result_dir else None, markerscale = 6, s = 3, alpha = 0.9)
utils.plot_latent(zs = [np.concatenate([z_ds_umaps[1][x] for x in range(0, len(z_ds_umaps[1]), 2)], axis = 0), np.concatenate([z_ds_umaps[1][x] for x in range(1, len(z_ds_umaps[1]), 2)], axis = 0)], 
                  annos = [np.concatenate([label_cond2[x] for x in range(0, len(label_cond2), 2)]), np.concatenate([label_cond2[x] for x in range(1, len(label_cond2), 2)])], 
                  mode = "separate", axis_label = "UMAP", figsize = (10,10), save = result_dir + comment+"diff2_cond2_sep.png" if result_dir else None, markerscale = 6, s = 3, alpha = 0.9)


# In[] 


# %%
