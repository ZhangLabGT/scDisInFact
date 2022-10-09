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
# sigma = 0.4
# n_diff_genes = 20
# diff = 8
# ngenes = 500
# ncells_total = 10000 
# n_batches = 6
# data_dir = f"../data/simulated/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
# result_dir = f"./simulated/prediction/1condition_{ncells_total}_{ngenes}_{sigma}_{n_diff_genes}_{diff}/"
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

data_dir = f"../data/simulated/" + sys.argv[1] + "/"
# lsa performs the best
result_dir = f"./simulated/prediction/" + sys.argv[1] + "/"
n_diff_genes = eval(sys.argv[1].split("_")[4])
n_batches = 6
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



'''
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

utils.plot_latent(x_umaps, annos = label_annos, mode = "modality", save = save_file, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent(x_umaps, annos = label_conditions, mode = "joint", save = save_file, figsize = (15,10), axis_label = "UMAP", markerscale = 6)
'''


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
# # train_joint is more efficient, but does not work as well compared to train
# model.train()
# losses = model.train_model(nepochs = nepochs, recon_loss = "NB", reg_contr = 0.01)

# end_time = time.time()
# print("time cost: {:.2f}".format(end_time - start_time))

# torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}.pth")
model.load_state_dict(torch.load(result_dir + f"model_{Ks}_{lambs}.pth", map_location = device))

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

# In[] Extract predicted matrix of scGEN and scDisInFact
plt.rcParams["font.size"] = 15
np.random.seed(0)

#-------------------------------------------------------------------------------------------------------------------
#
# scGEN
#
#-------------------------------------------------------------------------------------------------------------------
result_scgen = result_dir + "scGEN_scenario1/"
X_scgen_imputes = []
for batch_id in [2,3,4,5]:
    X_scgen_imputes.append(np.loadtxt(result_scgen + f"GxC{batch_id + 1}_ctrl_impute.txt").T)
    # make non-negative
    X_scgen_imputes[-1] = X_scgen_imputes[-1] * (X_scgen_imputes[-1] > 0)

#-------------------------------------------------------------------------------------------------------------------
#
# scPreGAN
#
#-------------------------------------------------------------------------------------------------------------------
result_scpregan = result_dir + "scPreGAN_scenario1/"
X_scpregan_imputes = []
for batch_id in [2,3,4,5]:
    X_scpregan_imputes.append(np.loadtxt(result_scpregan + f"GxC{batch_id + 1}_ctrl_impute.txt").T)
    # make non-negative
    X_scpregan_imputes[-1] = X_scpregan_imputes[-1] * (X_scpregan_imputes[-1] > 0)

#-------------------------------------------------------------------------------------------------------------------
#
# scDisInFact
#
#-------------------------------------------------------------------------------------------------------------------
z_cs = []
z_ds = []
zs = []
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


# NOTE: impute control for batches 3, 4, 5, 6, only removed of conditions
# control, batch 0 and 1
mean_ctrl = np.mean(np.concatenate([x[0] for x in z_ds[0:2]], axis = 0), axis = 0)[None,:]
# stim1, batch 2 and 3
mean_stim1 = np.mean(np.concatenate([x[0] for x in z_ds[2:4]], axis = 0), axis = 0)[None,:]
# stim2, batch 4 and 5
mean_stim2 = np.mean(np.concatenate([x[0] for x in z_ds[4:]], axis = 0), axis = 0)[None,:]

z_ds_ctrl = np.concatenate([x[0] for x in z_ds[0:2]], axis = 0)
# find the clostest
# mean_ctrl = z_ds_ctrl[[np.argmin(np.sum((z_ds_ctrl - mean_ctrl) ** 2, axis = 1))], :]

change_stim1_ctrl = mean_ctrl - mean_stim1
change_stim2_ctrl = mean_ctrl - mean_stim2


X_scdisinfact_imputes = []
zs = []
z_ds_impute = []
with torch.no_grad():
    for batch_id, dataset in enumerate(datasets):
        #-------------------------------------------------------------------------------------------------------------------
        #
        # Remove both batch effect and condition effect (Imputation of count matrices under the control condition)
        #
        #-------------------------------------------------------------------------------------------------------------------
        
        # removed of batch effect
        ref_batch = 0.0
        # still use the original batch_id as input, not change the latent embedding
        z_c, _ = model.Enc_c(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
        if batch_id in [0,1]:
            z_d = []
            for Enc_d in model.Enc_ds:
                # still use the original batch_id as input, not change the latent embedding
                _z_d, _ = Enc_d(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
                z_d.append(_z_d)        

        elif batch_id in [2,3,4,5]:
            # NOTE: removed of condition effect
            # manner one, use mean
            # z_d = [torch.tensor(mean_ctrl, device = model.device).expand(dataset.counts_stand.shape[0], Ks[1])]
            
            # manner two, random sample, better than mean
            # idx = np.random.choice(z_ds_ctrl.shape[0], z_c.shape[0], replace = True)
            # z_d = [torch.tensor(z_ds_ctrl[idx,:], device = model.device)]
            
            # manner three, latent space arithmetics
            if batch_id in [2,3]:
                z_d = z_ds[batch_id][0] + change_stim1_ctrl
            else:
                z_d = z_ds[batch_id][0] + change_stim2_ctrl 
            z_d = [torch.tensor(z_d, device = model.device)]

        z_ds_impute.append(z_d[0].detach().cpu().numpy())
        # change the batch_id into ref batch as input
        z = torch.concat([z_c] + z_d + [torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None]], axis = 1)        
        zs.append(z.cpu().detach().numpy())

        # NOTE: change the batch_id into ref batch as input, change the diff condition into control
        mu_impute, _, _ = model.Dec(torch.concat([z_c] + z_d, dim = 1), torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None])
        mu_impute = mu_impute.cpu().detach().numpy() 

        X_scdisinfact_imputes.append(mu_impute)
        

# In[] Compare with baseline
# -------------------------------------------------------------------------------
#
# cell-specific scores with ground truth
#
# -------------------------------------------------------------------------------
mses_scdisinfact = []
mses_diff_scdisinfact = []
pearsons_scdisinfact = []
r2_scdisinfact = []
r2_diff_scdisinfact = []

mses_scgen = []
mses_diff_scgen = []
pearsons_scgen = []
r2_scgen = []
r2_diff_scgen = []

mses_scpregan = []
mses_diff_scpregan = []
pearsons_scpregan = []
r2_scpregan = []
r2_diff_scpregan = []

for i, batch_id in enumerate([2, 3, 4, 5]):
    # load impute and ground truth matrices
    X_scdisinfact_impute = X_scdisinfact_imputes[batch_id]
    X_scgen_impute = X_scgen_imputes[i]
    X_scpregan_impute = X_scpregan_imputes[i]
    X_gt = counts_gt[batch_id]
    
    # normalize the counts
    X_scdisinfact_impute_norm = X_scdisinfact_impute/np.sum(X_scdisinfact_impute, axis = 1, keepdims = True)
    X_scgen_impute_norm = X_scgen_impute/np.sum(X_scgen_impute, axis = 1, keepdims = True)
    X_scpregan_impute_norm = X_scpregan_impute/np.sum(X_scpregan_impute, axis = 1, keepdims = True)
    X_gt_norm = X_gt/np.sum(X_gt, axis = 1, keepdims = True)

    # cell-specific normalized MSE
    mses_scdisinfact.append(np.sum((X_scdisinfact_impute_norm - X_gt_norm) ** 2, axis = 1))
    # cell-specific normalized MSE on diff genes
    mses_diff_scdisinfact.append(np.sum((X_scdisinfact_impute_norm[:,:2*n_diff_genes] - X_gt_norm[:,:2*n_diff_genes]) ** 2, axis = 1))   
    # cell-specific pearson correlation
    pearsons_scdisinfact.append(np.array([stats.pearsonr(X_scdisinfact_impute_norm[i,:], X_gt_norm[i,:])[0] for i in range(X_gt_norm.shape[0])]))
    # cell-specific R2 score
    r2_scdisinfact.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scdisinfact_impute_norm, X_gt_norm)]))
    # cell-specific R2 diff score
    r2_diff_scdisinfact.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scdisinfact_impute_norm[:,:2*n_diff_genes], X_gt_norm[:,:2*n_diff_genes])]))

    # cell-specific normalized MSE
    mses_scgen.append(np.sum((X_scgen_impute_norm - X_gt_norm) ** 2, axis = 1))
    # cell-specific normalized MSE on diff genes
    mses_diff_scgen.append(np.sum((X_scgen_impute_norm[:,:2*n_diff_genes] - X_gt_norm[:,:2*n_diff_genes]) ** 2, axis = 1))   
    # cell-specific pearson correlation
    pearsons_scgen.append(np.array([stats.pearsonr(X_scgen_impute_norm[i,:], X_gt_norm[i,:])[0] for i in range(X_gt_norm.shape[0])]))
    # cell-specific R2 score
    r2_scgen.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scgen_impute_norm, X_gt_norm)]))
    # cell-specific R2 diff score
    r2_diff_scgen.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scgen_impute_norm[:,:2*n_diff_genes], X_gt_norm[:,:2*n_diff_genes])]))

    # cell-specific normalized MSE
    mses_scpregan.append(np.sum((X_scpregan_impute_norm - X_gt_norm) ** 2, axis = 1))
    # cell-specific normalized MSE on diff genes
    mses_diff_scpregan.append(np.sum((X_scpregan_impute_norm[:,:2*n_diff_genes] - X_gt_norm[:,:2*n_diff_genes]) ** 2, axis = 1))   
    # cell-specific pearson correlation
    pearsons_scpregan.append(np.array([stats.pearsonr(X_scpregan_impute_norm[i,:], X_gt_norm[i,:])[0] for i in range(X_gt_norm.shape[0])]))
    # cell-specific R2 score
    r2_scpregan.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scpregan_impute_norm, X_gt_norm)]))
    # cell-specific R2 diff score
    r2_diff_scpregan.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scpregan_impute_norm[:,:2*n_diff_genes], X_gt_norm[:,:2*n_diff_genes])]))

mses_scdisinfact = np.concatenate(mses_scdisinfact, axis = 0)
mses_diff_scdisinfact = np.concatenate(mses_diff_scdisinfact, axis = 0)
pearsons_scdisinfact = np.concatenate(pearsons_scdisinfact, axis = 0)
r2_scdisinfact = np.concatenate(r2_scdisinfact, axis = 0)
r2_diff_scdisinfact = np.concatenate(r2_diff_scdisinfact, axis = 0)

mses_scgen = np.concatenate(mses_scgen, axis = 0)
mses_diff_scgen = np.concatenate(mses_diff_scgen, axis = 0)
pearsons_scgen = np.concatenate(pearsons_scgen, axis = 0)
r2_scgen = np.concatenate(r2_scgen, axis = 0)
r2_diff_scgen = np.concatenate(r2_diff_scgen, axis = 0)

mses_scpregan = np.concatenate(mses_scpregan, axis = 0)
mses_diff_scpregan = np.concatenate(mses_diff_scpregan, axis = 0)
pearsons_scpregan = np.concatenate(pearsons_scpregan, axis = 0)
r2_scpregan = np.concatenate(r2_scpregan, axis = 0)
r2_diff_scpregan = np.concatenate(r2_diff_scpregan, axis = 0)

scores_cell_specific = pd.DataFrame(columns = ["MSE (normalized)", "MSE diff (normalized)", "Pearson", "R2", "R2 diff", "Method"])
scores_cell_specific["MSE (normalized)"] = np.concatenate((mses_scdisinfact, mses_scgen, mses_scpregan), axis = 0)
scores_cell_specific["MSE diff (normalized)"] = np.concatenate((mses_diff_scdisinfact, mses_diff_scgen, mses_diff_scpregan), axis = 0)
scores_cell_specific["Pearson"] = np.concatenate((pearsons_scdisinfact, pearsons_scgen, pearsons_scpregan), axis = 0)
scores_cell_specific["R2"] = np.concatenate((r2_scdisinfact, r2_scgen, r2_scpregan), axis = 0)
scores_cell_specific["R2 diff"] = np.concatenate((r2_diff_scdisinfact, r2_diff_scgen, r2_diff_scpregan), axis = 0)
scores_cell_specific["Method"] = np.array(["scDisInFact"] * mses_scdisinfact.shape[0] + ["scGEN"] * mses_scgen.shape[0] + ["scPreGAN"] * mses_scpregan.shape[0])

scores_cell_specific.to_csv(result_dir + "scores_cell_specific.csv")


# plot violin plot
fig = plt.figure(figsize = (23,5))
ax = fig.subplots(nrows = 1, ncols = 5)
sns.boxplot(data = scores_cell_specific, x = "Method", y = "MSE (normalized)", ax = ax[0])
sns.boxplot(data = scores_cell_specific, x = "Method", y = "MSE diff (normalized)", ax = ax[1])
sns.boxplot(data = scores_cell_specific, x = "Method", y = "Pearson", ax = ax[2])
sns.boxplot(data = scores_cell_specific, x = "Method", y = "R2", ax = ax[3])
sns.boxplot(data = scores_cell_specific, x = "Method", y = "R2 diff", ax = ax[4])
plt.tight_layout()
fig.savefig(result_dir + "mse.png", bbox_inches = "tight")


# In[]
# -------------------------------------------------------------------------------
#
# cell-type-specific scores with ground truth
#
# -------------------------------------------------------------------------------
# TODO: R^2 calculation
mses_scdisinfact_celltype = []
mses_diff_scdisinfact_celltype = []
pearsons_scdisinfact_celltype = []
r2_scdisinfact_celltype = []
r2_diff_scdisinfact_celltype = []

mses_scgen_celltype = []
mses_diff_scgen_celltype = []
pearsons_scgen_celltype = []
r2_scgen_celltype = []
r2_diff_scgen_celltype = []

mses_scpregan_celltype = []
mses_diff_scpregan_celltype = []
pearsons_scpregan_celltype = []
r2_scpregan_celltype = []
r2_diff_scpregan_celltype = []

for i, batch_id in enumerate([2, 3, 4, 5]):
    # load impute and ground truth matrices
    X_scdisinfact_impute = X_scdisinfact_imputes[batch_id]
    X_scgen_impute = X_scgen_imputes[i]
    X_scpregan_impute = X_scpregan_imputes[i]
    X_gt = counts_gt[batch_id]
    label_anno = label_annos[batch_id]
    
    # normalize the counts
    X_scdisinfact_impute_norm = X_scdisinfact_impute/np.sum(X_scdisinfact_impute, axis = 1, keepdims = True)
    X_scgen_impute_norm = X_scgen_impute/np.sum(X_scgen_impute, axis = 1, keepdims = True)
    X_scpregan_impute_norm = X_scpregan_impute/np.sum(X_scpregan_impute, axis = 1, keepdims = True)
    X_gt_norm = X_gt/np.sum(X_gt, axis = 1, keepdims = True)

    # calculate cell type averaged expressions
    uniq_celltype = np.unique(label_anno)
    X_scdisinfact_impute_mean = []
    X_scgen_impute_mean = []
    X_scpregan_impute_mean = []
    X_gt_mean = []
    for celltype in uniq_celltype:
        X_scdisinfact_impute_mean.append(np.mean(X_scdisinfact_impute_norm[label_anno == celltype,:], axis = 0, keepdims = True))
        X_scgen_impute_mean.append(np.mean(X_scgen_impute_norm[label_anno == celltype,:], axis = 0, keepdims = True))
        X_scpregan_impute_mean.append(np.mean(X_scpregan_impute_norm[label_anno == celltype,:], axis = 0, keepdims = True))
        X_gt_mean.append(np.mean(X_gt_norm[label_anno == celltype,:], axis = 0, keepdims = True))
    X_scdisinfact_impute_mean = np.concatenate(X_scdisinfact_impute_mean, axis = 0)
    X_scgen_impute_mean = np.concatenate(X_scgen_impute_mean, axis = 0)
    X_scpregan_impute_mean = np.concatenate(X_scpregan_impute_mean, axis = 0)
    X_gt_mean = np.concatenate(X_gt_mean, axis = 0)

    # cell-type-specific normalized MSE
    mses_scdisinfact_celltype.append(np.sum((X_scdisinfact_impute_mean - X_gt_mean) ** 2, axis = 1))
    # cell-type-specific normalized MSE on diff genes
    mses_diff_scdisinfact_celltype.append(np.sum((X_scdisinfact_impute_mean[:,:2*n_diff_genes] - X_gt_mean[:,:2*n_diff_genes]) ** 2, axis = 1))   
    # cell-type-specific pearson correlation
    pearsons_scdisinfact_celltype.append(np.array([stats.pearsonr(X_scdisinfact_impute_mean[i,:], X_gt_mean[i,:])[0] for i in range(X_gt_mean.shape[0])]))
    # cell-type-specific R2 score
    r2_scdisinfact_celltype.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scdisinfact_impute_mean, X_gt_mean)]))
    # cell-type-specific R2 score
    r2_diff_scdisinfact_celltype.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scdisinfact_impute_mean[:,:2*n_diff_genes], X_gt_mean[:,:2*n_diff_genes])]))

    # cell-type-specific normalized MSE
    mses_scgen_celltype.append(np.sum((X_scgen_impute_mean - X_gt_mean) ** 2, axis = 1))
    # cell-type-specific normalized MSE on diff genes
    mses_diff_scgen_celltype.append(np.sum((X_scgen_impute_mean[:,:2*n_diff_genes] - X_gt_mean[:,:2*n_diff_genes]) ** 2, axis = 1))   
    # cell-type-specific pearson correlation
    pearsons_scgen_celltype.append(np.array([stats.pearsonr(X_scgen_impute_mean[i,:], X_gt_mean[i,:])[0] for i in range(X_gt_mean.shape[0])]))
    # cell-specific R2 score
    r2_scgen_celltype.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scgen_impute_mean, X_gt_mean)]))
    # cell-specific R2 diff score
    r2_diff_scgen_celltype.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scgen_impute_mean[:,:2*n_diff_genes], X_gt_mean[:,:2*n_diff_genes])]))

    # cell-type-specific normalized MSE
    mses_scpregan_celltype.append(np.sum((X_scpregan_impute_mean - X_gt_mean) ** 2, axis = 1))
    # cell-type-specific normalized MSE on diff genes
    mses_diff_scpregan_celltype.append(np.sum((X_scpregan_impute_mean[:,:2*n_diff_genes] - X_gt_mean[:,:2*n_diff_genes]) ** 2, axis = 1))   
    # cell-type-specific pearson correlation
    pearsons_scpregan_celltype.append(np.array([stats.pearsonr(X_scpregan_impute_mean[i,:], X_gt_mean[i,:])[0] for i in range(X_gt_mean.shape[0])]))
    # cell-specific R2 score
    r2_scpregan_celltype.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scpregan_impute_mean, X_gt_mean)]))
    # cell-specific R2 diff score
    r2_diff_scpregan_celltype.append(np.array([r2_score(y_pred = x, y_true = y) for x, y in zip(X_scpregan_impute_mean[:,:2*n_diff_genes], X_gt_mean[:,:2*n_diff_genes])]))

mses_scdisinfact_celltype = np.concatenate(mses_scdisinfact_celltype, axis = 0)
mses_diff_scdisinfact_celltype = np.concatenate(mses_diff_scdisinfact_celltype, axis = 0)
pearsons_scdisinfact_celltype = np.concatenate(pearsons_scdisinfact_celltype, axis = 0)
r2_scdisinfact_celltype = np.concatenate(r2_scdisinfact_celltype, axis = 0)
r2_diff_scdisinfact_celltype = np.concatenate(r2_diff_scdisinfact_celltype, axis = 0)

mses_scgen_celltype = np.concatenate(mses_scgen_celltype, axis = 0)
mses_diff_scgen_celltype = np.concatenate(mses_diff_scgen_celltype, axis = 0)
pearsons_scgen_celltype = np.concatenate(pearsons_scgen_celltype, axis = 0)
r2_scgen_celltype = np.concatenate(r2_scgen_celltype, axis = 0)
r2_diff_scgen_celltype = np.concatenate(r2_diff_scgen_celltype, axis = 0)

mses_scpregan_celltype = np.concatenate(mses_scpregan_celltype, axis = 0)
mses_diff_scpregan_celltype = np.concatenate(mses_diff_scpregan_celltype, axis = 0)
pearsons_scpregan_celltype = np.concatenate(pearsons_scpregan_celltype, axis = 0)
r2_scpregan_celltype = np.concatenate(r2_scpregan_celltype, axis = 0)
r2_diff_scpregan_celltype = np.concatenate(r2_diff_scpregan_celltype, axis = 0)

scores_cluster_specific = pd.DataFrame(columns = ["MSE (normalized)", "MSE diff (normalized)", "Pearson", "R2", "R2 diff", "Method"])
scores_cluster_specific["MSE (normalized)"] = np.concatenate((mses_scdisinfact_celltype, mses_scgen_celltype, mses_scpregan_celltype), axis = 0)
scores_cluster_specific["MSE diff (normalized)"] = np.concatenate((mses_diff_scdisinfact_celltype, mses_diff_scgen_celltype, mses_diff_scpregan_celltype), axis = 0)
scores_cluster_specific["Pearson"] = np.concatenate((pearsons_scdisinfact_celltype, pearsons_scgen_celltype, pearsons_scpregan_celltype), axis = 0)
scores_cluster_specific["R2"] = np.concatenate((r2_scdisinfact_celltype, r2_scgen_celltype, r2_scpregan_celltype), axis = 0)
scores_cluster_specific["R2 diff"] = np.concatenate((r2_diff_scdisinfact_celltype, r2_diff_scgen_celltype, r2_diff_scpregan_celltype), axis = 0)
scores_cluster_specific["Method"] = np.array(["scDisInFact"] * mses_scdisinfact_celltype.shape[0] + ["scGEN"] * mses_scgen_celltype.shape[0] + ["scPreGAN"] * mses_scgen_celltype.shape[0])

scores_cluster_specific.to_csv(result_dir + "scores_cluster_specific.csv")

# plot violin plot
fig = plt.figure(figsize = (23,5))
ax = fig.subplots(nrows = 1, ncols = 5)
sns.boxplot(data = scores_cluster_specific, x = "Method", y = "MSE (normalized)", ax = ax[0])
sns.boxplot(data = scores_cluster_specific, x = "Method", y = "MSE diff (normalized)", ax = ax[1])
sns.boxplot(data = scores_cluster_specific, x = "Method", y = "Pearson", ax = ax[2])
sns.boxplot(data = scores_cluster_specific, x = "Method", y = "R2", ax = ax[3])
sns.boxplot(data = scores_cluster_specific, x = "Method", y = "R2 diff", ax = ax[4])
plt.tight_layout()
fig.savefig(result_dir + "mse_cluster.png", bbox_inches = "tight")

# In[] Calculate batch mixing score
labels = np.concatenate(label_annos[2:], axis = 0)

X_scdisinfact_impute = np.concatenate(X_scdisinfact_imputes[2:], axis = 0)
# normalization
X_scdisinfact_impute_norm = X_scdisinfact_impute/(np.sum(X_scdisinfact_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scdisinfact_impute_norm = np.log1p(X_scdisinfact_impute_norm)
x_pca_scdisinfact = PCA(n_components = 100).fit_transform(X_scdisinfact_impute_norm)

X_scgen_impute = np.concatenate(X_scgen_imputes, axis = 0)
# normalization
X_scgen_impute_norm = X_scgen_impute/(np.sum(X_scgen_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scgen_impute_norm = np.log1p(X_scgen_impute_norm)
x_pca_scgen = PCA(n_components = 100).fit_transform(X_scgen_impute_norm)

X_scpregan_impute = np.concatenate(X_scpregan_imputes, axis = 0)
# normalization
X_scpregan_impute_norm = X_scpregan_impute/(np.sum(X_scpregan_impute, axis = 1, keepdims = True) + 1e-6) * 100
X_scpregan_impute_norm = np.log1p(X_scpregan_impute_norm)
x_pca_scpregan = PCA(n_components = 100).fit_transform(X_scpregan_impute_norm)

n_neighbors = 30
# graph connectivity score, check the batch mixing of the imputated gene expression data
gc_scdisinfact = bmk.graph_connectivity(X = x_pca_scdisinfact, groups = labels, k = n_neighbors)
print('GC (scDisInFact): {:.3f}'.format(gc_scdisinfact))
gc_scgen = bmk.graph_connectivity(X = x_pca_scgen, groups = labels, k = n_neighbors)
print('GC (scGEN): {:.3f}'.format(gc_scgen))
gc_scpregan = bmk.graph_connectivity(X = x_pca_scpregan, groups = labels, k = n_neighbors)
print('GC (scPreGAN): {:.3f}'.format(gc_scpregan))

# ARI score, check the separation of cell types? still needed?
nmi_scdisinfact = []
ari_scdisinfact = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scdisinfact = utils.leiden_cluster(X = x_pca_scdisinfact, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scdisinfact.append(bmk.nmi(group1 = labels, group2 = leiden_labels_scdisinfact))
    ari_scdisinfact.append(bmk.ari(group1 = labels, group2 = leiden_labels_scdisinfact))
print('NMI (scDisInFact): {:.3f}'.format(max(nmi_scdisinfact)))
print('ARI (scDisInFact): {:.3f}'.format(max(ari_scdisinfact)))

nmi_scgen = []
ari_scgen = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scgen = utils.leiden_cluster(X = x_pca_scgen, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scgen.append(bmk.nmi(group1 = labels, group2 = leiden_labels_scgen))
    ari_scgen.append(bmk.ari(group1 = labels, group2 = leiden_labels_scgen))
print('NMI (scGEN): {:.3f}'.format(max(nmi_scgen)))
print('ARI (scGEN): {:.3f}'.format(max(ari_scgen)))

nmi_scpregan = []
ari_scpregan = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scpregan = utils.leiden_cluster(X = x_pca_scpregan, knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scpregan.append(bmk.nmi(group1 = labels, group2 = leiden_labels_scpregan))
    ari_scpregan.append(bmk.ari(group1 = labels, group2 = leiden_labels_scpregan))
print('NMI (scPreGAN): {:.3f}'.format(max(nmi_scpregan)))
print('ARI (scPreGAN): {:.3f}'.format(max(ari_scpregan)))

# scores
scores = pd.DataFrame(columns = ["GC", "NMI", "ARI", "methods"])
scores["NMI"] = np.array([np.max(nmi_scdisinfact), np.max(nmi_scgen), np.max(nmi_scpregan)])
scores["ARI"] = np.array([np.max(ari_scdisinfact), np.max(ari_scgen), np.max(nmi_scpregan)])
scores["methods"] = np.array(["scDisInFact", "scGEN", "scPreGAN"])
scores["GC"] = np.array([gc_scdisinfact, gc_scgen, gc_scpregan])
scores.to_csv(result_dir + comment + "batch_mixing_scdisinfact.csv")


# In[] Visualize the impute matrix (scDisInFact)
plt.rcParams["font.size"] = 10
# NOTE: Impute, removed of both conditions and batch effect

# z_ds_impute = np.concatenate(z_ds_impute, axis = 0)
# z_ds_impute_umap = umap_op.fit_transform(z_ds_impute)


# UMAP visualization
pca_op = PCA(n_components = 30)
umap_op = UMAP(min_dist = 0.4, random_state = 0, n_neighbors = 30)
x_umap = umap_op.fit_transform(pca_op.fit_transform(X_scdisinfact_impute_norm))
# z_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

x_umaps = []
# z_umaps = []
# z_ds_impute_umaps = []
for batch in range(4):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + X_scgen_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        # z_umaps.append(z_umap[start_pointer:end_pointer,:])
        # z_ds_impute_umaps.append(z_ds_impute_umap[start_pointer:end_pointer,:])

    elif batch == (4 - 1):
        start_pointer = start_pointer + X_scgen_imputes[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        # z_umaps.append(z_umap[start_pointer:,:])
        # z_ds_impute_umaps.append(z_ds_impute_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + X_scgen_imputes[batch - 1].shape[0]
        end_pointer = start_pointer + X_scgen_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        # z_umaps.append(z_umap[start_pointer:end_pointer,:])
        # z_ds_impute_umaps.append(z_ds_impute_umap[start_pointer:end_pointer,:])

# # NOTE: Full latent space, the latent space should mix between batches to make sure that the output of the decoder to be similar between batches
# utils.plot_latent(zs = z_umaps, annos = label_annos, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"latent_celltypes_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
# utils.plot_latent(zs = z_umaps, annos = label_annos, mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"latent_celltypes_sep_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
# utils.plot_latent(zs = z_umaps, annos = label_batches, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"latent_batches_ctrl.png" if result_dir else None, markerscale = 6, s = 5)
# utils.plot_latent(zs = z_umaps, annos = label_conditions, mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"latent_condition_ctrl.png" if result_dir else None, markerscale = 6, s = 5)

# # NOTE: visualize only the diff dimension, the diff dimension should mix between batches for correct prediction
# utils.plot_latent(zs = z_ds_impute_umaps, annos = label_conditions, mode = "separate", axis_label = "UMAP", figsize = (5,13), save = None, markerscale = 6, s = 3)

# NOTE: visualize the imputed results
utils.plot_latent(zs = x_umaps, annos = label_annos[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_celltypes_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_annos[2:], mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"impute_celltypes_sep_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_batches[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_batches_ctrl.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_conditions[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_condition_ctrl.png" if result_dir else None, markerscale = 6, s = 5)


# In[] Visualize the impute matrix (scGEN)

# UMAP visualization
pca_op = PCA(n_components = 30)
umap_op = UMAP(min_dist = 0.4, random_state = 0, n_neighbors = 30)
x_umap = umap_op.fit_transform(pca_op.fit_transform(X_scgen_impute_norm))

x_umaps = []
for batch in range(4):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + X_scgen_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

    elif batch == (4 - 1):
        start_pointer = start_pointer + X_scgen_imputes[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + X_scgen_imputes[batch - 1].shape[0]
        end_pointer = start_pointer + X_scgen_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

comment = "scGEN_scenario1/"
# NOTE: visualize the imputed results
utils.plot_latent(zs = x_umaps, annos = label_annos[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_celltypes_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_annos[2:], mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"impute_celltypes_sep_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_batches[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_batches_ctrl.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_conditions[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_condition_ctrl.png" if result_dir else None, markerscale = 6, s = 5)

# In[] Visualize the impute matrix (scPreGAN)

# UMAP visualization
pca_op = PCA(n_components = 30)
umap_op = UMAP(min_dist = 0.4, random_state = 0, n_neighbors = 30)
x_umap = umap_op.fit_transform(pca_op.fit_transform(X_scpregan_impute_norm))

x_umaps = []
for batch in range(4):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + X_scpregan_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

    elif batch == (4 - 1):
        start_pointer = start_pointer + X_scpregan_imputes[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])

    else:
        start_pointer = start_pointer + X_scpregan_imputes[batch - 1].shape[0]
        end_pointer = start_pointer + X_scpregan_imputes[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

comment = "scPreGAN_scenario1/"
# NOTE: visualize the imputed results
utils.plot_latent(zs = x_umaps, annos = label_annos[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_celltypes_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_annos[2:], mode = "separate", axis_label = "UMAP", figsize = (10,20), save = result_dir + comment+"impute_celltypes_sep_ctrl.png" if result_dir else None , markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_batches[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_batches_ctrl.png" if result_dir else None, markerscale = 6, s = 5)
utils.plot_latent(zs = x_umaps, annos = label_conditions[2:], mode = "joint", axis_label = "UMAP", figsize = (10,5), save = result_dir + comment+"impute_condition_ctrl.png" if result_dir else None, markerscale = 6, s = 5)

# In[]
if True:
    # TODO: batch mixing (for batch 1, 2 there is no reference in scGEN)
    plt.rcParams["font.size"] = 20
    scores_cell = pd.DataFrame(columns = ["MSE (normalized)", "MSE diff (normalized)", "Pearson", "R2", "R2 diff", "Method"])
    scores_cluster = pd.DataFrame(columns = ["MSE (normalized)", "MSE diff (normalized)", "Pearson", "R2", "R2 diff", "Method"]) 
    status = []
    noise_levels = []
    ndiff_genes = []
    ndiffs = []
    result_dir = "./simulated/prediction/"
    # for dataset in ["0.2_20_2", "0.2_50_2", "0.2_100_2", "0.3_20_2", "0.3_50_2", "0.3_100_2", "0.4_20_2", "0.4_50_2", "0.4_100_2","0.2_20_4", "0.2_20_8", "0.3_20_4", "0.3_20_8", "0.4_20_4", "0.4_20_8"]:
    for dataset in ["0.2_20_2", "0.2_50_2", "0.2_100_2", "0.2_20_4", "0.2_50_4", "0.2_100_4", "0.2_20_8", "0.2_50_8", "0.2_100_8"]:
        noise_level, ndiff_gene, ndiff = dataset.split("_")

        # cell-level    
        score_cell = pd.read_csv(result_dir + "1condition_10000_500_" + dataset + "/scores_cell_specific.csv", index_col = 0)
        scores_cell = pd.concat((scores_cell, score_cell), axis = 0)

        # cluster-level
        score_cluster = pd.read_csv(result_dir + "1condition_10000_500_" + dataset + "/scores_cluster_specific.csv", index_col = 0)
        scores_cluster = pd.concat((scores_cluster, score_cluster), axis = 0)



    fig = plt.figure(figsize = (22,6))
    ax = fig.subplots(nrows = 1, ncols = 4)
    sns.boxplot(data = scores_cell, x = "Method", y = "MSE (normalized)", ax = ax[0])
    sns.boxplot(data = scores_cell, x = "Method", y = "MSE diff (normalized)", ax = ax[1])
    # sns.boxplot(data = scores_cell, x = "Method", y = "Pearson", ax = ax[2])
    sns.boxplot(data = scores_cell, x = "Method", y = "R2", ax = ax[2])
    sns.boxplot(data = scores_cell, x = "Method", y = "R2 diff", ax = ax[3])
    ax[2].set_ylabel("$R^2$")
    ax[3].set_ylabel("$R^2$ diff") 
    ax[0].set_xlabel(None)
    ax[1].set_xlabel(None)
    ax[2].set_xlabel(None)
    ax[3].set_xlabel(None)
    ax[0].set_ylim(0, 0.08)
    ax[1].set_ylim(0, 0.08)

    ax[0].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    ax[1].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    ax[2].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    ax[3].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    plt.suptitle("Cell-specific scores")
    plt.tight_layout()
    fig.savefig(result_dir + "scores_cell.png", bbox_inches = "tight")

    fig = plt.figure(figsize = (22,6))
    ax = fig.subplots(nrows = 1, ncols = 4)
    sns.boxplot(data = scores_cluster, x = "Method", y = "MSE (normalized)", ax = ax[0])
    sns.boxplot(data = scores_cluster, x = "Method", y = "MSE diff (normalized)", ax = ax[1])
    # sns.boxplot(data = scores_cluster, x = "Method", y = "Pearson", ax = ax[2])
    sns.boxplot(data = scores_cluster, x = "Method", y = "R2", ax = ax[2])
    sns.boxplot(data = scores_cluster, x = "Method", y = "R2 diff", ax = ax[3])
    ax[2].set_ylabel("$R^2$")
    ax[3].set_ylabel("$R^2$ diff") 
    ax[0].set_xlabel(None)
    ax[1].set_xlabel(None)
    ax[2].set_xlabel(None)
    ax[3].set_xlabel(None)
    ax[0].set_ylim(0, 0.08)
    ax[1].set_ylim(0, 0.08)

    ax[0].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    ax[1].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    ax[2].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    ax[3].set_xticklabels(["scDisInFact", "scGEN", "scPreGAN"], rotation = 0)
    plt.suptitle("Cell-type-specific scores")
    plt.tight_layout()
    fig.savefig(result_dir + "scores_cluster.png", bbox_inches = "tight")



# In[] Test prediction accuracy, old for reference
'''
plt.rcParams["font.size"] = 15
# one forward pass
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


# NOTE: impute control for batches 3, 4, 5, 6, only removed of conditions

# control, batch 0 and 1
mean_ctrl = np.mean(np.concatenate([x[0] for x in z_ds[0:2]], axis = 0), axis = 0)[None,:]
# stim1, batch 2 and 3
mean_stim1 = np.mean(np.concatenate([x[0] for x in z_ds[2:4]], axis = 0), axis = 0)[None,:]
# stim2, batch 4 and 5
mean_stim2 = np.mean(np.concatenate([x[0] for x in z_ds[4:]], axis = 0), axis = 0)[None,:]

z_ds_ctrl = np.concatenate([x[0] for x in z_ds[0:2]], axis = 0)
# find the clostest
# mean_ctrl = z_ds_ctrl[[np.argmin(np.sum((z_ds_ctrl - mean_ctrl) ** 2, axis = 1))], :]

change_stim1_ctrl = mean_ctrl - mean_stim1
change_stim2_ctrl = mean_ctrl - mean_stim2

mse = []
pearson = []
mse_diff = []
mse_ratio = []
pearson_ratio = []
mse_ratio_diff = []

np.random.seed(0)
with torch.no_grad():
    for batch_id, dataset in enumerate(datasets):
        #-------------------------------------------------------------------------------------------------------------------
        #
        # Remove both batch effect and condition effect (Imputation of count matrices under the control condition)
        #
        #-------------------------------------------------------------------------------------------------------------------
        
        # removed of batch effect
        ref_batch = 0.0
        # still use the original batch_id as input, not change the latent embedding
        z_c, _ = model.Enc_c(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
        if batch_id in [0,1]:
            z_d = []
            for Enc_d in model.Enc_ds:
                # still use the original batch_id as input, not change the latent embedding
                _z_d, _ = Enc_d(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
                z_d.append(_z_d)        

        elif batch_id in [2,3,4,5]:
            # NOTE: removed of condition effect
            # manner one, use mean
            # z_d = [torch.tensor(mean_ctrl, device = model.device).expand(dataset.counts_stand.shape[0], Ks[1])]
            
            # manner two, random sample, better than mean
            # idx = np.random.choice(z_ds_ctrl.shape[0], z_c.shape[0], replace = True)
            # z_d = [torch.tensor(z_ds_ctrl[idx,:], device = model.device)]
            
            # manner three, latent space arithmetics
            if batch_id in [2,3]:
                z_d = z_ds[batch_id][0] + change_stim1_ctrl
            else:
                z_d = z_ds[batch_id][0] + change_stim2_ctrl 
            z_d = [torch.tensor(z_d, device = model.device)]

        # NOTE: change the batch_id into ref batch as input, change the diff condition into control
        mu_impute, _, _ = model.Dec(torch.concat([z_c] + z_d, dim = 1), torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None])

        mu_impute = mu_impute.cpu().detach().numpy() 
        # NOTE: mu is normalized of library size
        mu_impute_norm = mu_impute/np.sum(mu_impute, axis = 1, keepdims = True)

        #-------------------------------------------------------------------------------------------------------------------
        #
        # Not remove either batch effect or condition effect (only denoise the count matrix) 
        #
        #-------------------------------------------------------------------------------------------------------------------
        # still use the original batch_id as input, not change the latent embedding
        z_c, _ = model.Enc_c(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
        z_d = []
        for Enc_d in model.Enc_ds:
            # still use the original batch_id as input, not change the latent embedding
            _z_d, _ = Enc_d(dataset.counts_stand.to(model.device), dataset.batch_id[:,None].to(model.device))
            z_d.append(_z_d)        
            
        # NOTE: change the batch_id into ref batch as input, only remove the batch effect but keep the condition effect
        # mu, _, _ = model.Dec(torch.concat([z_c] + z_d, dim = 1), torch.tensor([ref_batch] * dataset.counts_stand.shape[0], device = model.device)[:,None])
        mu, _, _ = model.Dec(torch.concat([z_c] + z_d, dim = 1), dataset.batch_id[:,None].to(model.device))
        mu = mu.cpu().detach().numpy() 
        # NOTE: mu is normalized of library size
        mu_norm = mu/np.sum(mu, axis = 1, keepdims = True)

        # change mu to input
        mu = dataset.counts.numpy()
        mu_norm = mu/np.sum(mu, axis = 1, keepdims = True)

        #-------------------------------------------------------------------------------------------------------------------
        #
        # TODO: challenge: remove only batch effect and keep the condition effect, then compared the mse with remove both batch effect and condition effect
        # Requires the accurate disentangling of both parts (condition and batch effect). 
        # (Alternative: remove only condition effect but keep batch effect, then compare) 
        #
        #-------------------------------------------------------------------------------------------------------------------

        # input count matrix
        # mu_obs = dataset.counts.cpu().detach().numpy()
        # mu_norm = mu_obs/np.sum(mu_obs, axis = 1, keepdims = True)

        # ground truth: real count
        mu_gt = counts_gt[batch_id]
        mu_gt_norm = mu_gt/np.sum(counts_gt[batch_id], axis = 1, keepdims = True)

        # mse is calculated by averaging over cells
        mse.append(np.mean(np.sum((mu_gt_norm - mu_impute_norm) ** 2, axis = 1)))

        # pearson is calculated by averaging over cells
        pearson.append(np.mean(np.array([stats.pearsonr(mu_gt_norm[i,:], mu_impute_norm[i,:])[0] for i in range(mu_impute_norm.shape[0])])))
        mse_diff.append(np.mean(np.sum((mu_gt_norm[:,:2*n_diff_genes] - mu_impute_norm[:,:2*n_diff_genes]) ** 2, axis = 1)))

        mse_ratio.append(
            np.mean(np.sum((mu_gt_norm - mu_impute_norm) ** 2, axis = 1))/np.mean(np.sum((mu_gt_norm - mu_norm) ** 2, axis = 1))
        )

        pearson_ratio.append(
            np.mean(np.array([stats.pearsonr(mu_gt_norm[i,:], mu_impute_norm[i,:])[0] for i in range(mu_impute_norm.shape[0])]))/np.mean(np.array([stats.pearsonr(mu_gt_norm[i,:], mu_norm[i,:])[0] for i in range(mu_norm.shape[0])]))
        )

        mse_ratio_diff.append(
            np.mean(np.sum((mu_gt_norm[:,:2*n_diff_genes] - mu_impute_norm[:,:2*n_diff_genes]) ** 2, axis = 1))/np.mean(np.sum((mu_gt_norm[:,:2*n_diff_genes] - mu_norm[:,:2*n_diff_genes]) ** 2, axis = 1))
        )

        # plot the correlationship
        fig = plt.figure(figsize = (14, 14))
        ax = fig.subplots(nrows = 2, ncols = 2)
        ax[0,0].scatter(mu_gt_norm[:,20:].reshape(-1), mu_impute_norm[:,20:].reshape(-1), s = 3)
        ax[0,0].scatter(mu_gt_norm[:,:20].reshape(-1), mu_impute_norm[:,:20].reshape(-1), s = 3, c = "r")
        ax[0,0].plot([0,1], [0,1], c = "k")
        ax[0,0].set_xlim(xmin = 0, xmax = 1)
        ax[0,0].set_ylim(ymin = 0, ymax = 1)
        ax[0,0].set_xlabel("True count")
        ax[0,0].set_ylabel("Impute count")
        ax[0,0].set_title(f"Impute batch {batch_id}")

        ax[0,1].scatter(mu_gt_norm[:,:20].reshape(-1), mu_impute_norm[:,:20].reshape(-1), s = 3)
        ax[0,1].plot([0,1], [0,1], c = "k")
        ax[0,1].set_xlim(xmin = 0, xmax = 0.1)
        ax[0,1].set_ylim(ymin = 0, ymax = 0.1)
        ax[0,1].set_xlabel("True count")
        ax[0,1].set_ylabel("Impute count")
        ax[0,1].set_title(f"Impute batch {batch_id} (diff genes)")
        fig.savefig(result_dir + comment + f"corr_batch_{batch_id}.png", bbox_inches = "tight")

        ax[1,0].scatter(mu_gt_norm[:,20:].reshape(-1), mu_norm[:,20:].reshape(-1), s = 3)
        ax[1,0].scatter(mu_gt_norm[:,:20].reshape(-1), mu_norm[:,:20].reshape(-1), s = 3, c = "r")
        ax[1,0].plot([0,1], [0,1], c = "k")
        ax[1,0].set_xlim(xmin = 0, xmax = 1)
        ax[1,0].set_ylim(ymin = 0, ymax = 1)
        ax[1,0].set_xlabel("True count")
        ax[1,0].set_ylabel("Observed count")
        ax[1,0].set_title(f"Denoised batch {batch_id}")

        ax[1,1].scatter(mu_gt_norm[:,:20].reshape(-1), mu_norm[:,:20].reshape(-1), s = 3)
        ax[1,1].plot([0,1], [0,1], c = "k")
        ax[1,1].set_xlim(xmin = 0, xmax = 0.1)
        ax[1,1].set_ylim(ymin = 0, ymax = 0.1)
        ax[1,1].set_xlabel("True count")
        ax[1,1].set_ylabel("Observed count")
        ax[1,1].set_title(f"Denoised batch {batch_id} (diff genes)")
        fig.savefig(result_dir + comment + f"corr_batch_{batch_id}.png", bbox_inches = "tight")

# print(mse_ratio)
# print(pearson_ratio)
# print(mse_ratio_diff)
np.savetxt(result_dir + comment + "mse.txt", np.array(mse))
np.savetxt(result_dir + comment + "mse_diff.txt", np.array(mse_diff))
np.savetxt(result_dir + comment + "pearson.txt", np.array(pearson))

np.savetxt(result_dir + comment + "mse_ratio.txt", np.array(mse_ratio))
np.savetxt(result_dir + comment + "mse_ratio_diff.txt", np.array(mse_ratio_diff))
np.savetxt(result_dir + comment + "pearson_ratio.txt", np.array(pearson_ratio))
'''
# %%
