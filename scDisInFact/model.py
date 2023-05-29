import sys, os
import torch
import numpy as np 
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import pandas as pd
import scipy.sparse as sp

import scDisInFact.base_model as base_model 
import scDisInFact.loss_function as loss_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class scdisinfact_dataset(Dataset):

    def __init__(self, counts, counts_norm, size_factor, diff_labels, batch_id, mmd_batch_id):
        """
        Preprocessing similar to the DCA (dca.io.normalize)
        """
        assert not len(counts) == 0, "Count is empty"
        self.counts = torch.FloatTensor(counts)
        self.counts_norm = torch.FloatTensor(counts_norm)
        self.size_factor = torch.FloatTensor(size_factor)
        # make sure the input time point are integer
        self.diff_labels = []
        # loop through all types of diff labels
        for diff_label in diff_labels:
            assert diff_label.shape[0] == self.counts_norm.shape[0]
            self.diff_labels.append(torch.LongTensor(diff_label))
        self.batch_id = torch.Tensor(batch_id)
        # batch and condition combination
        self.mmd_batch_id = torch.Tensor(mmd_batch_id)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"mmd_batch_id": self.mmd_batch_id[idx], 
                  "batch_id": self.batch_id[idx], 
                  "diff_labels": [x[idx] for x in self.diff_labels],  
                  "counts": self.counts[idx,:], 
                  "counts_norm": self.counts_norm[idx,:], 
                  "index": idx, 
                  "size_factor": self.size_factor[idx]}
        return sample



def create_scdisinfact_dataset(counts, meta_cells, condition_key, batch_key, batch_cond_key = None, log_trans = True):
    """\
    Description:
    --------------
        Create scDisInFact dataset using count matrices and meta data
    
    Parameters:
    --------------
        counts: 
            Input count matrix: can be one numpy count matrix including all data batches, or a list of count matrices. 
            Accept ``scipy.sparse.csr_matrix'' or ``numpy.array'' for count matrix format. 
            Count matrix should be of the shape (ncells, ngenes), and gene should match between different count matrices.

        meta_cells:
            Inpute dataframe/list of dataframe that includes the meta-information of each cell in count matrices. 
            The cell should match that of ``counts''
        
        condition_key:
            List of column labels in the ``meta_cells'' that corresponds to condition groups in different condition types.

        batch_key:
            The column label in the ``meta_cells'' that correspond to batches
        
        batch_cond_key:
            The column label in the ``meta_cells'' that correspond to batch-condition pairs. 
            The batch-condition pairs gives a unique label for each condition group under each data batch.
            Batch-condition pair is automatically calculated from the batch and condition labels when ``batch_cond_key'' is None.
            ``batch_cond_key'' is None by default.

        log_trans:
            Whether log transform the count matrix or not
    Return:
    --------------
        datasets_array: 
            an array of scdisinfact datasets
        meta_cells_array:
            an array of meta cells (match datasets)
        matching_dict:
            matching dictionary between condition/batch ID and condition/batch names
    """
    # Sanity check
    print("Sanity check...")
    if type(counts) == list:
        # check the meta_cells should also be list
        assert type(meta_cells) == list
        # check the length of counts match that of meta_cells
        assert len(counts) == len(meta_cells)
        for i in range(len(counts)):
            # check the number of cells in each count matrix match that of each meta_cells
            assert counts[i].shape[0] == meta_cells[i].shape[0]
            # check the condition_key is in the columns of meta_cells
            for cond in condition_key:
                assert cond in meta_cells[i].columns
            # check the batch_key is in the columns of meta_cells
            assert batch_key in meta_cells[i].columns
    else:
        # should include the same number of cells
        assert counts.shape[0] == meta_cells.shape[0]
        for cond in condition_key:
            assert cond in meta_cells.columns
        # check the batch_key is in the columns of meta_cells
        assert batch_key in meta_cells.columns
        if batch_cond_key is not None:
            # check if the batch_cond_key is in meta_cells columns
            assert batch_cond_key in meta_cells.columns
    print("Finished.")

    print("Create scDisInFact datasets...")
    # concatenate the count matrices and meta_cells if they are lists
    if type(counts) == list:
        try:
            if sp.issparse(counts[0]):
                counts = sp.vstack(counts, format = "csr").toarray()
            else:
                counts = np.concatenate(counts, axis = 0)
        except:
            raise ValueError("Genes are not match among count matrices.")
        meta_cells = pd.concat(meta_cells, axis = 0)
    else:
        # make sure to be dataframe
        meta_cells = pd.DataFrame(meta_cells)
        if sp.issparse(counts):
            counts = counts.toarray()
        
        
    # construct batch_cond pairs that combine batch id with condition types
    if batch_cond_key is None:
        meta_cells["batch_cond"] = meta_cells[[batch_key] + condition_key].apply(lambda row: '_'.join(row.to_numpy().astype(str)), axis=1)
    else:
        meta_cells["batch_cond"] = meta_cells[batch_cond_key].to_numpy()
    # transfer label to ids
    cond_ids = []
    cond_names = []
    for cond in condition_key:
        cond_id, cond_name = pd.factorize(meta_cells[cond].to_numpy().squeeze(), sort = True)
        meta_cells[cond + "_id"] = cond_id
        cond_ids.append(cond_id)
        cond_names.append(cond_name)
    
    batch_ids, batch_names = pd.factorize(meta_cells[batch_key].to_numpy().squeeze(), sort = True)
    meta_cells[batch_key + "_id"] = batch_ids
    batch_cond_ids, batch_cond_names = pd.factorize(meta_cells["batch_cond"].to_numpy().squeeze(), sort = True)
    meta_cells["batch_cond_id"] = batch_cond_ids

    # normalize the count matrix
    size_factor = np.tile(np.sum(counts, axis = 1, keepdims = True), (1, counts.shape[1]))/100
    # in scanpy, np.median(counts) is used instead of 100 here
    counts_norm = counts/size_factor
    if log_trans:
        counts_norm = np.log1p(counts_norm)
    # further standardize the count
    scaler = StandardScaler().fit(counts_norm)
    counts_norm = torch.FloatTensor(scaler.transform(counts_norm))

    datasets_array = []
    counts_array = []
    counts_norm_array = []
    size_factor_array = []
    meta_cells_array = []
    for batch_cond_id, batch_cond_name in enumerate(batch_cond_names):
        counts_array.append(counts[batch_cond_ids == batch_cond_id, :])
        counts_norm_array.append(counts_norm[batch_cond_ids == batch_cond_id, :])
        size_factor_array.append(size_factor[batch_cond_ids == batch_cond_id, :])

        meta_cells_array.append(meta_cells.iloc[batch_cond_ids == batch_cond_id, :])
        datasets_array.append(scdisinfact_dataset(counts = counts_array[-1], counts_norm = counts_norm_array[-1],
                                                size_factor = size_factor_array[-1],
                                                diff_labels = [x[batch_cond_ids == batch_cond_id] for x in cond_ids], 
                                                batch_id = batch_ids[batch_cond_ids == batch_cond_id],
                                                mmd_batch_id = batch_cond_ids[batch_cond_ids == batch_cond_id]
                                                ))
    print("Finished.")

    matching_dict = {"cond_names": cond_names, "batch_name": batch_names, "batch_cond_names": batch_cond_names}
    return {"datasets": datasets_array, "meta_cells": meta_cells_array, "matching_dict": matching_dict, "scaler": scaler}

class scdisinfact(nn.Module):
    """\
    Description:
    --------------
        Implementation of scDisInFact
    
    Parameters:
    --------------
        data_dict:
            dictionary returned by create_scdisinfact_dataset
        reg_mmd_comm:
            regularization weight of MMD loss on shared-bio factor, default value is 1e-4
        reg_mmd_diff:
            regularization weight of MMD loss on unshared-bio factor, default value is 1e-4
        reg_gl:
            regularization weight of group lasso loss, default value is 1
        reg_class:
            regularization weight of classification loss, default value is 1
        reg_tc:
            regularization weight of total correlation loss, default value is 0.5
        reg_kl:
            regularization weight of KL-divergence
        Ks:
            dimensions of latent factors, arranged following: (shared-bio factor, unshared-bio factor 1, ..., unshared-bio factor n), default value is [8,4]
        batch_size:
            size of mini-batch in stochastic gradient descent, default value is 64
        interval:
            number of epochs between each result printing, default 10
        lr:
            learning rate, default 5e-4
        seed:
            seed for reproducing the result, default is 0
        enc_injection:
            inject batch categories into the unshared encoder as additional dimension
        device:
            training device
    
    Examples:
    --------------
    >>> model = scdisinfact.scdisinfact(data_dict = data_dict, Ks = [8,4], batch_size = 8, interval = 10, lr = 5e-4, 
                                reg_mmd_comm = 1e-4, reg_mmd_diff = 1e-4, reg_gl = 1, reg_tc = 0.5, 
                                reg_kl = 1e-6, reg_class = 1, seed = 0, device = torch.device("cuda:0"))
    >>> model.train_model(nepochs = 100, recon_loss = "NB", reg_contr = 0.01)
    """
    def __init__(self, data_dict, reg_mmd_comm = 1e-4, reg_mmd_diff = 1e-4, reg_gl = 1, reg_class = 1, reg_tc = 0.5, 
                 reg_kl = 1e-5, Ks = [8, 4], batch_size = 64, interval = 10, lr = 5e-4, seed = 0, enc_injection = True, device = device):
        super().__init__()
        # initialize the parameters
        self.Ks = {"common_factor": Ks[0], "diff_factors": Ks[1:]}
        # number of diff factors
        self.n_diff_factors = len(self.Ks["diff_factors"])
        # mini-batch size
        self.batch_size = batch_size
        # test interval
        self.interval = interval
        # learning rate
        self.lr = lr
        # regularization parameters
        self.lambs = {"mmd_comm": reg_mmd_comm, "mmd_diff": reg_mmd_diff, "class": reg_class, "tc": reg_tc, "gl": reg_gl, "kl": reg_kl}
        # random seed
        self.seed = seed
        # device 
        self.device = device
        # dataset
        self.data_dict = data_dict

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # create data loaders
        self.train_loaders = []
        # create test data loaders
        self.test_loaders = []
        # store the number of cells for each data matrix
        self.ncells = []
        # store unique diff labels for each diff factor [[unique label type 1], [unique label type 2], ...]
        self.uniq_diff_labels = [[] for x in range(self.n_diff_factors)]
        self.uniq_batch_ids = []

        # loop through all data matrices/datasets
        for idx, dataset in enumerate(self.data_dict["datasets"]):
            assert self.n_diff_factors == len(dataset.diff_labels)
            # make sure that the genes are matched
            if idx == 0:
                self.ngenes = dataset.counts.shape[1]
            else:
                assert self.ngenes == dataset.counts.shape[1]

            # total number of cells
            self.ncells.append(dataset.counts.shape[0])
            # create train loader 
            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))
            # create test loader
            # subsampling the test dataset when the dataset is too large
            cutoff = 100
            if len(dataset) > cutoff:
                print("test dataset shrink to {:d}".format(cutoff))
                samples = torch.randperm(n = cutoff)[:cutoff]
                test_dataset = Subset(dataset, samples)
            else:
                test_dataset = dataset
            self.test_loaders.append(DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False))
            # unique diff labels for the current data matrix
            for i, diff_label in enumerate(dataset.diff_labels):
                self.uniq_diff_labels[i].extend([x.item() for x in torch.unique(diff_label)])
            # unique batch ids for the current data matrix
            self.uniq_batch_ids.extend([x.item() for x in torch.unique(dataset.batch_id)])

        # unique diff labels for each diff factor
        for diff_factor in range(self.n_diff_factors):
            self.uniq_diff_labels[diff_factor] = sorted(set(self.uniq_diff_labels[diff_factor])) 
        # unique data batches
        self.uniq_batch_ids = sorted(set(self.uniq_batch_ids))

        # create model
        # encoder for common biological factor
        self.Enc_c = base_model.Encoder(n_input = self.ngenes, n_output = self.Ks["common_factor"], n_layers = 2, n_hidden = 128, n_cat_list = [len(self.uniq_batch_ids)], dropout_rate  = 0.2, use_batch_norm = False).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = nn.ModuleList([])
        for diff_factor in range(self.n_diff_factors):
            self.Enc_ds.append(
                base_model.Encoder(n_input = self.ngenes, n_output = self.Ks["diff_factors"][diff_factor], n_layers = 1, n_hidden = 128, n_cat_list = [len(self.uniq_batch_ids)] if enc_injection == True else None, dropout_rate = 0.2, use_batch_norm = False).to(self.device)
            )
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = nn.ModuleList([])
        for diff_factor in range(self.n_diff_factors):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][diff_factor], len(self.uniq_diff_labels[diff_factor])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = base_model.Decoder(n_input = self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), n_output = self.ngenes, n_cat_list = [len(self.uniq_batch_ids)], n_layers = 2, n_hidden = 128, dropout_rate = 0.2, use_batch_norm = False).to(self.device)
        # Discriminator for factor vae
        self.disc = base_model.FCLayers(n_in=self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), n_out=2, n_cat_list=None, n_layers=3, n_hidden=2, dropout_rate=0.2, use_batch_norm = False).to(self.device)
        
        self.opt = opt.Adam(self.parameters(), lr = self.lr)



    def reparametrize(self, mu, logvar, clamp = 0):
        # exp(0.5*log_var) = exp(log(\sqrt{var})) = \sqrt{var}
        std = logvar.mul(0.5).exp_()
        if clamp > 0:
            # prevent the shrinkage of variance
            std = torch.clamp(std, min = clamp)

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def inference(self, counts, batch_ids, print_stat = False, clamp_comm = 0.0, clamp_diff = 0.0):
        # sanity check
        assert counts.shape[0] == batch_ids.shape[0]
        assert batch_ids.shape[1] == 1
        # pass through common encoder network
        mu_c, logvar_c = self.Enc_c(counts.to(self.device), batch_ids.to(self.device))
            
        # pass through diff encoder network
        mu_d = []
        logvar_d = []
        for diff_factor in range(self.n_diff_factors):
            _mu_d, _logvar_d = self.Enc_ds[diff_factor](counts.to(self.device), batch_ids.to(self.device))
            mu_d.append(_mu_d)
            logvar_d.append(_logvar_d)

        if self.training:
            # latent sampling
            z_c = self.reparametrize(mu_c, logvar_c, clamp = clamp_comm)                   
            z_d = []
            for diff_factor in range(self.n_diff_factors):
                z_d.append(self.reparametrize(mu_d[diff_factor], logvar_d[diff_factor], clamp = clamp_diff))
                    
            if print_stat:
                print("mean z_c: {:.5f}".format(torch.mean(mu_c).item()))
                print("mean var z_c: {:.5f}".format(torch.mean(logvar_c.mul(0.5).exp_()).item()))
                for i, x in enumerate(mu_d):
                    print("mean z_d{:d}: {:.5f}".format(i, torch.mean(x).item()))
                    print("mean var z_d{:d}: {:.5f}".format(i, torch.mean(logvar_d[i].mul(0.5).exp_())))
            
            return {"mu_c":mu_c, "logvar_c": logvar_c, "z_c": z_c, "mu_d": mu_d, "logvar_d": logvar_d, "z_d": z_d}

        else:
            # when evaluate the model, z_c=mu_c, z_d=mu_d
            return {"mu_c":mu_c, "logvar_c": logvar_c, "z_c": mu_c, "mu_d": mu_d, "logvar_d": logvar_d, "z_d": mu_d}
            

    def generative(self, z_c, z_d, batch_ids):
        """\
        Description:
        --------------
            generating high-dimensional data from shared-bio, unshared-bio factors
        
        Parameters:
        --------------
            z_c:
                shared-bio factor, of the shape (ncells, nlatents)
            z_d:
                list of unshared-bio factors, where one element correspond to unshared-bio factor of one condition type, each factor is of the shape (ncells, nlatents)
            batch_ids:
                the batch ids, of the shape (ncells, 1)
        
        Return:
        --------------
            dictionary of generative output:
                "mu": the mean of NB distribution, used as prediction/reconstruction result
                "theta": the dispersion parameter of NB distribution
                "z_pred": the prediction output of classifier
                "y_orig": the discriminator output of original samples
                "y_perm": the discriminator output of permuted samples
        """
        # sanity check
        assert len(z_d) == self.n_diff_factors
        # decoder
        mu, pi, theta = self.Dec(torch.cat([z_c] + z_d, dim = 1), batch_ids.to(self.device))
        # classifier
        z_pred = []
        for diff_factor in range(self.n_diff_factors):
            z_pred.append(self.classifiers[diff_factor](z_d[diff_factor]))

        # discriminator
        # create original samples
        orig_samples = torch.cat([z_c] + z_d, dim = 1)
        # create permuted samples
        perm_idx = []
        for diff_factor in range(self.n_diff_factors):
            perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
        perm_samples = torch.cat([z_c] + [x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)
        # pass through discriminator
        y_orig = self.disc(orig_samples)
        y_perm = self.disc(perm_samples)     
        return {"mu": mu, "pi": pi, "theta": theta, "z_pred": z_pred, "y_orig": y_orig, "y_perm": y_perm}
    
    def loss(self, dict_inf, dict_gen, size_factor, count, batch_id, diff_labels, recon_loss):
        """\
        Decription:
        --------------
            loss function of scDisInFact
          
        """
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        contr_loss = loss_func.CircleLoss(m=0.25, gamma=80)

        # 1.reconstruction loss
        if recon_loss == "ZINB":
            lamb_pi = 1e-5
            loss_recon = loss_func.ZINB(pi = dict_gen["pi"], theta = dict_gen["theta"], scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = dict_gen["mu"])
        elif recon_loss == "NB":
            loss_recon = loss_func.NB(theta = dict_gen["theta"], scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = dict_gen["mu"])
        elif recon_loss == "MSE":
            mse_loss = nn.MSELoss()
            loss_recon = mse_loss(dict_gen["mu"] * size_factor, count)
        else:
            raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")        

        # 2.kl divergence
        # loss_kl = torch.sum(dict_inf["mu_c"].pow(2).add_(dict_inf["logvar_c"].exp()).mul_(-1).add_(1).add_(dict_inf["logvar_c"])).mul_(-0.5)         
        # average instead of sum across data within a batch
        loss_kl = torch.mean(-0.5 * torch.sum(1 + dict_inf["logvar_c"] - dict_inf["mu_c"] ** 2 - dict_inf["logvar_c"].exp(), dim = 1), dim = 0)
        for diff_factor in range(self.n_diff_factors):
            # loss_kl += torch.sum(dict_inf["mu_d"][diff_factor].pow(2).add_(dict_inf["logvar_d"][diff_factor].exp()).mul_(-1).add_(1).add_(dict_inf["logvar_d"][diff_factor])).mul_(-0.5)
            loss_kl += torch.mean(-0.5 * torch.sum(1 + dict_inf["logvar_d"][diff_factor] - dict_inf["mu_d"][diff_factor] ** 2 - dict_inf["logvar_d"][diff_factor].exp(), dim = 1), dim = 0)    

        # 3.MMD loss
        # common mmd loss
        loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = dict_inf["mu_c"], batch_ids = batch_id, device = self.device)
        # condition specific mmd loss
        loss_mmd_diff = 0
        for diff_factor in range(self.n_diff_factors):
            for diff_label in self.uniq_diff_labels[diff_factor]:
                idx = diff_labels[diff_factor] == diff_label
                if any(idx):
                    loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = dict_inf["mu_d"][diff_factor][idx, :], batch_ids = batch_id[idx], device = self.device)
                else:
                    loss_mmd_diff += 0
        # 4.classifier loss and contr loss
        loss_class = 0
        loss_contr = 0
        for diff_factor in range(self.n_diff_factors):
            loss_class += ce_loss(input = dict_gen["z_pred"][diff_factor], target = diff_labels[diff_factor])
            loss_contr += contr_loss(F.normalize(dict_inf["z_d"][diff_factor]), diff_labels[diff_factor])
        
        # 5.total correlation
        loss_tc = ce_loss(input = torch.cat([dict_gen["y_orig"], dict_gen["y_perm"]], dim = 0), target =torch.tensor([0] * dict_gen["y_orig"].shape[0] + [1] * dict_gen["y_perm"].shape[0], device = self.device))

        # 6.group lasso
        loss_gl_d = 0
        for diff_factor in range(self.n_diff_factors):
            # TODO: decide on alpha
            loss_gl_d += loss_func.grouplasso(self.Enc_ds[diff_factor].fc.fc_layers[0][0].weight, alpha = 0.1)

        return loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d


    def train_model(self, nepochs = 50, recon_loss = "NB", reg_contr = 0.01):
        """\
        Decription:
        -------------
            Training function of scDisInFact
        
        Parameters:
        -------------
            nepochs: number of epochs
            recon_loss: choose from "NB", "ZINB", "MSE"
            reg_contr: regulation weight of contrastive loss
        """
        self.train()

        best_loss = 1e3
        trigger = 0
        clamp_comm = 0.01
        clamp_diff = 0.01


        loss_tests = []
        loss_recon_tests = []
        loss_kl_tests = []
        loss_mmd_comm_tests = []
        loss_mmd_diff_tests = []
        loss_class_tests = []
        loss_gl_d_tests = []
        loss_gl_c_tests = []
        loss_tc_tests = []

        for epoch in range(nepochs + 1):
            for data_batch in zip(*self.train_loaders):
                # loop through the data batches correspond to diffferent data matrices
                counts_norm = []
                batch_id = []
                mmd_batch_id = []
                size_factor = []
                count = []
                diff_labels = [[] for x in range(self.n_diff_factors)]

                # load count data
                for x in data_batch:
                    counts_norm.append(x["counts_norm"])
                    batch_id.append(x["batch_id"][:, None])
                    mmd_batch_id.append(x["mmd_batch_id"])
                    size_factor.append(x["size_factor"])
                    count.append(x["counts"])       
                    for diff_factor in range(self.n_diff_factors):
                        diff_labels[diff_factor].append(x["diff_labels"][diff_factor])

                counts_norm = torch.cat(counts_norm, dim = 0).to(self.device, non_blocking=True)
                batch_id = torch.cat(batch_id, dim = 0).to(self.device, non_blocking=True)
                mmd_batch_id = torch.cat(mmd_batch_id, dim = 0).to(self.device, non_blocking=True)
                size_factor = torch.cat(size_factor, dim = 0).to(self.device, non_blocking=True)
                count = torch.cat(count, dim = 0).to(self.device, non_blocking=True)
                for diff_factor in range(self.n_diff_factors):
                    diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0).to(self.device, non_blocking=True)

                # 1. 
                # freeze the gradient of diff_encoders, classifiers, and discriminators
                for diff_factor in range(self.n_diff_factors):
                    for x in self.Enc_ds[diff_factor].parameters():
                        x.requires_grad = False
                    for x in self.classifiers[diff_factor].parameters():
                        x.requires_grad = False
                for x in self.disc.parameters():
                    x.requires_grad = False
                # activate the common encoder, decoder
                for x in self.Enc_c.parameters():
                    x.requires_grad = True
                for x in self.Dec.parameters():
                    x.requires_grad = True

                # pass through the encoders
                dict_inf = self.inference(counts = counts_norm, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                # pass through the decoder
                dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d = self.loss(dict_inf = dict_inf, \
                    dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = mmd_batch_id, diff_labels = diff_labels, recon_loss = recon_loss)

                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl
                # print("\nstep1")
                # print(loss_recon)
                # print(self.lambs["mmd_comm"] * loss_mmd_comm)
                # print(self.lambs["kl"] * loss_kl)
                # print()

                loss.backward()
                # clip the gradient to prevent exploding
                nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

                # 2. 
                # activate the gradient of diff_encoders, classifiers, and discriminators
                for diff_factor in range(self.n_diff_factors):
                    for x in self.Enc_ds[diff_factor].parameters():
                        x.requires_grad = True
                    for x in self.classifiers[diff_factor].parameters():
                        x.requires_grad = True
                # freeze the common encoder and disciminators
                for x in self.Enc_c.parameters():
                    x.requires_grad = False
                for x in self.disc.parameters():
                    x.requires_grad = False
                # activate the decoder
                for x in self.Dec.parameters():
                    x.requires_grad = True

                # pass through the encoders
                dict_inf = self.inference(counts = counts_norm, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                # pass through the decoder
                dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d = self.loss(dict_inf = dict_inf, \
                    dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = mmd_batch_id, diff_labels = diff_labels, recon_loss = recon_loss)

                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * (loss_class + reg_contr * loss_contr) + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                # print("\nstep2")
                # print(loss_recon)
                # print(self.lambs["mmd_diff"] * loss_mmd_diff)
                # print(self.lambs["kl"] * loss_kl)
                # print(self.lambs["class"] * loss_class)
                # print(self.lambs["class"] * reg_contr * loss_contr)
                # print(self.lambs["gl"] * loss_gl_d)
                # print(self.lambs["tc"] * loss_tc)
                # print()

                loss.backward()
                # clip the gradient to prevent exploding
                nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()


                # 3. 
                # activate the gradient of discriminators, freeze all remaining
                for diff_factor in range(self.n_diff_factors):
                    for x in self.Enc_ds[diff_factor].parameters():
                        x.requires_grad = False
                    for x in self.classifiers[diff_factor].parameters():
                        x.requires_grad = False
                for x in self.Enc_c.parameters():
                    x.requires_grad = False
                for x in self.disc.parameters():
                    x.requires_grad = True
                # freeze the decoder                
                for x in self.Dec.parameters():
                    x.requires_grad = False

                # pass through the encoders
                dict_inf = self.inference(counts = counts_norm, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                # pass through the decoder
                dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d = self.loss(dict_inf = dict_inf, \
                    dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = mmd_batch_id, diff_labels = diff_labels, recon_loss = recon_loss)

                loss = self.lambs["tc"] * loss_tc
                # print("\nstep3")
                # print(self.lambs["tc"] * loss_tc)
                # print()

                loss.backward()
                # clip the gradient to prevent exploding
                nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()    

            # TEST
            if epoch % self.interval == 0:   
                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        # loop through the data batches correspond to diffferent data matrices
                        counts_norm = []
                        batch_id = []
                        mmd_batch_id = []
                        size_factor = []
                        count = []
                        diff_labels = [[] for x in range(self.n_diff_factors)]

                        # load count data
                        for x in data_batch:
                            counts_norm.append(x["counts_norm"])
                            batch_id.append(x["batch_id"][:, None])
                            mmd_batch_id.append(x["mmd_batch_id"])
                            size_factor.append(x["size_factor"])
                            count.append(x["counts"])          
                            for diff_factor in range(self.n_diff_factors):
                                diff_labels[diff_factor].append(x["diff_labels"][diff_factor])
                        
                        counts_norm = torch.cat(counts_norm, dim = 0).to(self.device, non_blocking=True)
                        batch_id = torch.cat(batch_id, dim = 0).to(self.device, non_blocking=True)
                        mmd_batch_id = torch.cat(mmd_batch_id, dim = 0).to(self.device, non_blocking=True)
                        size_factor = torch.cat(size_factor, dim = 0).to(self.device, non_blocking=True)
                        count = torch.cat(count, dim = 0).to(self.device, non_blocking=True)
                        for diff_factor in range(self.n_diff_factors):
                            diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0).to(self.device, non_blocking=True)

                        # pass through the encoders
                        dict_inf = self.inference(counts = counts_norm, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                        # pass through the decoder
                        dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                        loss_recon_test, loss_kl_test, loss_mmd_comm_test, loss_mmd_diff_test, loss_class_test, loss_contr_test, loss_tc_test, loss_gl_d_test = self.loss(dict_inf = dict_inf, \
                            dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = mmd_batch_id, diff_labels = diff_labels, recon_loss = recon_loss)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * (loss_class_test + reg_contr * loss_contr_test) \
                            + self.lambs["gl"] * loss_gl_d_test + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test
                  
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss contrastive: {:.5f}'.format(loss_contr_test.item()),
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test.item())
                        ]
                        for i in info:
                            print("\t", i)       
                               
                        print("GPU memory usage: {:f}MB".format(torch.cuda.memory_allocated(device)/1024/1024))

                        loss_tests.append(loss_test.item())
                        loss_recon_tests.append(loss_recon_test.item())
                        loss_mmd_comm_tests.append(loss_mmd_comm_test.item())
                        loss_mmd_diff_tests.append(loss_mmd_diff_test.item())
                        loss_class_tests.append(loss_class_test.item())
                        loss_kl_tests.append(loss_kl_test.item())
                        loss_gl_d_tests.append(loss_gl_d_test.item())
                        loss_tc_tests.append(loss_tc_test.item())        

                        # # update for early stopping 
                        # if loss_test.item() < best_loss:
                            
                        #     best_loss = loss_test.item()
                        #     torch.save(self.state_dict(), f'../check_points/model.pt')
                        #     trigger = 0
                        # else:
                        #     trigger += 1
                        #     print(trigger)
                        #     if trigger % 5 == 0:
                        #         self.opt.param_groups[0]['lr'] *= 0.5
                        #         print('Epoch: {}, shrink lr to {:.4f}'.format(epoch, self.opt.param_groups[0]['lr']))
                        #         if self.opt.param_groups[0]['lr'] <= 1e-6:
                        #             break
                        #         else:
                        #             self.load_state_dict(torch.load(f'../check_points/model.pt'))
                        #             trigger = 0                            

        self.eval()                
        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests


    def predict_counts(self, input_counts, meta_cells, condition_keys, batch_key, predict_conds = None, predict_batch = None):
        """\
        Description:
        -------------
            Function for condition effect prediction.

        Parameters:
        -------------
            input_counts:
                the input count matrix, of the shape (ncells, ngenes), of the type np.array()

            meta_cells:
                the meta cell data frame of input count matrix
            
            condition_keys:
                the list of columns of the meta_cell data frame that correspond to the conditions
            
            batch_key:
                the column of the meta_cell data frame that correspond to the batches

            predict_conds:
                the condition label of the predicted dataset, 
                predict_conds should have length equal to the number of condition type, 
                and should have one condition group id for each condition type,
                default None, where the condition is kept the same as predict_dataset

            predict_batch:
                the batch id of the predicted dataset, default None, where batch_ids is kept the same as predict_dataset

        Returns:
        -------------
            predicted counts
        
        """
        # number of condition types should match
        # assert len(self.uniq_diff_labels) == len(predict_conds)        
        # assert len(self.uniq_diff_labels) == len(condition_keys)
        # process the current condition labels
        curr_conds = []
        for idx, condition_key in enumerate(condition_keys):
            curr_cond = np.zeros(meta_cells.shape[0])
            for condition_id, condition in enumerate(self.data_dict["matching_dict"]["cond_names"][idx]):
                curr_cond[meta_cells[condition_key].values.squeeze() == condition] = condition_id
            curr_conds.append(curr_cond)
        
        # process the predict condition label
        if predict_conds is not None:
            for idx, condition in enumerate(predict_conds):
                predict_conds[idx] = np.array([np.where(self.data_dict["matching_dict"]["cond_names"][idx] == predict_conds[idx])[0][0]] * curr_conds[idx].shape[0])

        # process the batch labels
        curr_batch = torch.zeros(meta_cells.shape[0])
        for batch_id, batch in enumerate(self.data_dict["matching_dict"]["batch_name"]):
            curr_batch[meta_cells[batch_key].values.squeeze() == batch] = batch_id
        
        # process the input count matrix
        size_factor = np.tile(np.sum(input_counts, axis = 1, keepdims = True), (1, input_counts.shape[1]))/100
        counts_norm = np.log1p(input_counts/size_factor)
        # further standardize the count
        counts_norm = torch.FloatTensor(self.data_dict["scaler"].transform(counts_norm))

        # pass through training data to obtain delta
        with torch.no_grad():
            diff_labels = [[] for x in range(self.n_diff_factors)]
            z_ds_train = [[] for x in range(self.n_diff_factors)]

            for dataset in self.data_dict["datasets"]:
                # infer latent factor on train dataset
                dict_inf_train = self.inference(counts = dataset.counts_norm.to(self.device), batch_ids = dataset.batch_id[:,None].to(self.device), print_stat = True)
                # extract unshared-bio factors
                z_d = dict_inf_train["mu_d"]                
                for diff_factor, diff_label in enumerate(dataset.diff_labels):
                    # append condition label
                    diff_labels[diff_factor].extend([x for x in diff_label])
                    # append unshared-bio factor
                    z_ds_train[diff_factor].append(z_d[diff_factor])

            # pass through predicting data for prediction
            dic_inf_pred = self.inference(counts = counts_norm.to(self.device), batch_ids = curr_batch[:, None].to(self.device), print_stat = True)
            z_ds_pred = dic_inf_pred["mu_d"]
            
            diff_labels = [np.array(x) for x in diff_labels]
            z_ds_train = [torch.concat(x, dim = 0) for x in z_ds_train]   

            # store the centroid z_ds for each condition groups
            if predict_conds is not None:
                mean_zds = []
                for diff_factor in range(self.n_diff_factors):

                    # calculate the centroid of condition groups under diff_factor
                    mean_zds.append(torch.concat([torch.mean(z_ds_train[diff_factor][diff_labels[diff_factor] == x], dim = 0, keepdim = True) for x in self.uniq_diff_labels[diff_factor]], dim = 0))
                    # predicted centroid, of the shape (ncells, ndims)
                    pred_mean = mean_zds[diff_factor][predict_conds[diff_factor]]
                    # current centroid, of the shape (ncells, ndims)
                    curr_mean = mean_zds[diff_factor][curr_conds[diff_factor]]
                    # differences
                    delta = pred_mean - curr_mean
                    # latent space arithmetics
                    z_ds_pred[diff_factor] = z_ds_pred[diff_factor] + delta

            # generate data from the latent factors
            if predict_batch is None:   
                # predict_batch is kept the same         
                dict_gen = self.generative(z_c = dic_inf_pred["mu_c"], z_d = z_ds_pred, batch_ids = curr_batch[:,None].to(self.device))
            else:
                # predict_batch is given
                predict_batch = torch.tensor([[np.where(self.data_dict["matching_dict"]["batch_name"] == predict_batch)[0][0]] for x in range(curr_batch.shape[0])], device = self.device)
                dict_gen = self.generative(z_c = dic_inf_pred["mu_c"], z_d = z_ds_pred, batch_ids = predict_batch)

        return dict_gen["mu"].detach().cpu().numpy()
 

    def extract_gene_scores(self):
        """\
        Description:
        -------------
            Extract the condition-associated gene scores
        
        Return:
        -------------
            list containing scores of genes under each condition type
        """
        # checked
        scores = []
        # loop through all condition types
        for diff_factor in range(self.n_diff_factors):
            scores.append(self.Enc_ds[diff_factor].fc.fc_layers[0][0].weight.detach().cpu().pow(2).sum(dim=0).add(1e-8).pow(1/2.)[:self.ngenes].numpy())
        
        return scores



    # def predict_counts_optrans(self, input_counts, meta_cells, condition_keys, batch_key, predict_conds = None, predict_batch = None):
    #     """\
    #     Description:
    #     -------------
    #         Function for condition effect prediction.

    #     Parameters:
    #     -------------
    #         input_counts:
    #             the input count matrix, of the shape (ncells, ngenes), of the type np.array()

    #         meta_cells:
    #             the meta cell data frame of input count matrix
            
    #         condition_keys:
    #             the list of columns of the meta_cell data frame that correspond to the conditions
            
    #         batch_key:
    #             the column of the meta_cell data frame that correspond to the batches

    #         predict_conds:
    #             the condition label of the predicted dataset, 
    #             predict_conds should have length equal to the number of condition type, 
    #             and should have one condition group id for each condition type,
    #             default None, where the condition is kept the same as predict_dataset

    #         predict_batch:
    #             the batch id of the predicted dataset, default None, where batch_ids is kept the same as predict_dataset

    #     Returns:
    #     -------------
    #         predicted counts
        
    #     """
    #     # number of condition types should match
    #     # assert len(self.uniq_diff_labels) == len(predict_conds)        
    #     # assert len(self.uniq_diff_labels) == len(condition_keys)
    #     # process the current condition labels
    #     curr_conds = []
    #     for idx, condition_key in enumerate(condition_keys):
    #         curr_cond = np.zeros(meta_cells.shape[0])
    #         for condition_id, condition in enumerate(self.data_dict["matching_dict"]["cond_names"][idx]):
    #             curr_cond[meta_cells[condition_key].values.squeeze() == condition] = condition_id
    #         curr_conds.append(curr_cond)
        
    #     # process the predict condition label
    #     if predict_conds is not None:
    #         for idx, condition in enumerate(predict_conds):
    #             predict_conds[idx] = np.array([np.where(self.data_dict["matching_dict"]["cond_names"][idx] == predict_conds[idx])[0][0]] * curr_conds[idx].shape[0])

    #     # process the batch labels
    #     curr_batch = torch.zeros(meta_cells.shape[0])
    #     for batch_id, batch in enumerate(self.data_dict["matching_dict"]["batch_name"]):
    #         curr_batch[meta_cells[batch_key].values.squeeze() == batch] = batch_id
        
    #     # process the input count matrix
    #     size_factor = np.tile(np.sum(input_counts, axis = 1, keepdims = True), (1, input_counts.shape[1]))/100
    #     counts_norm = np.log1p(input_counts/size_factor)
    #     # further standardize the count
    #     counts_norm = torch.FloatTensor(self.data_dict["scaler"].transform(counts_norm))

    #     # pass through training data to obtain delta
    #     with torch.no_grad():
    #         diff_labels = [[] for x in range(self.n_diff_factors)]
    #         z_ds_train = [[] for x in range(self.n_diff_factors)]

    #         for dataset in self.data_dict["datasets"]:
    #             # infer latent factor on train dataset
    #             dict_inf_train = self.inference(counts = dataset.counts_norm.to(self.device), batch_ids = dataset.batch_id[:,None].to(self.device), print_stat = True)
    #             # extract unshared-bio factors
    #             z_d = dict_inf_train["mu_d"]                
    #             for diff_factor, diff_label in enumerate(dataset.diff_labels):
    #                 # append condition label
    #                 diff_labels[diff_factor].extend([x for x in diff_label])
    #                 # append unshared-bio factor
    #                 z_ds_train[diff_factor].append(z_d[diff_factor])

    #         # pass through predicting data for prediction
    #         dic_inf_pred = self.inference(counts = counts_norm.to(self.device), batch_ids = curr_batch[:, None].to(self.device), print_stat = True)
    #         z_ds_pred = dic_inf_pred["mu_d"]

    #         diff_labels = [np.array(x) for x in diff_labels]
    #         z_ds_train = [torch.concat(x, dim = 0) for x in z_ds_train]   

    #         # store the centroid z_ds for each condition groups
    #         if predict_conds is not None:
    #             for diff_factor in range(self.n_diff_factors):
    #                 # optimal transport
    #                 pred_distri = []
    #                 for cond in np.unique(predict_conds[diff_factor]):
    #                     pred_distri.append(z_ds_train[diff_factor][diff_labels[diff_factor] == cond,:].detach().cpu().numpy())
    #                 pred_distri = np.concatenate(pred_distri, axis = 0)                   

    #                 # curr_distri = []
    #                 # for cond in np.unique(curr_conds[diff_factor]):
    #                 #     curr_distri.append(z_ds_train[diff_factor][diff_labels[diff_factor] == cond,:].detach().cpu().numpy())
    #                 # curr_distri = np.concatenate(curr_distri, axis = 0)
                    
    #                 curr_distri = z_ds_pred[diff_factor].detach().cpu().numpy()
                    
    #                 # randomly downsample to reduce calculation time
    #                 np.random.seed(0)
    #                 pred_distri_anchor = pred_distri[np.random.choice(pred_distri.shape[0], 1000, replace = False), :]
    #                 curr_distri_anchor = curr_distri[np.random.choice(curr_distri.shape[0], 1000, replace = False), :]
                    
    #                 pdist_pred_anchor = pairwise_distances(pred_distri, pred_distri_anchor)
    #                 pdist_curr_anchor = pairwise_distances(curr_distri, curr_distri_anchor)
    #                 print(pred_distri.shape)
    #                 print(curr_distri.shape)

    #                 _, T = opt_trans.calc_trans(x1 = curr_distri_anchor, x2 = pred_distri_anchor, njobs = 1)
    #                 delta_anchor = pred_distri_anchor[np.argmax(T, axis = 1),:] - curr_distri_anchor
    #                 delta = delta_anchor[np.argmin(pdist_curr_anchor, axis = 1),:]                

    #                 z_ds_pred[diff_factor] = z_ds_pred[diff_factor] + torch.FloatTensor(delta).to(self.device)

    #         # generate data from the latent factors
    #         if predict_batch is None:   
    #             # predict_batch is kept the same         
    #             dict_gen = self.generative(z_c = dic_inf_pred["mu_c"], z_d = z_ds_pred, batch_ids = curr_batch[:,None].to(self.device))
    #         else:
    #             # predict_batch is given
    #             predict_batch = torch.tensor([[np.where(self.data_dict["matching_dict"]["batch_name"] == predict_batch)[0][0]] for x in range(curr_batch.shape[0])], device = self.device)
    #             dict_gen = self.generative(z_c = dic_inf_pred["mu_c"], z_d = z_ds_pred, batch_ids = predict_batch)

    #     return dict_gen["mu"].detach().cpu().numpy(), z_ds_pred, pred_distri, curr_distri
 

