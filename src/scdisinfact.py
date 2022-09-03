import sys, os
import torch
import numpy as np 
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

sys.path.append(".")
import model
import loss_function as loss_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dataset(Dataset):

    def __init__(self, counts, anno, diff_labels, batch_id):
        """
        Preprocessing similar to the DCA (dca.io.normalize)
        """
        assert not len(counts) == 0, "Count is empty"
        # normalize the count
        self.libsizes = np.tile(np.sum(counts, axis = 1, keepdims = True), (1, counts.shape[1]))
        # is the tile necessary?
        # in scanpy, np.median(counts) is used instead of 100 here
        self.counts_norm = counts/self.libsizes * 100
        self.counts_norm = np.log1p(self.counts_norm)
        self.counts = torch.FloatTensor(counts)

        # further standardize the count
        self.counts_stand = torch.FloatTensor(StandardScaler().fit_transform(self.counts_norm))
        if anno is not None:
            self.anno = torch.Tensor(anno)
        else:
            self.anno = None
        self.libsizes = torch.FloatTensor(self.libsizes)
        self.size_factor = self.libsizes / 100
        # make sure the input time point are integer
        self.diff_labels = []
        # loop through all types of diff labels
        for diff_label in diff_labels:
            assert diff_label.shape[0] == self.counts_stand.shape[0]
            self.diff_labels.append(torch.LongTensor(diff_label))
        self.batch_id = torch.Tensor(batch_id)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        if self.anno is not None:
            sample = {"batch_id": self.batch_id[idx], "diff_labels": [x[idx] for x in self.diff_labels], "count": self.counts[idx,:], "count_stand": self.counts_stand[idx,:], "index": idx, "anno": self.anno[idx], "size_factor": self.size_factor[idx]}
        else:
            sample = {"batch_id": self.batch_id[idx], "diff_labels": [x[idx] for x in self.diff_labels],  "count": self.counts[idx,:], "count_stand": self.counts_stand[idx,:], "index": idx, "size_factor": self.size_factor[idx]}
        return sample


class scdisinfact_ae(nn.Module):
    """\
    Description:
    --------------
        scDistinct

    """
    def __init__(self, datasets, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, lambs = [1,1,1,1,1,1], seed = 0, device = device, contr_loss = None):
        super().__init__()
        # initialize the parameters
        self.Ks = {"common_factor": Ks[0], "diff_factors": Ks[1:]}
        self.n_diff_types = len(self.Ks["diff_factors"])

        self.batch_size = batch_size
        self.interval = interval
        self.lr = lr
        self.lambs = lambs
        self.seed = seed 
        self.device = device
        self.contr = contr_loss

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # create data loaders
        self.train_loaders = []
        # create test data loaders
        self.test_loaders = []
        # store the number of cells for each batch
        self.ncells = []
        # store the number of unique diff labels [[unique label type 1], [unique label type 2], ...]
        self.diff_labels = [[] for x in range(self.n_diff_types)]
        for batch_id, dataset in enumerate(datasets):
            assert self.n_diff_types == len(dataset.diff_labels)
            # total number of cells
            self.ncells.append(dataset.counts.shape[0])
            # create train loader 
            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))
            # create test loader
            # prevent the case when the dataset is too large
            cutoff = 1000
            if len(dataset) > cutoff:
                print("test dataset shrink to {:d}".format(cutoff))
                idx = torch.randperm(n = cutoff)[:cutoff]
                test_dataset = Subset(dataset, idx)
            else:
                test_dataset = dataset
            self.test_loaders.append(DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False))

            # make sure that the genes are matched
            if batch_id == 0:
                self.ngenes = dataset.counts.shape[1]
            else:
                assert self.ngenes == dataset.counts.shape[1]
            # each dataset can have multiple diff_labels and each diff_label can have multiple conditions
            for idx, diff_label in enumerate(dataset.diff_labels):
                self.diff_labels[idx].extend([x.item() for x in torch.unique(diff_label)])
        
        for idx in range(self.n_diff_types):
            self.diff_labels[idx] = set(self.diff_labels[idx]) 

        # create model
        # encoder for common biological factor, + 1 here refers to the one batch ID
        self.Enc_c = model.Encoder(features = [self.ngenes, 256, 64, self.Ks["common_factor"]], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = []
        for idx in range(self.n_diff_types):
            self.Enc_ds.append(model.Encoder(features = [self.ngenes, 32, self.Ks["diff_factors"][idx]], dropout_rate = 0, negative_slope = 0.2).to(self.device))
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = []
        for idx in range(self.n_diff_types):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][idx], len(self.diff_labels[idx])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 64, 256, self.ngenes], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        
        # Discriminator for factor vae
        self.disc = model.Encoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 16, 8, 2], dropout_rate = -0, negative_slope = 0.2).to(self.device)

        # parameter when training the common biological factor
        self.param_common = nn.ModuleDict({"encoder_common": self.Enc_c, "decoder": self.Dec})
        # parameter when training the time factor
        self.param_diff = nn.ModuleDict({"encoder_diff": nn.ModuleList(self.Enc_ds), "classifier": nn.ModuleList(self.classifiers)})

        self.param_disc = nn.ModuleDict({"disc": self.disc})
        # declare optimizer for time factor and common biological factor separately
        self.opt = opt.Adam(
            [{'params': self.param_common.parameters()}, 
            {'params': self.param_diff.parameters()},
            {'params': self.param_disc.parameters()}], lr = self.lr
        )


    def train(self, nepochs = 50, recon_loss = "ZINB"):
        lamb_pi = 1e-5
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

        loss_tests = []
        loss_recon_tests = []
        loss_mmd_tests = []
        loss_class_tests = []
        loss_gl_d_tests = []
        loss_gl_c_tests = []
        loss_tc_tests = []

        for epoch in range(nepochs + 1):
            for data_batch in zip(*self.train_loaders):
                # 1. train on common factor
                loss_recon = 0
                loss_gl_c = 0
                z_cs = {"z": [], "batch_id": []}
                for x in data_batch:
                    # concatenate the batch ID with gene expression data as the input
                    z_c = self.Enc_c(x["count_stand"].to(self.device))
                    z_cs["z"].append(z_c)
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_t and classifier
                    with torch.no_grad():
                        z_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_d = Enc_d(x["count_stand"].to(self.device))
                            z_d.append(_z_d)
                            
                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["size_factor"].to(self.device), ridge_lambda = lamb_pi, device = self.device).loss(y_true = x["count"].to(self.device), y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = x["size_factor"].to(self.device), device = self.device).loss(y_true = x["count"].to(self.device), y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * x["size_factor"].to(self.device), x["count"].to(self.device))
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                    # GroupLasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss
                loss_mmd = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs[0] * loss_mmd + self.lambs[3] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_class = 0
                loss_mmd = 0
                loss_gl_d = 0
                loss_tc = 0

                z_ds = []
                for condi in range(self.n_diff_types):
                    z_ds.append({"diff_label": [], "z": [], "batch_id": []})

                for x in data_batch:
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_c = self.Enc_c(x["count_stand"].to(self.device))
                        
                    # loop through the diff encoder for each condition types
                    for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        _z_d = Enc_d(x["count_stand"].to(self.device))
                        
                        # NOTE: an alternative is to use z after sample
                        z_ds[condi]["z"].append(_z_d)
                        z_ds[condi]["diff_label"].append(x["diff_labels"][condi])
                        z_ds[condi]["batch_id"].append(x["batch_id"])

                        # make prediction
                        d_pred = classifier(_z_d)
                        # calculate the cross-entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][condi].to(self.device))
                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        
                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for condi in range(self.n_diff_types):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0]).to(self.device))

                # contrastive loss, note that the same data batch have cells from only one cluster, contrastive loss should be added jointly
                for condi in range(self.n_diff_types):
                    # contrastive loss, loop through all condition types
                    z_ds[condi]["z"] = torch.cat(z_ds[condi]["z"], dim = 0)
                    z_ds[condi]["diff_label"] = torch.cat(z_ds[condi]["diff_label"], dim = 0)
                    z_ds[condi]["batch_id"] = torch.cat(z_ds[condi]["batch_id"], dim = 0)

                    # condition specific mmd loss
                    for diff_label in range(self.n_diff_types):
                        idx = z_ds[condi]["diff_label"] == diff_label
                        loss_mmd += loss_func.maximum_mean_discrepancy(xs = z_ds[condi]["z"][idx, :], batch_ids = z_ds[condi]["batch_id"][idx], device = self.device)


                loss = self.lambs[0] * loss_mmd + self.lambs[1] * loss_class + self.lambs[3] * loss_gl_d + self.lambs[2] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_c = self.Enc_c(x["count_stand"].to(self.device))
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_d = Enc_d(x["count_stand"].to(self.device))
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for condi in range(self.n_diff_types):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0]).to(self.device))

                loss = self.lambs[2] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_mmd_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0

                z_ds = []
                z_cs = {"z": [], "batch_id": []}
                for condi in range(self.n_diff_types):
                    z_ds.append({"diff_label": [], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            # common encoder
                            z_c = self.Enc_c(x["count_stand"].to(self.device))
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            z_d = []
                            for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_d = Enc_d(x["count_stand"].to(self.device))
                                z_d.append(_z_d)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[condi]["z"].append(_z_d)
                                z_ds[condi]["diff_label"].append(x["diff_labels"][condi])
                                z_ds[condi]["batch_id"].append(x["batch_id"])

                                # make prediction for current condition type
                                d_pred = classifier(_z_d)
                                # calculate cross entropy loss
                                loss_class_test += ce_loss(input = d_pred, target = x["diff_labels"][condi].to(self.device))
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                
                            mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c] + z_d, dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for condi in range(self.n_diff_types):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0]).to(self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["size_factor"].to(self.device), ridge_lambda = lamb_pi, device = self.device).loss(y_true = x["count"].to(self.device), y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = x["size_factor"].to(self.device), device = self.device).loss(y_true = x["count"].to(self.device), y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * x["size_factor"].to(self.device), x["count"].to(self.device))
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        loss_mmd_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)

                        for condi in range(self.n_diff_types):
                            # condition specific mmd loss
                            z_ds[condi]["z"] = torch.cat(z_ds[condi]["z"], dim = 0)
                            z_ds[condi]["diff_label"] = torch.cat(z_ds[condi]["diff_label"], dim = 0)
                            z_ds[condi]["batch_id"] = torch.cat(z_ds[condi]["batch_id"], dim = 0)
                            for diff_label in range(self.n_diff_types):
                                idx = z_ds[condi]["diff_label"] == diff_label
                                loss_mmd_test += loss_func.maximum_mean_discrepancy(xs = z_ds[condi]["z"][idx, :], batch_ids = z_ds[condi]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon + self.lambs[0] * loss_mmd_test + self.lambs[1] * loss_class_test + self.lambs[3] * (loss_gl_d_test+loss_gl_c_test) + self.lambs[2] * loss_tc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss mmd: {:.5f}'.format(loss_mmd_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
                        ]
                        for i in info:
                            print("\t", i)              
                        
                        loss_tests.append(loss_test.item())
                        loss_recon_tests.append(loss_recon.item())
                        loss_mmd_tests.append(loss_mmd_test.item())
                        loss_class_tests.append(loss_class_test.item())
                        loss_gl_d_tests.append(loss_gl_d_test.item())
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())

        return loss_tests, loss_recon_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests


                
class scdisinfact(nn.Module):
    """\
    Description:
    --------------
        New model that separate the encoder and control backward gradient. (VARIATIONAL AUTOENCODER)

    """
    def __init__(self, datasets, reg_mmd_comm, reg_mmd_diff, reg_gl, reg_class = 1, reg_tc = 0.1, reg_kl = 1e-6, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, seed = 0, device = device):
        super().__init__()
        # initialize the parameters
        self.Ks = {"common_factor": Ks[0], "diff_factors": Ks[1:]}
        # number of diff factors
        self.n_diff_factors = len(self.Ks["diff_factors"])

        self.batch_size = batch_size
        self.interval = interval
        self.lr = lr
        self.lambs = {"mmd_comm": reg_mmd_comm, 
                     "mmd_diff": reg_mmd_diff, 
                     "class": reg_class, 
                     "tc": reg_tc, 
                     "gl": reg_gl, 
                     "kl": reg_kl}
        self.seed = seed 
        self.device = device

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # create data loaders
        self.train_loaders = []
        # create test data loaders
        self.test_loaders = []
        # store the number of cells for each batch
        self.ncells = []
        # store diff labels for each diff factor [[unique label type 1], [unique label type 2], ...]
        self.diff_labels = [[] for x in range(self.n_diff_factors)]
        for batch_id, dataset in enumerate(datasets):
            assert self.n_diff_factors == len(dataset.diff_labels)
            # total number of cells
            self.ncells.append(dataset.counts.shape[0])
            # create train loader 
            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))
            # create test loader
            # prevent the case when the dataset is too large, may not cover all the conditions if too small
            cutoff = 10000
            if len(dataset) > cutoff:
                print("test dataset shrink to {:d}".format(cutoff))
                idx = torch.randperm(n = cutoff)[:cutoff]
                test_dataset = Subset(dataset, idx)
            else:
                test_dataset = dataset
            self.test_loaders.append(DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False))

            # make sure that the genes are matched
            if batch_id == 0:
                self.ngenes = dataset.counts.shape[1]
            else:
                assert self.ngenes == dataset.counts.shape[1]
            # each dataset can have multiple diff_labels and each diff_label can have multiple conditions
            for idx, diff_label in enumerate(dataset.diff_labels):
                self.diff_labels[idx].extend([x.item() for x in torch.unique(diff_label)])
        
        # unique diff labels for each diff factor
        for diff_factor in range(self.n_diff_factors):
            self.diff_labels[diff_factor] = set(self.diff_labels[diff_factor]) 

        # create model
        # encoder for common biological factor, + 1 here refers to the one batch ID
        self.Enc_c = model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["common_factor"]], dropout_rate = 0.0, negative_slope = 0).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = nn.ModuleList([])
        for idx in range(self.n_diff_factors):
            self.Enc_ds.append(model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["diff_factors"][idx]], dropout_rate = 0.0, negative_slope = 0).to(self.device))
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = nn.ModuleList([])
        for idx in range(self.n_diff_factors):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][idx], len(self.diff_labels[idx])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]) + 1, 64, 256, self.ngenes], dropout_rate = 0.0, negative_slope = 0).to(self.device)
        
        # Discriminator for factor vae
        self.disc = model.Encoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 16, 8, 2], dropout_rate = 0.0, negative_slope = 0).to(self.device)

        # parameter when training the common biological factor
        self.param_common = nn.ModuleDict({"encoder_common": self.Enc_c, "decoder": self.Dec})
        # parameter when training the time factor
        self.param_diff = nn.ModuleDict({"encoder_diff": nn.ModuleList(self.Enc_ds), "classifier": nn.ModuleList(self.classifiers)})

        self.param_disc = nn.ModuleDict({"disc": self.disc})
        # declare optimizer for time factor and common biological factor separately
        self.opt = opt.Adam(
            [{'params': self.param_common.parameters()}, 
            {'params': self.param_diff.parameters()},
            {'params': self.param_disc.parameters()}], lr = self.lr
        )

    def reparametrize(self, mu, logvar, clamp = 0):
        # exp(0.5*log_var) = exp(log(\sqrt{var})) = \sqrt{var}
        std = logvar.mul(0.5).exp_()
        if clamp > 0:
            std = torch.clamp(std, min = clamp)

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def train_model(self, nepochs = 50, recon_loss = "NB", reg_contr = 0.0):
        lamb_pi = 1e-5
        best_loss = 1e3
        trigger = 0
        clamp_comm = 0.0
        clamp_diff = 0.0
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        # contr_loss = loss_func.SupConLoss(temperature=0.07, device = self.device)
        contr_loss = loss_func.CircleLoss(m=0.25, gamma=80)

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
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_d
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                                    
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss, between batches and conditions
                loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_recon = 0
                loss_class = 0
                loss_kl = 0
                loss_mmd_diff = 0
                loss_gl_d = 0
                loss_tc = 0
                loss_contr = 0

                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)

                    # loop through the diff encoder for each condition types
                    for diff_factor in range(self.n_diff_factors):
                        _z_mu_d, _z_logvar_d = self.Enc_ds[diff_factor](input)
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)

                        # NOTE: an alternative is to use z after sample
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")

                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                for diff_factor in range(self.n_diff_factors):
                    z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)

                    # contrastive loss
                    d_pred = F.normalize(z_ds[diff_factor]["z"])
                    loss_contr += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))
                    # loss_contr = torch.tensor([0], device = self.device)

                    # classifier loss
                    d_pred = self.classifiers[diff_factor](z_ds[diff_factor]["z"])
                    loss_class += ce_loss(input = d_pred, target = z_ds[diff_factor]["diff_label"].to(self.device))

                    # group lasso
                    loss_gl_d += loss_func.grouplasso(self.Enc_ds[diff_factor].fc.fc_layers[0].linear.weight, alpha = 0)

                    # condition specific mmd loss
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)


                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * (loss_class + reg_contr * loss_contr) + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    
                    
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi in range(self.n_diff_factors):
                            _z_mu_d, _z_logvar_d = self.Enc_ds[condi](input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0
                loss_contr_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor in range(self.n_diff_factors):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = self.Enc_ds[diff_factor](input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])
                                
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)

                            # contrastive loss
                            d_pred = F.normalize(z_ds[diff_factor]["z"])
                            loss_contr_test += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))
                            # loss_contr_test = torch.tensor([0], device = self.device)

                            # classifier loss
                            d_pred = self.classifiers[diff_factor](z_ds[diff_factor]["z"])
                            loss_class_test += ce_loss(input = d_pred, target = z_ds[diff_factor]["diff_label"].to(self.device))

                            # group lasso
                            loss_gl_d_test += loss_func.grouplasso(self.Enc_ds[diff_factor].fc.fc_layers[0].linear.weight, alpha = 0)

                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # group lasso
                        loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * (loss_class_test + reg_contr * loss_contr_test) \
                            + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss contrastive: {:.5f}'.format(loss_contr_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())                
                        
                        # update for early stopping 
                        if loss_test.item() < best_loss:
                            
                            best_loss = loss_test.item()
                            torch.save(self.state_dict(), f'../check_points/model.pt')
                            trigger = 0
                        else:
                            trigger += 1
                            print(trigger)
                            if trigger % 5 == 0:
                                self.opt.param_groups[0]['lr'] *= 0.5
                                print('Epoch: {}, shrink lr to {:.4f}'.format(epoch, self.opt.param_groups[0]['lr']))
                                if self.opt.param_groups[0]['lr'] <= 1e-6:
                                    break
                                else:
                                    self.load_state_dict(torch.load(f'../check_points/model.pt'))
                                    trigger = 0                            
                        
        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests



    def train_contr(self, nepochs = 50, recon_loss = "NB"):
        lamb_pi = 1e-5
        best_loss = 1e3
        trigger = 0
        clamp_comm = 0.0
        clamp_diff = 0.0
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        contr_loss = loss_func.SupConLoss(temperature=0.07, device = self.device)
        # contr_loss = loss_func.SNNLoss(temperature=0.07, device = self.device)
        contr_loss = loss_func.CircleLoss(m=0.25, gamma=80)
        # snnl_ce = loss_func.SNNLCrossEntropy()

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
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_d
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                                    
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss, between batches and conditions
                loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_recon = 0
                loss_class = 0
                loss_kl = 0
                loss_mmd_diff = 0
                loss_gl_d = 0
                loss_tc = 0

                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)

                    # loop through the diff encoder for each condition types
                    for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        _z_mu_d, _z_logvar_d = Enc_d(input)
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)

                        # NOTE: an alternative is to use z after sample
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")

                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                for diff_factor in range(self.n_diff_factors):
                    z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)

                    # contrastive loss
                    # d_pred = F.normalize(self.classifiers[diff_factor](z_ds[diff_factor]["z"]))                    
                    d_pred = F.normalize(z_ds[diff_factor]["z"])
                    loss_class += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))                    
                    # loss_class += contr_loss(d_pred.unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))
                    # loss_class += contr_loss(z_ds[diff_factor]["z"].unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))
                    # loss_class += snnl_ce.SNNL(z_ds[diff_factor]["z"], z_ds[diff_factor]["diff_label"].to(self.device), self.device) 

                    # condition specific mmd loss
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)


                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * loss_class + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                # loss = self.lambs["class"] * loss_class
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    
                    
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = Enc_d(input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])
                                
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)

                            d_pred = F.normalize(z_ds[diff_factor]["z"])
                            loss_class_test += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))
                            # make prediction for current condition type
                            # d_pred = F.normalize(self.classifiers[diff_factor](z_ds[diff_factor]["z"]))
                            # # calculate the contrastive loss
                            # loss_class_test += contr_loss(d_pred.unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))
                            # loss_class_test += snnl_ce.SNNL(z_ds[diff_factor]["z"], z_ds[diff_factor]["diff_label"].to(self.device), device = self.device) 
                            # loss_class_test += contr_loss(z_ds[diff_factor]["z"].unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))

                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * loss_class_test \
                            + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())                
                        
                        '''
                        # update for early stopping 
                        if loss_test.item() < best_loss:
                            
                            best_loss = loss_test.item()
                            torch.save(self.state_dict(), f'../check_points/model.pt')
                            trigger = 0
                        else:
                            trigger += 1
                            print(trigger)
                            if trigger % 5 == 0:
                                self.opt.param_groups[0]['lr'] *= 0.5
                                print('Epoch: {}, shrink lr to {:.4f}'.format(epoch, self.opt.param_groups[0]['lr']))
                                if self.opt.param_groups[0]['lr'] <= 1e-6:
                                    break
                                else:
                                    self.load_state_dict(torch.load(f'../check_points/model.pt'))
                                    trigger = 0                            
                        '''                
        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests


    def train_norecon(self, nepochs = 50, recon_loss = "ZINB"):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

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
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_d
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                                    
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss, between batches and conditions
                loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_recon = 0
                loss_class = 0
                loss_kl = 0
                loss_mmd_diff = 0
                loss_gl_d = 0
                loss_tc = 0

                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)

                    # loop through the diff encoder for each condition types
                    for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        _z_mu_d, _z_logvar_d = Enc_d(input)
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)

                        # NOTE: an alternative is to use z after sample
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        # make prediction
                        d_pred = classifier(_z_d)
                        # calculate the cross-entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                for diff_factor in range(self.n_diff_factors):
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)

                    # condition specific mmd loss
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)


                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * loss_class + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    
                    
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = Enc_d(input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                                # make prediction for current condition type
                                d_pred = classifier(_z_d)
                                # calculate cross entropy loss
                                loss_class_test += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * loss_class_test \
                            + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests

    def train_joint(self, nepochs = 50, recon_loss = "ZINB"):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        contr_loss = loss_func.SupConLoss(temperature=0.07)

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
                # 1. train on encoder and decoder
                loss_recon = 0
                loss_mmd_comm = 0
                loss_mmd_diff = 0
                loss_class = 0
                loss_gl_c = 0
                loss_gl_d = 0
                loss_kl = 0
                loss_tc = 0
                # latent space
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    

                    # common encoder
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    # store results
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                                             
                    z_cs["batch_id"].append(x["batch_id"])

                    # loop through all diff encoders
                    for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        # diff encoder for one condition
                        _z_mu_d, _z_logvar_d = Enc_d(input)
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                        # store results
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        """
                        # make prediction for current condition type
                        d_pred = classifier(_z_d)
                        # calculate cross entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                        """
                        loss_class += contr_loss(_z_d, x["diff_labels"][diff_factor].to(self.device))

                        # calculate group lasso for diff encoder
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                        # calculate the kl divergence
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        
                    
                    # decoder
                    mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))

                    # use discriminator
                    # create original samples, 
                    # NOTE: loss cannot stabilize with z_c.detach()
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    # NOTE: cannot stabilize with z_c.detach()
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
                    # group lasso for common encoder
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                    # reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                
                # MMD between batches for common encoder
                loss_mmd_comm += loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # MMD between batches for diff encoder
                for diff_factor in range(self.n_diff_factors):
                    # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += 10 * loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["mmd_diff"] * loss_mmd_diff \
                    + self.lambs["class"] * loss_class + self.lambs["gl"] * (loss_gl_d+loss_gl_c) + self.lambs["kl"] * loss_kl + self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()                   

                
                # 2. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_mmd_comm_test = 0
                loss_mmd_diff_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = Enc_d(input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                                # make prediction for current condition type
                                d_pred = classifier(_z_d)
                                # calculate cross entropy loss
                                loss_class_test += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test \
                            + self.lambs["class"] * loss_class_test + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test \
                                + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd (common): {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd (diff): {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests

    def test_model(self, counts, batch_ids, print_stat = False):

        assert counts.shape[0] == batch_ids.shape[0]
        assert batch_ids.shape[1] == 1
        # common encoder network
        input = torch.cat([counts, batch_ids], dim = 1).to(self.device)
        z_c, logvar_c = self.Enc_c(input)
        # diff encoder network
        z_d = []
        logvar_d = []
        for diff_factor in range(self.n_diff_factors):
            _z_d, _logvar_d = self.Enc_ds[diff_factor](input)
            z_d.append(_z_d)
            logvar_d.append(_logvar_d)

        # decoder
        z = torch.cat([z_c] + z_d + [batch_ids.to(self.device)], dim = 1)
        mu, pi, theta = self.Dec(z)

        if print_stat:
            print("mean z_c")
            print(torch.mean(z_c))
            print("mean var z_c")
            print(torch.mean(logvar_c.mul(0.5).exp_()))
            print("mean z_d")
            print([torch.mean(x) for x in z_d])
            print("mean var z_d")
            print([torch.mean(x.mul(0.5).exp_()) for x in logvar_d])
        
        return z_c, z_d, z, mu



                
class scdisinfact_scvi(nn.Module):
    """\
    Description:
    --------------
        New model that separate the encoder and control backward gradient. (VARIATIONAL AUTOENCODER)

    """
    def __init__(self, datasets, reg_mmd_comm, reg_mmd_diff, reg_gl, reg_class = 1, reg_tc = 0.1, reg_kl = 1e-6, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, seed = 0, device = device):
        super().__init__()
        # initialize the parameters
        self.Ks = {"common_factor": Ks[0], "diff_factors": Ks[1:]}
        # number of diff factors
        self.n_diff_factors = len(self.Ks["diff_factors"])

        self.batch_size = batch_size
        self.interval = interval
        self.lr = lr
        self.lambs = {"mmd_comm": reg_mmd_comm, 
                     "mmd_diff": reg_mmd_diff, 
                     "class": reg_class, 
                     "tc": reg_tc, 
                     "gl": reg_gl, 
                     "kl": reg_kl}
        self.seed = seed 
        self.device = device

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # create data loaders
        self.train_loaders = []
        # create test data loaders
        self.test_loaders = []
        # store the number of cells for each batch
        self.ncells = []
        # store diff labels for each diff factor [[unique label type 1], [unique label type 2], ...]
        self.diff_labels = [[] for x in range(self.n_diff_factors)]
        for batch_id, dataset in enumerate(datasets):
            assert self.n_diff_factors == len(dataset.diff_labels)
            # total number of cells
            self.ncells.append(dataset.counts.shape[0])
            # create train loader 
            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))
            # create test loader
            # prevent the case when the dataset is too large, may not cover all the conditions if too small
            cutoff = 10000
            if len(dataset) > cutoff:
                print("test dataset shrink to {:d}".format(cutoff))
                idx = torch.randperm(n = cutoff)[:cutoff]
                test_dataset = Subset(dataset, idx)
            else:
                test_dataset = dataset
            self.test_loaders.append(DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False))

            # make sure that the genes are matched
            if batch_id == 0:
                self.ngenes = dataset.counts.shape[1]
            else:
                assert self.ngenes == dataset.counts.shape[1]
            # each dataset can have multiple diff_labels and each diff_label can have multiple conditions
            for idx, diff_label in enumerate(dataset.diff_labels):
                self.diff_labels[idx].extend([x.item() for x in torch.unique(diff_label)])
        
        # unique diff labels for each diff factor
        for diff_factor in range(self.n_diff_factors):
            self.diff_labels[diff_factor] = set(self.diff_labels[diff_factor]) 

        # create model
        # encoder for common biological factor, + 1 here refers to the one batch ID
        self.Enc_c = model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["common_factor"]], dropout_rate = 0.0, negative_slope = 0).to(self.device)
        self.Enc_c = model.Encoder_scvi(            
            n_input = self.ngenes,
            n_output = self.Ks["common_factor"],
            n_cat_list = [],
            dropout_rate  = 0,
        )

        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = nn.ModuleList([])
        for idx in range(self.n_diff_factors):
            self.Enc_ds.append(model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["diff_factors"][idx]], dropout_rate = 0.0, negative_slope = 0).to(self.device))
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = nn.ModuleList([])
        for idx in range(self.n_diff_factors):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][idx], len(self.diff_labels[idx])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]) + 1, 64, 256, self.ngenes], dropout_rate = 0.0, negative_slope = 0).to(self.device)
        
        # Discriminator for factor vae
        self.disc = model.Encoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 16, 8, 2], dropout_rate = 0.0, negative_slope = 0).to(self.device)

        # parameter when training the common biological factor
        self.param_common = nn.ModuleDict({"encoder_common": self.Enc_c, "decoder": self.Dec})
        # parameter when training the time factor
        self.param_diff = nn.ModuleDict({"encoder_diff": nn.ModuleList(self.Enc_ds), "classifier": nn.ModuleList(self.classifiers)})

        self.param_disc = nn.ModuleDict({"disc": self.disc})
        # declare optimizer for time factor and common biological factor separately
        self.opt = opt.Adam(
            [{'params': self.param_common.parameters()}, 
            {'params': self.param_diff.parameters()},
            {'params': self.param_disc.parameters()}], lr = self.lr
        )

    def reparametrize(self, mu, logvar, clamp = 0):
        # exp(0.5*log_var) = exp(log(\sqrt{var})) = \sqrt{var}
        std = logvar.mul(0.5).exp_()
        if clamp > 0:
            std = torch.clamp(std, min = clamp)

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def train_model(self, nepochs = 50, recon_loss = "NB", reg_contr = 0.0):
        lamb_pi = 1e-5
        best_loss = 1e3
        trigger = 0
        clamp_comm = 0.0
        clamp_diff = 0.0
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        # contr_loss = loss_func.SupConLoss(temperature=0.07, device = self.device)
        contr_loss = loss_func.CircleLoss(m=0.25, gamma=80)

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
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_d
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                                    
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss, between batches and conditions
                loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_recon = 0
                loss_class = 0
                loss_kl = 0
                loss_mmd_diff = 0
                loss_gl_d = 0
                loss_tc = 0
                loss_contr = 0

                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)

                    # loop through the diff encoder for each condition types
                    for diff_factor in range(self.n_diff_factors):
                        _z_mu_d, _z_logvar_d = self.Enc_ds[diff_factor](input)
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)

                        # NOTE: an alternative is to use z after sample
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")

                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                for diff_factor in range(self.n_diff_factors):
                    z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)

                    # contrastive loss
                    d_pred = F.normalize(z_ds[diff_factor]["z"])
                    loss_contr += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))
                    # loss_contr = torch.tensor([0], device = self.device)

                    # classifier loss
                    d_pred = self.classifiers[diff_factor](z_ds[diff_factor]["z"])
                    loss_class += ce_loss(input = d_pred, target = z_ds[diff_factor]["diff_label"].to(self.device))

                    # group lasso
                    loss_gl_d += loss_func.grouplasso(self.Enc_ds[diff_factor].fc.fc_layers[0].linear.weight, alpha = 0)

                    # condition specific mmd loss
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)


                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * (loss_class + reg_contr * loss_contr) + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    
                    
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi in range(self.n_diff_factors):
                            _z_mu_d, _z_logvar_d = self.Enc_ds[condi](input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0
                loss_contr_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor in range(self.n_diff_factors):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = self.Enc_ds[diff_factor](input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])
                                
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)

                            # contrastive loss
                            d_pred = F.normalize(z_ds[diff_factor]["z"])
                            loss_contr_test += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))
                            # loss_contr_test = torch.tensor([0], device = self.device)

                            # classifier loss
                            d_pred = self.classifiers[diff_factor](z_ds[diff_factor]["z"])
                            loss_class_test += ce_loss(input = d_pred, target = z_ds[diff_factor]["diff_label"].to(self.device))

                            # group lasso
                            loss_gl_d_test += loss_func.grouplasso(self.Enc_ds[diff_factor].fc.fc_layers[0].linear.weight, alpha = 0)

                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # group lasso
                        loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * (loss_class_test + reg_contr * loss_contr_test) \
                            + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss contrastive: {:.5f}'.format(loss_contr_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())                
                        
                        # update for early stopping 
                        if loss_test.item() < best_loss:
                            
                            best_loss = loss_test.item()
                            torch.save(self.state_dict(), f'../check_points/model.pt')
                            trigger = 0
                        else:
                            trigger += 1
                            print(trigger)
                            if trigger % 5 == 0:
                                self.opt.param_groups[0]['lr'] *= 0.5
                                print('Epoch: {}, shrink lr to {:.4f}'.format(epoch, self.opt.param_groups[0]['lr']))
                                if self.opt.param_groups[0]['lr'] <= 1e-6:
                                    break
                                else:
                                    self.load_state_dict(torch.load(f'../check_points/model.pt'))
                                    trigger = 0                            
                        
        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests



    def train_contr(self, nepochs = 50, recon_loss = "NB"):
        lamb_pi = 1e-5
        best_loss = 1e3
        trigger = 0
        clamp_comm = 0.0
        clamp_diff = 0.0
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        contr_loss = loss_func.SupConLoss(temperature=0.07, device = self.device)
        # contr_loss = loss_func.SNNLoss(temperature=0.07, device = self.device)
        contr_loss = loss_func.CircleLoss(m=0.25, gamma=80)
        # snnl_ce = loss_func.SNNLCrossEntropy()

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
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_d
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                                    
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss, between batches and conditions
                loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_recon = 0
                loss_class = 0
                loss_kl = 0
                loss_mmd_diff = 0
                loss_gl_d = 0
                loss_tc = 0

                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)

                    # loop through the diff encoder for each condition types
                    for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        _z_mu_d, _z_logvar_d = Enc_d(input)
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)

                        # NOTE: an alternative is to use z after sample
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")

                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                for diff_factor in range(self.n_diff_factors):
                    z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)

                    # contrastive loss
                    # d_pred = F.normalize(self.classifiers[diff_factor](z_ds[diff_factor]["z"]))                    
                    d_pred = F.normalize(z_ds[diff_factor]["z"])
                    loss_class += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))                    
                    # loss_class += contr_loss(d_pred.unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))
                    # loss_class += contr_loss(z_ds[diff_factor]["z"].unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))
                    # loss_class += snnl_ce.SNNL(z_ds[diff_factor]["z"], z_ds[diff_factor]["diff_label"].to(self.device), self.device) 

                    # condition specific mmd loss
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)


                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * loss_class + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                # loss = self.lambs["class"] * loss_class
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    
                    
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c, clamp = clamp_comm)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = Enc_d(input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d, clamp = clamp_diff)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])
                                
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            z_ds[diff_factor]["z"] = torch.cat(z_ds[diff_factor]["z"], dim = 0)

                            d_pred = F.normalize(z_ds[diff_factor]["z"])
                            loss_class_test += contr_loss(d_pred, z_ds[diff_factor]["diff_label"].to(self.device))
                            # make prediction for current condition type
                            # d_pred = F.normalize(self.classifiers[diff_factor](z_ds[diff_factor]["z"]))
                            # # calculate the contrastive loss
                            # loss_class_test += contr_loss(d_pred.unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))
                            # loss_class_test += snnl_ce.SNNL(z_ds[diff_factor]["z"], z_ds[diff_factor]["diff_label"].to(self.device), device = self.device) 
                            # loss_class_test += contr_loss(z_ds[diff_factor]["z"].unsqueeze(1), z_ds[diff_factor]["diff_label"].to(self.device))

                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * loss_class_test \
                            + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())                
                        
                        '''
                        # update for early stopping 
                        if loss_test.item() < best_loss:
                            
                            best_loss = loss_test.item()
                            torch.save(self.state_dict(), f'../check_points/model.pt')
                            trigger = 0
                        else:
                            trigger += 1
                            print(trigger)
                            if trigger % 5 == 0:
                                self.opt.param_groups[0]['lr'] *= 0.5
                                print('Epoch: {}, shrink lr to {:.4f}'.format(epoch, self.opt.param_groups[0]['lr']))
                                if self.opt.param_groups[0]['lr'] <= 1e-6:
                                    break
                                else:
                                    self.load_state_dict(torch.load(f'../check_points/model.pt'))
                                    trigger = 0                            
                        '''                
        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests


    def train_norecon(self, nepochs = 50, recon_loss = "ZINB"):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

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
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_d
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d + [batch_id], dim = 1))
                    # calculate the reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                                    
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss, between batches and conditions
                loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_recon = 0
                loss_class = 0
                loss_kl = 0
                loss_mmd_diff = 0
                loss_gl_d = 0
                loss_tc = 0

                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)

                    # loop through the diff encoder for each condition types
                    for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        _z_mu_d, _z_logvar_d = Enc_d(input)
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)

                        # NOTE: an alternative is to use z after sample
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        # make prediction
                        d_pred = classifier(_z_d)
                        # calculate the cross-entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

                    # NOTE: calculate the total correlation, use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    # not updating the discriminator
                    for x in self.disc.parameters():
                        x.requires_grad = False
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                for diff_factor in range(self.n_diff_factors):
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)

                    # condition specific mmd loss
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)


                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * loss_class + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    
                    
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = Enc_d(input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                                # make prediction for current condition type
                                d_pred = classifier(_z_d)
                                # calculate cross entropy loss
                                loss_class_test += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test + self.lambs["class"] * loss_class_test \
                            + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd common: {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd diff: {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests

    def train_joint(self, nepochs = 50, recon_loss = "ZINB"):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
        contr_loss = loss_func.SupConLoss(temperature=0.07)

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
                # 1. train on encoder and decoder
                loss_recon = 0
                loss_mmd_comm = 0
                loss_mmd_diff = 0
                loss_class = 0
                loss_gl_c = 0
                loss_gl_d = 0
                loss_kl = 0
                loss_tc = 0
                # latent space
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = []
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    

                    # common encoder
                    z_mu_c, z_logvar_c = self.Enc_c(input)
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    # store results
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                                             
                    z_cs["batch_id"].append(x["batch_id"])

                    # loop through all diff encoders
                    for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        # diff encoder for one condition
                        _z_mu_d, _z_logvar_d = Enc_d(input)
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                        # store results
                        z_ds[diff_factor]["z"].append(_z_d)
                        z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                        z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                        z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                        z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                        """
                        # make prediction for current condition type
                        d_pred = classifier(_z_d)
                        # calculate cross entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                        """
                        loss_class += contr_loss(_z_d, x["diff_labels"][diff_factor].to(self.device))

                        # calculate group lasso for diff encoder
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                        # calculate the kl divergence
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        
                    
                    # decoder
                    mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))

                    # use discriminator
                    # create original samples, 
                    # NOTE: loss cannot stabilize with z_c.detach()
                    orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    # NOTE: cannot stabilize with z_c.detach()
                    perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc -= ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
                    # group lasso for common encoder
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                    # reconstruction loss
                    # ZINB
                    if recon_loss == "ZINB":
                        loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                    # NB
                    elif recon_loss == "NB":
                        loss_recon += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                    # MSE
                    elif recon_loss == "MSE":
                        mse_loss = nn.MSELoss()
                        loss_recon += mse_loss(mu * size_factor, count)
                    else:
                        raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                
                # MMD between batches for common encoder
                loss_mmd_comm += loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # MMD between batches for diff encoder
                for diff_factor in range(self.n_diff_factors):
                    # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                    z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                    z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                    z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                    for diff_label in self.diff_labels[diff_factor]:
                        idx = z_ds[diff_factor]["diff_label"] == diff_label
                        loss_mmd_diff += 10 * loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                # total loss
                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["mmd_diff"] * loss_mmd_diff \
                    + self.lambs["class"] * loss_class + self.lambs["gl"] * (loss_gl_d+loss_gl_c) + self.lambs["kl"] * loss_kl + self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()                   

                
                # 2. train on discriminator
                loss_tc = 0
                for x in self.disc.parameters():
                    x.requires_grad = True
                
                for x in data_batch:
                    count_stand = x["count_stand"].to(self.device, non_blocking=True)
                    batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                    size_factor = x["size_factor"].to(self.device, non_blocking=True)
                    count = x["count"].to(self.device, non_blocking=True)

                    # concatenate the batch ID with gene expression data as the input
                    input = torch.concat([count_stand, batch_id], dim = 1)                    

                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(input)
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(input)
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                
                    # NOTE: use discriminator
                    # create original samples
                    orig_samples = torch.cat([z_c] + z_d, dim = 1)
                    # create permuted samples
                    perm_idx = []
                    for diff_factor in range(self.n_diff_factors):
                        perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                    perm_samples = torch.cat([z_c, torch.cat([x[perm_idx[i], :] for i, x in enumerate(z_d)], dim = 1)], dim = 1)
                    # discriminator
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc += ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_mmd_comm_test = 0
                loss_mmd_diff_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for diff_factor in range(self.n_diff_factors):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            count_stand = x["count_stand"].to(self.device, non_blocking=True)
                            batch_id = x["batch_id"][:, None].to(self.device, non_blocking=True)
                            size_factor = x["size_factor"].to(self.device, non_blocking=True)
                            count = x["count"].to(self.device, non_blocking=True)

                            # concatenate the batch ID with gene expression data as the input
                            input = torch.concat([count_stand, batch_id], dim = 1)                    

                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(input)
                            # sampling
                            z_c = self.reparametrize(z_mu_c, z_logvar_c)
                            # calculate kl divergence
                            loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                            # NOTE: an alternative is to use z after sample
                            z_cs["z"].append(z_c)
                            z_cs["z_logvar"].append(z_logvar_c)
                            z_cs["z_mu"].append(z_mu_c)                                             
                            z_cs["batch_id"].append(x["batch_id"])

                            # diff encoder
                            for diff_factor, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                _z_mu_d, _z_logvar_d = Enc_d(input)
                                _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[diff_factor]["z"].append(_z_d)
                                z_ds[diff_factor]["z_logvar"].append(_z_logvar_d)
                                z_ds[diff_factor]["z_mu"].append(_z_mu_d)
                                z_ds[diff_factor]["diff_label"].append(x["diff_labels"][diff_factor])
                                z_ds[diff_factor]["batch_id"].append(x["batch_id"])

                                # make prediction for current condition type
                                d_pred = classifier(_z_d)
                                # calculate cross entropy loss
                                loss_class_test += ce_loss(input = d_pred, target = x["diff_labels"][diff_factor].to(self.device))
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 0)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)                        

                            mu, pi, theta = self.Dec(torch.concat([z_c] + [x["z"][-1] for x in z_ds] + [batch_id], dim = 1))
                            
                            # use discriminator
                            # create original samples
                            orig_samples = torch.cat([z_c, torch.cat([x["z"][-1] for x in z_ds], dim = 1)], dim = 1)
                            # create permuted samples
                            perm_idx = []
                            for diff_factor in range(self.n_diff_factors):
                                perm_idx.append(torch.randperm(n = orig_samples.shape[0]))
                            perm_samples = torch.cat([z_c, torch.cat([x["z"][-1][perm_idx[i], :] for i, x in enumerate(z_ds)], dim = 1)], dim = 1)
                            y_orig = self.disc(orig_samples)
                            y_perm = self.disc(perm_samples)
                            # total correlation correspond to cross entropy loss
                            loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                        target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0], device = self.device))
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 0)
                            # ZINB
                            if recon_loss == "ZINB":
                                loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = size_factor, ridge_lambda = lamb_pi, device = self.device).loss(y_true = count, y_pred = mu)
                            # NB
                            elif recon_loss == "NB":
                                loss_recon_test += loss_func.NB(theta = theta, scale_factor = size_factor, device = self.device).loss(y_true = count, y_pred = mu)
                            # MSE
                            elif recon_loss == "MSE":
                                mse_loss = nn.MSELoss()
                                loss_recon_test += mse_loss(mu * size_factor, count)
                            else:
                                raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")
                        
                        loss_mmd_comm_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                        

                        loss_mmd_diff_test = 0
                        for diff_factor in range(self.n_diff_factors):
                            # condition specific mmd loss: For each diff_factor, make sure that the cells with the same condition(diff label) across batches to be mixed 
                            z_ds[diff_factor]["z_mu"] = torch.cat(z_ds[diff_factor]["z_mu"], dim = 0)
                            z_ds[diff_factor]["diff_label"] = torch.cat(z_ds[diff_factor]["diff_label"], dim = 0)
                            z_ds[diff_factor]["batch_id"] = torch.cat(z_ds[diff_factor]["batch_id"], dim = 0)
                            for diff_label in self.diff_labels[diff_factor]:
                                idx = z_ds[diff_factor]["diff_label"] == diff_label
                                loss_mmd_diff_test += loss_func.maximum_mean_discrepancy(xs = z_ds[diff_factor]["z_mu"][idx, :], batch_ids = z_ds[diff_factor]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = loss_recon_test + self.lambs["mmd_comm"] * loss_mmd_comm_test + self.lambs["mmd_diff"] * loss_mmd_diff_test \
                            + self.lambs["class"] * loss_class_test + self.lambs["gl"] * (loss_gl_d_test+loss_gl_c_test) + self.lambs["kl"] * loss_kl_test \
                                + self.lambs["tc"] * loss_tc_test_disc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd (common): {:.5f}'.format(loss_mmd_comm_test.item()),
                            'loss mmd (diff): {:.5f}'.format(loss_mmd_diff_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                            'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                            'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
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
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests

    def test_model(self, counts, batch_ids, print_stat = False):

        assert counts.shape[0] == batch_ids.shape[0]
        assert batch_ids.shape[1] == 1
        # common encoder network
        input = torch.cat([counts, batch_ids], dim = 1).to(self.device)
        z_c, logvar_c = self.Enc_c(input)
        # diff encoder network
        z_d = []
        logvar_d = []
        for diff_factor in range(self.n_diff_factors):
            _z_d, _logvar_d = self.Enc_ds[diff_factor](input)
            z_d.append(_z_d)
            logvar_d.append(_logvar_d)

        # decoder
        z = torch.cat([z_c] + z_d + [batch_ids.to(self.device)], dim = 1)
        mu, pi, theta = self.Dec(z)

        if print_stat:
            print("mean z_c")
            print(torch.mean(z_c))
            print("mean var z_c")
            print(torch.mean(logvar_c.mul(0.5).exp_()))
            print("mean z_d")
            print([torch.mean(x) for x in z_d])
            print("mean var z_d")
            print([torch.mean(x.mul(0.5).exp_()) for x in logvar_d])
        
        return z_c, z_d, z, mu

