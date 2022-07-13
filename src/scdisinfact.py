import sys, os
import torch
import numpy as np 
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

sys.path.append(".")
import model
import loss_function as loss_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dataset(Dataset):

    def __init__(self, counts, anno, diff_labels, batch_id):

        assert not len(counts) == 0, "Count is empty"
        # normalize the count
        self.libsizes = np.tile(np.sum(counts, axis = 1, keepdims = True), (1, counts.shape[1]))
        # is the tile necessary?
        
        self.counts_norm = counts/self.libsizes * 100
        self.counts_norm = np.log1p(self.counts_norm)
        self.counts = torch.FloatTensor(counts)

        # further standardize the count
        self.counts_stand = torch.FloatTensor(StandardScaler().fit_transform(self.counts_norm))
        self.anno = torch.Tensor(anno)
        self.libsizes = torch.FloatTensor(self.libsizes)
        # make sure the input time point are integer
        self.diff_labels = []
        # loop through all types of diff labels
        for diff_label in diff_labels:
            self.diff_labels.append(torch.LongTensor(diff_label))
        self.batch_id = torch.Tensor(batch_id)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        if self.anno is not None:
            sample = {"batch_id": self.batch_id[idx], "diff_labels": [x[idx] for x in self.diff_labels], "count": self.counts[idx,:], "count_stand": self.counts_stand[idx,:], "index": idx, "anno": self.anno[idx], "libsize": self.libsizes[idx]}
        else:
            sample = {"batch_id": self.batch_id[idx], "diff_labels": [x[idx] for x in self.diff_labels],  "count": self.counts[idx,:], "count_stand": self.counts_stand[idx,:], "index": idx, "libsize": self.libsizes[idx]}
        return sample


class scdisinfact_ae(nn.Module):
    """\
    Description:
    --------------
        scDistinct

    """
    def __init__(self, datasets, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, lambs = [1,1,1], seed = 0, device = device, contr_loss = None):
        super().__init__()
        # initialize the parameters
        self.datasets = datasets
        self.Ks = {"common_factor": Ks[0], "diff_factor": Ks[1]}

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
        # store the number of cells for each batch
        self.ncells = []
        # store the number of unique diff labels
        self.diff_labels = []
        for batch_id, dataset in enumerate(self.datasets):
            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))

            self.ncells.append(dataset.counts.shape[0])

            # make sure that the genes are matched
            if batch_id == 0:
                self.ngenes = dataset.counts.shape[1]
            else:
                assert self.ngenes == dataset.counts.shape[1]
            # make sure that each dataset has one unique diff label
            diff_label = [x.item() for x in torch.unique(dataset.diff_label)]
            assert len(diff_label) == 1
            self.diff_labels.extend(diff_label)
        self.diff_labels = set(self.diff_labels) 

        # create model
        # encoder for common biological factor
        self.Enc_c = model.Encoder(features = [self.ngenes, 128, 64, self.Ks["common_factor"]], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # encoder for time factor
        self.Enc_d = model.Encoder(features = [self.ngenes, 128, 64, self.Ks["diff_factor"]], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(features = [self.Ks["common_factor"] + self.Ks["diff_factor"], 64, 128, self.ngenes], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # self.classifier = model.classifier(features = [self.Ks["diff_factor"], 4, len(self.diff_labels)]).to(self.device)
        # use a linear classifier as stated in the paper
        self.classifier = nn.Linear(self.Ks["diff_factor"], len(self.diff_labels)).to(self.device)

        # parameter when training the common biological factor
        self.param_common = [
            {'params': self.Enc_c.parameters()},
            {'params': self.Dec.parameters()}            
        ]

        # parameter when training the time factor
        self.param_diff = [
            {'params': self.Enc_d.parameters()},
            {'params': self.classifier.parameters()}            
        ]

        # declare optimizer for time factor and common biological factor separately
        self.opt_common = opt.Adam(self.param_common, lr = self.lr)
        self.opt_diff = opt.Adam(self.param_diff, lr = self.lr)

    
    def train(self, nepochs = 50):
        lamb_pi = 1e-5
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
                
        loss_tests = []
        loss_recon_tests = []
        loss_mmd_tests = []
        loss_class_tests = []
        loss_gl_d_tests = []
        loss_gl_c_tests = []

        for epoch in range(nepochs + 1):
            for data_batch in zip(*self.train_loaders):
                loss_recon = 0
                loss_class = 0
                loss_contr = 0
                loss_gl_d = 0
                loss_gl_c = 0
                
                zs_contr = {}
                zs_contr['diff_label'] = []
                zs_contr['x'] = []
                zs_mmd = []
                # 1. train on common factor
                for batch_id, x in enumerate(data_batch):
                    z_c = self.Enc_c(x["count_stand"].to(self.device))
                    # freeze the gradient of Enc_d and classifier
                    with torch.no_grad():
                        z_d = self.Enc_d(x["count_stand"].to(self.device))
                    z = torch.concat((z_c, z_d), dim = 1)
                    mu, pi, theta = self.Dec(z)
                    # calculate the reconstruction loss
                    loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(self.device), ridge_lambda = lamb_pi, device = self.device).loss(y_true = x["count"].to(self.device), y_pred = mu)
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                    zs_mmd.append(z_c)
                                
                loss_mmd = loss_func.maximum_mean_discrepancy(xs = zs_mmd, ref_batch = 1, device = self.device)
                loss = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd + self.lambs[4] * loss_gl_c                
                loss.backward()

                # NOTE: check the gradient of Enc_t and classifier to be None or 0
                with torch.no_grad():
                    for x in self.Enc_d.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for x in self.classifier.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for x in self.Enc_c.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.Dec.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                
                # update common encoder parameter
                self.opt_common.step()
                self.opt_common.zero_grad()

                # 2. train on diff factor
                for batch_id, x in enumerate(data_batch):
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_c = self.Enc_c(x["count_stand"].to(self.device))
                    z_d = self.Enc_d(x["count_stand"].to(self.device))
                    zs_contr['x'].append(z_d)
                    zs_contr['diff_label'].append(x["diff_label"])
                    d_pred = self.classifier(z_d)
                    # calculate the group lasso for diff encoder        
                    loss_gl_d += loss_func.grouplasso(self.Enc_d.fc.fc_layers[0].linear.weight)
                    # calculate the cross-entropy loss
                    loss_class += ce_loss(input = d_pred, target = x["diff_label"].to(self.device))
                # calculate the contrastive loss, note that the same data batch have cells from only one cluster, contrastive loss should be added jointly
                loss_contr = self.contr(torch.cat(zs_contr['x']), torch.cat(zs_contr['diff_label']))
                loss = self.lambs[2] * loss_class + self.lambs[3] * loss_contr + self.lambs[4] * loss_gl_d
                loss.backward()

                # NOTE: check the gradient of Enc_c and Dec to be 0 or None
                with torch.no_grad():
                    for x in self.Enc_d.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.classifier.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.Enc_c.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for x in self.Dec.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)

                # update the diff encoder parameter
                self.opt_diff.step()
                self.opt_diff.zero_grad()
            
            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_mmd_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                
                zs_mmd = []
                with torch.no_grad():
                    for dataset in self.datasets:
                        z_c = self.Enc_c(dataset.counts_stand.to(self.device))
                        z_d = self.Enc_d(dataset.counts_stand.to(self.device))
                        z = torch.concat((z_c, z_d), dim = 1)
                        mu, pi, theta = self.Dec(z)
                        d_pred = self.classifier(z_d)
                        zs_mmd.append(z_c)

                        loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                        loss_gl_d_test += loss_func.grouplasso(self.Enc_d.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                        loss_class_test += ce_loss(input = d_pred, target = dataset.diff_label.to(self.device))
                        loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = dataset.libsizes.to(self.device), ridge_lambda = lamb_pi, device = self.device).loss(y_true = dataset.counts.to(self.device), y_pred = mu)
                    
                    loss_mmd_test = loss_func.maximum_mean_discrepancy(xs = zs_mmd, ref_batch = 1, device = self.device)
                    loss_test = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd_test + self.lambs[2] * loss_class_test + self.lambs[4] * (loss_gl_d_test+loss_gl_c_test)
                    # loss_test = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd_test + self.lambs[2] * loss_class_test

                    print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                    info = [
                        'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                        'loss mmd: {:.5f}'.format(loss_mmd_test.item()),
                        'loss classification: {:.5f}'.format(loss_class_test.item()),
                        'loss group lasso common: {:.5f}'.format(loss_gl_d_test.item()), 
                        'loss group lasso diff: {:.5f}'.format(loss_gl_c_test.item()), 
                    ]
                    for i in info:
                        print("\t", i)              
                    loss_tests.append(loss_test.item())
                    loss_recon_tests.append(loss_recon_test.item())
                    loss_mmd_tests.append(loss_mmd_test.item())
                    loss_class_tests.append(loss_class_test.item())
                    loss_gl_d_tests.append(loss_gl_d_test.item())
                    loss_gl_c_tests.append(loss_gl_c_test.item())

        return loss_tests, loss_recon_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests

                
class scdisinfact_vae(nn.Module):
    """\
    Description:
    --------------
        New model that separate the encoder and control backward gradient. (VARIATIONAL AUTOENCODER)

    """
    def __init__(self, datasets, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, lambs = [1,1,1,1,1,1], seed = 0, device = device, contr_loss = None):
        super().__init__()
        # initialize the parameters
        self.datasets = datasets
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
        # store the number of cells for each batch
        self.ncells = []
        # store the number of unique diff labels [[unique label type 1], [unique label type 2], ...]
        self.diff_labels = [[] for x in range(self.n_diff_types)]
        for batch_id, dataset in enumerate(self.datasets):
            assert self.n_diff_types == len(dataset.diff_labels)

            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))
            self.ncells.append(dataset.counts.shape[0])

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
        self.Enc_c = model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["common_factor"]], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = []
        for idx in range(self.n_diff_types):
            self.Enc_ds.append(model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["diff_factors"][idx]], dropout_rate = 0, negative_slope = 0.2).to(self.device))
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = []
        for idx in range(self.n_diff_types):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][idx], len(self.diff_labels[idx])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 64, 256, self.ngenes], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        
        # parameter when training the common biological factor
        self.param_common = nn.ModuleDict({"encoder_common": self.Enc_c, "decoder": self.Dec})
        # parameter when training the time factor
        self.param_diff = nn.ModuleDict({"encoder_diff": nn.ModuleList(self.Enc_ds), "classifier": nn.ModuleList(self.classifiers)})

        # declare optimizer for time factor and common biological factor separately
        self.opt = opt.Adam(
            [{'params': self.param_common.parameters()}, 
            {'params': self.param_diff.parameters()}], lr = self.lr
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def total_correlation(self, z_c, z_d, mu_c, mu_d, logvar_c, logvar_d, mode = "MWS"):
        # total number of cells, dataset_size
        N = sum(self.ncells)
        batch_size = z_c.shape[0]
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        # logqz of the shape (i, j, k) for q(z_i[k]|x_j)
        _logqz_c = loss_func.log_gaussian(sample = z_c.view(batch_size, 1, self.Ks["common_factor"]), 
                                         mu = mu_c.view(1, batch_size, self.Ks["common_factor"]), 
                                         logvar = logvar_c.view(1, batch_size, self.Ks["common_factor"]),
                                         device = device
                                        )
        _logqz_u = loss_func.log_gaussian(sample = z_d.view(batch_size, 1, sum(self.Ks["diff_factors"])), 
                                         mu = mu_d.view(1, batch_size, sum(self.Ks["diff_factors"])), 
                                         logvar = logvar_d.view(1, batch_size, sum(self.Ks["diff_factors"])),
                                         device = device
                                        )
        _logqz = loss_func.log_gaussian(sample = torch.cat([z_c, z_d], dim = 1).view(batch_size, 1, self.Ks["common_factor"] + sum(self.Ks["diff_factors"])), 
                                         mu = torch.cat([mu_c, mu_d], dim = 1).view(1, batch_size, self.Ks["common_factor"] + sum(self.Ks["diff_factors"])), 
                                         logvar = torch.cat([logvar_c, logvar_d], dim = 1).view(1, batch_size, self.Ks["common_factor"] + sum(self.Ks["diff_factors"])),
                                         device = device       
                                        )
        # minibatch weighted sampling
        # first sum over k to obtain q(z_i|x_j) from q(z_i[k]|x_j)
        # then sum over j to marginalize
        if mode == "MWS":
            logqz_c = (loss_func.logsumexp(_logqz_c.sum(2), dim=1, keepdim=False) - np.log(batch_size * N))
            logqz_u = (loss_func.logsumexp(_logqz_u.sum(2), dim=1, keepdim=False) - np.log(batch_size * N))
            logqz = (loss_func.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - np.log(batch_size * N))
        else:
            logqz_c = (loss_func.logsumexp(_logqz_c.sum(2), dim=1, keepdim=False) - np.log(batch_size))
            logqz_u = (loss_func.logsumexp(_logqz_u.sum(2), dim=1, keepdim=False) - np.log(batch_size))
            logqz = (loss_func.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - np.log(batch_size))
            

        # then average over i
        tc = logqz.sum()/batch_size - logqz_c.sum()/batch_size - logqz_u.sum()/batch_size
        return tc

    def train(self, nepochs = 50):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

        loss_tests = []
        loss_recon_tests = []
        loss_kl_tests = []
        loss_mmd_tests = []
        loss_class_tests = []
        loss_gl_d_tests = []
        loss_gl_c_tests = []
        loss_contr_tests = []
        loss_tc_tests = []

        for epoch in range(nepochs + 1):
            for data_batch in zip(*self.train_loaders):
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for batch_id, x in enumerate(data_batch):
                    # concatenate the batch ID with gene expression data as the input
                    z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])

                    # freeze the gradient of Enc_t and classifier
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)
                        
                        z_ds["z"].append(torch.cat(z_d, dim = 1))
                        z_ds["z_mu"].append(torch.cat(z_mu_d, dim = 1))
                        z_ds["z_logvar"].append(torch.cat(z_logvar_d, dim = 1))
                        z_ds["batch_id"].append(x["batch_id"])

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))

                    # calculate the reconstruction loss
                    loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(self.device), ridge_lambda = lamb_pi).loss(y_true = x["count"].to(self.device), y_pred = mu)
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss
                loss_mmd = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # calculate the total correlation
                # loss_tc = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                #                                 z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0))
                loss_tc = 0
                # total loss
                loss = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd + self.lambs[5] * loss_kl + self.lambs[4] * loss_gl_c + self.lambs[6] * loss_tc
                loss.backward()

                # check the gradient of Enc_t and classifier to be None or 0
                with torch.no_grad():
                    for Enc_d in self.Enc_ds:
                        for x in Enc_d.parameters():
                            assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for classifier in self.classifiers:
                        for x in classifier.parameters():
                            assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6
                    for x in self.Enc_c.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.Dec.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                     
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_class = 0
                loss_kl = 0
                loss_mmd = 0
                loss_gl_d = 0
                loss_contr = 0

                zd_mmd = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for condi in range(self.n_diff_types):
                    zd_mmd.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for batch_id, x in enumerate(data_batch):
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_cs["z"].append(z_c)
                        z_cs["z_logvar"].append(z_logvar_c)
                        z_cs["z_mu"].append(z_mu_c)                        
                        z_cs["batch_id"].append(x["batch_id"])

                    # loop through the diff encoder for each condition types
                    for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        z_mu_d, z_logvar_d = Enc_d(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                        # sample the latent space for each diff encoder
                        z_d = self.reparametrize(z_mu_d, z_logvar_d)

                        # NOTE: an alternative is to use z after sample
                        zd_mmd[condi]["z"].append(z_d)
                        zd_mmd[condi]["z_logvar"].append(z_logvar_d)
                        zd_mmd[condi]["z_mu"].append(z_mu_d)
                        zd_mmd[condi]["diff_label"].append(x["diff_labels"][condi])
                        zd_mmd[condi]["batch_id"].append(x["batch_id"])

                        # make prediction
                        d_pred = classifier(z_d)
                        # calculate the cross-entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][condi].to(self.device))
                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(z_mu_d.pow(2).add_(z_logvar_d.exp()).mul_(-1).add_(1).add_(z_logvar_d)).mul_(-0.5)
                    
                    z_ds["z"].append(torch.cat([x["z"][batch_id] for x in zd_mmd], dim = 1))
                    z_ds["z_logvar"].append(torch.cat([x["z_logvar"][batch_id] for x in zd_mmd], dim = 1))
                    z_ds["z_mu"].append(torch.cat([x["z_mu"][batch_id] for x in zd_mmd], dim = 1))

                # contrastive loss, note that the same data batch have cells from only one cluster, contrastive loss should be added jointly
                for condi in range(self.n_diff_types):
                    # contrastive loss, loop through all condition types
                    zd_mmd[condi]["z_mu"] = torch.cat(zd_mmd[condi]["z_mu"], dim = 0)
                    zd_mmd[condi]["diff_label"] = torch.cat(zd_mmd[condi]["diff_label"], dim = 0)
                    zd_mmd[condi]["batch_id"] = torch.cat(zd_mmd[condi]["batch_id"], dim = 0)
                    loss_contr += self.contr(zd_mmd[condi]["z_mu"], zd_mmd[condi]["diff_label"])  

                    # condition specific mmd loss
                    for diff_label in range(self.n_diff_types):
                        idx = zd_mmd[condi]["diff_label"] == diff_label
                        loss_mmd += loss_func.maximum_mean_discrepancy(xs = zd_mmd[condi]["z_mu"][idx, :], batch_ids = zd_mmd[condi]["batch_id"][idx], device = self.device)

                # calculate the total correlation
                if epoch >= 0.75 * nepochs:
                    loss_tc = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                                                    z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0))
                else:
                    loss_tc = 0

                loss = self.lambs[1] * loss_mmd + self.lambs[2] * loss_class + self.lambs[5] * loss_kl + self.lambs[3] * loss_contr + self.lambs[4] * loss_gl_d + self.lambs[6] * loss_tc
                loss.backward()

                # check the gradient of Enc_c and Dec to be 0 or None
                with torch.no_grad():
                    for Enc_d in self.Enc_ds:
                        for x in Enc_d.parameters():
                            assert torch.sum(x.grad.data.abs()) != 0
                    for classifier in self.classifiers:
                        for x in classifier.parameters():
                            assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.Enc_c.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6
                    for x in self.Dec.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6

                self.opt.step()
                self.opt.zero_grad()
            
            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_mmd_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0
                loss_contr_test = 0

                zd_mmd = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for condi in range(self.n_diff_types):
                    zd_mmd.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for batch_id, dataset in enumerate(self.datasets):
                        # common encoder
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([dataset.counts_stand, torch.FloatTensor([[batch_id]]).expand(dataset.counts_stand.shape[0], 1)], dim = 1).to(self.device))
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        # calculate kl divergence
                        loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                        # NOTE: an alternative is to use z after sample
                        z_cs["z"].append(z_c)
                        z_cs["z_logvar"].append(z_logvar_c)
                        z_cs["z_mu"].append(z_mu_c)                                             
                        z_cs["batch_id"].append(dataset.batch_id)

                        # diff encoder
                        z_d = []
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            # loop through all condition types, and conduct sampling
                            z_mu_d, z_logvar_d = Enc_d(torch.concat([dataset.counts_stand, torch.FloatTensor([[batch_id]]).expand(dataset.counts_stand.shape[0], 1)], dim = 1).to(self.device))
                            _z_d = self.reparametrize(z_mu_d, z_logvar_d)
                            z_d.append(_z_d)
                            
                            # NOTE: an alternative is to use z after sample
                            zd_mmd[condi]["z"].append(_z_d)
                            zd_mmd[condi]["z_logvar"].append(z_logvar_d)
                            zd_mmd[condi]["z_mu"].append(z_mu_d)
                            zd_mmd[condi]["diff_label"].append(dataset.diff_labels[condi])
                            zd_mmd[condi]["batch_id"].append(dataset.batch_id)

                            # make prediction for current condition type
                            d_pred = classifier(_z_d)
                            # calculate cross entropy loss
                            loss_class_test += ce_loss(input = d_pred, target = dataset.diff_labels[condi].to(self.device))
                            # calculate group lasso
                            loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                            # calculate the kl divergence
                            loss_kl_test += torch.sum(z_mu_d.pow(2).add_(z_logvar_d.exp()).mul_(-1).add_(1).add_(z_logvar_d)).mul_(-0.5)                        

                        z_ds["z"].append(torch.cat([x["z"][batch_id] for x in zd_mmd], dim = 1))
                        z_ds["z_logvar"].append(torch.cat([x["z_logvar"][batch_id] for x in zd_mmd], dim = 1))
                        z_ds["z_mu"].append(torch.cat([x["z_mu"][batch_id] for x in zd_mmd], dim = 1))

                        mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))
                        
                        loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                        loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = dataset.libsizes.to(self.device), ridge_lambda = lamb_pi).loss(y_true = dataset.counts.to(self.device), y_pred = mu)
                        
                    loss_mmd_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0))
                    # calculate the total correlation
                    loss_tc_test = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                                                    z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0), mode = None)
                    loss_tc_test_mws = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                                                    z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0))
                    
                    for condi in range(self.n_diff_types):
                        # contrastive loss, loop through all condition types
                        loss_contr_test += self.contr(torch.cat(zd_mmd[condi]['z_mu']), torch.cat(zd_mmd[condi]['diff_label']))  
                        # condition specific mmd loss
                        zd_mmd[condi]["z_mu"] = torch.cat(zd_mmd[condi]["z_mu"], dim = 0)
                        zd_mmd[condi]["diff_label"] = torch.cat(zd_mmd[condi]["diff_label"], dim = 0)
                        zd_mmd[condi]["batch_id"] = torch.cat(zd_mmd[condi]["batch_id"], dim = 0)
                        for diff_label in range(self.n_diff_types):
                            idx = zd_mmd[condi]["diff_label"] == diff_label
                            loss_mmd_test += loss_func.maximum_mean_discrepancy(xs = zd_mmd[condi]["z_mu"][idx, :], batch_ids = zd_mmd[condi]["batch_id"][idx], device = self.device)

                    # total loss
                    loss_test = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd_test + self.lambs[2] * loss_class_test + self.lambs[3] * loss_contr + self.lambs[4] * (loss_gl_d_test+loss_gl_c_test) + self.lambs[5] * loss_kl_test + self.lambs[6] * loss_tc
                    
                    print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                    info = [
                        'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                        'loss kl: {:.5f}'.format(loss_kl_test.item()),
                        'loss mmd: {:.5f}'.format(loss_mmd_test.item()),
                        'loss classification: {:.5f}'.format(loss_class_test.item()),
                        'loss contrastive: {:.5f}'.format(loss_contr_test.item()),
                        'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                        'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                        'loss total correlation: {:.5f}'.format(loss_tc_test.item()),
                        'loss total correlation (NWS): {:.5f}'.format(loss_tc_test_mws.item())
                    ]
                    for i in info:
                        print("\t", i)              
                    
                    loss_tests.append(loss_test.item())
                    loss_recon_tests.append(loss_recon.item())
                    loss_mmd_tests.append(loss_mmd_test.item())
                    loss_class_tests.append(loss_class_test.item())
                    loss_contr_tests.append(loss_contr_test.item())
                    loss_kl_tests.append(loss_kl_test.item())
                    loss_gl_d_tests.append(loss_gl_d_test.item())
                    loss_gl_c_tests.append(loss_gl_c_test.item())
                    loss_tc_tests.append(loss_tc_test.item())

                    # # update for early stopping 
                    # if loss_test.item() < best_loss:# - 0.01 * abs(best_loss):
                        
                    #     best_loss = loss.item()
                    #     torch.save(self.state_dict(), f'../check_points/model.pt')
                    #     count = 0
                    # else:
                    #     count += 1
                    #     print(count)
                    #     if count % int(nepochs/self.interval) == 0:
                    #         self.opt_time.param_groups[0]['lr'] *= 0.5
                    #         self.opt_common.param_groups[0]['lr'] *= 0.5
                    #         print('Epoch: {}, shrink lr to {:.4f}'.format(epoch + 1, self.opt_time.param_groups[0]['lr']))
                    #         if self.opt_time.param_groups[0]['lr'] < 1e-6:
                    #             break
                    #         else:
                    #             self.load_state_dict(torch.load(f'../check_points/model.pt'))
                    #             count = 0                            

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests
    
                
class scdisinfact_factorvae(nn.Module):
    """\
    Description:
    --------------
        New model that separate the encoder and control backward gradient. (VARIATIONAL AUTOENCODER)

    """
    def __init__(self, datasets, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, lambs = [1,1,1,1,1,1], seed = 0, device = device, contr_loss = None):
        super().__init__()
        # initialize the parameters
        self.datasets = datasets
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
        # store the number of cells for each batch
        self.ncells = []
        # store the number of unique diff labels [[unique label type 1], [unique label type 2], ...]
        self.diff_labels = [[] for x in range(self.n_diff_types)]
        for batch_id, dataset in enumerate(self.datasets):
            assert self.n_diff_types == len(dataset.diff_labels)

            self.train_loaders.append(DataLoader(dataset, batch_size = self.batch_size, shuffle = True))
            self.ncells.append(dataset.counts.shape[0])

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
        self.Enc_c = model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["common_factor"]], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = []
        for idx in range(self.n_diff_types):
            self.Enc_ds.append(model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["diff_factors"][idx]], dropout_rate = 0, negative_slope = 0.2).to(self.device))
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = []
        for idx in range(self.n_diff_types):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][idx], len(self.diff_labels[idx])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 64, 256, self.ngenes], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        
        # Discriminator for factor vae
        self.disc = model.Encoder(features = [self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), 8, 2], dropout_rate = -0, negative_slope = 0.2).to(self.device)

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

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def total_correlation(self, z_c, z_d, mu_c, mu_d, logvar_c, logvar_d, mode = "MWS"):
        # total number of cells, dataset_size
        N = sum(self.ncells)
        batch_size = z_c.shape[0]
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        # logqz of the shape (i, j, k) for q(z_i[k]|x_j)
        _logqz_c = loss_func.log_gaussian(sample = z_c.view(batch_size, 1, self.Ks["common_factor"]), 
                                         mu = mu_c.view(1, batch_size, self.Ks["common_factor"]), 
                                         logvar = logvar_c.view(1, batch_size, self.Ks["common_factor"]),
                                         device = device
                                        )
        _logqz_u = loss_func.log_gaussian(sample = z_d.view(batch_size, 1, sum(self.Ks["diff_factors"])), 
                                         mu = mu_d.view(1, batch_size, sum(self.Ks["diff_factors"])), 
                                         logvar = logvar_d.view(1, batch_size, sum(self.Ks["diff_factors"])),
                                         device = device
                                        )
        _logqz = loss_func.log_gaussian(sample = torch.cat([z_c, z_d], dim = 1).view(batch_size, 1, self.Ks["common_factor"] + sum(self.Ks["diff_factors"])), 
                                         mu = torch.cat([mu_c, mu_d], dim = 1).view(1, batch_size, self.Ks["common_factor"] + sum(self.Ks["diff_factors"])), 
                                         logvar = torch.cat([logvar_c, logvar_d], dim = 1).view(1, batch_size, self.Ks["common_factor"] + sum(self.Ks["diff_factors"])),
                                         device = device       
                                        )
        # minibatch weighted sampling
        # first sum over k to obtain q(z_i|x_j) from q(z_i[k]|x_j)
        # then sum over j to marginalize
        if mode == "MWS":
            logqz_c = (loss_func.logsumexp(_logqz_c.sum(2), dim=1, keepdim=False) - np.log(batch_size * N))
            logqz_u = (loss_func.logsumexp(_logqz_u.sum(2), dim=1, keepdim=False) - np.log(batch_size * N))
            logqz = (loss_func.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - np.log(batch_size * N))
        else:
            logqz_c = (loss_func.logsumexp(_logqz_c.sum(2), dim=1, keepdim=False) - np.log(batch_size))
            logqz_u = (loss_func.logsumexp(_logqz_u.sum(2), dim=1, keepdim=False) - np.log(batch_size))
            logqz = (loss_func.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - np.log(batch_size))
            

        # then average over i
        tc = logqz.sum()/batch_size - logqz_c.sum()/batch_size - logqz_u.sum()/batch_size
        return tc

    def train(self, nepochs = 50):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

        loss_tests = []
        loss_recon_tests = []
        loss_kl_tests = []
        loss_mmd_tests = []
        loss_class_tests = []
        loss_gl_d_tests = []
        loss_gl_c_tests = []
        loss_contr_tests = []
        loss_tc_tests = []

        for epoch in range(nepochs + 1):
            for data_batch in zip(*self.train_loaders):
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for batch_id, x in enumerate(data_batch):
                    # concatenate the batch ID with gene expression data as the input
                    z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])

                    # freeze the gradient of Enc_t and classifier
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)
                        
                        z_ds["z"].append(torch.cat(z_d, dim = 1))
                        z_ds["z_mu"].append(torch.cat(z_mu_d, dim = 1))
                        z_ds["z_logvar"].append(torch.cat(z_logvar_d, dim = 1))
                        z_ds["batch_id"].append(x["batch_id"])

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))

                    # calculate the reconstruction loss
                    loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(self.device), ridge_lambda = lamb_pi).loss(y_true = x["count"].to(self.device), y_pred = mu)
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss
                loss_mmd = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # calculate the total correlation
                loss_tc = 0
                # total loss
                loss = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd + self.lambs[5] * loss_kl + self.lambs[4] * loss_gl_c + self.lambs[6] * loss_tc
                loss.backward()

                # check the gradient of Enc_t and classifier to be None or 0
                with torch.no_grad():
                    for Enc_d in self.Enc_ds:
                        for x in Enc_d.parameters():
                            assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for classifier in self.classifiers:
                        for x in classifier.parameters():
                            assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6
                    for x in self.Enc_c.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.Dec.parameters():
                        assert torch.sum(x.grad.data.abs()) != 0
                    # check the gradient of discriminator to be None or 0
                    for x in self.disc.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                     
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_class = 0
                loss_kl = 0
                loss_mmd = 0
                loss_gl_d = 0
                loss_contr = 0

                zd_mmd = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for condi in range(self.n_diff_types):
                    zd_mmd.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for batch_id, x in enumerate(data_batch):
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_cs["z"].append(z_c)
                        z_cs["z_logvar"].append(z_logvar_c)
                        z_cs["z_mu"].append(z_mu_c)                        
                        z_cs["batch_id"].append(x["batch_id"])

                    # loop through the diff encoder for each condition types
                    for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        z_mu_d, z_logvar_d = Enc_d(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                        # sample the latent space for each diff encoder
                        z_d = self.reparametrize(z_mu_d, z_logvar_d)

                        # NOTE: an alternative is to use z after sample
                        zd_mmd[condi]["z"].append(z_d)
                        zd_mmd[condi]["z_logvar"].append(z_logvar_d)
                        zd_mmd[condi]["z_mu"].append(z_mu_d)
                        zd_mmd[condi]["diff_label"].append(x["diff_labels"][condi])
                        zd_mmd[condi]["batch_id"].append(x["batch_id"])

                        # make prediction
                        d_pred = classifier(z_d)
                        # calculate the cross-entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][condi].to(self.device))
                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(z_mu_d.pow(2).add_(z_logvar_d.exp()).mul_(-1).add_(1).add_(z_logvar_d)).mul_(-0.5)
                    
                    z_ds["z"].append(torch.cat([x["z"][batch_id] for x in zd_mmd], dim = 1))
                    z_ds["z_logvar"].append(torch.cat([x["z_logvar"][batch_id] for x in zd_mmd], dim = 1))
                    z_ds["z_mu"].append(torch.cat([x["z_mu"][batch_id] for x in zd_mmd], dim = 1))

                # contrastive loss, note that the same data batch have cells from only one cluster, contrastive loss should be added jointly
                for condi in range(self.n_diff_types):
                    # contrastive loss, loop through all condition types
                    zd_mmd[condi]["z_mu"] = torch.cat(zd_mmd[condi]["z_mu"], dim = 0)
                    zd_mmd[condi]["diff_label"] = torch.cat(zd_mmd[condi]["diff_label"], dim = 0)
                    zd_mmd[condi]["batch_id"] = torch.cat(zd_mmd[condi]["batch_id"], dim = 0)
                    loss_contr += self.contr(zd_mmd[condi]["z_mu"], zd_mmd[condi]["diff_label"])  

                    # condition specific mmd loss
                    for diff_label in range(self.n_diff_types):
                        idx = zd_mmd[condi]["diff_label"] == diff_label
                        loss_mmd += loss_func.maximum_mean_discrepancy(xs = zd_mmd[condi]["z_mu"][idx, :], batch_ids = zd_mmd[condi]["batch_id"][idx], device = self.device)

                # calculate the total correlation
                # if epoch >= 0.75 * nepochs:
                #     loss_tc = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                #                                     z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0))
                # else:
                #     loss_tc = 0
                # NOTE: use discriminator
                # create original samples
                orig_samples = torch.cat([torch.cat(z_cs["z"]), torch.cat(z_ds["z"])], dim = 1)
                # create permuted samples
                perm_idx = torch.randperm(n = orig_samples.shape[0])
                perm_samples = torch.cat([torch.cat(z_cs["z"]), torch.cat(z_ds["z"])[perm_idx, :]], dim = 1)
                # not updating the discriminator
                for x in self.disc.parameters():
                    x.requires_grad = False
                y_orig = self.disc(orig_samples)
                y_perm = self.disc(perm_samples)
                # total correlation correspond to cross entropy loss
                loss_tc = - ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0]).to(self.device))

                loss = self.lambs[1] * loss_mmd + self.lambs[2] * loss_class + self.lambs[5] * loss_kl + self.lambs[3] * loss_contr + self.lambs[4] * loss_gl_d + self.lambs[6] * loss_tc
                loss.backward()

                # check the gradient of Enc_c and Dec to be 0 or None
                with torch.no_grad():
                    for Enc_d in self.Enc_ds:
                        for x in Enc_d.parameters():
                            assert torch.sum(x.grad.data.abs()) != 0
                    for classifier in self.classifiers:
                        for x in classifier.parameters():
                            assert torch.sum(x.grad.data.abs()) != 0
                    for x in self.Enc_c.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6
                    for x in self.Dec.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6
                    # check the gradient of discriminator
                    for x in self.disc.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)

                self.opt.step()
                self.opt.zero_grad()

                
                # 3. train on discriminator
                for x in self.disc.parameters():
                    x.requires_grad = True
                z_cs = {"z": []}
                z_ds = {"z": []}
                for batch_id, x in enumerate(data_batch):
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_cs["z"].append(z_c)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(torch.concat([x["count_stand"], torch.FloatTensor([[batch_id]]).expand(x["count_stand"].shape[0], 1)], dim = 1).to(self.device))
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                        z_d = torch.cat(z_d, dim = 1)
                        z_ds["z"].append(z_d)

                # NOTE: use discriminator
                # create original samples
                orig_samples = torch.cat([torch.cat(z_cs["z"]), torch.cat(z_ds["z"])], dim = 1)
                # create permuted samples
                perm_idx = torch.randperm(n = orig_samples.shape[0])
                perm_samples = torch.cat([torch.cat(z_cs["z"]), torch.cat(z_ds["z"])[perm_idx, :]], dim = 1)
                # discriminator
                y_orig = self.disc(orig_samples)
                y_perm = self.disc(perm_samples)
                # total correlation correspond to cross entropy loss
                loss_tc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), target = torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0]).to(self.device))

                loss = self.lambs[6] * loss_tc
                loss.backward()

                # check the gradient of the discriminator to be not 0
                with torch.no_grad():
                    for Enc_d in self.Enc_ds:
                        for x in Enc_d.parameters():
                            assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for classifier in self.classifiers:
                        for x in classifier.parameters():
                            assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                    for x in self.Enc_c.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6
                    for x in self.Dec.parameters():
                        assert (x.grad is None) or (torch.sum(x.grad.data.abs()) == 0)
                        # assert torch.sum(x.grad.data.abs()) < 1e-6     

                self.opt.step()
                self.opt.zero_grad()    

            # TEST:
            if epoch % self.interval == 0:
                loss_recon_test = 0
                loss_mmd_test = 0
                loss_class_test = 0
                loss_gl_c_test = 0
                loss_gl_d_test = 0
                loss_kl_test = 0
                loss_contr_test = 0

                zd_mmd = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                z_ds = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for condi in range(self.n_diff_types):
                    zd_mmd.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for batch_id, dataset in enumerate(self.datasets):
                        # common encoder
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([dataset.counts_stand, torch.FloatTensor([[batch_id]]).expand(dataset.counts_stand.shape[0], 1)], dim = 1).to(self.device))
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        # calculate kl divergence
                        loss_kl_test += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                        # NOTE: an alternative is to use z after sample
                        z_cs["z"].append(z_c)
                        z_cs["z_logvar"].append(z_logvar_c)
                        z_cs["z_mu"].append(z_mu_c)                                             
                        z_cs["batch_id"].append(dataset.batch_id)

                        # diff encoder
                        z_d = []
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            # loop through all condition types, and conduct sampling
                            z_mu_d, z_logvar_d = Enc_d(torch.concat([dataset.counts_stand, torch.FloatTensor([[batch_id]]).expand(dataset.counts_stand.shape[0], 1)], dim = 1).to(self.device))
                            _z_d = self.reparametrize(z_mu_d, z_logvar_d)
                            z_d.append(_z_d)
                            
                            # NOTE: an alternative is to use z after sample
                            zd_mmd[condi]["z"].append(_z_d)
                            zd_mmd[condi]["z_logvar"].append(z_logvar_d)
                            zd_mmd[condi]["z_mu"].append(z_mu_d)
                            zd_mmd[condi]["diff_label"].append(dataset.diff_labels[condi])
                            zd_mmd[condi]["batch_id"].append(dataset.batch_id)

                            # make prediction for current condition type
                            d_pred = classifier(_z_d)
                            # calculate cross entropy loss
                            loss_class_test += ce_loss(input = d_pred, target = dataset.diff_labels[condi].to(self.device))
                            # calculate group lasso
                            loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                            # calculate the kl divergence
                            loss_kl_test += torch.sum(z_mu_d.pow(2).add_(z_logvar_d.exp()).mul_(-1).add_(1).add_(z_logvar_d)).mul_(-0.5)                        

                        z_ds["z"].append(torch.cat([x["z"][batch_id] for x in zd_mmd], dim = 1))
                        z_ds["z_logvar"].append(torch.cat([x["z_logvar"][batch_id] for x in zd_mmd], dim = 1))
                        z_ds["z_mu"].append(torch.cat([x["z_mu"][batch_id] for x in zd_mmd], dim = 1))

                        mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))
                        
                        loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                        loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = dataset.libsizes.to(self.device), ridge_lambda = lamb_pi).loss(y_true = dataset.counts.to(self.device), y_pred = mu)
                        
                    loss_mmd_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0))
                    # calculate the total correlation
                    loss_tc_test = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                                                    z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0), mode = None)
                    loss_tc_test_mws = self.total_correlation(z_c = torch.cat(z_cs["z"], dim = 0), mu_c = torch.cat(z_cs["z_mu"], dim = 0), logvar_c = torch.cat(z_cs["z_logvar"], dim = 0),\
                                                    z_d = torch.cat(z_ds["z"], dim = 0), mu_d = torch.cat(z_ds["z_mu"], dim = 0), logvar_d = torch.cat(z_ds["z_logvar"], dim = 0))
                    # use discriminator
                    # create original samples
                    orig_samples = torch.cat([torch.cat(z_cs["z"]), torch.cat(z_ds["z"])], dim = 1)
                    # create permuted samples
                    perm_idx = torch.randperm(n = orig_samples.shape[0])
                    perm_samples = torch.cat([torch.cat(z_cs["z"]), torch.cat(z_ds["z"])[perm_idx, :]], dim = 1)
                    y_orig = self.disc(orig_samples)
                    y_perm = self.disc(perm_samples)
                    # total correlation correspond to cross entropy loss
                    loss_tc_test_disc = ce_loss(input = torch.cat([y_orig, y_perm], dim = 0), 
                                                target =torch.tensor([0] * y_orig.shape[0] + [1] * y_perm.shape[0]).to(self.device))



                    for condi in range(self.n_diff_types):
                        # contrastive loss, loop through all condition types
                        loss_contr_test += self.contr(torch.cat(zd_mmd[condi]['z_mu']), torch.cat(zd_mmd[condi]['diff_label']))  
                        # condition specific mmd loss
                        zd_mmd[condi]["z_mu"] = torch.cat(zd_mmd[condi]["z_mu"], dim = 0)
                        zd_mmd[condi]["diff_label"] = torch.cat(zd_mmd[condi]["diff_label"], dim = 0)
                        zd_mmd[condi]["batch_id"] = torch.cat(zd_mmd[condi]["batch_id"], dim = 0)
                        for diff_label in range(self.n_diff_types):
                            idx = zd_mmd[condi]["diff_label"] == diff_label
                            loss_mmd_test += loss_func.maximum_mean_discrepancy(xs = zd_mmd[condi]["z_mu"][idx, :], batch_ids = zd_mmd[condi]["batch_id"][idx], device = self.device)

                    # total loss
                    loss_test = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd_test + self.lambs[2] * loss_class_test + self.lambs[3] * loss_contr + self.lambs[4] * (loss_gl_d_test+loss_gl_c_test) + self.lambs[5] * loss_kl_test + self.lambs[6] * loss_tc
                    
                    print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                    info = [
                        'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                        'loss kl: {:.5f}'.format(loss_kl_test.item()),
                        'loss mmd: {:.5f}'.format(loss_mmd_test.item()),
                        'loss classification: {:.5f}'.format(loss_class_test.item()),
                        'loss contrastive: {:.5f}'.format(loss_contr_test.item()),
                        'loss group lasso common: {:.5f}'.format(loss_gl_c_test.item()), 
                        'loss group lasso diff: {:.5f}'.format(loss_gl_d_test.item()), 
                        'loss total correlation: {:.5f}'.format(loss_tc_test.item()),
                        'loss total correlation (NWS): {:.5f}'.format(loss_tc_test_mws.item()),
                        'loss total correlation (disc): {:.5f}'.format(loss_tc_test_disc.item())
                    ]
                    for i in info:
                        print("\t", i)              
                    
                    loss_tests.append(loss_test.item())
                    loss_recon_tests.append(loss_recon.item())
                    loss_mmd_tests.append(loss_mmd_test.item())
                    loss_class_tests.append(loss_class_test.item())
                    loss_contr_tests.append(loss_contr_test.item())
                    loss_kl_tests.append(loss_kl_test.item())
                    loss_gl_d_tests.append(loss_gl_d_test.item())
                    loss_gl_c_tests.append(loss_gl_c_test.item())
                    loss_tc_tests.append(loss_tc_test.item())

                    # # update for early stopping 
                    # if loss_test.item() < best_loss:# - 0.01 * abs(best_loss):
                        
                    #     best_loss = loss.item()
                    #     torch.save(self.state_dict(), f'../check_points/model.pt')
                    #     count = 0
                    # else:
                    #     count += 1
                    #     print(count)
                    #     if count % int(nepochs/self.interval) == 0:
                    #         self.opt_time.param_groups[0]['lr'] *= 0.5
                    #         self.opt_common.param_groups[0]['lr'] *= 0.5
                    #         print('Epoch: {}, shrink lr to {:.4f}'.format(epoch + 1, self.opt_time.param_groups[0]['lr']))
                    #         if self.opt_time.param_groups[0]['lr'] < 1e-6:
                    #             break
                    #         else:
                    #             self.load_state_dict(torch.load(f'../check_points/model.pt'))
                    #             count = 0                            

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests
    

                
class scdisinfact(nn.Module):
    """\
    Description:
    --------------
        New model that separate the encoder and control backward gradient. (VARIATIONAL AUTOENCODER)

    """
    def __init__(self, datasets, Ks = [16, 8], batch_size = 64, interval = 10, lr = 5e-4, lambs = [1,1,1,1,1,1], seed = 0, device = device, contr_loss = None):
        super().__init__()
        # initialize the parameters
        self.datasets = datasets
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
        for batch_id, dataset in enumerate(self.datasets):
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
        self.Enc_c = model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["common_factor"]], dropout_rate = 0, negative_slope = 0.2).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = []
        for idx in range(self.n_diff_types):
            self.Enc_ds.append(model.Encoder_var(features = [self.ngenes + 1, 256, 64, self.Ks["diff_factors"][idx]], dropout_rate = 0, negative_slope = 0.2).to(self.device))
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

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def train(self, nepochs = 50):
        lamb_pi = 1e-5
        # TODO: in the vanilla vae, beta should be 1, in beta-vae, beta >1, the visualization for both cases are not good. Try beta with smaller value
        ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

        loss_tests = []
        loss_recon_tests = []
        loss_kl_tests = []
        loss_mmd_tests = []
        loss_class_tests = []
        loss_gl_d_tests = []
        loss_gl_c_tests = []
        loss_contr_tests = []
        loss_tc_tests = []

        for epoch in range(nepochs + 1):
            for data_batch in zip(*self.train_loaders):
                # 1. train on common factor
                loss_recon = 0
                loss_kl = 0
                loss_gl_c = 0
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for x in data_batch:
                    # concatenate the batch ID with gene expression data as the input
                    z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                    # sampling
                    z_c = self.reparametrize(z_mu_c, z_logvar_c)
                    # calculate kl divergence
                    loss_kl += torch.sum(z_mu_c.pow(2).add_(z_logvar_c.exp()).mul_(-1).add_(1).add_(z_logvar_c)).mul_(-0.5)     
                    z_cs["z"].append(z_c)
                    z_cs["z_logvar"].append(z_logvar_c)
                    z_cs["z_mu"].append(z_mu_c)                        
                    z_cs["batch_id"].append(x["batch_id"])
                    # freeze the gradient of Enc_t and classifier
                    with torch.no_grad():
                        z_d = []
                        z_mu_d = []
                        z_logvar_d = []
                        # loop through the encoder for each condition type
                        for Enc_d in self.Enc_ds:
                            _z_mu_d, _z_logvar_d = Enc_d(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                            # sampling
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
                            z_d.append(_z_d)
                            z_mu_d.append(_z_mu_d)
                            z_logvar_d.append(_z_logvar_d)

                    mu, pi, theta = self.Dec(torch.concat([z_c] + z_d, dim = 1))
                    # calculate the reconstruction loss
                    loss_recon += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(self.device), ridge_lambda = lamb_pi).loss(y_true = x["count"].to(self.device), y_pred = mu)
                    # NOTE: calculate the group lasso for common encoder, we corrently don't need to use group lasso
                    loss_gl_c += 0 # loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight)
                
                # calculate global mmd loss
                loss_mmd = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0), device = self.device)
                # total loss
                loss = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd + self.lambs[5] * loss_kl + self.lambs[4] * loss_gl_c
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                
                # 2. train on diff factor
                loss_class = 0
                loss_kl = 0
                loss_mmd = 0
                loss_gl_d = 0
                loss_contr = 0
                loss_tc = 0

                z_ds = []
                for condi in range(self.n_diff_types):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                for x in data_batch:
                    # freeze the gradient of Enc_c
                    with torch.no_grad():
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                        # sampling
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)

                    # loop through the diff encoder for each condition types
                    for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                        _z_mu_d, _z_logvar_d = Enc_d(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                        # sample the latent space for each diff encoder
                        _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)

                        # NOTE: an alternative is to use z after sample
                        z_ds[condi]["z"].append(_z_d)
                        z_ds[condi]["z_logvar"].append(_z_logvar_d)
                        z_ds[condi]["z_mu"].append(_z_mu_d)
                        z_ds[condi]["diff_label"].append(x["diff_labels"][condi])
                        z_ds[condi]["batch_id"].append(x["batch_id"])

                        # make prediction
                        d_pred = classifier(_z_d)
                        # calculate the cross-entropy loss
                        loss_class += ce_loss(input = d_pred, target = x["diff_labels"][condi].to(self.device))
                        # calculate the group lasso for each diff encoder     
                        loss_gl_d += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight)
                        # calculate kl divergence with prior distribution
                        loss_kl += torch.sum(_z_mu_d.pow(2).add_(_z_logvar_d.exp()).mul_(-1).add_(1).add_(_z_logvar_d)).mul_(-0.5)

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
                    z_ds[condi]["z_mu"] = torch.cat(z_ds[condi]["z_mu"], dim = 0)
                    z_ds[condi]["diff_label"] = torch.cat(z_ds[condi]["diff_label"], dim = 0)
                    z_ds[condi]["batch_id"] = torch.cat(z_ds[condi]["batch_id"], dim = 0)
                    loss_contr += self.contr(z_ds[condi]["z_mu"], z_ds[condi]["diff_label"])  

                    # condition specific mmd loss
                    for diff_label in range(self.n_diff_types):
                        idx = z_ds[condi]["diff_label"] == diff_label
                        loss_mmd += loss_func.maximum_mean_discrepancy(xs = z_ds[condi]["z_mu"][idx, :], batch_ids = z_ds[condi]["batch_id"][idx], device = self.device)


                loss = self.lambs[1] * loss_mmd + self.lambs[2] * loss_class + self.lambs[5] * loss_kl + self.lambs[3] * loss_contr + self.lambs[4] * loss_gl_d + self.lambs[6] * loss_tc
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
                        z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                        z_c = self.reparametrize(z_mu_c, z_logvar_c)
                        z_d = []
                        # loop through the diff encoder for each condition types
                        for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                            _z_mu_d, _z_logvar_d = Enc_d(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                            _z_d = self.reparametrize(_z_mu_d, _z_logvar_d)
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

                loss = self.lambs[6] * loss_tc
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
                loss_kl_test = 0
                loss_contr_test = 0

                z_ds = []
                z_cs = {"z_mu":[], "z_logvar":[], "z": [], "batch_id": []}
                for condi in range(self.n_diff_types):
                    z_ds.append({"diff_label": [], "z_mu":[], "z_logvar":[], "z": [], "batch_id": []})

                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        for x in data_batch:
                            # common encoder
                            z_mu_c, z_logvar_c = self.Enc_c(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
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
                            z_d = []
                            for condi, (Enc_d, classifier) in enumerate(zip(self.Enc_ds, self.classifiers)):
                                # loop through all condition types, and conduct sampling
                                z_mu_d, z_logvar_d = Enc_d(torch.concat([x["count_stand"], x["batch_id"][:, None]], dim = 1).to(self.device))
                                _z_d = self.reparametrize(z_mu_d, z_logvar_d)
                                z_d.append(_z_d)
                                
                                # NOTE: an alternative is to use z after sample
                                z_ds[condi]["z"].append(_z_d)
                                z_ds[condi]["z_logvar"].append(z_logvar_d)
                                z_ds[condi]["z_mu"].append(z_mu_d)
                                z_ds[condi]["diff_label"].append(x["diff_labels"][condi])
                                z_ds[condi]["batch_id"].append(x["batch_id"])

                                # make prediction for current condition type
                                d_pred = classifier(_z_d)
                                # calculate cross entropy loss
                                loss_class_test += ce_loss(input = d_pred, target = x["diff_labels"][condi].to(self.device))
                                # calculate group lasso
                                loss_gl_d_test += loss_func.grouplasso(Enc_d.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                                # calculate the kl divergence
                                loss_kl_test += torch.sum(z_mu_d.pow(2).add_(z_logvar_d.exp()).mul_(-1).add_(1).add_(z_logvar_d)).mul_(-0.5)                        

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
        
                            loss_gl_c_test += loss_func.grouplasso(self.Enc_c.fc.fc_layers[0].linear.weight, alpha = 1e-2)
                            loss_recon_test += loss_func.ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(self.device), ridge_lambda = lamb_pi).loss(y_true = x["count"].to(self.device), y_pred = mu)

                        loss_mmd_test = loss_func.maximum_mean_discrepancy(xs = torch.cat(z_cs["z_mu"], dim = 0), batch_ids = torch.cat(z_cs["batch_id"], dim = 0))
                        


                        for condi in range(self.n_diff_types):
                            # contrastive loss, loop through all condition types
                            loss_contr_test += self.contr(torch.cat(z_ds[condi]['z_mu']), torch.cat(z_ds[condi]['diff_label']))  
                            # condition specific mmd loss
                            z_ds[condi]["z_mu"] = torch.cat(z_ds[condi]["z_mu"], dim = 0)
                            z_ds[condi]["diff_label"] = torch.cat(z_ds[condi]["diff_label"], dim = 0)
                            z_ds[condi]["batch_id"] = torch.cat(z_ds[condi]["batch_id"], dim = 0)
                            for diff_label in range(self.n_diff_types):
                                idx = z_ds[condi]["diff_label"] == diff_label
                                loss_mmd_test += loss_func.maximum_mean_discrepancy(xs = z_ds[condi]["z_mu"][idx, :], batch_ids = z_ds[condi]["batch_id"][idx], device = self.device)

                        # total loss
                        loss_test = self.lambs[0] * loss_recon + self.lambs[1] * loss_mmd_test + self.lambs[2] * loss_class_test + self.lambs[3] * loss_contr + self.lambs[4] * (loss_gl_d_test+loss_gl_c_test) + self.lambs[5] * loss_kl_test + self.lambs[6] * loss_tc
                    
                        print('Epoch {}, Validating Loss: {:.4f}'.format(epoch, loss_test.item()))
                        info = [
                            'loss reconstruction: {:.5f}'.format(loss_recon_test.item()),
                            'loss kl: {:.5f}'.format(loss_kl_test.item()),
                            'loss mmd: {:.5f}'.format(loss_mmd_test.item()),
                            'loss classification: {:.5f}'.format(loss_class_test.item()),
                            'loss contrastive: {:.5f}'.format(loss_contr_test.item()),
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
                        loss_contr_tests.append(loss_contr_test.item())
                        loss_kl_tests.append(loss_kl_test.item())
                        loss_gl_d_tests.append(loss_gl_d_test.item())
                        loss_gl_c_tests.append(loss_gl_c_test.item())
                        loss_tc_tests.append(loss_tc_test_disc.item())

        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests
