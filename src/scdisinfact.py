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


class scdisinfact(nn.Module):
    """\
    Description:
    --------------
        New model that separate the encoder and control backward gradient. (VARIATIONAL AUTOENCODER)

    """
    def __init__(self, datasets, reg_mmd_comm, reg_mmd_diff, reg_gl, reg_class = 1, reg_tc = 0.1, reg_kl = 1e-6, Ks = [8, 4], batch_size = 64, interval = 10, lr = 5e-4, seed = 0, device = device):
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
        for idx, dataset in enumerate(datasets):
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
            cutoff = 10000
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
            self.uniq_diff_labels[diff_factor] = set(self.uniq_diff_labels[diff_factor]) 
        # unique data batches
        self.uniq_batch_ids = set(self.uniq_batch_ids)

        # create model
        # encoder for common biological factor
        self.Enc_c = model.Encoder(n_input = self.ngenes, n_output = self.Ks["common_factor"], n_layers = 2, n_hidden = 128, n_cat_list = [len(self.uniq_batch_ids)], dropout_rate  = 0.0).to(self.device)
        # encoder for time factor, + 1 here refers to the one batch ID
        self.Enc_ds = nn.ModuleList([])
        for diff_factor in range(self.n_diff_factors):
            self.Enc_ds.append(
                model.Encoder(n_input = self.ngenes, n_output = self.Ks["diff_factors"][diff_factor], n_layers = 1, n_hidden = 128, n_cat_list = [len(self.uniq_batch_ids)], dropout_rate = 0.0).to(self.device)
            )
        # NOTE: classify the time point, out dim = number of unique time points, currently use only time dimensions as input, update the last layer to be linear
        # use a linear classifier as stated in the paper
        self.classifiers = nn.ModuleList([])
        for diff_factor in range(self.n_diff_factors):
            self.classifiers.append(nn.Linear(self.Ks["diff_factors"][diff_factor], len(self.uniq_diff_labels[diff_factor])).to(self.device))
        # NOTE: reconstruct the original data, use all latent dimensions as input
        self.Dec = model.Decoder(n_input = self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), n_output = self.ngenes, n_cat_list = [len(self.uniq_batch_ids)], n_layers = 2, n_hidden = 128, dropout_rate = 0.0).to(self.device)
        # Discriminator for factor vae
        self.disc = model.FCLayers(n_in=self.Ks["common_factor"] + sum(self.Ks["diff_factors"]), n_out=2, n_cat_list=None, n_layers=3, n_hidden=2, dropout_rate=0.0).to(self.device)

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
            # prevent the shrinkage of variance
            std = torch.clamp(std, min = clamp)

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def inference(self, counts, batch_ids, print_stat = False, eval_model = False, clamp_comm = 0.0, clamp_diff = 0.0):
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

        if not eval_model:
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
            return {"mu_c":mu_c, "logvar_c": logvar_c, "mu_d": mu_d, "logvar_d": logvar_d}
            

    def generative(self, z_c, z_d, batch_ids):
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
        loss_kl = torch.sum(dict_inf["mu_c"].pow(2).add_(dict_inf["logvar_c"].exp()).mul_(-1).add_(1).add_(dict_inf["logvar_c"])).mul_(-0.5)         
        for diff_factor in range(self.n_diff_factors):
            loss_kl += torch.sum(dict_inf["mu_d"][diff_factor].pow(2).add_(dict_inf["logvar_d"][diff_factor].exp()).mul_(-1).add_(1).add_(dict_inf["logvar_d"][diff_factor])).mul_(-0.5)

        # 3.MMD loss
        # common mmd loss
        loss_mmd_comm = loss_func.maximum_mean_discrepancy(xs = dict_inf["mu_c"], batch_ids = batch_id, device = self.device)
        # condition specific mmd loss
        loss_mmd_diff = 0
        for diff_factor in range(self.n_diff_factors):
            for diff_label in self.uniq_diff_labels[diff_factor]:
                idx = diff_labels[diff_factor] == diff_label
                loss_mmd_diff += loss_func.maximum_mean_discrepancy(xs = dict_inf["mu_d"][diff_factor][idx, :], batch_ids = batch_id[idx], device = self.device)
        
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


    def train_model(self, nepochs = 50, recon_loss = "NB", reg_contr = 0.0):
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
                count_stand = []
                batch_id = []
                size_factor = []
                count = []
                diff_labels = [[] for x in range(self.n_diff_factors)]

                # load count data
                for x in data_batch:
                    count_stand.append(x["count_stand"].to(self.device, non_blocking=True))
                    batch_id.append(x["batch_id"][:, None].to(self.device, non_blocking=True))
                    size_factor.append(x["size_factor"].to(self.device, non_blocking=True))
                    count.append(x["count"].to(self.device, non_blocking=True))          
                    for diff_factor in range(self.n_diff_factors):
                        diff_labels[diff_factor].append(x["diff_labels"][diff_factor].to(self.device, non_blocking=True))
                
                count_stand = torch.cat(count_stand, dim = 0)
                batch_id = torch.cat(batch_id, dim = 0)
                size_factor = torch.cat(size_factor, dim = 0)
                count = torch.cat(count, dim = 0)
                for diff_factor in range(self.n_diff_factors):
                    diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

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
                dict_inf = self.inference(counts = count_stand, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                # pass through the decoder
                dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d = self.loss(dict_inf = dict_inf, \
                    dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = batch_id.squeeze(), diff_labels = diff_labels, recon_loss = recon_loss)

                loss = loss_recon + self.lambs["mmd_comm"] * loss_mmd_comm + self.lambs["kl"] * loss_kl
                loss.backward()
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
                dict_inf = self.inference(counts = count_stand, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                # pass through the decoder
                dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d = self.loss(dict_inf = dict_inf, \
                    dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = batch_id.squeeze(), diff_labels = diff_labels, recon_loss = recon_loss)

                loss = loss_recon + self.lambs["mmd_diff"] * loss_mmd_diff + self.lambs["class"] * (loss_class + reg_contr * loss_contr) + self.lambs["kl"] * loss_kl + self.lambs["gl"] * loss_gl_d + self.lambs["tc"] * loss_tc
                loss.backward()
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
                dict_inf = self.inference(counts = count_stand, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                # pass through the decoder
                dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                loss_recon, loss_kl, loss_mmd_comm, loss_mmd_diff, loss_class, loss_contr, loss_tc, loss_gl_d = self.loss(dict_inf = dict_inf, \
                    dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = batch_id.squeeze(), diff_labels = diff_labels, recon_loss = recon_loss)

                loss = self.lambs["tc"] * loss_tc
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()    

            # TEST
            if epoch % self.interval == 0:   
                with torch.no_grad():
                    # use the whole dataset for validation
                    for data_batch in zip(*self.test_loaders):
                        # loop through the data batches correspond to diffferent data matrices
                        count_stand = []
                        batch_id = []
                        size_factor = []
                        count = []
                        diff_labels = [[] for x in range(self.n_diff_factors)]

                        # load count data
                        for x in data_batch:
                            count_stand.append(x["count_stand"].to(self.device, non_blocking=True))
                            batch_id.append(x["batch_id"][:, None].to(self.device, non_blocking=True))
                            size_factor.append(x["size_factor"].to(self.device, non_blocking=True))
                            count.append(x["count"].to(self.device, non_blocking=True))          
                            for diff_factor in range(self.n_diff_factors):
                                diff_labels[diff_factor].append(x["diff_labels"][diff_factor].to(self.device, non_blocking=True))
                        
                        count_stand = torch.cat(count_stand, dim = 0)
                        batch_id = torch.cat(batch_id, dim = 0)
                        size_factor = torch.cat(size_factor, dim = 0)
                        count = torch.cat(count, dim = 0)
                        for diff_factor in range(self.n_diff_factors):
                            diff_labels[diff_factor] = torch.cat(diff_labels[diff_factor], dim = 0)

                        # pass through the encoders
                        dict_inf = self.inference(counts = count_stand, batch_ids = batch_id, print_stat = False, clamp_comm = clamp_comm, clamp_diff = clamp_diff)
                        # pass through the decoder
                        dict_gen = self.generative(z_c = dict_inf["z_c"], z_d = dict_inf["z_d"], batch_ids = batch_id)

                        loss_recon_test, loss_kl_test, loss_mmd_comm_test, loss_mmd_diff_test, loss_class_test, loss_contr_test, loss_tc_test, loss_gl_d_test = self.loss(dict_inf = dict_inf, \
                            dict_gen = dict_gen, size_factor = size_factor, count = count, batch_id = batch_id.squeeze(), diff_labels = diff_labels, recon_loss = recon_loss)

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
                        
        return loss_tests, loss_recon_tests, loss_kl_tests, loss_mmd_comm_tests, loss_mmd_diff_tests, loss_class_tests, loss_gl_d_tests, loss_gl_c_tests, loss_tc_tests
