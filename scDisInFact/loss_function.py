import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scDisInFact.zinb import *
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------------------------
#
# Maximum mean discrepency
#
#--------------------------------------------------------------------------
def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y, device):
    sigmas = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6], device = device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result



def maximum_mean_discrepancy(xs, batch_ids, device, ref_batch = None): #Function to calculate MMD value
    # number of cells
    assert batch_ids.shape[0] == xs.shape[0]
    batches = torch.unique(batch_ids, sorted = True)
    nbatches = batches.shape[0]
    if ref_batch is None:
        # select the first batch, the batches are equal sizes
        ref_batch = batches[0]
    # assuming batch 0 is the reference batch
    cost = 0
    # within batch
    for batch in batches:
        xs_batch = xs[batch_ids == batch, :]
        if batch == ref_batch:
            cost += (nbatches - 1) * torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))
        else:
            cost += torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))
    
    # between batches
    xs_refbatch = xs[batch_ids == ref_batch]
    for batch in batches:
        if batch != ref_batch:
            xs_batch = xs[batch_ids == batch, :]
            cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs_refbatch, xs_batch, device))
    
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.tensor([0.0], device = device)

    return cost

#--------------------------------------------------------------------------
#
# Calculate contrastive loss
#
#--------------------------------------------------------------------------
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nelem(x):
    nelem = torch.reduce_sum(torch.float(~torch.isnan(x), torch.float32))
    return torch.float(torch.where(torch.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.reduce_sum(x), nelem)

def convert_label_to_similarity(normed_feature, label):
    # print(torch.sum(normed_feature))
    # print(normed_feature.shape)
    # print(torch.sum(label))
    # print(label.shape)
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    # print(torch.sum(similarity_matrix))
    # print(similarity_matrix.shape)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    # print(torch.sum(label_matrix))
    # print(label_matrix.shape)
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)
    # print(positive_matrix)
    # print(negative_matrix)

    
    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class TripletLoss(nn.Module):
    def __init__(self,margin=0.3):
      super(TripletLoss,self).__init__()
      self.margin=margin
      self.ranking_loss=nn.MarginRankingLoss(margin=margin)
      
    def forward(self,inputs,labels):
      n=inputs.size(0)

      # calculate pairwise distance (Euclidean, sqrt(x1^2 + x2^2 - 2x1x2)), using latent embeddings, of the shape (n_cells, n_cells)
      dist=torch.pow(inputs,2).sum(dim=1,keepdim=True).expand(n,n)
      dist=dist+dist.t()
      dist.addmm_(1,-2,inputs,inputs.t())
      dist=dist.clamp(min=1e-12).sqrt()
      
      # mask matrix of the shape (n_cells, n_cells), the element (n1, n2) -> 1 if cell n1 and cell n2 are of the same class, 0 otherwise
      mask=labels.expand(n,n).eq(labels.expand(n,n).t())
      #print(mask.shape)
      #print(mask[0])
      dist_ap,dist_an=[],[]
      for i in range(n):
        #print(i)
        # find the largest distance to cell i in all positive pairs
        dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        # find the smallest distance to cell i in all negative pairs
        dist_an.append(dist[i][mask[i]==0].min().unsqueeze(0))
      # list of max positive distances for each cell
      dist_ap=torch.cat(dist_ap)
      # list of min negative distances for each cell
      dist_an=torch.cat(dist_an)
      
      y=torch.ones_like(dist_an)
      # max(- y * (dist_an - dist_ap), 0), minimize positive distances and maximize negative distances
      # if use for multi-classes, then different margin for different classes
      loss=self.ranking_loss(dist_an,dist_ap,y)
      return loss

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, inputs, labels):
        sp, sn = convert_label_to_similarity(inputs, labels)
        alpha_p = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m
        
        logit_p = - alpha_p * (sp - delta_p) * self.gamma
        logit_n = alpha_n * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class SNNLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SNNLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            # torch.matmul(anchor_feature, contrast_feature.T)
            -self.pairwise_euclid_distance(anchor_feature, contrast_feature),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

#--------------------------------------------------------------------------
#
# Group LASSO loss
#
#--------------------------------------------------------------------------
def grouplasso(W, alpha = 1e-4):
    '''
    Definition:
    -----------
        Calculate the L1/2 norm or group lasso of parameter matrix W, with smoothing
    Parameters:
    -----------
        W: of the shape (out_features, in_features), l2 norm is calculated on out_features (axis = 0), l1 is calculated on in_features (axis = 1)
        alpha: smooth parameter, no smoothing if alpha = 0
    Returns:
    -----------
        loss_gl: the L1/2 norm loss term
    '''
    # l2 norm on rows
    l2_norm = W.pow(2).sum(dim=0).add(1e-8).pow(1/2.)
    # group lasso + smoothing term
    loss_gl = torch.mean((l2_norm >= alpha) * l2_norm + (l2_norm < alpha) * (W.pow(2).sum(dim=0).add(1e-8)/(2*alpha + 1e-8) + alpha/2))
    return loss_gl

#--------------------------------------------------------------------------
#
# Calculate marginal distribution
#
#--------------------------------------------------------------------------
def estimate_entropies(qz_samples, mus, logvars, device):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.

    Then we can estimate numerically stable negative log likelihood:
        - log q(z) = log N - \log\sum_n=1^N\exp log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, S) Variable
        qz_params  (N, K, nparams) Variable
    """

    # qz_samples of the shape (K, S), where K is the sample dimension, S is the total number of samples 
    # select 10000 samples from S 
    qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.shape[1])[:10000].to(device)))
    # K is the latent dimensions, S is the number of samples (maximum 10000)
    K, S = qz_samples.shape
    # N is the total number of samples
    N, _ = mus.shape
    assert(K == mus.shape[1])
    assert(K == logvars.shape[1])
    assert(N == logvars.shape[0])

    # 1 dimension for the whole z
    joint_entropy = torch.FloatTensor([0]).to(device)

    k = 0
    # loop through every sample in the batch with batch_size as stepsize
    while k < S:
        # batchsize is 10
        batch_size = min(10, S - k)
        # logqz_i of the shape (N, K, 10), which calculate the log distribution of each dimension separately
        # actually calculate logq(z_m[i]|x_j), m is the mth sample, j is the jth sample, and i is the ith dimension
        logqz_i = log_gaussian(
            # select qz_samples (N, K, 10), 10 is the batch size (total S)
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            # mu (N, K, 10), 10 is the batch size (total S)
            mus.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size], 
            # logvar (N, K, 10), 10 is the batch size (total S)
            logvars.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size]
            )
        k += batch_size

        # logqz of the shape: (N, 10) (sum over K)
        logqz = logqz_i.sum(1)  
        # first logsumexp calculate \log \sum_{j=1}^Nq(z_i|x_j)
        # then calculate \log N - \log \sum_{j=1}^Nq(z_i|x_j)
        # finally sum over i \sum_{i=1}^S [\log N - \log \sum_{j=1}^Nq(z_i|x_j)]
        joint_entropy += (np.log(N) - logsumexp(logqz, dim=0, keepdim=False).data).sum(0)

    # calculate the average: E_{p(x)} E_{q(z|x)} [-log q(z)]
    joint_entropy /= S

    return joint_entropy


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if torch.is_tensor(sum_exp):
            return m + torch.log(sum_exp)
        else:
            return m + np.log(sum_exp)

def _log_gaussian(sample, mu, logvar, device):
    """
    calculate the log likelihood of Gaussian distribution
    """
    # inv_sigma = 1/exp(logvar)
    inv_sigma = torch.exp(-logvar)
    loglkl = -0.5 * (np.log(2 * np.pi) + 2 * logvar + (sample - mu) * (sample - mu) * inv_sigma * inv_sigma)
    return loglkl

def log_gaussian(sample, mu, logvar, device):
    """
    calculate the log likelihood of Gaussian distribution
    """
    mu = mu.type_as(sample)
    logvar = logvar.type_as(sample)
    c = Variable(torch.Tensor([np.log(2 * np.pi)])).type_as(sample.data)
    inv_sigma = torch.exp(-logvar)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logvar + c)