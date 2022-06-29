import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from zinb import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y, device):
    sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result



def _maximum_mean_discrepancy(xs, ref_batch = 0, device = device): #Function to calculate MMD value
    nbatches = len(xs)
    # assuming batch 0 is the reference batch
    cost = 0
    # within batch
    for batch in range(nbatches):
        if batch == ref_batch:
            cost += (nbatches - 1) * torch.mean(_gaussian_kernel_matrix(xs[batch], xs[batch], device))
        else:
            cost += torch.mean(_gaussian_kernel_matrix(xs[batch], xs[batch], device))
    
    # between batches
    for batch in range(1, nbatches):
        cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs[ref_batch], xs[batch], device))
    
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost

def maximum_mean_discrepancy(xs, batch_ids, ref_batch = None, device = device): #Function to calculate MMD value
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
        cost = torch.FloatTensor([0.0]).to(device)

    return cost


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
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    
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
class MMD_LOSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss_MSE = nn.MSELoss().forward(y_pred, y_true)
        loss_MMD = maximum_mean_discrepancy(y_pred, y_true)
        return (loss_MMD + loss_MSE)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362
        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss


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
    loss_gl = torch.sum((l2_norm >= alpha) * l2_norm + (l2_norm < alpha) * (W.pow(2).sum(dim=0).add(1e-8)/(2*alpha + 1e-8) + alpha/2))
    return loss_gl