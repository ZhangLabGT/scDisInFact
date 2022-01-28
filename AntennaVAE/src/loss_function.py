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


def _gaussian_kernel_matrix(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result


def maximum_mean_discrepancy(x, y): #Function to calculate MMD value
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cost = torch.mean(_gaussian_kernel_matrix(x, x))
    cost += torch.mean(_gaussian_kernel_matrix(y, y))
    cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(x, y))
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost

# Loss
def pinfo_loss(model, mask, norm = "l2"):
    W = None
    for _, layers in model.fc_layers.named_children():
        for name, layer in layers.named_children():
            if name[:3] == "lin":
                if W is None:
                    W = layer.weight
                else:
                    W = torch.mm(layer.weight,W)
    if norm == "l2":
        loss = torch.norm(mask.T * W, p = "fro")
    else:
        # l1 norm
        loss = torch.sum(torch.abs(mask.T * W))
    return loss



def dist_loss(z, diff_sim, mask = None, mode = "mse"):
    # cosine similarity loss
    latent_sim = compute_pairwise_distances(z, z)
    if mode == "mse":
        latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
        diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')

        if mask is not None:
            loss_dist = torch.norm((diff_sim - latent_sim) * mask, p = 'fro')
        else:   
            loss_dist = torch.norm(diff_sim - latent_sim, p = 'fro')
    
    elif mode == "kl":
        # latent_sim = 1/(1 + latent_sim)
        # diff_sim = 1/(1 + diff_sim)
        Q_dist = latent_sim / torch.sum(latent_sim) + 1e-12
        P_dist = diff_sim / torch.sum(diff_sim) + 1e-12
        loss_dist = torch.sum(Q_dist * torch.log(Q_dist / P_dist))
    return loss_dist

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

'''
class NB():
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):
        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        
        y_true = y_true.type(torch.float32)
        y_pred = y_pred.type(torch.float32) * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        # Clip theta
        theta = torch.minimum(self.theta, torch.tensor(1e6).to(device))

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.reduce_sum(final), nelem)
            else:
                final = torch.reduce_mean(final)

        return final  

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        if not torch.is_tensor(pi):
            self.pi = torch.tensor(pi).to(device)
        else:
            self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        # reuse existing NB neg.log.lik.
        # mean is always False here, because everything is calculated
        # element-wise. we take the mean only in the end
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log((torch.tensor(1.0+eps).to(device)-self.pi))
        y_true = y_true.type(torch.float32)
        
        y_pred = y_pred.type(torch.float32) * scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6).to(device))

        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)


        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result) 
            else:
                result = torch.mean(result)

        result = _nan2inf(result)
        
        return result
'''


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

class MMD_LOSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss_MSE = nn.MSELoss().forward(y_pred, y_true)
        loss_MMD = maximum_mean_discrepancy(y_pred, y_true)
        return (loss_MMD + loss_MSE)
# MMD func from Kaggle

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def MMD(x, y):
#     '''
#     Using gaussian kernel for MMD
#     '''
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
#     print(xx, xx.diag(),xx.diag().unsqueeze(0), rx)
#     dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
#     XX, YY, XY = (torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device))
    
#     # applying kernel method
#     sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
#     for sigma in sigmas:
#         XX += torch.exp(-0.5*dxx/sigma)
#         YY += torch.exp(-0.5*dyy/sigma)
#         XY += torch.exp(-0.5*dxy/sigma)

#     return torch.mean(XX + YY - 2. * XY)
