import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# code adapted from DCA: https://github.com/theislab/dca


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

class NB():
    """\
    Description:
    ------------
        The loss term of negative binomial
    Usage:
    ------------
        nb = NB(theta = theta, scale_factor = libsize, masking = False)
        nb_loss = nb.loss(y_true = mean_x, y_pred = x)        
    """
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False, device = device):
        """\
        Parameters:
        -----------
            theta: theta is the dispersion parameter of the negative binomial distribution. the output of the estimater
            scale_factor: scaling factor of y_pred (observed count), of the shape the same as the observed count. 
            scope: not sure
        """
        # for numerical stability, 1e-10 might not be enough, make it larger when the loss becomes nan
        self.eps = 1e-6
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta
        self.device = device

    def loss(self, y_true, y_pred, mean=True):
        """\
        Parameters:
        -----------
            y_true: the mean estimation. should be the output of the estimator
            y_pred: the observed counts.
            mean: calculate the mean of loss
        """
        scale_factor = self.scale_factor
        eps = self.eps
        
        y_true = y_true.type(torch.float32)
        y_pred = y_pred.type(torch.float32) * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        # Clip theta
        # theta = torch.minimum(self.theta, torch.tensor(1e6).to(self.device))
        theta = self.theta.clamp(min = None, max = 1e6).to(self.device)

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.reduce_sum(final), nelem)
            else:
                final = torch.mean(final)

        return final  

class ZINB(NB):
    """\
    Description:
    ------------
        The loss term of zero inflated negative binomial (ZINB)
    Usage:
    ------------
        zinb = ZINB(pi = pi, theta = theta, scale_factor = libsize, ridge_lambda = 1e-5)
        zinb_loss = zinb.loss(y_true = mean_x, y_pred = x)
    """
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        """\
        Parameters:
        -----------
            pi: the zero-inflation parameter, the probability of a observed zero value not sampled from negative binomial distribution. Should be the output of the estimater
            ridge_lambda: ridge regularization for pi, not in the likelihood function, of the form ``ridge_lambda * ||pi||^2'', set to 0 if not needed.
            scope: not sure
            kwargs includes: 
                theta: theta is the dispersion parameter of the negative binomial distribution. the output of the estimater
                scale_factor: scaling factor of y_pred (observed count), of the shape the same as the observed count. 
                masking:
        """
        super().__init__(scope=scope, **kwargs)
        if not torch.is_tensor(pi):
            self.pi = torch.tensor(pi).to(self.device)
        else:
            self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        """\
        Parameters:
        -----------
            y_true: the mean estimation. should be the output of the estimator
            y_pred: the observed counts.
            mean: calculate the mean of loss
        """
        # set the scaling factor (libsize) for each observed (normalized) counts
        scale_factor = self.scale_factor
        # the margin for 0
        eps = self.eps

        # calculate the negative log-likelihood of nb distribution. reuse existing NB neg.log.lik.
        # mean is always False here, because everything is calculated
        # element-wise. we take the mean only in the end
        # -log((1-pi) * NB(x; mu, theta)) = - log(1 - pi) + nb.loss(x; mu, theta)
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log((torch.tensor(1.0+eps).to(self.device)-self.pi))
        y_true = y_true.type(torch.float32)
        
        # scale the observed (normalized) counts by the scaling factor
        y_pred = y_pred.type(torch.float32) * scale_factor
        # compute elementwise minimum between self.theta and 1e6, make sure all values are not inf
        # theta = torch.minimum(self.theta, torch.tensor(1e6).to(device))
        theta = self.theta.clamp(min = None, max = 1e6).to(self.device)

        # calculate the negative log-likelihood of the zero inflation part
        # first calculate zero_nb = (theta/(theta + x))^theta
        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        # then calculate the negative log-likelihood of the zero inflation part 
        # -log(pi + (1 - pi)*zero_nb)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        # when observation is 0, negative log likelihood equals to zero_case, or equals to nb_case
        # result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        result = torch.where(y_true < 1e-8, zero_case, nb_case)

        # regularization term
        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        # calculate the mean of all likelihood over genes and cells
        if mean:
            if self.masking:
                result = _reduce_mean(result) 
            else:
                result = torch.mean(result)

        result = _nan2inf(result)
        
        return result