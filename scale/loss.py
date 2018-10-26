#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Mon 23 Apr 2018 08:26:32 PM CST

# File Name: loss_function.py
# Description:

"""
import torch
import torch.nn.functional as F

import math

def kl_divergence(mu, logvar):
    """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


def elbo(recon_x, x, z_params, binary=True):
    """
    elbo = likelihood - kl_divergence
    L = -elbo

    Params:
        recon_x:
        x:
    """
    mu, logvar = z_params
    kld = kl_divergence(mu, logvar)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x)
    else:
        likelihood = -F.mse_loss(recon_x, x)
    return torch.sum(likelihood), torch.sum(kld)
    # return likelihood, kld


def elbo_SCALE(recon_x, x, gamma, c_params, z_params, binary=True):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    mu_c, var_c, pi = c_params; #print(mu_c.size(), var_c.size(), pi.size())
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(x|z)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x) #;print(logvar_expand.size()) #, torch.exp(logvar_expand)/var_c)
    else:
        likelihood = -F.mse_loss(recon_x, x)

    # log p(z|c)
    logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           torch.exp(logvar_expand)/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
    # log p(c)
    logpc = torch.sum(gamma*torch.log(pi), 1)

    # log q(z|x) or q entropy    
    qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma*torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx

    return torch.sum(likelihood), torch.sum(kld)


