#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Mon 23 Apr 2018 08:25:48 PM CST

# File Name: model.py
# Description:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import numpy as np
from itertools import repeat

from .layer import Encoder, Decoder, build_mlp
from .loss import elbo, elbo_VaDE
		

class VAE(nn.Module):
	def __init__(self, dims, bn=False, dropout=0, binary=True):
		"""
		Variational Autoencoder [Kingma 2013] model
		consisting of an encoder/decoder pair for which
		a variational distribution is fitted to the
		encoder. Also known as the M1 model in [Kingma 2014].

		:param dims: x, z and hidden dimensions of the networks
		"""
		super(VAE, self).__init__()
		[x_dim, z_dim, encode_dim, decode_dim] = dims
		self.binary = binary
		if binary:
			decode_activation = nn.Sigmoid()
		else:
			decode_activation = None
		
		self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
		self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)
		
		self.reset_parameters()
		
	def reset_parameters(self):
		"""
		Initialize weights
		"""
		for m in self.modules():
			if isinstance(m, nn.Linear):
				init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
	
	def forward(self, x, y=None):
		"""
		Runs a data point through the model in order
		to provide its reconstruction and q distribution
		parameters.

		:param x: input data
		:return: reconstructed input
		"""
		z, mu, logvar = self.encoder(x)
		recon_x = self.decoder(z)

		return recon_x
	
	
	def loss_function(self, x):
		z, mu, logvar = self.encoder(x)
		recon_x = self.decoder(z)
		likelihood, kld = elbo(recon_x, x, (mu, logvar), binary=self.binary)
		self.likelihood = likelihood
		self.kld = kld
		return -(likelihood - kld) 
	
	def get_feature(self, x):
		self.eval()
		
		z = self.encoder(x)[1].detach().data.cpu().numpy()
		return z

	def load_model(self, path):
		pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict) 
		self.load_state_dict(model_dict)
		
		
	
class SCALE(VAE):
	def __init__(self, dims, n_centroids, beta=repeat(1)):
		super().__init__(dims)
		self.beta = beta
		self.n_centroids = n_centroids
		z_dim = dims[1]
		
		# init c_params
		self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
		self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
		self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2
		
	def loss_function(self, x):
		# c_params = self.pi, self.mu_c, self.var_c
		z, mu, logvar = self.encoder(x)
		recon_x = self.decoder(z)
		gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
		likelihood, kld = elbo_VaDE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)
		self.likelihood = likelihood
		self.kld = kld
		# return -(likelihood - kld)
		return -(likelihood - next(self.beta)*kld) 
		
	def get_gamma(self, z): # , n_centroids, c_params):
		"""
		Inference c from z

		gamma is q(c|x)
		q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
		"""
		pi, mu_c, var_c = self.pi, self.mu_c, self.var_c
		n_centroids = self.n_centroids
		
		N = z.size(0)
		z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
		pi = pi.repeat(N,1) # NxK
		mu_c = mu_c.repeat(N,1,1) # NxDxK
		var_c = var_c.repeat(N,1,1) # NxDxK

		# p(c,z) = p(c)*p(z|c) as p_c_z
		p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

		return gamma, mu_c, var_c, pi
	
	def predict(self, x, method='kmeans'):
		"""
		Inference c from x
		
		:param: x, data
				method, one of 'vade', 'kmeans'
		
		"""
		
		from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
		self.eval()
		if method == 'kmeans':
			kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
			feature = self.get_feature(x)
			try:
				pred = kmeans.fit_predict(feature); 
			except:
				pred = np.random.choice(range(self.n_centroids), size=len(feature)) # random pred
			
			return pred
		elif method == 'vade':
			
			z = self.encoder(x)[0].detach()
			logits = self.get_gamma(z)[0].cpu().detach()
			pred = np.argmax(logits.numpy(), axis=1)
			return pred
	
	