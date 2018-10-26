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
from sklearn.mixture import GaussianMixture

from .layer import Encoder, Decoder, build_mlp, DeterministicWarmup
from .loss import elbo, elbo_SCALE


class VAE(nn.Module):
	def __init__(self, dims, device='cpu', bn=False, dropout=0, binary=True):
		"""
		Variational Autoencoder [Kingma 2013] model
		consisting of an encoder/decoder pair for which
		a variational distribution is fitted to the
		encoder. Also known as the M1 model in [Kingma 2014].

		:param dims: x, z and hidden dimensions of the networks
		"""
		super(VAE, self).__init__()
		[x_dim, z_dim, encode_dim, decode_dim] = dims
		self.device = device
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

	def get_feature(self, data):
		"""
		obtain latent features from torch tensor data
		"""
		# return self.encoder(data)[0].detach().data.cpu().numpy()
		return self.encoder(data)[0].cpu().detach().numpy()

	def get_imputed_data(self, data):
		"""
		obtain imputed data from torch tensor data
		"""
		return self.forward(data).cpu().detach().numpy()

	def predict(self, data):
		"""
		Predict assignments applying k-means on latent feature

		Input: 
			x, data matrix
		Return:
			predicted cluster assignments
		"""
		self.eval()
		feature = self.get_feature(data)
		from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
		kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
		try:
			pred = kmeans.fit_predict(feature); 
		except:
			pred = np.random.choice(range(self.n_centroids), size=len(feature)) # random pred
		return pred

	def load_model(self, path):
		pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict) 
		self.load_state_dict(model_dict)

	def fit(self, dataloader,
		epochs=300,
		lr=0.002, 
		weight_decay=5e-4,
		print_interval=10, 
		device=None,
		verbose=True,
	   ):
		if device is None:
			device = self.device
		else:
			device = device

		optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) 
		for epoch in range(epochs):
			epoch_lr = adjust_learning_rate(lr, optimizer, epoch)	
			epoch_loss = 0
			self.train()
			for i, x in enumerate(dataloader):
				x = x[0].to(device)
				optimizer.zero_grad()
				loss = self.loss_function(x)/len(x)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			avg_loss = epoch_loss/len(dataloader)

			# Display Training Process
			if (epoch+1) % print_interval == 0 or epoch==0:
				if verbose:
					print('[Epoch {:3d}] Loss: {:.3f} lr: {:.4f}'.format(epoch+1, epoch_loss/len(dataloader), epoch_lr))



class SCALE(VAE):
	def __init__(self, dims, n_centroids, device='cpu'):
		super(SCALE, self).__init__(dims, device)
		self.beta = DeterministicWarmup(n=100, t_max=1)
		self.n_centroids = n_centroids
		z_dim = dims[1]

		# init c_params
		self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
		self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
		self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2

	def loss_function(self, x):
		z, mu, logvar = self.encoder(x)
		recon_x = self.decoder(z)
		gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
		likelihood, kld = elbo_SCALE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)
		self.likelihood = likelihood
		self.kld = kld

		return -(likelihood - next(self.beta)*kld) 

	def get_gamma(self, z):
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

	def init_gmm_params(self, data):
		"""
		Init SCALE model with GMM model parameters
		"""
		z = self.get_feature(data)
		gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
		gmm.fit(z)
		self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
		self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))



def adjust_learning_rate(init_lr, optimizer, epoch):
	lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr
	return lr	

