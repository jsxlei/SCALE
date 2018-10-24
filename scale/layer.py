#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Mon 23 Apr 2018 08:25:55 PM CST

# File Name: layer.py
# Description:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

import math
import numpy as np


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
	"""
	Build multilayer linear perceptron
	"""
	net = []
	for i in range(1, len(layers)):
		net.append(nn.Linear(layers[i-1], layers[i]))
		if bn:
			net.append(nn.BatchNorm1d(layers[i]))
		net.append(activation)
		if dropout > 0:
			net.append(nn.Dropout(dropout))
	return nn.Sequential(*net)


class Encoder(nn.Module):
	def __init__(self, dims, bn=False, dropout=0):
		"""
		Inference network

		Attempts to infer the probability distribution
		p(z|x) from the data by fitting a variational
		distribution q_φ(z|x). Returns the two parameters
		of the distribution (µ, log σ²).

		:param dims: dimensions of the networks
		   given by the number of neurons on the form
		   [input_dim, [hidden_dims], latent_dim].
		"""
		super(Encoder, self).__init__()

		[x_dim, h_dim, z_dim] = dims
		# self.hidden = build_mlp([x_dim, *h_dim], bn=bn, dropout=dropout)
		self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
		self.sample = GaussianSample(([x_dim]+h_dim)[-1], z_dim)
		# self.sample = GaussianSample([x_dim, *h_dim][-1], z_dim)

	def forward(self, x):
		x = self.hidden(x)
		return self.sample(x)


class Decoder(nn.Module):
	def __init__(self, dims, bn=False, dropout=0, output_activation=nn.Sigmoid()):
		"""
		Generative network

		Generates samples from the original distribution
		p(x) by transforming a latent representation, e.g.
		by finding p_θ(x|z).

		:param dims: dimensions of the networks
			given by the number of neurons on the form
			[latent_dim, [hidden_dims], input_dim].
		"""
		super(Decoder, self).__init__()

		[z_dim, h_dim, x_dim] = dims

		# self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
		self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
		# self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
		self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)

		self.output_activation = output_activation

	def forward(self, x):
		x = self.hidden(x)
		if self.output_activation is not None:
			return self.output_activation(self.reconstruction(x))
		else:
			return self.reconstruction(x)

class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[Sønderby 2016].
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n

	def __iter__(self):
		return self

	def __next__(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t

	def next(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t


###################
###################
class Stochastic(nn.Module):
	"""
	Base stochastic layer that uses the
	reparametrization trick [Kingma 2013]
	to draw a sample from a distribution
	parametrised by mu and log_var.
	"""
	def reparametrize(self, mu, logvar):
		if self.training:
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu

class GaussianSample(Stochastic):
	"""
	Layer that represents a sample from a
	Gaussian distribution.
	"""
	def __init__(self, in_features, out_features):
		super(GaussianSample, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.mu = nn.Linear(in_features, out_features)
		self.log_var = nn.Linear(in_features, out_features)

	def forward(self, x):
		mu = self.mu(x)
		log_var = self.log_var(x)

		return self.reparametrize(mu, log_var), mu, log_var


class GaussianMerge(GaussianSample):
	"""
	Precision weighted merging of two Gaussian
	distributions.
	Merges information from z into the given
	mean and log variance and produces
	a sample from this new distribution.
	"""
	def __init__(self, in_features, out_features):
		super(GaussianMerge, self).__init__(in_features, out_features)

	def forward(self, z, mu1, log_var1):
		# Calculate precision of each distribution
		# (inverse variance)
		mu2 = self.mu(z)
		log_var2 = F.softplus(self.log_var(z))
		precision1, precision2 = (1/torch.exp(log_var1), 1/torch.exp(log_var2))

		# Merge distributions into a single new
		# distribution
		mu = ((mu1 * precision1) + (mu2 * precision2)) / (precision1 + precision2)

		var = 1 / (precision1 + precision2)
		log_var = torch.log(var + 1e-8)

		return self.reparametrize(mu, log_var), mu, log_var

