#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Sat 28 Apr 2018 08:31:29 PM CST

# File Name: SCALE.py
# Description: Single-cell ATAC-seq analysis via feature extraction.
	Input: 
		scATAC-seq data
	Output:
		1. latent GMM feature
		2. cluster assignment
		3. imputation data
		4. cell type specific peaks
"""


import time
import torch

import numpy as np
import pandas as pd
import argparse

from scale import SCALE
from scale.utils import *
from scale.plot import VisdomLinePlotter
from scale import config
		
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='SCALE: Single-cell ATAC-seq analysis via feature extraction')
	parser.add_argument('--data', '-d', type=str, help='input data matrix peaks x samples')
	parser.add_argument('--n_centroids', '-k', type=int, default=None, help='cluster number')
	parser.add_argument('--sep', type=str, default='\t', help='input data sep format \t or , ')
	parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
	parser.add_argument('--visdom', '-v', action='store_true', help='Show training process with visdom')
	parser.add_argument('--save', type=bool, default=True, help='Save the results')
	parser.add_argument('--verbose', type=bool, default=True, help='Print loss of training process')
	parser.add_argument('--reference', '-r', type=str, default='', help='Whether ground truth available')
	parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
	parser.add_argument('--epochs', '-e', type=int, default=None, help='Training epochs')
	parser.add_argument('--gpu', '-g', type=int, default=0, help='Use gpu when training')
	parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
	parser.add_argument('--input_dim', type=int, default=None, help='Force input dim')
	## data preprocessing
	parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
	parser.add_argument('--gene_filter', action='store_true', help='Perform gene filter as SC3')
	parser.add_argument('-x', '--pct', type=float, default=6, help='Percent of genes when performing gene filter as SC3')
	
	args = parser.parse_args()

	# Set random seed
	seed = args.seed
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available:
		torch.cuda.manual_seed_all(seed)

	# path = os.path.join(args.path, args.dataset) #;print(path)
	device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
	
	# Load data and labels
	data_params_ = get_loader(args.data, 
							  args.input_dim, 
							  sep=args.sep,
							  batch_size=config.batch_size, 
							  X=args.pct,
							  gene_filter=args.gene_filter,
							  log_transform=args.log_transform)
	dataloader, data, data_params = data_params_[0], data_params_[1], data_params_[2:]
	input_dim = data.shape[1]		
	
	if args.n_centroids is None:
		est_k = estimate_k(data.t())
		print('Estimate k: {}'.format(est_k))
		k = est_k
	else:
		k = args.n_centroids
		
	if args.visdom:
		visdom = VisdomLinePlotter()
	else:
		visdom = None
		
	if args.epochs is None:
		epochs = config.epochs
	else:
		epochs = args.epochs
	
	print("********** SCALE ***********\n")
	print("======== Parameters ========")
	print('Cell number: {}\nInput_dim: {}\nn_centroids: {}\nEpoch: {}\nSeed: {}'.format(
		data.shape[0], input_dim, k, epochs, args.seed))
	print("============================")
	
	dims = [input_dim, config.latent, config.encode_dim, []]
	model = SCALE(dims, n_centroids=k).to(device)
	if not args.pretrain:
		print('## Training Model ##\n')
		t0 = time.time()
		model.init_gmm_params(data.to(device))
		model.fit(dataloader,
				  lr=config.lr, 
				  weight_decay=config.weight_decay, 
			      epochs=epochs, 
				  device=device,
			      visdom=visdom,
				  verbose=args.verbose
				   )
		print('Running Time: {:.2f} s'.format(time.time()-t0))
	else:
		print('## Loading Model {} ##\n'.format(args.pretrain))
		model.load_model(args.pretrain)
		model.to(device)
		
	# Clustering Report
	if args.reference:
		ref, classes = read_labels(args.reference)
		pred = model.predict(data.to(device))
		cluster_report(ref, pred, classes)
							 
	outdir = args.outdir
	if args.save:
		save_results(model, data, data_params, args.outdir, device)
	