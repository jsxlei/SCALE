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
import os
from visdom import Visdom
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scale import SCALE
from scale import init_gmm_params, fit_GMM, DeterministicWarmup
from scale.utils import *
from scale.plot import *
from scale.plot import VisdomLinePlotter


def main():
	
	if not args.pretrain:
		print('## Training Model ##\n')
		t0 = time.time()
		fit(model, dataloader, data,
			lr=args.lr, 
			weight_decay=args.weight_decay, 
			epochs=args.epochs, 
			device=device)
		print('Running Time',time.time()-t0,'s')
	else:
		print('## Loading Pretrain Model ##\n')
		
	model.load_model(model_file)
	model.to(device)
	
	### output ###
	# 1. latent GMM feature
	feature = model.get_feature(data.to(device))
	
	# 2. cluster assignments
	pred = model.predict(data.to(device))
	if args.reference:
		f1_score, pred = cluster_acc(pred, ref)
	
	# 3. imputation data
	recon_x = model(data.to(device)).data.cpu().numpy()
	recon_x = norm.inverse_transform(recon_x)
	
	# 4. cell type specific peaks
	weight = model.state_dict()['decoder.reconstruction.weight'].cpu().numpy()
	specific_peaks = peak_selection(weight, weight_index, kind='both', cutoff=2.5)
	
	if args.save:
		assign_file = os.path.join(out_dir, 'cluster_assignments.txt')
		feature_file = os.path.join(out_dir, 'feature.txt')
		# index_file = os.path.join(out_dir, 'weight_index.txt')
		impute_file = os.path.join(out_dir, 'imputed_data.txt')
	
		pd.Series(pred).to_csv(assign_file, sep='\t', header=False) # save cluster assignments
		pd.DataFrame(feature).to_csv(feature_file, sep='\t', header=False) # save latent feature
		# open(index_file, 'w').write('\n'.join(map(str,weight_index)))
		pd.DataFrame(recon_x.T, index=weight_index, columns=columns).loc[raw_index].to_csv(impute_file, sep='\t') # save imputed data
		
		# save specific peaks
		all_peaks = []
		for i, peaks in enumerate(specific_peaks):
			peak_file = os.path.join(peak_dir, 'peak_index{}.txt'.format(i))
			open(peak_file, 'w').write('\n'.join(list(peaks)))
			all_peaks.append(peaks)
		peak_file = os.path.join(peak_dir, 'peak_index.txt')
		open(peak_file, 'w').write('\n'.join(set(peaks)))
		# torch.save(model.state_dict(), model_file)
	if args.plot:
		emb_png = os.path.join(plot_dir, 'feature_embeddng.pdf')
		plot_embedding(feature, ref, classes, markersize=10, figsize=(4,4), save=emb_png)
		
	# Clustering Report
	if args.reference and args.report:
		cm = cluster_report(ref, pred, classes)
		
	# compare_with_other_methods(model, data, k, args.hidden, device, ref)
	

def fit(model, dataloader, data, lr, weight_decay, epochs, device='cpu'):
	init_gmm_params(model, data.to(device))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	
	best_score = -1
	best_epoch = 0
	for epoch in range(epochs):
		epoch_lr = adjust_learning_rate(lr, optimizer, epoch)	
		epoch_loss = 0

		# Train Model
		model.train()
		for i, x in enumerate(dataloader):
			x = x[0].to(device)
			
			optimizer.zero_grad()
			loss = model.loss_function(x)/len(x)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
		
		avg_loss = epoch_loss/len(dataloader)
		
		pred = model.predict(data.to(device))
		silhouette_score = cal_silhouette(data, pred)
		if args.reference:
			f1_score, pred = cluster_acc(pred, ref)
			ari_score = adjusted_rand_score(ref, pred)
			
		# Display Training Process
		if epoch % args.interval == 0 and args.visdom:
			line.plot('_loss', 'loss', epoch, epoch_loss/len(dataloader))
			line.plot('_silhouette', 'SCALE', epoch, silhouette_score)
			if args.reference:
				line.plot('', 'ARI', epoch, ari_score)
				line.plot('', 'F1', epoch, f1_score)
				
		# Model Selection by Silhouette Score
		if best_score < silhouette_score and epoch >= 0.6*epochs:
			best_score = silhouette_score
			best_epoch = epoch
			torch.save(model.state_dict(), model_file) # save model state_dict
			
		if args.verbose:
			if (epoch+1) % args.interval == 0:
				print('[Epoch {:3d}] Loss: {:.3f} lr: {:.3f}'.format(epoch+1, epoch_loss/len(dataloader), epoch_lr))
	
	print('Best epoch: {} Best silhouette_score: {:.3f} \n'.format(best_epoch, best_score))

		
def adjust_learning_rate(init_lr, optimizer, epoch):
	lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr
	return lr	
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='SCALE: Single-cell ATAC-seq analysis via feature extraction')
	parser.add_argument('--dataset', '-d', type=str, help='dataset name')
	parser.add_argument('--n_centroids', '-k', type=int, default=None, help='cluster number')
	parser.add_argument('--sep', type=str, default='\t', help='input data sep format \t or , ')
	parser.add_argument('--path', type=str, default='data/', help='Data stored path')
	parser.add_argument('--outpath', '-o', type=str, default='results/', help='Output path')
	
	## parameters of model training are suggested as default
	parser.add_argument('--batch-size', '-b', type=int, default=16)
	parser.add_argument('--epochs', '-e', type=int, default=300)
	parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate') 
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
	parser.add_argument('--hidden', type=int, default=10, help='dim of latent feature') # hidden dim
	parser.add_argument('--encode_dim', type=int, nargs='*', default=[3200, 1600, 800, 400], help='encoder structure')
	parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='decoder structure')
	parser.add_argument('--input_dim', '-i', type=int, default=None, help='Force input dimension') 
	
	## visualization, report and other 
	parser.add_argument('--visdom', '-v', action='store_true', help='Show training process with visdom')
	parser.add_argument('--report', action='store_true', help='Report the final clustering results')
	parser.add_argument('--save', action='store_true', help='Save the results')
	parser.add_argument('--plot', action='store_true', help='Save figures')
	parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
	parser.add_argument('--interval', type=int, default=10, help='Print epoch intervals')
	parser.add_argument('--reference', '-r', type=str, default='labels.txt', help='Whether ground truth available')
	parser.add_argument('--pretrain', action='store_true', help='Reload the trained model')
	parser.add_argument('--gpu', '-g', type=int, default=0, help='Use gpu when training')
	parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
	
	## data preprocessing
	parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
	parser.add_argument('--gene_filter', action='store_true', help='Perform gene filter as SC3')
	parser.add_argument('-x', '--pct_dropout', type=float, default=6, help='Percent of genes when performing gene filter as SC3')
	
	args = parser.parse_args()

	# Set random seed
	seed = args.seed
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available:
		torch.cuda.manual_seed_all(seed)

	path = os.path.join(args.path, args.dataset) #;print(path)
	device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
	
	# Load data and labels
	data_file = os.path.join(path, 'data.txt')
	dataloader, data, weight_index, raw_index, columns, norm = get_loader(data_file, 
														  args.input_dim, 
														  sep=args.sep,
														  batch_size=args.batch_size, 
														  X=args.pct_dropout,
														  gene_filter=args.gene_filter,
														  log_transform=args.log_transform)
	input_dim = data.shape[1]
	
	if args.reference:
		reference_file = os.path.join(path, args.reference)
		ref, classes = read_labels(reference_file)
	
	if args.n_centroids is None:
		est_k = estimate_k(data.t())
		print('Estimate k: {}'.format(est_k))
		k = est_k
	else:
		k = args.n_centroids
		
	
	# Define Model
	beta = DeterministicWarmup(n=100, t_max=1) # warmup
	dims = [input_dim, args.hidden, args.encode_dim, []]
	model = SCALE(dims, n_centroids=k, beta=beta).to(device)
	

	if args.visdom:
		line = VisdomLinePlotter(args.dataset)
		viz = Visdom()

	# Define output file
	out_dir = os.path.join(args.outpath, args.dataset)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	plot_dir = os.path.join(out_dir, 'png')
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	peak_dir = os.path.join(out_dir, 'specific_peaks')
	if not os.path.exists(peak_dir):
		os.makedirs(peak_dir)
	model_file = os.path.join(out_dir, 'model.pt')
	
	print("======== Parameters ========")
	print('Dataset: {}\nInput_dim: {}\nn_centroids: {}\nEpoch: {}\nSeed: {}'.format(
		args.dataset, input_dim, k, args.epochs, args.seed))
	print("============================")
	
	main()

	# scATAC-seq
	# ./DeepSubtype.py --save --report -d GM12878vsHEK -k 2
	# ./DeepSubtype.py --save  --report -d InSilico -k 6 
	# ./DeepSubtype.py --save --report -d Leukemia -k 6