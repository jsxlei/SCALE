#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Sat 28 Apr 2018 08:31:29 PM CST

# File Name: SCALE.py
# Description: Single-Cell ATAC-seq Analysis via Latent feature Extraction.
    Input: 
        scATAC-seq data
    Output:
        1. latent feature
        2. cluster assignment
        3. imputation data
"""


import time
import torch

import numpy as np
import pandas as pd
import os
import scanpy as sc
import argparse

from scale import SCALE
from scale.dataset import load_dataset
from scale.utils import read_labels, cluster_report, estimate_k, binarization
from scale.plot import plot_embedding

from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction')
    parser.add_argument('--data_list', '-d', type=str, nargs='+', default=[])
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=30)
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[1024, 128], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    parser.add_argument('--min_peaks', type=float, default=100, help='Remove low quality cells with few peaks')
    parser.add_argument('--min_cells', type=float, default=0.01, help='Remove low quality peaks')
    parser.add_argument('--n_feature', type=int, default=100000, help='Keep the number of highly variable peaks')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--max_iter', '-i', type=int, default=30000, help='Max iteration')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--impute', action='store_true', help='Save the imputed data in layer impute')
    parser.add_argument('--binary', action='store_true', help='Save binary imputed data in layer binary')
    parser.add_argument('--embed', type=str, default='UMAP')
    parser.add_argument('--reference', type=str, default='celltype')
    parser.add_argument('--cluster_method', type=str, default='leiden')

    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device='cpu'
    batch_size = args.batch_size
    
    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
    print("**********************************************************************\n")

    adata, trainloader, testloader = load_dataset(
        args.data_list,
        batch_categories=None, 
        join='inner', 
        batch_key='batch', 
        batch_name='batch',
        min_genes=args.min_peaks,
        min_cells=args.min_cells,
        batch_size=args.batch_size, 
        n_top_genes=args.n_feature,
        log=None,
    )

    cell_num = adata.shape[0] 
    input_dim = adata.shape[1] 	
    
#     if args.n_centroids is None:
#         k = estimate_k(adata.X.T)
#         print('Estimate k = {}'.format(k))
#     else:
#         k = args.n_centroids
    lr = args.lr
    k = args.n_centroids
    
    outdir = args.outdir+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    print("\n======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}'.format(
        cell_num, input_dim, k, args.max_iter, batch_size, args.min_peaks, args.min_cells))
    print("============================")

    dims = [input_dim, args.latent, args.encode_dim, args.decode_dim]
    model = SCALE(dims, n_centroids=k)
    print(model)

    if not args.pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  weight_decay=args.weight_decay,
                  verbose=args.verbose,
                  device = device,
                  max_iter=args.max_iter,
#                   name=name,
                  outdir=outdir
                   )
        torch.save(model.state_dict(), os.path.join(outdir, 'model.pt')) # save model
    else:
        print('\n## Loading Model: {}\n'.format(args.pretrain))
        model.load_model(args.pretrain)
        model.to(device)
    
    ### output ###
    print('outdir: {}'.format(outdir))
    # 1. latent feature
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')

    # 2. cluster
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    if args.cluster_method == 'leiden':
        sc.tl.leiden(adata)
    elif args.cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)

#     if args.reference in adata.obs:
#         cluster_report(adata.obs[args.reference].cat.codes, adata.obs[args.cluster_method].astype(int))

    sc.settings.figdir = outdir
    sc.set_figure_params(dpi=80, figsize=(6,6), fontsize=10)
    if args.embed == 'UMAP':
        sc.tl.umap(adata, min_dist=0.1)
        color = [c for c in ['celltype', args.cluster_method] if c in adata.obs]
        sc.pl.umap(adata, color=color, save='.png', wspace=0.4, ncols=4)
    elif args.embed == 'tSNE':
        sc.tl.tsne(adata, use_rep='latent')
        color = [c for c in ['celltype', args.cluster_method] if c in adata.obs]
        sc.pl.tsne(adata, color=color, save='.png', wspace=0.4, ncols=4)
    
    if args.impute:
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
    if args.binary:
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
        adata.obsm['binary'] = binarization(adata.obsm['impute'], adata.X)
        del adata.obsm['impute']
    
    adata.write(outdir+'adata.h5ad', compression='gzip')
