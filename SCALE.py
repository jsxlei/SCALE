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


import argparse

from scale import SCALE_function


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
    parser.add_argument('--min_cells', type=float, default=3, help='Remove low quality peaks')
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

    adata = SCALE_function(
        args.data_list, 
        n_centroids = args.n_centroids,
        outdir = args.outdir,
        verbose = args.verbose,
        pretrain = args.pretrain,
        lr = args.lr,
        batch_size = args.batch_size,
        gpu = args.gpu,
        seed = args.seed,
        encode_dim = args.encode_dim,
        decode_dim = args.decode_dim,
        latent = args.latent,
        min_peaks = args.min_peaks,
        min_cells = args.min_cells,
        n_feature = args.n_feature,
        log_transform = args.log_transform,
        max_iter = args.max_iter,
        weight_decay = args.weight_decay,
        impute = args.impute,
        binary = args.binary,
        embed = args.embed,
        reference = args.reference,
        cluster_method = args.cluster_method,
    )

    
