#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Sat 28 Apr 2018 08:31:29 PM CST

# File Name: SCALE.py
# Description: Single-Cell ATAC-seq Analysis via Latent feature Extraction.
    Input: 
        scATAC-seq data
    Output:
        1. latent GMM feature
        2. cluster assignment
        3. imputation data
"""


import time
import torch

import numpy as np
import pandas as pd
import argparse

from scale import SCALE
from scale.dataset import SingleCellDataset
from scale.utils import read_labels, cluster_report
from scale import config

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction')
    parser.add_argument('--data', '-d', type=str, help='input data matrix peaks x samples')
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number')
    parser.add_argument('--sep', type=str, default='\t', help='input data sep format \t or , ')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--no_results', action='store_true', help='Not Save the results')
    parser.add_argument('--verbose', action='store_false', help='Print loss of training process')
    parser.add_argument('--reference', '-r', type=str, default=None, help='Whether ground truth available')
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--epochs', '-e', type=int, default=None, help='Training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Use gpu when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    parser.add_argument("--local_rank", type=int)
#     parser.add_argument('--input_dim', type=int, default=None, help='Force input dim')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
#     parser.add_argument('--gene_filter', action='store_true', help='Perform gene filter as SC3')
#     parser.add_argument('-x', '--pct', type=float, default=6, help='Percent of genes when performing gene filter as SC3')

    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.device = args.device if torch.cuda.is_available() and args.device!="cpu" else "cpu" 
    device = torch.device(args.device)
    if args.batch_size is None:
        batch_size = config.batch_size
    else:
        batch_size = args.batch_size

    normalizer = MinMaxScaler()
    dataset = SingleCellDataset(args.data, args.reference, transforms=[normalizer.fit_transform])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    device = args.device

    cell_num = dataset.shape[0] 
    input_dim = dataset.shape[1] 	

    k = args.n_centroids

    if args.epochs is None:
        epochs = config.epochs
    else:
        epochs = args.epochs
    if args.lr is None:
        lr = config.lr
    else:
        lr = args.lr

    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq analysis via Latent feature Extraction")
    print("**********************************************************************\n")
    print("======== Parameters ========")
    print('Cell number: {}\nInput_dim: {}\nn_centroids: {}\nEpoch: {}\nSeed: {}\nDevice: {}'.format(
        cell_num, input_dim, k, epochs, args.seed, args.device))
    print("============================")

    dims = [input_dim, config.latent, config.encode_dim, config.decode_dim]
    model = SCALE(dims, n_centroids=k)
    print(model)
    # torch.distributed.init_process_group(backend='gloo', world_size=4, init_method='env://')
    # model = torch.nn.parallel.DistributedDataParallelCPU(model)
#     print(model)

    if not args.pretrain:
        print('\n## Training Model ##')
        t0 = time.time()
        model.init_gmm_params(dataloader)
        model.fit(dataloader,
                  lr=lr, 
                  weight_decay=config.weight_decay, 
                  epochs=epochs, 
                  verbose=args.verbose,
                  print_interval=config.print_interval,
                  device = device
                   )
        print('\nRunning Time: {:.2f} s'.format(time.time()-t0))
    else:
        print('\n## Loading Model {} ##\n'.format(args.pretrain))
        model.load_model(args.pretrain)

    # Clustering Report
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if args.reference:
        pred = model.predict(dataloader, device)
        cluster_report(dataset.labels, pred, dataset.CellType)

    outdir = args.outdir
    import os
    if not args.no_results:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        model.eval()
        torch.save(model.state_dict(), os.path.join(outdir, 'model.pt')) # save model file

        ### output ###
        # 1. latent GMM feature
        
        feature = model.encodeBatch(dataloader, device=device, out='z')

        # 2. cluster assignments
        pred = model.predict(dataloader, device)

        # 3. imputed data
        recon_x = model.encodeBatch(dataloader, device, out='x', transforms=[normalizer.inverse_transform])

        assign_file = os.path.join(outdir, 'cluster_assignments.txt')
        feature_file = os.path.join(outdir, 'feature.txt')
        impute_file = os.path.join(outdir, 'imputed_data.txt')

        pd.Series(pred).to_csv(assign_file, sep='\t', header=False) # save cluster assignments
        pd.DataFrame(feature).to_csv(feature_file, sep='\t', header=False) # save latent feature
        pd.DataFrame(recon_x.T, index=dataset.peaks, columns=dataset.sample_id).to_csv(impute_file, sep='\t') # save imputed data

