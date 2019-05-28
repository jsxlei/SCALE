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
import os
import argparse

from scale import SCALE
from scale.dataset import SingleCellDataset
from scale.utils import read_labels, cluster_report, estimate_k, binarization

from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction')
    parser.add_argument('--dataset', '-d', type=str, help='input dataset path')
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--no_results', action='store_true', help='Not Save the results')
    parser.add_argument('--binary', action='store_true', help='Transform imputed data to binary values')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Use gpu when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    # parser.add_argument("--beta", type=int, default=1)
    # parser.add_argument('-n', type=int, default=200)
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[1024, 128], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    parser.add_argument('-x', type=int, default=0, help='Remove peaks (signal > 0) in less than X percent of cells')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--max_iter', '-i', type=int, default=30000, help='Max iteration')
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.device = args.device if torch.cuda.is_available() and args.device!="cpu" else "cpu" 
    device = torch.device(args.device)
    batch_size = args.batch_size

    normalizer = MaxAbsScaler()
    dataset = SingleCellDataset(args.dataset, X=args.x, transforms=[normalizer.fit_transform])
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    device = args.device

    cell_num = dataset.shape[0] 
    input_dim = dataset.shape[1] 	
    
    if args.n_centroids is None:
        k = estimate_k(dataset.data.T)
        print('Estimate k {}'.format(k))
    else:
        k = args.n_centroids
    lr = args.lr
    name = args.dataset.strip('/').split('/')[-1]
        
    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq analysis via Latent feature Extraction")
    print("**********************************************************************\n")
    print("======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}'.format(
        cell_num, input_dim, k, args.max_iter, batch_size))
    print("============================")

    dims = [input_dim, args.latent, args.encode_dim, args.decode_dim]
    model = SCALE(dims, n_centroids=k)
#     print(model)

    if not args.pretrain:
        print('\n## Training Model ##')
        t0 = time.time()
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  weight_decay=args.weight_decay,
                  verbose=args.verbose,
                  device = device,
                  max_iter=args.max_iter,
#                   n=args.n,
#                   beta=args.beta,
                  name=name
                   )
        print('\nRunning Time: {:.2f} s'.format((time.time()-t0)/60))
    else:
        print('\n## Loading Model {} ##\n'.format(args.pretrain))
        model.load_model(args.pretrain)
        model.to(device)

    outdir = args.outdir
    if not args.no_results:
        print('outidr: {}'.format(outdir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        ### output ###
        # 1. latent GMM feature
        feature = model.encodeBatch(testloader, device=device, out='z')
        pd.DataFrame(feature).to_csv(os.path.join(outdir, 'feature.txt'), sep='\t', header=False)
        
        # 2. cluster assignments
        pred = model.predict(testloader, device)
        pd.Series(pred).to_csv(os.path.join(outdir, 'cluster_assignments.txt'), sep='\t', header=False)
                               
        # gmm_pred = model.predict(testloader, device, method='gmm')
        # pd.Series(gmm_pred).to_csv(os.path.join(outdir, 'gmm_predict.txt'), sep='\t', header=False)
        
        if dataset.celltype is not None:
            y = dataset.CellType[dataset.celltype]
        else:
            y = pred
            
        # 3. imputed data
        recon_x = model.encodeBatch(testloader, device, out='x', transforms=[normalizer.inverse_transform])
        recon_x = pd.DataFrame(recon_x.T, index=dataset.peaks, columns=dataset.cell_id)
        if args.binary:
            recon_x = binarization(recon_x, raw)
        recon_x.to_csv(os.path.join(outdir, 'imputed_data.txt'), sep='\t') 

        torch.save(model.to('cpu').state_dict(), os.path.join(outdir, 'model.pt')) # save model
        np.savetxt(os.path.join(args.outdir, 'mu_c.txt'), model.mu_c.cpu().detach().numpy())

        from scale.plot import plot_embedding
        plot_embedding(feature, y, #marker=model.mu_c.cpu().detach().numpy().T, 
                       save=os.path.join(outdir, 'tsne.pdf'), save_emb=os.path.join(outdir, 'tsne.txt'))
        
