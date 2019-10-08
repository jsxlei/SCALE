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
import argparse

from scale import SCALE
from scale.dataset import SingleCellDataset
from scale.utils import read_labels, cluster_report, estimate_k, binarization
from scale.plot import plot_embedding

from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction')
    parser.add_argument('--dataset', '-d', type=str, help='input dataset path')
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[1024, 128], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    parser.add_argument('--low', '-x', type=float, default=0.01, help='Remove low ratio peaks')
    parser.add_argument('--high', type=float, default=0.9, help='Remove high ratio peaks')
    parser.add_argument('--min_peaks', type=float, default=200, help='Remove low quality cells with few peaks')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--max_iter', '-i', type=int, default=30000, help='Max iteration')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--impute', action='store_true', help='Save the imputed data')
    parser.add_argument('--binary', action='store_true', help='Save binary imputed data')
    parser.add_argument('--no_tsne', action='store_true', help='Not save the tsne embedding')
    parser.add_argument('--reference', '-r', default=None, type=str, help='Reference celltypes')
    parser.add_argument('--transpose', '-t', action='store_true', help='Transpose the input matrix')

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

    normalizer = MaxAbsScaler()
    dataset = SingleCellDataset(args.dataset, low=args.low, high=args.high, min_peaks=args.min_peaks,
                                transpose=args.transpose, transforms=[normalizer.fit_transform])
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    cell_num = dataset.shape[0] 
    input_dim = dataset.shape[1] 	
    
    if args.n_centroids is None:
        k = estimate_k(dataset.data.T)
        print('Estimate k {}'.format(k))
    else:
        k = args.n_centroids
    lr = args.lr
    name = args.dataset.strip('/').split('/')[-1]
    args.min_peaks = int(args.min_peaks) if args.min_peaks >= 1 else args.min_peaks
    
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
    print("**********************************************************************\n")
    print("======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\nrare peak filter: {}\ncommon peak filter: {}'.format(
        cell_num, input_dim, k, args.max_iter, batch_size, args.min_peaks, args.low, args.high))
    print("============================")

    dims = [input_dim, args.latent, args.encode_dim, args.decode_dim]
    model = SCALE(dims, n_centroids=k)
#     print(model)

    if not args.pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  weight_decay=args.weight_decay,
                  verbose=args.verbose,
                  device = device,
                  max_iter=args.max_iter,
                  name=name,
                  outdir=outdir
                   )
#         torch.save(model.to('cpu').state_dict(), os.path.join(outdir, 'model.pt')) # save model
    else:
        print('\n## Loading Model: {}\n'.format(args.pretrain))
        model.load_model(args.pretrain)
        model.to(device)

    ### output ###
    print('outdir: {}'.format(outdir))
    # 1. latent feature
    feature = model.encodeBatch(testloader, device=device, out='z')
    pd.DataFrame(feature).to_csv(os.path.join(outdir, 'feature.txt'), sep='\t', header=False)

    # 2. cluster assignments
    pred = model.predict(testloader, device)
    pd.Series(pred, index=dataset.barcode).to_csv(os.path.join(outdir, 'cluster_assignments.txt'), sep='\t', header=False)
            
    # 3. imputed data
    if args.impute or args.binary:
        recon_x = model.encodeBatch(testloader, device, out='x', transforms=[normalizer.inverse_transform])
        if args.binary:
            import scipy
            print("Saving binary imputed data")
            recon_x = binarization(recon_x, dataset.data).T
            imputed_dir = outdir+'/binary_imputed/'
            os.makedirs(imputed_dir, exist_ok=True)
            scipy.io.mmwrite(imputed_dir+'count.mtx', recon_x)
            pd.Series(dataset.peaks).to_csv(imputed_dir+'peak.txt', sep='\t', index=False, header=None)
            pd.Series(dataset.barcode).to_csv(imputed_dir+'barcode.txt', sep='\t', index=False, header=None)
        elif args.impute:
            print("Saving imputed data")
            recon_x = pd.DataFrame(recon_x.T, index=dataset.peaks, columns=dataset.barcode)
            recon_x.to_csv(os.path.join(outdir, 'imputed_data.txt'), sep='\t') 
        
#     torch.save(model.to('cpu').state_dict(), os.path.join(outdir, 'model.pt')) # save model
    
    if not args.no_tsne:
        print("Plotting tSNE embedding")
        if args.reference:
            ref = pd.read_csv(args.reference, sep='\t', header=None, index_col=0)[1]
            labels = ref.reindex(dataset.barcode, fill_value='unknown')
        else:
            labels = pred
        plot_embedding(feature, labels, 
                       save=os.path.join(outdir, 'tsne.pdf'), save_emb=os.path.join(outdir, 'tsne.txt'))
        
