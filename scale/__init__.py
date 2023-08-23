#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 31 May 2018 08:45:33 PM CST

# File Name: __init__.py
# Description:

"""

__author__ = "Lei Xiong"
__email__ = "jsxlei@gmail.com"

from .layer import *
from .model import *
from .loss import *


#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 31 May 2018 08:45:33 PM CST

# File Name: __init__.py
# Description:

"""

__author__ = "Lei Xiong"
__email__ = "jsxlei@gmail.com"

from .layer import *
from .model import *
from .loss import *
from .dataset import load_dataset
from .utils import estimate_k, binarization


import time
import torch

import numpy as np
import pandas as pd
import os
import scanpy as sc
from anndata import AnnData

from typing import Union, List

def SCALE_function(
        data_list:Union[str, List], 
        n_centroids:int = 30,
        outdir:bool = None,
        verbose:bool = False,
        pretrain:str = None,
        lr:float = 0.0002,
        batch_size:int = 64,
        gpu:int = 0,
        seed:int = 18,
        encode_dim:List = [1024, 128],
        decode_dim:List = [],
        latent:int = 10,
        min_peaks:int = 100,
        min_cells:Union[float, int] = 3,
        n_feature:int = 100000,
        log_transform:bool = False,
        max_iter:int = 30000,
        weight_decay:float = 5e-4,
        impute:bool = False,
        binary:bool = False,
        embed:str = 'UMAP',
        reference:str = 'cell_type',
        cluster_method:str = 'leiden',
    )->AnnData:
    
    """
    Use SCALE [Xiong19]_ to cluster and impute on scATAC-seq

    Parameters
    ----------
    data_list
        A path of input data, could be AnnData, or file in format of h5ad, txt, csv or dir containing mtx file
    n_centroids
        Initial n_centroids, default is 30
    outdir
        output of dir where results will be saved, if None, will return an AnnData object
    verbose
        Verbosity, default is False
    pretrain
        Use the pretrained model to project on new data or repeat results
    lr
        learning rate for training model
    batch_size
        batch_size of one iteration for training model 
    gpu
        Use id of gpu device 
    seed
        Random seed
    encode_dim
        encode architecture 
    decode_dim
        decode architecture
    latent
        dimensions of latent
    min_peaks
        min peaks for filtering cells
    min_cells
        min cells for filtering peaks, will be ratio if small than 1, default is 3 
    n_feature
        number of the most highly variable peaks will be used for training model
    log_transform
        log transform the data, recommended for scRNA-seq data
    max_iter
        Max iteration for training model
    weight_decay
        weight decay for training model
    impute
        output the imputed data in adata if true
    binary
        Change the imputed data to binary format
    embed
        Embedding method, UMAP or tSNE, default is UMAP
    reference
        reference annotations for evaluation, default is cell_type
    cluster_method
        cluster method, default is Leiden

    Returns
    -------
    adata
        An AnnData object


    Example
    -------
    >>> from scale import SCALE_function
    >>> adata = sc.read('Your/scATAC/Data')
    >>> adata = SCALE_function(adata)

    if want imputed data   

    >>> adata = SCALE_function(adata, impute=True)

    imputed data will be saved in adata.obsm['impute']
    binary version of imputed data will be saved in adata.obsm['binary']

    )->AnnData:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
    print("**********************************************************************\n")
    

    adata, trainloader, testloader = load_dataset(
        data_list,
        min_genes=min_peaks,
        min_cells=min_cells,
        n_top_genes=n_feature,
        batch_size=batch_size, 
        log=None,
    )

    cell_num = adata.shape[0] 
    input_dim = adata.shape[1] 	
    
    k = n_centroids
    
    if outdir:
        outdir =  outdir+'/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('outdir: {}'.format(outdir))
        
    print("\n======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}'.format(
        cell_num, input_dim, max_iter, batch_size,  min_peaks,  min_cells))
    print("============================")

    latent = 10
    encode_dim = [1024, 128]
    decode_dim = []
    dims = [input_dim,  latent,  encode_dim,  decode_dim]
    model = SCALE(dims, n_centroids=k)
    # print(model)

    if not pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  verbose= verbose,
                  device = device,
                  max_iter= max_iter,
                  outdir=outdir
                   )
        if outdir:
            torch.save(model.state_dict(), os.path.join(outdir, 'model.pt')) # save model
    else:
        print('\n## Loading Model: {}\n'.format(pretrain))
        model.load_model(pretrain)
        model.to(device)
    
    ### output ###

    # 1. latent feature
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')

    # 2. cluster
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    if cluster_method == 'leiden':
        sc.tl.leiden(adata)
    elif cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)


    sc.set_figure_params(dpi=80, figsize=(6,6), fontsize=10)
    if outdir:
        sc.settings.figdir = outdir
        save = '.png'
    else:
        save = None
    if  embed == 'UMAP':
        sc.tl.umap(adata, min_dist=0.1)
        color = [c for c in ['celltype',  'kmeans', 'leiden', 'cell_type'] if c in adata.obs]
        sc.pl.umap(adata, color=color, save=save, show=False, wspace=0.4, ncols=4)
    elif  embed == 'tSNE':
        sc.tl.tsne(adata, use_rep='latent')
        color = [c for c in ['celltype',  'kmeans', 'leiden', 'cell_type'] if c in adata.obs]
        sc.pl.tsne(adata, color=color, save=save, show=False, wspace=0.4, ncols=4)
    
    if  impute:
        print("Imputation")
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
        adata.obsm['binary'] = binarization(adata.obsm['impute'], adata.X)
    
    if outdir:
        adata.write(outdir+'adata.h5ad', compression='gzip')
    
    return adata