#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Wed 26 Dec 2018 03:46:19 PM CST

# File Name: dataset.py
# Description:

"""
import time
import os
import numpy as np
import pandas as pd
import scipy
from glob import glob
from scipy.io import mmread
import csv
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):
    """
    Single-cell dataset
    """

    def __init__(self, path, 
                 X = 0,
                 min_peaks = 0,
                 transforms=[]):
        
        self.data, self.peaks, self.cell_id = load_data(path)
        
        if min_peaks > 0:
            self.filter_cell(min_peaks)
        if X>0:
            self.filter_peak(X)
        
        for transform in transforms:
            self.data = transform(self.data)
        
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index];
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return data
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nPeak number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')
        
    def filter_peak(self, X=4):
        """
        Removes rare peaks with (signal > 0) in less than X% of cells 
        """
        total_cells = self.data.shape[0]
        
        count = np.array((self.data >0).sum(0)).squeeze()
        indices = np.where(count > 0.01*X*total_cells)[0]
        self.data = self.data[:, indices]
        self.peaks = self.peaks[indices]
        
    def filter_cell(self, min_peaks=0):
        if min_peaks < 1:
            min_peaks = len(self.peaks)*min_peaks
        indices = np.where(np.sum(self.data>0, 1)>=min_peaks)[0]
        self.data = self.data[indices]
        self.cell_id = self.cell_id[indices]
        

def load_data(path, min_trans=600, ratio=0):
    print("Loading  data ...")
    t0 = time.time()
    if os.path.isdir(path):
        data, peaks, cell_id = read_mtx(path)
    elif os.path.isfile(path):
        data, peaks, cell_id = read_csv(path)
    else:
        raise ValueError("File {} not exists".format(path))
    print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
    return data, peaks, cell_id


def read_mtx(path):
    for filename in glob(path+'/*'):
        if (('count' in filename) or ('matrix' in filename)) and ('mtx' in filename):
            count = mmread(filename).T.tocsr().astype('float32')
        if 'barcode' in filename:
            cell_id = pd.read_csv(filename, sep='\t', header=None)[0].values
        if 'gene' in filename or 'peak' in filename:
            feature = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
    return count, feature, cell_id
    

def read_csv(path):
    if ('.txt' in path) or ('tsv' in path):
        sep = '\t'
    elif '.csv' in path:
        sep = ','
    else:
        raise ValueError("File {} not in format txt or csv".format(path))
    data = pd.read_csv(path, sep=sep, index_col=0).T.astype('float32')
    genes = data.columns.values
    cell_id = data.index.values
    return scipy.sparse.csr_matrix(data.values), genes, cell_id