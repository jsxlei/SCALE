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
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):
    """
    Single-cell dataset
    """

    def __init__(self, path, 
                 low = 0,
                 high = 0.9,
                 min_peaks = 0,
                 transpose = False,
                 transforms=[]):
        
        self.data, self.peaks, self.barcode = load_data(path, transpose)
        
        if min_peaks > 0:
            self.filter_cell(min_peaks)
        
        self.filter_peak(low, high)
        
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
        
    def filter_peak(self, low=0, high=0.9):
        """
        Removes rare peaks and common peaks 
            low: low ratio threshold to remove the rare peaks
            high: high ratio threshold to remove the common peaks
        """
        total_cells = self.data.shape[0]
        count = np.array((self.data >0).sum(0)).squeeze()
#         indices = np.where(count > 0.01*X*total_cells)[0] 
        indices = np.where((count > low*total_cells) & (count < high*total_cells))[0] 
        self.data = self.data[:, indices]
        self.peaks = self.peaks[indices]
        
    def filter_cell(self, min_peaks=0):
        """
        Remove low quality cells by threshold of min_peaks
            min_peaks: if >= 1 means the min_peaks number else is the ratio
        """
        if min_peaks < 1:
            min_peaks = len(self.peaks)*min_peaks
        indices = np.where(np.sum(self.data>0, 1)>=min_peaks)[0]
        self.data = self.data[indices]
        self.barcode = self.barcode[indices]
        

def load_data(path, transpose=False):
    print("Loading  data ...")
    t0 = time.time()
    if os.path.isdir(path):
        count, peaks, barcode = read_mtx(path)
    elif os.path.isfile(path):
        count, peaks, barcode = read_csv(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if transpose: 
        count = count.transpose()
    print('Original data contains {} cells x {} peaks'.format(*count.shape))
    assert (len(barcode), len(peaks)) == count.shape
    print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
    return count, peaks, barcode


def read_mtx(path):
    for filename in glob(path+'/*'):
        basename = os.path.basename(filename)
        if (('count' in basename) or ('matrix' in basename)) and ('mtx' in basename):
            count = mmread(filename).T.tocsr().astype('float32')
        elif 'barcode' in basename:
            barcode = pd.read_csv(filename, sep='\t', header=None)[0].values
        elif 'gene' in basename or 'peak' in basename:
            feature = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values

    return count, feature, barcode
    

def read_csv(path):
    if ('.txt' in path) or ('tsv' in path):
        sep = '\t'
    elif '.csv' in path:
        sep = ','
    else:
        raise ValueError("File {} not in format txt or csv".format(path))
    data = pd.read_csv(path, sep=sep, index_col=0).T.astype('float32')
    genes = data.columns.values
    barcode = data.index.values
    return scipy.sparse.csr_matrix(data.values), genes, barcode