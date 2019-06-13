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
import scipy.io
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
        
        self.load_data(path)
        
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
    
    def load_data(self, path):
        print("Loading  data ...")
        t0 = time.time()
        exist_file = False
        for data_file in ['data.txt', 'data.txt.gz', 'data.mtx', 'data.mtx.gz']:
            data_file = os.path.join(path, data_file)
            if os.path.exists(data_file):
                if 'txt' in data_file:
                    data = pd.read_csv(data_file, sep='\t', index_col=0).T
                    self.peaks = data.columns.values
                    self.cell_id = data.index.values
                    self.data = data.values
                    self.dense = True
                elif 'mtx' in data_file:
                    self.data = scipy.io.mmread(data_file).T.tocsr()
                    peaks = pd.read_csv(os.path.join(path, 'peaks.txt'), sep='\t', header=None)
                    peaks = peaks[0].astype('str') + '_' + peaks[1].astype('str') + '_' + peaks[2].astype('str')
                    self.peaks = peaks.values
                    self.cell_id = [row[0] for row in csv.reader(
                        open(os.path.join(path, 'cell_id.txt')), delimiter="\t")]
                    self.peaks, self.cell_id = np.array(self.peaks), np.array(self.cell_id)
                    self.dense = False
                exist_file = True
                break
                
        if not exist_file:
            raise "Error: No data.txt or data.txt.gz file in {}".format(path)
            
        print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
  
    
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
        
