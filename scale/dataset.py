#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Wed 26 Dec 2018 03:46:19 PM CST

# File Name: dataset.py
# Description:

"""

import os
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):
    """
    Single-cell dataset
    """

    def __init__(self, data_file,
                 celltype_file = None,
                 transforms=[]):

        self.data_file = data_file
        self.celltype_file = celltype_file
        if self.check_exist_file():
            self.data, self.celltype = self.load_processed_data()
        else:
            self.data, self.celltype = self.process()
            
        for transform in transforms:
            self.data = transform(self.data)
        
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if self.celltype is None:
            return data
        else:
            celltype = self.celltype[index]
            return data, celltype

    def check_exist_file(self):
        if not os.path.exists(self.data_file):
            print('Error: No such file: {}'.format(self.data_file))
            return False
        if self.celltype_file is not None:
            if not os.path.exists(self.celltype_file):
                print('Error: No such file: {}'.format(self.celltype_file))
                return False
        return True
    
    def load_processed_data(self):
        print("Loading processed data ...")
        
        data = pd.read_csv(self.data_file, sep='\t', index_col=0).T
        self.peaks = data.columns.values
        self.sample_id = data.index.values
        
        if self.celltype_file is not None:
            celltype = pd.read_csv(self.celltype_file, sep='\t', header=None, index_col=0)[1].values
            encoder = LabelEncoder()
            celltype = encoder.fit_transform(celltype)
            self.CellType = encoder.classes_
            self.n_celltype = len(self.CellType)
        else:
            celltype = None
            
        print("Finished loading!")
        
        return data.values, celltype
        
    def downsampling_cell(self, size=1., seed=124):
        size = int(size * self.n_cells) if type(size) is not int else size
        np.random.seed(seed)
        indices = np.random.choice(self.n_cells, size=size, replace=False)
        return indices
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nPeak number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')

    def process(self):
        raise NotImplementedError
        
    def peak_filter(self, X=4, inplace=True):
        """
        Similar to Gene filter in SC3:
            Removes peaks that are either (1) or (2)
            (1). rare peaks:
                Peaks (signal > 0) in less than X% of cells 
            (2). ubiquitous peaks:
                Peaks (signal > 0) in at least (100 - X)% of cells
        Input data:
            data is an array of shape (n_peak, n_cell)
        """
        total_cells = self.data.shape[1]
        count = (self.data >0).sum(0); print(count.shape)
        indices1 = np.where(count > 0.01*X*total_cells)[0]
        indices2 = np.where(count < 0.01*(100-X)*total_cells)[0]
        indices = set(indices1) & set(indices2)
        self.data = self.data[:, indices]
        self.peaks = self.peaks[indices]
        
