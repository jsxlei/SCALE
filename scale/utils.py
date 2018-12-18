#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Tue 24 Apr 2018 08:05:21 PM CST

# File Name: utils.py
# Description:

"""


import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score

# ============== Data Processing ==============
# =============================================

def get_loader(data_file, input_dim=None, sep='\t', 
               batch_size=16, gene_filter=False, X=6, log_transform=False,
               normalize=True):
    """
    Load data
    Input:
        data file: peaks x cells matrix
    Return:
        dataloader, data, data index, raw data index, columns, and normalizer
    """
    data = pd.read_csv(data_file, index_col=0, sep=sep)
    # data[data>2] = 2
    raw_index = data.index
    columns = data.columns
    if gene_filter:
        data = gene_filter_(data, X);print('* Gene Filter *')
    if log_transform:
        data = np.log2(data+1);print('* Log Transform *')
    data = data.T

    input_dim = input_dim if input_dim else data.shape[1]
    data = sort_by_mad(data).iloc[:,:input_dim]
    index = data.columns.values

    norm = MinMaxScaler()
    if normalize:
        data = norm.fit_transform(data)

    data = torch.Tensor(data)
    dataloader = DataLoader(TensorDataset(data), batch_size, shuffle=True, num_workers=1)

    return dataloader, data, index, raw_index, columns, norm	


def read_labels(ref, return_enc=False):
    """
    Read labels and encode to 0, 1 .. k with class names 
    """
    # if isinstance(ref, str):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes


def sort_by_mad(data, axis=0):
    """
    Sort genes by mad to select input features
    """
    genes = data.mad(axis=axis).sort_values(ascending=False).index
    data = data.loc[:, genes]
    return data

def gene_filter_(data, X=6):
    """
    Gene filter in SC3:
        Removes genes/transcripts that are either (1) or (2)
        (1). rare genes/transcripts:
            Expressed (expression value > 2) in less than X% of cells 
        (2). ubiquitous genes/transcripts:
            Expressed (expression value > 0) in at least (100 - X)% of cells
    Input data:
        data is an array of shape (p,n)
    """
    total_cells = data.shape[1]
    count_1 = data[data > 2].count(axis=1)
    count_2 = data[data > 0].count(axis=1)

    genelist_1 = count_1[count_1 > 0.01*X * total_cells].index
    genelist_2 = count_2[count_2 < 0.01*(100-X) * total_cells].index
    genelist = set(genelist_1) & set(genelist_2)
    data = data.loc[genelist]
    return data


# =========== scATAC Preprocessing =============
# ==============================================
def peak_filter(data, x=10, n_reads=2):
    count = data[data >= n_reads].count(1)
    index = count[count >= x].index
    data = data.loc[index]
    return data

def cell_filter(data):
    thresh = data.shape[0]/50
    # thresh = min(min_peaks, data.shape[0]/50)
    data = data.loc[:, data.sum(0) > thresh]
    return data

def sample_filter(data, x=10, n_reads=2):
    data = peak_filter(data, x, n_reads)
    data = cell_filter(data)
    return data


# =================== Other ====================
# ==============================================
def estimate_k(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (p,n) matrix, p is feature, n is sample
    """
    p, n = data.shape

    x = scale(data)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)
    bd = 3.273 * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k

def peak_selection(weight, weight_index, kind='both', cutoff=2.5):
    """
    Select represented peaks of each components of each peaks, 
    correlations between peaks and features are quantified by decoder weight,
    weight is a Gaussian distribution, 
    filter peaks with weights more than cutoff=2.5 standard deviations from the mean.

    Input:
        model_file: saved model of SCALE model.pt, containing the weight of decoder
        weight_index_file: weight_index.txt, match to original peaks index of data for weight index, 
                           because saved weight index is disorder result by sort_by_mad operation. 
        kind: both for both side, pos for positive side and neg for negative side.
        cutoff: cutoff of standard deviations from mean.
    """
    std = weight.std(0)
    mean = weight.mean(0)
    specific_peaks = []
    for i in range(10):
        w = weight[:,i]
        if kind == 'both':
            index = np.where(np.abs(w-mean[i]) > cutoff*std[i])[0]
        if kind == 'pos':
            index = np.where(w-mean[i] > cutoff*std[i])[0]
        if kind == 'neg':
            index = np.where(mean[i]-w > cutoff*std[i])[0]
        specific_peaks.append(weight_index[index])
    return specific_peaks


def save_results(model, data, data_params, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model.eval()
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pt')) # save model file

    weight_index, raw_index, columns, norm = data_params

    ### output ###
    # 1. latent GMM feature
    feature = model.get_feature(data)

    # 2. cluster assignments
    pred = model.predict(data)

    # 3. imputed data
    recon_x = model.get_imputed_data(data)
    recon_x = norm.inverse_transform(recon_x)

    # 4. cell type specific peaks
    weight = model.state_dict()['decoder.reconstruction.weight'].cpu().numpy()
    specific_peaks = peak_selection(weight, weight_index, kind='both', cutoff=2.5)

    assign_file = os.path.join(outdir, 'cluster_assignments.txt')
    feature_file = os.path.join(outdir, 'feature.txt')
    impute_file = os.path.join(outdir, 'imputed_data.txt')

    pd.Series(pred).to_csv(assign_file, sep='\t', header=False) # save cluster assignments
    pd.DataFrame(feature).to_csv(feature_file, sep='\t', header=False) # save latent feature
    pd.DataFrame(recon_x.T, index=weight_index, columns=columns).loc[raw_index].to_csv(impute_file, sep='\t') # save imputed data

    # save specific peaks
    # peak_dir = outdir+'/specific_peaks/'
    # if not os.path.exists(peak_dir):
        # os.makedirs(peak_dir)
    # all_peaks = []
    # for i, peaks in enumerate(specific_peaks):
        # peak_file = os.path.join(peak_dir, 'peak_index{}.txt'.format(i))
        # open(peak_file, 'w').write('\n'.join(list(peaks)))
        # all_peaks += list(peaks)
    # peak_file = os.path.join(peak_dir, 'peak_index.txt')
    # open(peak_file, 'w').write('\n'.join(set(all_peaks)))
    

def pairwise_pearson(A, B):
    from scipy.stats import pearsonr
    corrs = []
    for i in range(A.shape[0]):
        if A.shape == B.shape:
            corr = pearsonr(A.iloc[i], B.iloc[i])[0]
        else:
            corr = pearsonr(A.iloc[i], B)[0]
        corrs.append(corr)
    return corrs

# ================= Metrics ===================
# =============================================

def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred==i)] = j
        return y_
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)


def cluster_report(ref, pred, classes):
    """
    Print Cluster Report
    """
    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    print('\n## Confusion matrix ##\n')
    print(cm)
    print('\n## Cluster Report ##\n')
    print(classification_report(ref, pred, target_names=classes))
    ari_score = adjusted_rand_score(ref, pred)
    print("\nAdjusted Rand score : {:.4f}".format(ari_score))







