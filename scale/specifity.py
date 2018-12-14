#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 01 Nov 2018 04:42:30 PM CST

# File Name: scale/specifity.py
# Description:

"""
import numpy as np
import pandas as pd
import scipy as sp

def jsd(p, q, base=np.e):
    """
        Jensen Shannon_divergence
    """
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.

def jsd_sp(p, q, base=np.e):
    """
        Define specificity score:
        score = 1 - sqrt(jsd(p, q))
    """
    return 1- jsd(p, q, base=np.e)**0.5

def log2norm(e):
    """
        log2(e+1) normalization 
    """
    loge = np.log2(e+1)
    return loge/sum(loge)

def predefined_pattern(t, labels):
    q = np.zeros(len(labels))
    q[np.where(labels==t)[0]] = 1
    return q

def vec_specificity_score(e, t, labels):
    """
        Calculate a vector specificity for cluster t
    """
    e = log2norm(e)
    et = log2norm(predefined_pattern(t, labels))
    return jsd_sp(e, et)

def mat_specificity_score(mat, labels):
    """
        Calculate all peaks or genes specificity across all clusters
        Return:
            peaks/genes x clusters dataframe
    """
    scores = []
    for i in np.unique(labels):
        score = mat.apply(lambda x: vec_specificity_score(x, i, labels), axis=1)
        scores.append(score)
    return pd.concat(scores, axis=1)

def cluster_specific(score_mat, classes=None, top=0):
    """
        Identify top specific peaks for each cluster
        
        Input: 
            score_mat calculated by mat_specificity_score
        Return:
            specific peaks index and peaks labels
    """
    scores = score_mat.max(1)
    peak_labels = np.argmax(score_mat.values, axis=1)
    inds = []
    labels = []
    if classes is None:
        classes = np.unique(peak_labels)
    for i in classes:
        index = np.where(peak_labels==i)[0]
        ind = np.argsort(scores[index])[-top:]
        ind = index[ind]
        inds.append(ind)
        labels.append(peak_labels[ind])
    return np.concatenate(inds), np.concatenate(labels)
    

