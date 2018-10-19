#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Tue 24 Apr 2018 08:05:21 PM CST

# File Name: utils.py
# Description:

"""


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix, adjusted_rand_score


# ================= Metrics ===================
# =============================================

def cluster_acc(Y_pred, Y):
	"""
	Calculate clustering accuracy
	Args:
		Y_pred: predict y classes
		Y: true y classes
	Return:
		f1_score: clustering f1 score
		y_pred: reassignment index predict y classes
		indices: classes assignment
	"""
	from sklearn.utils.linear_assignment_ import linear_assignment
	assert Y_pred.size == Y.size
	D = max(Y_pred.max(), Y.max())+1
	w = np.zeros((D,D), dtype=np.int64)
	for i in range(Y_pred.size):
		w[Y_pred[i], Y[i]] += 1
	ind = linear_assignment(w.max() - w)

	# acc = sum([w[i,j] for i,j in ind])*1.0/Y_pred.size
	y_ = reassign_cluster(Y_pred, ind)
	f1 = f1_score(Y, y_, average='micro')
	return f1, y_

def reassign_cluster(y_pred, index):
	y_ = np.zeros_like(y_pred)
	for i, j in index:
		y_[np.where(y_pred==i)] = j
	return y_


def cal_silhouette(data, pred, metric='cosine'):
	"""
	Calculate silhouette_score
	"""
	try:
		score = silhouette_score(data, pred, random_state=0, metric=metric)
	except Exception as error:
		print(error)
		score = 0

	return score


def cluster_report(ref, pred, classes):
	"""
	Print Cluster Report
	"""
	f1, pred = cluster_acc(pred, ref)
	cm = confusion_matrix(ref, pred)
	print(cm)
	print('\n')
	print(classification_report(ref, pred, target_names=classes))
	return cm


def compare_with_other_methods(model, data, k, hidden=10, device='cpu', ref=None):
	"""
	Compare with other methods including:
		k-means, 
		hclust,
		PCA+k-means,
		tSNE+k-means,
		GMM
		...
	"""
	from sklearn.cluster import KMeans, AgglomerativeClustering
	from sklearn.mixture import GaussianMixture
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA

	methods = ['k-means', 'hclust', 'PCA+k-means', 'tSNE+k-means', 'SCALE']
	print('-'*60)
	if ref is not None:
		print('Method\t\tSilhouette\tF1_score\tARI')
	else:
		print('Method\t\tSilhouette')
	print('-'*60)
	for method in methods:
		if method == 'SCALE':
			pred = model.predict(data.to(device))
		elif method == 'Imputed':
			recon_x = model(data.to(device))
			pred = model.predict(recon_x)
		elif method == 'hclust':
			pred = AgglomerativeClustering(n_clusters=k).fit_predict(data)
		elif method == 'GMM':
			pred = GaussianMixture(n_components=n_centroids, covariance_type='diag').fit_predict(data)
		elif method == 'k-means':
			pred = KMeans(n_clusters=k).fit_predict(data)
		elif method == 'PCA+k-means':
			feature = PCA(n_components=hidden).fit_transform(data)
			pred = KMeans(n_clusters=k).fit_predict(feature)
		elif method == 'tSNE+k-means':
			feature = TSNE(n_components=2).fit_transform(data)
			pred = KMeans(n_clusters=k).fit_predict(feature)
		else:
			print('No such method')
		score = cal_silhouette(data, pred)

		if ref is not None:
			f1_score, pred = cluster_acc(pred, ref)
			ari_score = adjusted_rand_score(ref, pred)
			print('{:12}\t{:.3f}\t\t{:.3f}\t\t{:.3f}'.format(method, score, f1_score, ari_score))
		else:
			print('{:12}\t{:.4f}'.format(method, score))

	print('-'*60)



# ============== Data Processing ==============
# =============================================

def get_loader(data_file, input_dim=None, sep='\t', batch_size=16, gene_filter=False, X=6, log_transform=False, normalize=True):
	"""
	Load data
	Input:
		data file: peaks x cells matrix
	Return:
		dataloader, data, data index, raw data index, columns, and normalizer
	"""
	# data = pd.read_csv(data_file, index_col=0, sep='\t').iloc[:,:input_dim]
	data = pd.read_csv(data_file, index_col=0, sep=sep)
	raw_index = data.index
	columns = data.columns
	if gene_filter:
		data = gene_filter_(data, X);print('* Gene Filter *')
	if log_transform:
		data = np.log2(data+1);print('* Log Transform *')
	data = data.T
	print('data shape: {}'.format(data.shape))

	input_dim = input_dim if input_dim else data.shape[1]
	# if input_dim != data.shape[1]:
	data = sort_by_mad(data).iloc[:,:input_dim]
	index = data.columns.values

	norm = MinMaxScaler()
	if normalize:
		data = norm.fit_transform(data)

	data = torch.Tensor(data)
	dataloader = DataLoader(TensorDataset(data), batch_size, shuffle=True, num_workers=4)

	return dataloader, data, index, raw_index, columns, norm	


def read_labels(ref):
	"""
	Read labels and encode to 0, 1 .. k with class names 
	"""
	# if isinstance(ref, str):
	ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

	encode = LabelEncoder()
	ref = encode.fit_transform(ref.values.squeeze())
	classes = encode.classes_
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
	#if format in ['RPKM', 'FPKM', 'CPM', 'TPM', 'READS', 'UMI']:
	count_1 = data[data > 2].count(axis=1)
	count_2 = data[data > 0].count(axis=1)

	genelist_1 = count_1[count_1 > 0.01*X * total_cells].index
	genelist_2 = count_2[count_2 < 0.01*(100-X) * total_cells].index
	genelist = set(genelist_1) & set(genelist_2)
	data = data.loc[genelist]
	# data = np.log2(data+1).T

	# print(data.shape)
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
	print(data.shape)
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


def feature_specifity(feature_file, assignment_file):
	"""
	Calculate the feature specifity:

	Input:
		feature_file: output feature.txt file
	"""
	feature = pd.read_csv(feature_file, sep='\t', header=None, index_col=0)
	ref, classes = read_labels(assignment_file)
	n_cluster = max(ref) + 1
	pvalue_mat = np.zeros((10, n_cluster))
	for cluster in range(n_cluster):
		for feat in range(10):
			a = feature.iloc[:, feat][ref == cluster]
			b = feature.iloc[:, feat][ref != cluster]
			pvalue = f_oneway(a,b)[1]
			pvalue_mat[feat, cluster] = pvalue

	plt.figure(figsize=(6, 6))
	grid = sns.heatmap(-np.log10(pvalue_mat), cmap='RdBu_r', 
					   vmax=20,
					   yticklabels=np.arange(10)+1, 
					   xticklabels=classes[:n_cluster])
	grid.set_ylabel('Feature', fontsize=18)
	grid.set_xticklabels(labels=classes[:n_cluster], rotation=45, fontsize=18)
	grid.set_yticklabels(labels=np.arange(10)+1, fontsize=16)
#     grid.set_title(dataset, fontsize=18)
	cbar = grid.collections[0].colorbar
	cbar.set_label('-log10 (Pvalue)', fontsize=18) #, rotation=0, x=-0.9, y=0)