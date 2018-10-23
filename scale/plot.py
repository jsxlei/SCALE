#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Mon 09 Apr 2018 07:36:48 PM CST

# File Name: plotting.py
# Description:

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['figure.dpi'] = 300


def plot_confusion_matrix(cm, x_classes, y_classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues,
						  figsize=(5,5),
						  mark=True,
						  save=None,
						  rotation=45,
						  show_cbar=True
						):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.

	Params:
		cm: confusion matrix, MxN 
		x_classes: N
		y_classes: M
	"""
	import itertools
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig = plt.figure(figsize=figsize)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)

	plt.title(title)

	x_tick_marks = np.arange(len(x_classes))
	y_tick_marks = np.arange(len(y_classes))
	plt.xticks(x_tick_marks, x_classes, rotation=rotation, ha='right')
	plt.yticks(y_tick_marks, y_classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	if mark:
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if cm[i, j] > 0:
				plt.text(j, i, format(cm[i, j], fmt),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('Predicted Cluster')
	if show_cbar:
		plt.colorbar(shrink=0.8) 
	if save:
		plt.savefig(save, format='pdf')
	else:
		plt.show()


def plot_heatmap(X, y, classes, 
				 cax_title='', xlabel='', ylabel='', yticklabels='', legend_font=10, 
				 show_legend=True, show_cax=True, tick_color='black',
				 bbox_to_anchor=(0.5, 1.3), position=(0.8, 0.78, .1, .04),
				 save=None, **kw):
	"""
	plot hidden code heatmap with labels

	Params:
		X: nxf array, n is sample number, f is feature
		y: a array of labels for n elements or a list of array
		classes: n_centroids classes
	"""
	import matplotlib.patches as mpatches  # add legend
	colormap = plt.cm.tab20
	index = np.argsort(y)
	X = X.iloc[:,index]
	y = y[index]

	n_centroids = max(y)+1
	colors = [colormap(i) for i in range(n_centroids)]
	col_colors = [ colors[i] for i in y ]
	legend_TN = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, classes)]


	if show_legend:
		kw.update({'col_colors':col_colors})

	cbar_kws={"orientation": "horizontal"}
	grid = sns.clustermap(X, yticklabels=True, cbar_kws=cbar_kws, **kw)
	if show_cax:
		grid.cax.set_position(position)
		grid.cax.tick_params(length=1, labelsize=4, rotation=0)
		# grid.cax.set_title(cax_title,fontsize=5, rotation='vertical',x=-0.55, y=0.8)
		grid.cax.set_title(cax_title, fontsize=6, y=0.35)
	# grid.ax_heatmap.set_position((0.3,0.,0.7,0.9))

	if show_legend:
		grid.ax_heatmap.legend(loc='upper center', 
							   bbox_to_anchor=bbox_to_anchor, 
							   handles=legend_TN, 
							   fontsize=legend_font, 
							   frameon=False, 
							   ncol=3)
		grid.ax_col_colors.tick_params(labelsize=6, length=0, labelcolor='orange')

	grid.ax_heatmap.set_xlabel(xlabel)
	grid.ax_heatmap.set_ylabel(ylabel, fontsize=8)
	grid.ax_heatmap.set_xticklabels('')
	grid.ax_heatmap.set_yticklabels(yticklabels, color=tick_color)
	grid.ax_heatmap.tick_params(axis='x', length=0)
	grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0, rotation=0)
	grid.ax_row_dendrogram.set_visible(False)
	grid.cax.set_visible(show_cax)
	grid.row_color_labels = classes

	if save:
		plt.savefig(save, format='pdf')
	else:
		plt.show()


def plot_embedding(X, y, classes, method='TSNE', figsize=(4,4), markersize=10, save=None, name='', legend=True):
	"""
	Visualize TSNE embedding with labels

	Params:
		X: nxf array, n is sample number, f is feature
		y: a array of labels for n elements 
		labels: n_centroids classes
	"""
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA, FastICA

	if X.shape[1] != 2:
		if method == 'PCA':
			pca = PCA(n_components=2, random_state=124)
			X = pca.fit_transform(X)
			ratio = pca.explained_variance_ratio_
			x_label = 'PCA dim1 {:.2%}'.format(ratio[0])
			y_label = 'PCA dim2 {:.2%}'.format(ratio[1])
		elif method == 'TSNE':
			X = TSNE(n_components=2, random_state=124).fit_transform(X)
			x_label = 'tSNE dim 1'
			y_label = 'tSNE dim 2'
		elif method == 'ICA':
			ica = FastICA(n_components=2, random_state=124)
			X = ica.fit_transform(X)
			x_label = 'ICA dim 1'
			y_label = 'ICA dim 2'

	n_centroids = max(y)+1
	assert n_centroids == len(classes), 'Mismatch n_centroids with classes'
	# fig = plt.figure(figsize=(10,10))
	if n_centroids > 20:
		colormap = plt.cm.gist_ncar
		colors = [colormap(i) for i in np.linspace(0, 0.9, n_centroids)]
	else:
		colormap = plt.cm.tab20
		# n = 20 // n_centroids
		colors = [colormap(i) for i in range(n_centroids)]

	plt.figure(figsize=figsize)
	for i,c in enumerate(classes):
		plt.scatter(X[y==i,0], X[y==i,1], s=markersize, color=colors[i], label=c)
	if legend:
		plt.legend(loc=9, bbox_to_anchor=(0.5,1.2), fontsize=10, ncol=3, frameon=False, markerscale=2.5)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.xlabel(x_label, fontsize=12)
	plt.ylabel(y_label, fontsize=12)

	if save:
		plt.savefig(save, format='pdf')
	else:
		plt.show()


def corr_heatmap(X, ref, classes, save=None, **kw):
	"""
	Plot cell-to-cell correlation matrix heatmap
	"""
	index = np.argsort(ref)
	X = X.iloc[:,index]
	ref = ref[index]
	corr = X.corr()

	import matplotlib.patches as mpatches  # add legend
	colormap = plt.cm.tab20

	n_centroids = max(ref)+1
	colors = [colormap(i) for i in range(n_centroids)]
	row_colors = col_colors = [ colors[i] for i in ref ]
	bbox_to_anchor = (0.4, 1.2)
	legend_TN = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, classes)]

	cbar_kws={"orientation": "horizontal", "ticks":[0, 0.5, 1]}
	grid = sns.clustermap(corr, cmap='RdBu_r', 
#                           row_colors=row_colors, 
						  col_colors=col_colors, 
						  row_cluster=False,
						  col_cluster=False,
						  cbar_kws=cbar_kws, 
						  **kw
						 )
	grid.ax_heatmap.set_xticklabels('')
	grid.ax_heatmap.set_yticklabels('')
	grid.ax_heatmap.tick_params(axis='x', length=0)
	grid.ax_heatmap.tick_params(axis='y', length=0)
	grid.ax_heatmap.legend(loc='upper center', 
						   bbox_to_anchor=bbox_to_anchor, 
						   handles=legend_TN, 
						   fontsize=6, 
						   frameon=False, 
						   ncol=3)

	grid.cax.set_position((0.8, 0.76, .1, .02)) 
	# grid.cax.set_position((0.95, 0.3, .02, .1)) 
	grid.cax.tick_params(length=1, labelsize=4, rotation=0)
	grid.cax.set_title('Pearson', fontsize=6, y=0.8)

	if save:
		plt.savefig(save, format='pdf')
	else:
		plt.show()
		
		
def feature_specifity(feature, ref, classes):
	"""
	Calculate the feature specifity:

	Input:
		feature: latent feature
		ref: cluster assignments
		classes: cluster classes
	"""
	from scipy.stats import f_oneway
	n_cluster = max(ref) + 1
	dim = feature.shape[1] # feature dimension
	pvalue_mat = np.zeros((dim, n_cluster))
	for cluster in range(n_cluster):
		for feat in range(dim):
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
	grid.set_yticklabels(labels=np.arange(dim)+1, fontsize=16)
#     grid.set_title(dataset, fontsize=18)
	cbar = grid.collections[0].colorbar
	cbar.set_label('-log10 (Pvalue)', fontsize=18) #, rotation=0, x=-0.9, y=0)


