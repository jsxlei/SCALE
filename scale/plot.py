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

def sort_by_classes(X, y, classes):
    if classes is None:
        classes = np.unique(y)
    index = []
    for c in classes:
        ind = np.where(y==c)[0]
        index.append(ind)
    index = np.concatenate(index)
    X = X.iloc[:, index]
    y = y[index]
    return X, y, classes, index


def plot_confusion_matrix(cm, x_classes=None, y_classes=None,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues,
                          figsize=(5,5),
                          mark=True,
                          save=None,
                          rotation=45,
                          show_cbar=True,
                          show_xticks=True,
                          show_yticks=True,
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
    
    ax=plt.gca()
    if not show_xticks:
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticklabels([])
    if not show_yticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticklabels([])
    else:
        plt.ylabel('Predicted Cluster')


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if mark:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j] > 0:
                plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if show_cbar:
        plt.colorbar(shrink=0.8) 
    if save:
        plt.savefig(save, format='pdf')
    else:
        plt.show()


def plot_heatmap(X, y, classes=None, y_pred=None, row_labels=None, colormap=None, 
                 cax_title='', xlabel='', ylabel='', yticklabels='', legend_font=10, 
                 show_legend=True, show_cax=True, tick_color='black', ncol=3,
                 bbox_to_anchor=(0.5, 1.3), position=(0.8, 0.78, .1, .04), return_grid=False,
                 save=None, **kw):
    """
    plot hidden code heatmap with labels

    Params:
        X: fxn array, n is sample number, f is feature
        y: a array of labels for n elements or a list of array
    """

    import matplotlib.patches as mpatches  # add legend
    # if classes is not None:
    X, y, classes, index = sort_by_classes(X, y, classes)
    # else:
        # classes = np.unique(y)

    if y_pred is not None:
        y_pred = y_pred[index]
        classes = list(classes) + list(np.unique(y_pred)) 
        if colormap is None:
            colormap = plt.cm.tab20
            colors = {c:colormap(i) for i, c in enumerate(classes)}
        else:
            colors = {c:colormap[i] for i, c in enumerate(classes)}
        col_colors = []
        col_colors.append([colors[c] for c in y])
        col_colors.append([colors[c] for c in y_pred])
    else:
        if colormap is None:
            colormap = plt.cm.tab20
            colors = {c:colormap(i) for i, c in enumerate(classes)}
        else:
            colors = {c:colormap[i] for i, c in enumerate(classes)}
        col_colors = [ colors[c] for c in y ]
        
    legend_TN = [mpatches.Patch(color=color, label=c) for c, color in colors.items()]

    if row_labels is not None:
        row_colors = [ colors[c] for c in row_labels ]
        kw.update({'row_colors':row_colors})

    kw.update({'col_colors':col_colors})

    cbar_kws={"orientation": "horizontal"}
    grid = sns.clustermap(X, yticklabels=True, 
            col_cluster=False,
            cbar_kws=cbar_kws, **kw)
    if show_cax:
        grid.cax.set_position(position)
        grid.cax.tick_params(length=1, labelsize=4, rotation=0)
        grid.cax.set_title(cax_title, fontsize=6, y=0.35)

    if show_legend:
        grid.ax_heatmap.legend(loc='upper center', 
                               bbox_to_anchor=bbox_to_anchor, 
                               handles=legend_TN, 
                               fontsize=legend_font, 
                               frameon=False, 
                               ncol=ncol)
        grid.ax_col_colors.tick_params(labelsize=6, length=0, labelcolor='orange')
    if 'row_cluster' in kw:
        if (kw['row_cluster']==True) and (yticklabels is not ''):
            yticklabels = yticklabels[grid.dendrogram_row.reordered_ind]

    grid.ax_heatmap.set_xlabel(xlabel)
    grid.ax_heatmap.set_ylabel(ylabel, fontsize=8)
    grid.ax_heatmap.set_xticklabels('')
    grid.ax_heatmap.set_yticklabels(yticklabels, color=tick_color)
    grid.ax_heatmap.yaxis.set_label_position('left')
    grid.ax_heatmap.tick_params(axis='x', length=0)
    grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0, rotation=0, labelleft=True, labelright=False)
    grid.ax_row_dendrogram.set_visible(False)
    grid.cax.set_visible(show_cax)
    grid.row_color_labels = classes

    if save:
        plt.savefig(save, format='pdf')
    else:
        plt.show()
    if return_grid:
        return grid


def plot_embedding(X, labels, classes=None, method='TSNE', cmap=None, 
                   return_emb=False, save=False, show_legend=True, **legend_params):
    if X.shape[1] != 2:
        if method == 'TSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=30, min_dist=0.3, metric='correlation').fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
        
    plt.figure(figsize=(8, 8))
    if classes is None:
        classes = np.unique(labels)

    if len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))
        
    for i, c in enumerate(classes):
        plt.scatter(X[labels==c, 0], X[labels==c, 1], color=colors[i], \
                                    label=c, s=10, edgecolors='none', alpha=0.8)
    plt.axis("off")
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 10,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(legend_params_)
    if show_legend:
        plt.legend(**legend_params)
    if save:
        if os.path.isfile(save):
            print('remove previous figure {}'.format(save))
            os.remove(save)
        plt.savefig(save, format='pdf', bbox_inches='tight')
    else:
        plt.show()
        
    if return_emb:
        return X


def corr_heatmap(X, y=None, classes=None, 
        cmap='RdBu_r',
        show_legend=True, 
        show_cbar=True, 
        figsize=(5,5), 
        ncol=3, 
        distance='pearson',
        ticks=None, 
        save=None,
        **kw):
    """
    Plot cell-to-cell correlation matrix heatmap
    """
    import matplotlib.patches as mpatches  # add legend
    colormap = plt.cm.tab20

    if y is not None:
        if classes is None:
            classes = np.unique(y)
        X, y, classes, index = sort_by_classes(X, y, classes)

        colors = {c:colormap(i) for i,c in enumerate(classes)}
        col_colors = [ colors[c] for c in y ]
        bbox_to_anchor = (0.4, 1.2)
        legend_TN = [mpatches.Patch(color=color, label=c) for c,color in colors.items()]
    else:
        col_colors = None
    # else:
    # index = np.argsort(ref)
    # X = X.iloc[:,index]
    # ref = ref[index]
    corr = X.corr(method=distance)



    cbar_kws={"orientation": "horizontal", "ticks":ticks}
    grid = sns.clustermap(corr, cmap=cmap, 
                          col_colors=col_colors, 
                          figsize=figsize,
                          row_cluster=False,
                          col_cluster=False,
                          cbar_kws=cbar_kws, 
                          **kw
                         )
    grid.ax_heatmap.set_xticklabels('')
    grid.ax_heatmap.set_yticklabels('')
    grid.ax_heatmap.tick_params(axis='x', length=0)
    grid.ax_heatmap.tick_params(axis='y', length=0)

    if show_legend and (y is not None):
        grid.ax_heatmap.legend(loc='upper center', 
                           bbox_to_anchor=bbox_to_anchor, 
                           handles=legend_TN, 
                           fontsize=6, 
                           frameon=False, 
                           ncol=ncol)
    if show_cbar:
        grid.cax.set_position((0.8, 0.76, .1, .02)) 
        grid.cax.tick_params(length=1, labelsize=4, rotation=0)
        grid.cax.set_title(distance, fontsize=6, y=0.8)
    else:
        grid.cax.set_visible(False)

    if save:
        plt.savefig(save, format='pdf')
    else:
        plt.show()


def feature_specifity(feature, ref, classes, figsize=(6,6)):
    """
    Calculate the feature specifity:

    Input:
        feature: latent feature
        ref: cluster assignments
        classes: cluster classes
    """
    from scipy.stats import f_oneway
    # n_cluster = max(ref) + 1
    n_cluster = len(classes)
    dim = feature.shape[1] # feature dimension
    pvalue_mat = np.zeros((dim, n_cluster))
    for i,cluster in enumerate(classes):
        for feat in range(dim):
            a = feature.iloc[:, feat][ref == cluster]
            b = feature.iloc[:, feat][ref != cluster]
            pvalue = f_oneway(a,b)[1]
            pvalue_mat[feat, i] = pvalue

    plt.figure(figsize=figsize)
    grid = sns.heatmap(-np.log10(pvalue_mat), cmap='RdBu_r', 
                       vmax=20,
                       yticklabels=np.arange(10)+1, 
                       xticklabels=classes[:n_cluster],
                       )
    grid.set_ylabel('Feature', fontsize=18)
    grid.set_xticklabels(labels=classes[:n_cluster], rotation=45, fontsize=18)
    grid.set_yticklabels(labels=np.arange(dim)+1, fontsize=16)
    cbar = grid.collections[0].colorbar
    cbar.set_label('-log10 (Pvalue)', fontsize=18) #, rotation=0, x=-0.9, y=0)



