# Cell type specific elements


    from scale.plot import plot_heatmap
    from scale.utils import read_labels
    
    feature = pd.read_csv('../output/feature.txt', sep='\t', index_col=0, header=None)
    ref, classes, le = read_labels('../data/labels.txt', return_enc=True) # or predicted cluster assignments
    imputed_data = pd.read_csv('../output/imputed_data.txt', sep='\t', index_col=0)
    y = le.inverse_transform(ref)
    
    
## cell type specific peaks
feature has 10 components(dimensions), each components has its most correlation peaks

    score_mat = mat_specificity_score(imputed_data, pred)
    peak_index, peak_labels = cluster_specific(score_mat, classes=classes_, top=200);print(len(peak_index))
    row_labels = ['cluster'+str(c+1) for c in peak_labels]

save top specific peaks

    f = open(out_dir+'specific_peaks.txt', 'w')
    for peak in imputed_data.index[peak_index].values:
        f.write(peak.replace('_', '\t')+'\n')
        
plot specific peaks heatmap

    plot_heatmap(imputed_data.iloc[peak_index], y=y, classes=classes, y_pred=y_pred, row_labels=row_labels, 
                 ncol=3,cmap='Reds', vmax=1, row_cluster=False, legend_font=6, cax_title='Peak Value',
                 figsize=(6, 8), bbox_to_anchor=(0.4, 1.2), position=(0.8, 0.76, 0.1, 0.015))
    
## cell type enriched motifs
Apply [chromVAR](https://github.com/GreenleafLab/chromVAR) on specific peaks  
We offer an Rscript ["chromVAR"](../scripts/chromVAR) in the scripts folder which can be run directly by:

    e.g. chromVAR -i input_dir -o output_dir
    ! chromVAR -i ../output/
Input dir is output dir of SCALE including imputed_data.txt and specific_peaks.txt
    
    
Plot deviations heatmap of chromVAR

    dev = pd.read_csv(out_dir + 'dev.txt'.format(dataset), index_col=0, sep='\t').fillna(0)
    var = pd.read_csv(out_dir + 'var.txt'.format(dataset), index_col=0, sep='\t')

    figsize, N, bbox_to_anchor, position = (6, 8), 50, (0.4, 1.2), (0.8, 0.78, .1, .01)
    index = var.sort_values(by='variability', ascending=False).index[:N]
    # index = var.p_value_adj[var.p_value_adj < 0.05].index
    yticklabels = var.loc[index].name.values
    plot_heatmap(dev.loc[index], y=y, classes=classes,
                 row_cluster=True,  
                 yticklabels=yticklabels,
                 vmax=3, vmin=-3, 
                 figsize=figsize, 
                 legend_font=6,
                 bbox_to_anchor=bbox_to_anchor,
                 position=position,
                 cax_title='TF deviation',
                 cmap='RdBu_r')
