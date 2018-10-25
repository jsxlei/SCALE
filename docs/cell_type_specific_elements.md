# Cell type specific peaks



    from scale.plot import plot_heatmap
    from scale.utils import read_labels
    
    feature = pd.read_csv('../output/feature.txt', sep='\t', index_col=0, header=None)
    ref, classes = read_labels('../data/labels.txt') # or predicted cluster assignments 
    
## feature heatmap
 
    plot_heatmap(feature.T, ref, classes, 
                 figsize=(8, 3), cmap='RdBu_r', #vmax=8, vmin=-8,
                 ylabel='Feature components', yticklabels=np.arange(10)+1, 
                 cax_title='Feature value',
                 row_cluster=False, legend_font=6, 
                 col_cluster=False, center=0)
                 
## feature specifity

    feature_specifity(feature, ref, classes)
    
## cell type specific peaks
    
    for i in range(feature.shape[1])[-2:]: # show the represented peaks of last two components of feature
        peak_file = specific_peak_dir+'peak_index{}.txt'.format(i)
        peak_index = open(peak_file).read().split()
        peak_data = impute_data.loc[peak_index]
        plot_heatmap(peak_data, ref, classes,
                     cmap='Reds', 
                     figsize=(8,3), 
                     cax_title='Read counts', 
                     ylabel='{} peaks of feature {}'.format(len(peak_index), i+1),
                     vmax=1, vmin=0, legend_font=8,
                     col_cluster=False, row_cluster=False,
                     show_legend=True,
                     show_cax = True,
                     bbox_to_anchor=(0.4, 1.32),
                    )
                    
                 

                 
                 
