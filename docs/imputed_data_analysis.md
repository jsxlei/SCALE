# Imputed data analysis

    from scale.plot import plot_embedding, corr_heatmap
    from scale.utils import read_labels
    
    ref, classes = read_labels('../data/labels.txt') # ground truth, or predicted cluster assignments
    imputed_data = pd.read_csv('../output/imputed_data.txt', sep='\t', index_col=0) # read imputed data


## Cell-to-cell correlations
Cell-to-cell correlations matrix heatmap

    corr_heatmap(imputed_data, ref, classes, figsize=(5,5))
![corr_heatmap](png/corr_heatmap.png)
