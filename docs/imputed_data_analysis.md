# Imputed data analysis

    from scale.plot import plot_embedding, corr_heatmap
    import seaborn as sns
    
    y = pd.read_csv('../data/labels.txt', index_col=0, header=None, sep='\t')[0] # ground truth
    imputed_data = pd.read_csv('../output/imputed_data.txt', sep='\t', index_col=0) # read imputed data
    raw_data = pd.read_csv('../data/data.txt', sep='\t', index_col=0) # read raw data
    classes = np.unique(y)

## Correlations of cells of imputed data with meta-cells of raw data

    cell_corr = []
    for c in classes:
        cells = np.where(y==c)[0]
        cell_corr.append(pairwise_pearson(imputed_data.iloc[:, cells].T, raw_data.iloc[:, cells].sum(1)))
    
    g = sns.boxplot(data=cell_corr, width=0.5)
    sns.despine(offset=10, trim=True)
    g.set_xticklabels(classes, rotation=45)
    
![corr_heatmap](png/correlation_with_metacell.png)
