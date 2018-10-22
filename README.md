# SCALE
Single-Cell ATAC-seq Analysis via Latent feature Extraction

### Installation

#### Installation from GitHub

To clone the repository and install manually, run the following from a terminal:

    git clone git://github.com/jsxlei/SCALE.git
    cd SCALE
    python setup.py install --user

### Usage

#### Quick Start

The following code runs SCALE on test data located in the SCALE repository.

SCALE -d data/test_data.txt -k 6

    import magic
    import pandas as pd
    import matplotlib.pyplot as plt
    X = pd.read_csv("MAGIC/data/test_data.csv")
    magic_operator = magic.MAGIC()
    X_magic = magic_operator.fit_transform(X, genes=['VIM', 'CDH1', 'ZEB1'])
    plt.scatter(X_magic['VIM'], X_magic['CDH1'], c=X_magic['ZEB1'], s=1, cmap='inferno')
    plt.show()
    magic.plot.animate_magic(X, gene_x='VIM', gene_y='CDH1', gene_color='ZEB1', operator=magic_operator)

#### Tutorials

