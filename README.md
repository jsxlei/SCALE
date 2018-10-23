# SCALE
Single-Cell ATAC-seq Analysis via Latent feature Extraction

### Installation

#### Installation from GitHub

To clone the repository and install manually, run the following from a terminal:

    git clone git://github.com/jsxlei/SCALE.git
    cd SCALE
	pip install -r requirements.txt
    python setup.py install --user

### Usage

#### Quick Start

The following code runs SCALE on test data located in the SCALE repository.

	SCALE -d data/data.txt -k 6
	
Look for more usage of SCALE

	SCALE --help 

Use functions in SCALE.

    import scale
	from scale import *
    from scale.plot import *
	from scale.utils import *
    

#### Tutorials
A demo on SCALE usage for single-cell ATAC-seq data can be found in this notebook: 
https://github.com/jsxlei/SCALE/tree/master/notebooks
