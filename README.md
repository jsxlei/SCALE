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

	SCALE -d data/data.txt -k 6

Use functions in SCALE.

    import scale
	from scale import *
    from scale.plot import *
	from scale.utils import *
    

#### Tutorials
https://github.com/jsxlei/SCALE/tree/master/notebooks/Demo.ipynb
