# SCALE
Single-Cell ATAC-seq analysis via Latent feature Extraction

### Installation  

SCALE neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running SCALE on CUDA is recommended if available.   
Make sure Pytorch is working correctly with CUDA by:  

	import torch
	print(torch.cuda.is_available())
	
	True
	
Currently SCALE currently requires Python 3 and does not work with Python 2.7

#### Installation from GitHub

To clone the repository and install manually, run the following from a terminal:

    git clone git://github.com/jsxlei/SCALE.git
    cd SCALE
    pip install -r requirements.txt
    python setup.py install --user

### Usage

#### Quick Start

The following code runs SCALE on test data located in the SCALE repository.

	SCALE.py -d data/data.txt -k 6

or check clustering results with ground truth labels and save results in a specific folder

	SCALE.py -d data/data.txt -k 6 -e 1000 -r data/labels.txt -o output/
	
Results will be saved in the output folder including:
* model.pt
* feature.txt
* cluster_assignments.txt
* imputed_data.txt

	
Look for more usage of SCALE

	SCALE.py --help 

Use functions in SCALE packages.

	import scale
	from scale import *
	from scale.plot import *
	from scale.utils import *
    

#### Tutorials
A demo on SCALE usage for single-cell ATAC-seq data can be found in this notebook: 
https://github.com/jsxlei/SCALE/tree/master/notebooks/tutorial.ipynb


### Documentation

* [Preprocessing](docs/preprocessing.md)
* [Model introduction](docs/model_introduction.md)
* [Inputs and Outputs](docs/inputs_and_outputs.md)
* [Feature embedding](docs/feature_embedding.md)
* [Cluster_assignments](docs/cluster_assignments.md)
* [Denoising and Imputation](docs/denoising_and_imputation.md)
* [Cell type specific elements](docs/cell_type_specific_elements.md)
