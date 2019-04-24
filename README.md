# SCALE
Single-Cell ATAC-seq analysis via Latent feature Extraction

### Installation  

SCALE neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running SCALE on CUDA is recommended if available.   
Make sure Pytorch is working correctly with CUDA by:  

	import torch
	print(torch.cuda.is_available())
	
	True
	
Currently, SCALE requires Python 3 and does not work with Python 2.7

#### Installation from GitHub

To clone the repository and install manually, run the following from a terminal:

    git clone git://github.com/jsxlei/SCALE.git
    cd SCALE
    pip install -r requirements.txt
    python setup.py install --user
    
Installation only requieres a few minutes.

### Usage

#### Quick Start

Run SCALE with default parameters by providing a count matrix file and predefined cluster number:  

	SCALE.py -d [count_matrix] -k [n_cluster]

We also provide other options such as:  
* check clustering results with ground truth labels: [-r] or [--ref]
* save results in a specific folder: [-o] or [--outdir] 
* modify the learning rate, default is 0.0002: [--lr]  
* change the batch size, default is 32: [--batch_size]
* add more epochs if the loss not converge, default is 300: [-e] or [--epochs]  

Run with custom options on example data located in the SCALE repository:  

	SCALE.py -d data/data.txt -k 6 -e 1000 --lr 0.002
	
#### Note 
We recommend:   
For dataset with 1,000+ cells, set learning rate = 0.0002 with option (--lr 0.0002).  
For dataset with 10,000+ cells, increase batch size to reduce running time with option (--batch_size 128 or bigger).    
	
#### Results
Results will be saved in the output folder including:
* model.pt
* feature.txt
* cluster_assignments.txt
* imputed_data.txt

#### Help
Look for more usage of SCALE

	SCALE.py --help 

Use functions in SCALE packages.

	import scale
	from scale import *
	from scale.plot import *
	from scale.utils import *
	
#### Running time
Apply SCALE with example data:    
Running on a normal desktop computer needs half an hour or more.  
Running on GPU only needs a few minutes.  
Increase batch size appropriately can also reduce the running time.   

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
