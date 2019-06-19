# Single-Cell ATAC-seq analysis via Latent feature Extraction
![](https://github.com/jsxlei/SCALE/wiki/png/model.png)

## Installation  

SCALE neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running SCALE on CUDA is recommended if available.   
	
#### install from GitHub

	git clone git://github.com/jsxlei/SCALE.git
	cd SCALE
	python setup.py install
    
Installation only requieres a few minutes.  
If you have any problem with installing pytorch, use

	conda install pytorch torchvision -c pytorch

or refer to [this](https://pytorch.org/get-started/locally/) for more detail

## Quick Start

Input dir of scATAC-seq data should contain one of: 
* **dense format**:    
	* **data.txt** / **data.txt.gz**: count matrix of tab separated format
* **sparse format**:   
	* **data.mtx** / **data.mtx.gz**: count matrix of mtx format   
	* **peaks.txt**: at least 3-column bed format without header e.g. chr/tstart/tend     
	* **cell_id.txt**: 1-column of cell ids without header

#### Run SCALE with known cluster number k:  

    SCALE.py -d [input_dir] -k [k]

#### Or Run SCALE with k estimated by SCALE if k is unknown: 

    SCALE.py -d [input_dir]

#### Data availability  
Download all the **provided datasets** [[Download]](https://cloud.tsinghua.edu.cn/u/d/a776d93940dc43c5aad6/)  

#### Useful options  
* save results in a specific folder: [-o] or [--outdir] 
* filter rare peaks if the peaks quality if not good or too many: [-x]
* modify the initial learning rate, default is 0.002: [--lr]  
* change the batch size, default is 32: [--batch_size]
* change iterations by watching the convergence of loss, default is 30000: [-i] or [--max_iter]  
* change random seed for parameter initialization, default is 18: [--seed]
* binarize the imputation values: [--binary]
	
#### Note    
If come across the nan loss, 
* try another random seed
* filter peaks with harsher threshold like -x 4 or -x 6
* change the initial learning rate to 0.0002 
	
#### Results
Results will be saved in the output folder including:
* model.pt
* feature.txt
* cluster_assignments.txt
* imputed_data.txt
* tsne.txt
* tsne.pdf

#### Help
Look for more usage of SCALE

	SCALE.py --help 

Use functions in SCALE packages.

	import scale
	from scale import *
	from scale.plot import *
	from scale.utils import *
	
#### Running time
<p float="left">
  <img src="https://github.com/jsxlei/SCALE/wiki/png/runtime.png" width="350" />
  <img src="https://github.com/jsxlei/SCALE/wiki/png/memory.png" width="350" /> 
</p>

## Tutorial


**[Tutorial Forebrain](https://github.com/jsxlei/SCALE/wiki/Forebrain)**   Run SCALE on dense matrix **Forebrain** dataset (k=8, 2088 cells)
	
**[Tutorial Mouse Atlas](https://github.com/jsxlei/SCALE/wiki/Mouse-Atlas)**   Run SCALE on sparse matrix **Mouse Atlas** dataset (k=30, ~80,000 cells)
