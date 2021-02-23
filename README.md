# Single-Cell ATAC-seq analysis via Latent feature Extraction
![](https://github.com/jsxlei/SCALE/wiki/png/model.png)

## News 
2021.01.14 Update to compatible with [h5ad](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) file and [scanpy](https://scanpy.readthedocs.io/en/stable/index.html)

## Installation  

SCALE neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running SCALE on CUDA is recommended if available.   
	
#### install from GitHub

	git clone git://github.com/jsxlei/SCALE.git
	cd SCALE
	python setup.py install
    
Installation only requires a few minutes.  

## Quick Start

#### Input
* h5ad file
* **count matrix file**:  
	* row is peak and column is barcode, in **txt** / **tsv** (sep=**"\t"**) or **csv** (sep=**","**) format
* mtx **folder** contains **three files**:   
	* **count file**: count in **mtx** format, filename contains key word **"count"** / **"matrix"**    
	* **peak file**: 1-column of peaks **chr_start_end**, filename contains key word **"peak"**  
	* **barcode file**: 1-column of barcodes, filename contains key word **"barcode"**

#### Run 

    SCALE.py -d [input]
    
if cluster number k is known:

    SCALE.py -d [input] -k [k]

#### Output
Output will be saved in the output folder including:
* **model.pt**:  saved model to reproduce results cooperated with option --pretrain
* **adata.h5ad**:  saved data including Leiden cluster assignment, latent feature matrix and UMAP results.
* **umap.pdf**:  visualization of 2d UMAP embeddings of each cell

#### Imputation  
Get binary imputed data in folder **binary_imputed** with option **--binary** (recommended for saving storage)

    SCALE.py -d [input] --binary  
    
or get numerical imputed data in file **imputed_data.txt** with option **--impute**

    SCALE.py -d [input] --impute
     
#### Useful options  
* save results in a specific folder: [-o] or [--outdir] 
* embed feature by tSNE or UMAP: [--embed]  tSNE/UMAP
* filter low quality cells by valid peaks number, default 100: [--min_peaks] 
* filter low quality peaks by valid cells number, default 10: [--min_cells]
* modify the initial learning rate, default is 0.002: [--lr]  
* change iterations by watching the convergence of loss, default is 30000: [-i] or [--max_iter]  
* change random seed for parameter initialization, default is 18: [--seed]
* binarize the imputation values: [--binary]
	

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


#### Data availability  
* [Forebrain](http://zhanglab.net/SCALE_SOURCE_DATA/Forebrain.h5ad)
* [Splenocyte](http://zhanglab.net/SCALE_SOURCE_DATA/Splenocyte.h5ad)
* [mouse_atlas](http://zhanglab.net/SCALE_SOURCE_DATA/mouse_atlas.h5ad)
* [InSilico](http://zhanglab.net/SCALE_SOURCE_DATA/InSilico.h5ad)
* [Leukemia](http://zhanglab.net/SCALE_SOURCE_DATA/Leukemia.h5ad)
* [GM12878vsHEK](http://zhanglab.net/SCALE_SOURCE_DATA/GM12878vsHEK.h5ad)
* [GM12878vsHL](http://zhanglab.net/SCALE_SOURCE_DATA/GM12878vsHL.h5ad)
* [Breast_Tumor](http://zhanglab.net/SCALE_SOURCE_DATA/Breast_Tumor.h5ad)


## Reference
[Lei Xiong, Kui Xu, Kang Tian, Yanqiu Shao, Lei Tang, Ge Gao, Michael Zhang, Tao Jiang & Qiangfeng Cliff Zhang. SCALE method for single-cell ATAC-seq analysis via latent feature extraction. Nature Communications, (2019).](https://www.nature.com/articles/s41467-019-12630-7)
