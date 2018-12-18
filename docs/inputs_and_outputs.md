# Inputs and Outputs

## Inputs
test inputs are in the [data/](../data) folder including: data.txt, labels.txt and peaks.txt  

* **peak count matrix**: (required) for training model, peaks(rows) x cells(columns). e.g. [data/data.txt](../data/data.txt)
* **ground truth labels**: (optional) for comparing predicted clustering assignments. e.g. [data/labels.txt](../data/labels.txt)

## Outputs
outputs are saved in default folder output/ or user-specified folder including:

* **model.pt**: trained SCALE model
* **feature.txt**: latent feature of input data
* **cluster_assignments.txt**: predicted cluster assignments by k-means on latent feature 
* **imputed_data.txt**: imputed data reconstructed from latent feature via decoder of SCALE
