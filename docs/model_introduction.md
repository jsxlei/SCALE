# Model introduction

![model_introduction](png/model_introduction.png)

## The evidence lower bound of SCALE
SCALE combines variational autoencoder (VAE) and Gaussian Mixture Model (GMM) to model the distribution of high dimensional sparse scATAC-seq data. 

joint probaility p(x,z,c) can be factorized into given x and c are independent conditioned on z:
> <a href="https://www.codecogs.com/eqnedit.php?latex=p(x,z,c)&space;=&space;p(x|z)p(z|c)p(c)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x,z,c)&space;=&space;p(x|z)p(z|c)p(c)" title="p(x,z,c) = p(x|z)p(z|c)p(c)" /></a>  

while:  
> <a href="https://www.codecogs.com/eqnedit.php?latex=p(c)&space;=&space;Discrete(c|\pi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c)&space;=&space;Discrete(c|\pi)" title="p(c) = Discrete(c|\pi)" /></a>  
> <a href="https://www.codecogs.com/eqnedit.php?latex=p(z|c)&space;=&space;N(z|\mu&space;_{c},&space;\sigma_{c}^{2}I)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(z|c)&space;=&space;N(z|\mu&space;_{c},&space;\sigma_{c}^{2}I)" title="p(z|c) = N(z|\mu _{c}, \sigma_{c}^{2}I)" /></a>  
> <a href="https://www.codecogs.com/eqnedit.php?latex=p(x|z)&space;=&space;N(x|\mu&space;_{x},&space;\sigma_{x}^{2}I)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?
> https://latex.codecogs.com/gif.latex?p(x|z)&space;=&space;Ber(x|\mu&space;_{x})
                                                                                                                             
The log-likelihood of data:  
> <a href="https://www.codecogs.com/eqnedit.php?latex=log&space;p(x)&space;=log\int_{z}^{&space;}&space;\sum_{c}^{&space;}&space;p(x,z,c)dz" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log&space;p(x)&space;=log\int_{z}^{&space;}&space;\sum_{c}^{&space;}&space;p(x,z,c)dz" title="log p(x) =log\int_{z}^{ } \sum_{c}^{ } p(x,z,c)dz" /></a>  

can be transformed into maximizing evidence lower bound (ELBO):  
> <a href="https://www.codecogs.com/eqnedit.php?latex=\geq&space;E_{q(z,c|x)}[log\frac{p(x,z,c)}{p(z,c|x)}]&space;=&space;L_{ELBO}(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\geq&space;E_{q(z,c|x)}[log\frac{p(x,z,c)}{p(z,c|x)}]&space;=&space;L_{ELBO}(x)" title="\geq E_{q(z,c|x)}[log\frac{p(x,z,c)}{p(z,c|x)}] = L_{ELBO}(x)" /></a>  

which can be writen into a reconstruction term and regularization term:  
> <a href="https://www.codecogs.com/eqnedit.php?latex=L_{ELBO}(x)&space;=&space;E_{q(z,c|x)}[logp(x\z)]&space;-&space;D_{KL}(q(z,c|x)||p(z,c))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{ELBO}(x)&space;=&space;E_{q(z,c|x)}[logp(x\z)]&space;-&space;D_{KL}(q(z,c|x)||p(z,c))" title="L_{ELBO}(x) = E_{q(z,c|x)}[logp(x\z)] - D_{KL}(q(z,c|x)||p(z,c))" /></a>  

The **reconstruction term** encourages imputed data similar to input data.  
The **regularization term** is Kullback-Leibeler divergence which regularizes the latent variable z to a GMM manifold.

## Model structure
SCALE is consisted by encoder and decoder.     
> **encoder**: a four-layer neural network (3200-1600-800-400) with ReLU activation function.    
> **decoder**: has no hidden layers, latent variable (feature) connected with output layer (peaks) with Sigmoid.   
**Initialize**: A GMM model is applied to initialize GMM parameters μ_c and σ_c of SCALE.   
**Optimizer**: Adam   
**weight decay**: 5e-4   
**Learning rate**: initializes at 0.002 with decaying 10% every 10 epochs until 0.0002.   
**Batch size**: 16    
**Epoch number**: 1000 for leukemia mixture and 300 for other data.  
