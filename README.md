# Multi-scale_t-SNE
Multi-scale t-SNE, a perplexity-free version of t-SNE.

----------

## IMPORTANT NOTE

At the end of this file, a demo presents how this python code can be used. Running this file (python mstSNE.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes.

---------

The mstSNE.py python code implements multi-scale t-SNE, which is a neighbor embedding algorithm for nonlinear dimensionality reduction (DR). It is a perplexity-free version of t-SNE. 


This method is presented in the articles: 

- [Fast Multiscale Neighbor Embedding](https://ieeexplore.ieee.org/document/9308987), from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in IEEE Transactions on Neural Networks and Learning Systems, in 2020. 

- [Perplexity-free t-SNE and twice Student tt-SNE](https://github.com/cdebodt/Multi-scale_t-SNE) (CdB-et-al_MstSNE-ttSNE_ESANN-2018.pdf file), from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in the proceedings of the ESANN 2018 conference. 

Quality assessment criteria for both supervised and unsupervised dimensionality reduction are also implemented in this file. 

At the end of this file, a demo presents how this python code can be used. Running this file (python mstSNE.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. The tested versions of the imported packages are specified at the end of the header. 


If you use this code or one of the articles, please cite as: 

- C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.

- BibTeX entry:
```
@article{CdB2020FMsNE,
 author={C. {de Bodt} and D. {Mulders} and M. {Verleysen} and J. A. {Lee}}, 
 journal={{IEEE} Trans. Neural Netw. Learn. Syst.}, 
 title={{F}ast {M}ultiscale {N}eighbor {E}mbedding}, 
 year={2020}, 
 volume={}, 
 number={}, 
 pages={1-15}, 
 doi={10.1109/TNNLS.2020.3042807}}
 ```

and/or as:

- de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In ESANN (pp. 123-128).

- BibTeX entry:
```
@inproceedings{CdB2018mstsne,
 title={Perplexity-free {t-SNE} and twice {Student} {tt-SNE}}, 
 author={de Bodt, C. and Mulders, D. and Verleysen, M. and Lee, J. A.}, 
 booktitle={ESANN}, 
 pages={123--128}, 
 year={2018}}
```

## Running 
The main functions of this file are:

- 'mstsne': nonlinear dimensionality reduction through multi-scale t-SNE (Ms t-SNE), as presented in the references [1, 7] below. This function enables reducing the dimension of a data set. 

- 'eval_dr_quality': unsupervised evaluation of the quality of a low-dimensional embedding, as introduced in [3, 4] and employed and summarized in [1, 2, 5, 7]. This function enables computing quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them.

- 'knngain': supervised evaluation of the quality of a low-dimensional embedding, as presented in [6]. This function enables computing criteria related to the accuracy of a KNN classifier in the low-dimensional space. The documentation of the function explains the meaning of the criteria and how to interpret them.

- 'viz_2d_emb' and 'viz_qa': visualization of a 2-D embedding and of the quality criteria. These functions respectively enable to: 

---> 'viz_2d_emb': plot a 2-D embedding. 

---> 'viz_qa': depict the quality criteria computed by 'eval_dr_quality' and 'knngain'.

The documentations of the functions describe their parameters. The demo shows how they can be used. 


## Notations
- DR: dimensionality reduction.

- HD: high-dimensional.

- LD: low-dimensional.

- HDS: HD space.

- LDS: LD space.

- SNE: stochastic neighbor embedding.

- t-SNE: Student t-distributed SNE.

- Ms SNE: multi-scale SNE.

- Ms t-SNE: multi-scale t-SNE.

## Related codes
Note that further implementations are also available [here](https://github.com/cdebodt/Fast_Multi-scale_NE). They provide python codes for: 

- multi-scale SNE, which has a O(N**2 log(N)) time complexity, where N is the number of data points;

- multi-scale t-SNE, which has a O(N**2 log(N)) time complexity;

- a fast acceleration of multi-scale SNE, which has a O(N (log(N))**2) time complexity;

- a fast acceleration of multi-scale t-SNE, which has a O(N (log(N))**2) time complexity;

- DR quality criteria quantifying the neighborhood preservation from the HDS to the LDS. 


In comparison, [the present python code](https://github.com/cdebodt/Multi-scale_t-SNE) implements:

- DR quality criteria as described above in the main functions of this file;

- multi-scale t-SNE, with a O(N**2 log(N)) time complexity, in the 'mstsne' function. As described in its documentation, the 'mstsne' function can be employed using any HD distances. On the other hand, the implementations of multi-scale SNE, multi-scale t-SNE, fast multi-scale SNE and fast multi-scale t-SNE provided [here](https://github.com/cdebodt/Fast_Multi-scale_NE) only deal with Euclidean distances in both the HDS and the LDS. 


Also, the [other implementations](https://github.com/cdebodt/Fast_Multi-scale_NE) rely on the python programming language, but involve some C and Cython codes for performance purposes. As [further detailed](https://github.com/cdebodt/Fast_Multi-scale_NE), a C compiler is hence required. On the other hand, the present python code in this file is based on numpy, numba and scipy; no prior compilation is hence needed. 


Note that the implementations available [here](https://github.com/cdebodt/DR-with-Missing-Data) provide python codes for the multi-scale SNE algorithm, which has a O(N**2 log(N)) time complexity. These codes are analogous to the present code in this file for the multi-scale t-SNE algorithm: any HD distances can be employed, and no prior compilation is needed as the code is based on numpy, numba and scipy. On the other hand, the [python implementation of multi-scale SNE](https://github.com/cdebodt/Fast_Multi-scale_NE) only deals with Euclidean distances in both the HDS and the LDS, and requires a C compiler as it involves some C and Cython components. 


Also, the python code available [here](https://github.com/cdebodt/cat-SNE) implements cat-SNE, a supervised version of t-SNE. [As detailed](https://github.com/cdebodt/cat-SNE), any HD distances can be employed and no prior compilation is needed as the code is based on numpy, numba and scipy. 


## References

[1] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In ESANN (pp. 123-128).

[2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.

[3] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.

[4] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.

[5] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.

[6] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).

[7] C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.

## Contact - Misc
author: Cyril de Bodt (Human Dynamics - MIT Media Lab, and ICTEAM - UCLouvain)

@email: cdebodt __at__ mit __dot__ edu, or cyril __dot__ debodt __at__ uclouvain.be

Last modification date: Jan 27th, 2021

Copyright (c) 2021 Université catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.


This code was tested with Python 3.8.5 (Anaconda distribution, Continuum Analytics, Inc.). It uses the following modules:

- numpy: version 1.19.1 tested

- numba: version 0.50.1 tested

- scipy: version 1.5.0 tested

- matplotlib: version 3.3.1 tested

- scikit-learn: version 0.23.1 tested


You can use, modify and redistribute this software freely, but not for commercial purposes. 

The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

