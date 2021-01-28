#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

#
# %%%% !!! IMPORTANT NOTE !!! %%%%
# At the end of this file, a demo presents how this python code can be used. Running this file (python mstSNE.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes.
# %%%% !!!                !!! %%%%

#     mstSNE.py

# This python code implements multi-scale t-SNE, which is a neighbor embedding algorithm for nonlinear dimensionality reduction (DR). It is a perplexity-free version of t-SNE. 

# This method is presented in the articles: 
# - "Fast Multiscale Neighbor Embedding", from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in IEEE Transactions on Neural Networks and Learning Systems, in 2020. 
# ----> Link to retrieve the article: https://ieeexplore.ieee.org/document/9308987
# - "Perplexity-free t-SNE and twice Student tt-SNE", from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in the proceedings of the ESANN 2018 conference. 
# ----> The PDF of this paper is freely available in the GitHub repository of this code (at https://github.com/cdebodt/Multi-scale_t-SNE), with 'CdB-et-al_MstSNE-ttSNE_ESANN-2018.pdf' as file name. 

# Quality assessment criteria for both supervised and unsupervised dimensionality reduction are also implemented in this file. 
# At the end of this file, a demo presents how this python code can be used. Running this file (python mstSNE.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. The tested versions of the imported packages are specified at the end of the header. 

# If you use this code or one of the articles, please cite as: 
# - C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.
# - BibTeX entry:
# @article{CdB2020FMsNE,
#  author={C. {de Bodt} and D. {Mulders} and M. {Verleysen} and J. A. {Lee}},
#  journal={{IEEE} Trans. Neural Netw. Learn. Syst.},
#  title={{F}ast {M}ultiscale {N}eighbor {E}mbedding}, 
#  year={2020},
#  volume={},
#  number={},
#  pages={1-15},
#  doi={10.1109/TNNLS.2020.3042807}}
# 
# and/or as:
# - de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In ESANN (pp. 123-128).
# - BibTeX entry:
#@inproceedings{CdB2018mstsne,
#  title={Perplexity-free {t-SNE} and twice {Student} {tt-SNE}},
#  author={de Bodt, C. and Mulders, D. and Verleysen, M. and Lee, J. A.},
#  booktitle={ESANN},
#  pages={123--128},
#  year={2018}
#}

# The main functions of this file are:
# - 'mstsne': nonlinear dimensionality reduction through multi-scale t-SNE (Ms t-SNE), as presented in the references [1, 7] below. This function enables reducing the dimension of a data set. 
# - 'eval_dr_quality': unsupervised evaluation of the quality of a low-dimensional embedding, as introduced in [3, 4] and employed and summarized in [1, 2, 5, 7]. This function enables computing quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them.
# - 'knngain': supervised evaluation of the quality of a low-dimensional embedding, as presented in [6]. This function enables computing criteria related to the accuracy of a KNN classifier in the low-dimensional space. The documentation of the function explains the meaning of the criteria and how to interpret them.
# - 'viz_2d_emb' and 'viz_qa': visualization of a 2-D embedding and of the quality criteria. These functions respectively enable to: 
# ---> 'viz_2d_emb': plot a 2-D embedding. 
# ---> 'viz_qa': depict the quality criteria computed by 'eval_dr_quality' and 'knngain'.
# The documentations of the functions describe their parameters. The demo shows how they can be used. 

# Notations:
# - DR: dimensionality reduction.
# - HD: high-dimensional.
# - LD: low-dimensional.
# - HDS: HD space.
# - LDS: LD space.
# - SNE: stochastic neighbor embedding.
# - t-SNE: Student t-distributed SNE.
# - Ms SNE: multi-scale SNE.
# - Ms t-SNE: multi-scale t-SNE.

# Note that further implementations are also available at "https://github.com/cdebodt/Fast_Multi-scale_NE". They provide python codes for: 
# - multi-scale SNE, which has a O(N**2 log(N)) time complexity, where N is the number of data points;
# - multi-scale t-SNE, which has a O(N**2 log(N)) time complexity;
# - a fast acceleration of multi-scale SNE, which has a O(N (log(N))**2) time complexity;
# - a fast acceleration of multi-scale t-SNE, which has a O(N (log(N))**2) time complexity;
# - DR quality criteria quantifying the neighborhood preservation from the HDS to the LDS. 

# In comparison, the present python code in this file (available at https://github.com/cdebodt/Multi-scale_t-SNE) implements:
# - DR quality criteria as described above in the main functions of this file;
# - multi-scale t-SNE, with a O(N**2 log(N)) time complexity, in the 'mstsne' function. As described in its documentation, the 'mstsne' function can be employed using any HD distances. On the other hand, the implementations of multi-scale SNE, multi-scale t-SNE, fast multi-scale SNE and fast multi-scale t-SNE provided at "https://github.com/cdebodt/Fast_Multi-scale_NE" only deal with Euclidean distances in both the HDS and the LDS. 

# Also, the implementations provided at "https://github.com/cdebodt/Fast_Multi-scale_NE" rely on the python programming language, but involve some C and Cython codes for performance purposes. As further detailed at "https://github.com/cdebodt/Fast_Multi-scale_NE", a C compiler is hence required. On the other hand, the present python code in this file is based on numpy, numba and scipy; no prior compilation is hence needed. 

# Note that the implementations available at "https://github.com/cdebodt/DR-with-Missing-Data" provide python codes for the multi-scale SNE algorithm, which has a O(N**2 log(N)) time complexity. These codes are analogous to the present code in this file for the multi-scale t-SNE algorithm: any HD distances can be employed, and no prior compilation is needed as the code is based on numpy, numba and scipy. On the other hand, the python implementation of multi-scale SNE provided at "https://github.com/cdebodt/Fast_Multi-scale_NE" only deals with Euclidean distances in both the HDS and the LDS, and requires a C compiler as it involves some C and Cython components. 

# Also, the python code available at "https://github.com/cdebodt/cat-SNE" implements cat-SNE, a supervised version of t-SNE. As detailed at "https://github.com/cdebodt/cat-SNE", any HD distances can be employed and no prior compilation is needed as the code is based on numpy, numba and scipy. 

# References:
# [1] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In ESANN (pp. 123-128).
# [2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# [3] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
# [4] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
# [5] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
# [6] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
# [7] C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.

# author: Cyril de Bodt (Human Dynamics - MIT Media Lab, and ICTEAM - UCLouvain)
# @email: cdebodt __at__ mit __dot__ edu, or cyril __dot__ debodt __at__ uclouvain.be
# Last modification date: Jan 27th, 2021
# Copyright (c) 2021 Université catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

# This code was tested with Python 3.8.5 (Anaconda distribution, Continuum Analytics, Inc.). It uses the following modules:
# - numpy: version 1.19.1 tested
# - numba: version 0.50.1 tested
# - scipy: version 1.5.0 tested
# - matplotlib: version 3.3.1 tested
# - scikit-learn: version 0.23.1 tested

# You can use, modify and redistribute this software freely, but not for commercial purposes. 
# The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

########################################################################################################
########################################################################################################

import numpy as np, numba, sklearn.decomposition, scipy.spatial.distance, matplotlib.pyplot as plt, scipy.optimize, time, os, sklearn.datasets, sklearn.manifold

# Name of this file
module_name = "mstSNE.py"

##############################
############################## 
# General functions used by others in the code. 
####################

@numba.jit(nopython=True)
def close_to_zero(v):
    """
    Check whether v is close to zero or not.
    In:
    - v: a scalar or numpy array.
    Out:
    A boolean or numpy array of boolean of the same shape as v, with True when the entry is close to 0 and False otherwise.
    """
    return np.absolute(v) <= 10.0**(-8.0)

@numba.jit(nopython=True)
def arange_except_i(N, i):
    """
    Create a 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    In:
    - N: a strictly positive integer.
    - i: a positive integer which is strictly smaller than N.
    Out:
    A 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    """
    arr = np.arange(N)
    return np.hstack((arr[:i], arr[i+1:]))

@numba.jit(nopython=True)
def fill_diago(M, v):
    """
    Replace the elements on the diagonal of a square matrix M with some value v.
    In:
    - M: a 2-D numpy array storing a square matrix.
    - v: some value.
    Out:
    M, but in which the diagonal elements have been replaced with v.
    """
    for i in range(M.shape[0]):
        M[i,i] = v
    return M

@numba.jit(nopython=True)
def contains_ident_ex(X):
    """
    Returns True if the data set contains two identical samples, False otherwise.
    In:
    - X: a 2-D numpy array with one example per row and one feature per column.
    Out:
    A boolean being True if and only if X contains two identical rows.
    """
    # Number of samples and of features
    N, M = X.shape
    # Tolerance
    atol = 10.0**(-8.0)
    # For each sample
    for i in range(N):
        if np.any(np.absolute(np.dot((np.absolute(X[i,:]-X[i+1:,:]) > atol).astype(np.float64), np.ones(shape=M, dtype=np.float64)))<=atol):
            return True
    return False

def eucl_dist_matr(X):
    """
    Compute the pairwise Euclidean distances in a data set. 
    In:
    - X: a 2-D np.ndarray with shape (N,M) containing one example per row and one feature per column.
    Out:
    A 2-D np.ndarray dm with shape (N,N) containing the pairwise Euclidean distances between the data points in X, such that dm[i,j] stores the Euclidean distance between X[i,:] and X[j,:].
    """
    return scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X, metric='euclidean'), force='tomatrix')

##############################
############################## 
# Nonlinear dimensionality reduction through perplexity-free multi-scale t-SNE (Ms t-SNE) [1, 7]. 
# The main function is 'mstsne'. 
# See its documentation for details. 
# The demo at the end of this file presents how to use it. 
####################

# Default random seed for Ms t-SNE. Only used if seed_mstsne is set to None in mstsne.
seed_MstSNE_def = 40
# Maximum number of iterations in L-BFGS. 
dr_nitmax = 30
# The iterations of L-BFGS stop as soon as max{|g_i | i = 1, ..., n} <= dr_gtol, where g_i is the i-th component of the gradient. 
dr_gtol = 10**(-5)
# The iterations of L-BFGS stop when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= dr_ftol, where f^k is the value of the cost function at iteration k.
dr_ftol = 2.2204460492503131e-09
# Maximum number of line search steps per L-BFGS-B iteration.
dr_maxls = 30
# The maximum number of variable metric corrections used to define the limited memory matrix of L-BFGS.
dr_maxcor = 6

n_eps_np_float64 = np.finfo(dtype=np.float64).eps

@numba.jit(nopython=True)
def ms_perplexities(N, K_star=2, L_min=-1, L_max=-1):
    """
    Define exponentially growing multi-scale perplexities, as in [1, 2, 7].
    In:
    - N: number of data points.
    - K_star: K_{*} as defined in [2], to set the multi-scale perplexities.
    - L_min: if -1, set as in [2]. 
    - L_max: if -1, set as in [2]. 
    Out:
    A tuple with:
    - L: as defined in [2]
    - K_h: 1-D numpy array, with the perplexities in increasing order.
    """
    if L_min == -1:
        L_min = 1
    if L_max == -1:
        L_max = int(round(np.log2(np.float64(N)/np.float64(K_star))))
    L = L_max-L_min+1
    K_h = (np.float64(2.0)**(np.linspace(L_min-1, L_max-1, L).astype(np.float64)))*np.float64(K_star)
    return L, K_h

def init_lds(X_hds, N, init='pca', n_components=2, rand_state=None, var=1.0):
    """
    Initialize the LD embedding.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column, or None. If X_hds is set to None, init cannot be equal to 'pca', otherwise an error is raised. 
    - N: number of examples in the data set. If X_hds is not None, N must be equal to X_hds.shape[0]. 
    - init: determines the initialization of the LD embedding. 
    ---> If isinstance(init, str) is True:
    ------> If init is equal to 'pca', the LD embedding is initialized with the first n_components principal components of X_hds. X_hds cannot be None in this case, otherwise an error is raised. 
    ------> If init is equal to 'random', the LD embedding is initialized randomly, using a uniform Gaussian distribution with a variance equal to var. X_hds may be set to None in this case.
    ------> Otherwise an error is raised.
    ---> If isinstance(init, np.ndarray) is True:
    ------> init must in this case be a 2-D numpy array, with N rows and n_components columns. It stores the LD positions to use for the initialization, with one example per row and one LD dimension per column. init[i,:] contains the initial LD coordinates for the HD sample X_hds[i,:]. X_hds may be set to None in this case. If init.ndim != 2 or init.shape[0] != N or init.shape[1] != n_components, an error is raised.
    ---> Otherwise, an error is raised.
    - n_components: number of dimensions in the LD space.
    - rand_state: random state to use. Such a random state can be created using the function 'np.random.RandomState'. If it is None, it is set to np.random. 
    - var: variance employed when init is equal to 'random'. 
    Out:
    A numpy ndarray with shape (N, n_components), containing the initialization of the LD data set, with one example per row and one LD dimension per column.
    """
    global module_name
    if rand_state is None:
        rand_state = np.random
    if isinstance(init, str):
        if init == "pca":
            if X_hds is None:
                raise ValueError("Error in function init_lds of module {module_name}: init cannot be set to 'pca' if X_hds is None.".format(module_name=module_name))
            return sklearn.decomposition.PCA(n_components=n_components, whiten=False, copy=True, svd_solver='auto', iterated_power='auto', tol=0.0, random_state=rand_state).fit_transform(X_hds)
        elif init == 'random':
            return var * rand_state.randn(N, n_components)
        else:
            raise ValueError("Error in function init_lds of module {module_name}: unknown value '{init}' for init parameter.".format(module_name=module_name, init=init))
    elif isinstance(init, np.ndarray):
        if init.ndim != 2:
            raise ValueError("Error in function init_lds of module {module_name}: init must be 2-D.".format(module_name=module_name))
        if init.shape[0] != N:
            raise ValueError("Error in function init_lds of module {module_name}: init must have {N} rows, but init.shape[0] = {v}.".format(module_name=module_name, N=N, v=init.shape[0]))
        if init.shape[1] != n_components:
            raise ValueError("Error in function init_lds of module {module_name}: init must have {n_components} columns, but init.shape[1] = {v}.".format(module_name=module_name, n_components=n_components, v=init.shape[1]))
        return init
    else:
        raise ValueError("Error in function init_lds of module {module_name}: unknown type value '{v}' for init parameter.".format(module_name=module_name, v=type(init)))

@numba.jit(nopython=True)
def sne_sim(dsi, vi, i, compute_log=True):
    """
    Compute the SNE asymmetric similarities, as well as their log.
    N refers to the number of data points. 
    In:
    - dsi: numpy 1-D array of floats with N squared distances with respect to data point i. Element k is the squared distance between data points k and i.
    - vi: bandwidth of the exponentials in the similarities with respect to i.
    - i: index of the data point with respect to which the similarities are computed, between 0 and N-1.
    - compute_log: boolean. If True, the logarithms of the similarities are also computed, and otherwise not.
    Out:
    A tuple with two elements:
    - A 1-D numpy array of floats with N elements. Element k is the SNE similarity between data points i and k.
    - If compute_log is True, a 1-D numpy array of floats with N element. Element k is the log of the SNE similarity between data points i and k. By convention, element i is set to 0. If compute_log is False, it is set to np.empty(shape=N, dtype=np.float64).
    """
    N = dsi.size
    si = np.empty(shape=N, dtype=np.float64)
    si[i] = 0.0
    log_si = np.empty(shape=N, dtype=np.float64)
    indj = arange_except_i(N=N, i=i)
    dsij = dsi[indj]
    log_num_sij = (dsij.min()-dsij)/vi
    si[indj] = np.exp(log_num_sij)
    den_si = si.sum()
    si /= den_si
    if compute_log:
        log_si[i] = 0.0
        log_si[indj] = log_num_sij - np.log(den_si)
    return si, log_si

@numba.jit(nopython=True)
def sne_bsf(dsi, vi, i, log_perp):
    """
    Function on which a binary search is performed to find the HD bandwidth of the i^th data point in SNE.
    In: 
    - dsi, vi, i: same as in sne_sim function.
    - log_perp: logarithm of the targeted perplexity.
    Out:
    A float corresponding to the current value of the entropy of the similarities with respect to i, minus log_perp.
    """
    si, log_si = sne_sim(dsi=dsi, vi=vi, i=i, compute_log=True)
    return -np.dot(si, log_si) - log_perp

@numba.jit(nopython=True)
def sne_bs(dsi, i, log_perp, x0=1.0):
    """
    Binary search to find the root of sne_bsf over vi. 
    In:
    - dsi, i, log_perp: same as in sne_bsf function.
    - x0: starting point for the binary search. Must be strictly positive.
    Out:
    A strictly positive float vi such that sne_bsf(dsi, vi, i, log_perp) is close to zero. 
    """
    fx0 = sne_bsf(dsi=dsi, vi=x0, i=i, log_perp=log_perp)
    if close_to_zero(v=fx0):
        return x0
    elif not np.isfinite(fx0):
        raise ValueError("Error in function sne_bs: fx0 is nan.")
    elif fx0 > 0:
        x_up, x_low = x0, x0/2.0
        fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_low):
            return x_low
        elif not np.isfinite(fx_low):
            # WARNING: cannot find a valid root!
            return x_up
        while fx_low > 0:
            x_up, x_low = x_low, x_low/2.0
            fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_low):
                return x_low
            if not np.isfinite(fx_low):
                return x_up
    else: 
        x_up, x_low = x0*2.0, x0
        fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_up):
            return x_up
        elif not np.isfinite(fx_up):
            return x_low
        while fx_up < 0:
            x_up, x_low = 2.0*x_up, x_up
            fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_up):
                return x_up
    while True:
        x = (x_up+x_low)/2.0
        fx = sne_bsf(dsi=dsi, vi=x, i=i, log_perp=log_perp)
        if close_to_zero(v=fx):
            return x
        elif fx > 0:
            x_up = x
        else:
            x_low = x

@numba.jit(nopython=True)
def sne_hd_similarities(dsm_hds, perp, compute_log=True, start_bs=np.ones(shape=1, dtype=np.float64)):
    """
    Computes the matrix of SNE asymmetric HD similarities, as well as their log.
    In:
    - dsm_hds: 2-D numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between i and j.
    - perp: perplexity. Must be > 1.
    - compute_log: boolean. If true, the logarithms of the similarities are also computed. Otherwise not.
    - start_bs: 1-D numpy array with N elements. Element at index i is the starting point of the binary search for the ith data point. If start_bs has only one element, it will be set to np.ones(shape=N, dtype=np.float64).
    Out:
    A tuple with three elements:
    - A 2-D numpy array with shape (N, N) and in which element [i,j] = the HD similarity between i and j. The similarity between i and i is set to 0.
    - If compute_log is True, 2-D numpy array with shape (N, N) and in which element [i,j] = the log of the HD similarity between i and j. By convention, log(0) is set to 0. If compute_log is False, it is set to np.empty(shape=(N,N), dtype=np.float64).
    - A 1-D numpy array with N elements, where element i is the denominator of the exponentials of the HD similarities with respect to data point i. 
    """
    if perp <= 1:
        raise ValueError("""Error in function sne_hd_similarities of module mstSNE.py: the perplexity should be >1.""")
    N = dsm_hds.shape[0]
    if start_bs.size == 1:
        start_bs = np.ones(shape=N, dtype=np.float64)
    log_perp = np.log(min(np.float64(perp), np.floor(0.99*np.float64(N))))
    # Computing the N**2 HD similarities, for i, j = 0, ..., N-1.
    si = np.empty(shape=(N,N), dtype=np.float64)
    # Even when compute_log is False, we cannot set log_si to None. We need to define it as an array, to be compatible with numba.
    log_si = np.empty(shape=(N,N), dtype=np.float64)
    arr_vi = np.empty(shape=N, dtype=np.float64)
    for i in range(N):
        # Computing the denominator of the exponentials of the HD similarities with respect to data point i. 
        vi = sne_bs(dsi=dsm_hds[i,:], i=i, log_perp=log_perp, x0=start_bs[i])
        # Computing the HD similarities between i and j for j=0, ..., N-1.
        tmp = sne_sim(dsi=dsm_hds[i,:], vi=vi, i=i, compute_log=compute_log)
        si[i,:] = tmp[0]
        if compute_log:
            log_si[i,:] = tmp[1]
        arr_vi[i] = vi
    return si, log_si, arr_vi

@numba.jit(nopython=True)
def ms_hd_similarities(dsm_hds, arr_perp):
    """
    Compute the matrix of multi-scale HD similarities sigma_{ij}, as defined in [1, 2, 7].
    In:
    - dsm_hds: 2-D numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between i and j.
    - arr_perp: numpy 1-D array containing the perplexities for all scales. All the perplexities must be > 1.
    Out:
    A tuple with:
    - A 2-D numpy array with shape (N, N) and in which element [i,j] = the multi-scale HD similarity sigma_{ij}.
    - A 2-D numpy array with shape (arr_perp.size, N) and in which element [h,i] = tau_{hi} = 2/pi_{hi}, following the notations of [2].
    - sim_hij: 3-D numpy array with shape (arr_perp.size, N, N) where sim_hij[h,:,:] contains the HD similarities at scale arr_perp[h].
    """
    # Number of data points
    N = dsm_hds.shape[0]
    # Number of perplexities
    L = arr_perp.size
    # Matrix storing the multi-scale HD similarities sigma_{ij}. Element [i,j] contains sigma_{ij}. sigma_{ii} is set to 0.
    sigma_ij = np.zeros(shape=(N,N), dtype=np.float64)
    # Matrix storing the HD similarities sigma_{hij} at each scale.
    sim_hij = np.empty(shape=(L,N,N), dtype=np.float64)
    # Matrix storing the HD tau_{hi}. Element [h,i] contains tau_{hi}.
    tau_hi = np.empty(shape=(L,N), dtype=np.float64)
    # For each perplexity
    for h, perp in enumerate(arr_perp):
        # Using the bandwidths found at the previous scale to initialize the binary search at the current scale.
        if h > 0:
            start_bs = tau_hi[h-1,:]
        else:
            start_bs = np.ones(shape=N, dtype=np.float64)
        # Computing the N**2 HD similarities sigma_{hij}
        sim_hij[h,:,:], dum, tau_hi[h,:] = sne_hd_similarities(dsm_hds=dsm_hds, perp=perp, compute_log=False, start_bs=start_bs)
        # Updating the multi-scale HD similarities
        sigma_ij += sim_hij[h,:,:]
    # Scaling the multi-scale HD similarities
    sigma_ij /= np.float64(L)
    # Returning
    return sigma_ij, tau_hi, sim_hij

@numba.jit(nopython=True)
def mstsne_ld_sim(dsm_ld):
    """
    Computes the LD similarities t_{i,j} for i, j = 0, ..., N-1, with N being the number of data points, using a Student-t distribution with one degree of freedom, as well as their log.
    In:
    - dsm_ld: 2-D numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared LD distance between the i^{th} and j^{th} data points.
    Out:
    A tuple with three elements:
    - A 2-D numpy array with shape (N, N) and in which element [i,j] = t_{i,j}.
    - A 2-D numpy array with shape (N, N) and in which element [i,j] = log(t_{i,j}). By convention, log(t_{i,i}) is set to 0.
    - 1/(1+dsm_ld)
    """
    global n_eps_np_float64
    dsm_ld_one = 1.0+dsm_ld
    inv_dsm_ld_one = 1.0/np.maximum(n_eps_np_float64, dsm_ld_one)
    t_ij = inv_dsm_ld_one.copy()
    log_t_ij = -np.log(dsm_ld_one)
    # Diagonal indexes
    t_ij = fill_diago(M=t_ij, v=0.0)
    log_t_ij = fill_diago(M=log_t_ij, v=0.0)
    # Denominator of the t_{ij}'s
    den_t_ij = t_ij.sum()
    # Computing the t_ij's
    t_ij /= np.maximum(n_eps_np_float64, den_t_ij)
    # Computing the log(t_ij)'s
    log_t_ij -= np.log(den_t_ij)
    return t_ij, log_t_ij, inv_dsm_ld_one

def mstsne_obj(x, tau_ij, N, n_components, arr_one, prod_N_nc):
    """
    Computes the value of the part of the objective function of multi-scale t-SNE which depends on the LD coordinates. 
    In:
    - x: numpy 1-D array with N*n_components elements, containing the current values of the LD coordinates. np.reshape(a=x, newshape=(N, n_components)) should yield a 2-D array with one example per row and one LD dimension per column.
    - tau_ij: numpy 2-D array with shape (N,N). Element [i,j] contains the multi-scale HD similarity between the i^{th} and j^{th} data points, as defined in [1, 7]. Diagonal elements must be equal to 0.
    - N: number of data points.
    - n_components: dimension of the LD embedding.
    - arr_one: must be equal to np.ones(shape=N, dtype=np.float64)
    - prod_N_nc: product of N and n_components
    Out:
    A scalar equal to the part of the KL divergence that depends on the LD coordinates.
    Remark:
    - In order to use the scipy optimization functions, the functions mstsne_obj and mstsne_grad must have the same arguments.
    """
    X_lds = np.reshape(a=x, newshape=(N, n_components))
    # Computing the log of the LD similarities. To compute the LD distances, we use the "sqeuclidean" metric instead of the "euclidean" one to avoid squaring the distances.
    log_t_ij = mstsne_ld_sim(dsm_ld=scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_lds, metric='sqeuclidean'), force='tomatrix'))[1]
    # Returning
    return -np.dot(tau_ij.ravel(), log_t_ij.ravel())

def mstsne_grad(x, tau_ij, N, n_components, arr_one, prod_N_nc):
    """
    Computes the value of the gradient of the objective function of multi-scale t-SNE at some LD coordinates.
    In:
    - x: numpy 1-D array with N*n_components elements, containing the current values of the LD coordinates. np.reshape(a=x, newshape=(N, n_components)) should yield a 2-D array with one example per row and one LD dimension per column.
    - tau_ij: numpy 2-D array with shape (N,N). Element [i,j] contains the multi-scale HD similarity between the i^{th} and j^{th} data points, as defined in [1, 7]. Diagonal elements must be equal to 0.
    - N: number of data points.
    - n_components: dimension of the LD embedding.
    - arr_one: must be equal to np.ones(shape=N, dtype=np.float64)
    - prod_N_nc: product of N and n_components
    Out:
    A 1-D numpy array with N*n_components elements, where element i is the coordinate of the gradient associated to x[i].
    Remark:
    - In order to use the scipy optimization functions, the functions mstsne_obj and mstsne_grad should have the same arguments.
    """
    X_lds = np.reshape(a=x, newshape=(N, n_components))
    # Computing the LD similarities. To compute the LD distances, we use the "sqeuclidean" metric instead of the "euclidean" one to avoid squaring the distances.
    t_ij, log_t_ij, inv_dsm_ld_one = mstsne_ld_sim(dsm_ld=scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_lds, metric='sqeuclidean'), force='tomatrix'))
    # Computing the multiplication factor in the gradient
    c_ij = 4.0*(tau_ij-t_ij)*inv_dsm_ld_one
    # Computing the gradient 
    grad_ld = (X_lds.T*np.dot(c_ij, arr_one)).T - np.dot(a=c_ij, b=X_lds)
    # Returning the reshaped gradient
    return np.reshape(a=grad_ld, newshape=prod_N_nc)

def mstsne_sim_hd(X_hds, K_h, dsm_hds=None):
    """
    Return the HD similarities at the different scales.
    In:
    - X_hds: 2-D numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column. If None, dsm_hds must be specified and different from None, otherwise an error is raised. X_hds is only used if dsm_hds is set to None.
    - K_h: 1-D numpy array with the perplexities at each scale in increasing order.
    - dsm_hds: (optional) 2-D numpy.ndarray with shape (N,N) containing the pairwise SQUARED HD distances between the data points in X_hds. dsm_hds[i,j] stores the SQUARED HD distance between X_hds[i,:] and X_hds[j,:]. If dsm_hds is specified and not None, X_hds is not used and can be set to None. If dsm_hds is None, it is deduced from X_hds using squared Euclidean distances. Hence, if dsm_hds is None, X_hds cannot be None, otherwise an error is raised.
    Out:
    A 3-D numpy array sim_hij with shape (K_h.size, N, N) where sim_hij[h,:,:] contains the HD similarities at scale K_h[h].
    """
    global module_name
    # Computing the squared HD distances. 
    if dsm_hds is None:
        if X_hds is None:
            raise ValueError("Error in function mstsne_sim_hd of module {module_name}: if dsm_hds is None, X_hds cannot be None.".format(module_name=module_name))
        dsm_hds = scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_hds, metric='sqeuclidean'), force='tomatrix')
    # Returning the HD similarities at the scales indicated by K_h
    return ms_hd_similarities(dsm_hds=dsm_hds, arr_perp=K_h)[2]

def mstsne_manage_seed(seed_mstsne=None):
    """
    Manage the random seed in mstsne.
    In:
    - seed_mstsne: an integer or None. If it is None, it is set to seed_MstSNE_def. If it is not an integer, an error is raised. 
    Out:
    If seed_mstsne > 0, np.random.RandomState(seed_mstsne) is returned. Otherwise, np.random is returned. 
    """
    global seed_MstSNE_def, module_name
    if seed_mstsne is None:
        seed_mstsne = seed_MstSNE_def
    if seed_mstsne != int(round(seed_mstsne)):
        raise ValueError("Error in function mstsne_manage_seed of module {module_name}: seed_mstsne must be an integer.".format(module_name=module_name))
    if seed_mstsne > 0:
        return np.random.RandomState(seed_mstsne)
    else:
        return np.random

def mstsne(X_hds, init='pca', n_components=2, dm_hds=None, seed_mstsne=None):
    """
    This function applies perplexity-free multi-scale t-SNE to reduce the dimension of a data set [1, 7].
    In:
    - X_hds: 2-D numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column, or None. If X_hds is not None, it is assumed that it does not contain duplicated examples. X_hds can only be None if dm_hds is not None and init is not set to 'pca', otherwise an error is raised. 
    - init: determines the initialization of the LD embedding. 
    ---> If isinstance(init, str) is True:
    ------> If init is equal to 'pca', the LD embedding is initialized with the first n_components principal components of X_hds. X_hds cannot be None in this case, even if dm_hds is specified, otherwise an error is raised. 
    ------> If init is equal to 'random', the LD embedding is initialized randomly, using a uniform Gaussian distribution with small variance. X_hds may be set to None in this case if dm_hds is specified.
    ------> Otherwise an error is raised.
    ---> If isinstance(init, np.ndarray) is True:
    ------> init must in this case be a 2-D numpy array, with N rows and n_components columns. It stores the LD positions to use for the initialization, with one example per row and one LD dimension per column. init[i,:] contains the initial LD coordinates for the HD sample X_hds[i,:]. X_hds may be set to None in this case if dm_hds is specified. If init.ndim != 2 or init.shape[0] != N or init.shape[1] != n_components, an error is raised.
    ---> Otherwise, an error is raised.
    - n_components: dimension of the LDS.
    - dm_hds: (optional) 2-D numpy array with the pairwise HD distances (NOT squared) between the data points. If dm_hds is None, it is deduced from X_hds using Euclidean distances. If dm_hds is not None, then the pairwise HD distances are not recomputed and X_hds may either be None or defined; if X_hds is not None, it will only be used if init is set to 'pca', to initialize the LD embedding to the first n_components principal components of X_hds. Hence, if both dm_hds and X_hds are not None and if init is set to 'pca', dm_hds and X_hds are assumed to be compatible, i.e. dm_hds[i,j] is assumed to store the HD distance between X_hds[i,:] and X_hds[j,:].
    - seed_mstsne: seed to use for the random state. Check mstsne_manage_seed for a description. 
    Out:
    A 2-D numpy.ndarray X_lds with shape (N, n_components), containing the LD data set, with one example per row and one dimension per column. X_lds[i,:] contains the LD coordinates of the HD sample X_hds[i,:]. 
    Remarks:
    - L-BFGS algorithm is used, as suggested in [2, 7].
    - Multi-scale optimization is performed, as presented in [2, 7].
    - Euclidean distances are employed to evaluate the pairwise HD similarities by default, as in [1, 2, 7]. Other distances can be used in the HD space by specifying the dm_hds parameter. Euclidean distances are employed in the LD embedding. 
    """
    global dr_nitmax, dr_gtol, dr_ftol, dr_maxls, dr_maxcor, module_name
    # Checking the value of dm_hds
    dm_hds_none = dm_hds is None
    if dm_hds_none:
        dsm_hds = None
        if X_hds is None:
            raise ValueError("Error in function mstsne of module {module_name}: X_hds and dm_hds cannot both be None.".format(module_name=module_name))
    else:
        dsm_hds = dm_hds**2
        dsm_hds = dsm_hds.astype(np.float64)
    # Defining the random state
    rand_state = mstsne_manage_seed(seed_mstsne)
    # Number of data points
    if dm_hds_none:
        N = X_hds.shape[0]
    else:
        N = dsm_hds.shape[0]
    # Product of N and n_components
    prod_N_nc = N*n_components
    
    # Defining K_star for the multi-scale perplexities, following the notations of [2]. 
    K_star = 2
    # Computing the multi-scale perplexities, following the notations of [2]. 
    L, K_h = ms_perplexities(N=N, K_star=K_star)
    
    # Initializing the LD embedding.
    X_lds = init_lds(X_hds=X_hds, N=N, init=init, n_components=n_components, rand_state=rand_state)
    
    # Computing the HD similarities at the different scales. 
    sim_hij_allh = mstsne_sim_hd(X_hds=X_hds, K_h=K_h, dsm_hds=dsm_hds)
    
    # Reshaping X_lds as the optimization functions work with 1-D arrays.
    X_lds = np.reshape(a=X_lds, newshape=prod_N_nc)
    
    # Matrix storing the multi-scale HD similarities sigma_{ij}, as introduced in [1, 2, 7]. Element [i,j] contains sigma_{ij}. sigma_{ii} is set to 0. We need to recompute them progressively by adding perplexities one at the time as we perform multi-scale optimization.
    sigma_ij = np.zeros(shape=(N,N), dtype=np.float64)
    # Matrix storing the symmetrized multi-scale HD similarities. 
    tau_ij = np.empty(shape=(N,N), dtype=np.float64)
    
    # Array with N ones, used in the computation of the gradient of multi-scale t-SNE
    arr_one = np.ones(shape=N, dtype=np.float64)
    N_2 = np.float64(2*N)
    
    # Multi-scale optimization. n_perp is the number of currently considered perplexities.
    for n_perp in range(1, L+1, 1):
        # Index of the currently added perplexity.
        h = L-n_perp
        
        # Updating the multi-scale HD similarities
        sigma_ij = (sigma_ij*(np.float64(n_perp)-1.0) + sim_hij_allh[h,:,:])/np.float64(n_perp)
        
        # Symmetrizing the multi-scale HD similarities
        tau_ij = (sigma_ij + sigma_ij.T)/N_2
        
        # Defining the arguments of the L-BFGS algorithm
        args = (tau_ij, N, n_components, arr_one, prod_N_nc)
        
        # Running L-BFGS
        res = scipy.optimize.minimize(fun=mstsne_obj, x0=X_lds, args=args, method='L-BFGS-B', jac=mstsne_grad, bounds=None, callback=None, options={'disp':False, 'maxls':dr_maxls, 'gtol':dr_gtol, 'maxiter':dr_nitmax, 'maxcor':dr_maxcor, 'maxfun':np.inf, 'ftol':dr_ftol})
        X_lds = res.x
    
    # Reshaping the result
    X_lds = np.reshape(a=X_lds, newshape=(N, n_components))
    
    # Returning
    return X_lds

##############################
############################## 
# Unsupervised DR quality assessment: rank-based criteria measuring the HD neighborhood preservation in the LD embedding [3, 4]. 
# These criteria are used in the experiments reported in [1, 7]. 
# The main function is 'eval_dr_quality'. 
# See its documentation for details. It explains the meaning of the quality criteria and how to interpret them. 
# The demo at the end of this file presents how to use the 'eval_dr_quality' function. 
####################

def coranking(d_hd, d_ld):
    """
    Computation of the co-ranking matrix, as described in [4]. 
    The time complexity of this function is O(N**2 log(N)), where N is the number of data points.
    In:
    - d_hd: 2-D numpy array representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array representing the redundant matrix of pairwise distances in the LDS.
    Out:
    The (N-1)x(N-1) co-ranking matrix, where N = d_hd.shape[0].
    """
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS. 
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')
    
    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)
    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j,i],j] = i
    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [2].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [2], with a log scale for K=1 to arr.size. 
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.float64(np.dot(arr, i_all_k))/(i_all_k.sum())

@numba.jit(nopython=True)
def eval_rnx(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [5]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding. 
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def eval_dr_quality(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2, 3, 4, 5] and as employed in the experiments reported in [1, 7].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS. 
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed. 
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K, where N refers to the number of data points and \cap to the set intersection operator. 
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K. 
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1. 
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail. 
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random. 
    In: 
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [2].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2). 
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)

##############################
############################## 
# Supervised DR quality assessment: accuracy of a KNN classifier in the LD embedding [6]. 
# See the documentation of the 'knngain' function for details. It explains the meaning of the supervised quality criteria and how to interpret them. 
####################

@numba.jit(nopython=True)
def knngain(d_hd, d_ld, labels):
    """
    Compute the KNN gain curve and its AUC, as defined in [6]. 
    If c_i refers to the class label of data point i, v_i^K (resp. n_i^K) to the set of the K nearest neighbors of data point i in the HDS (resp. LDS), and N to the number of data points, the KNN gain develops as G_{NN}(K) = (1/N) * \sum_{i=1}^{N} (|{j \in n_i^K such that c_i=c_j}|-|{j \in v_i^K such that c_i=c_j}|)/K.
    It averages the gain (or loss, if negative) of neighbors of the same class around each point, after DR. 
    Hence, a positive value correlates with likely improved KNN classification performances.
    As the R_{NX}(K) curve from the unsupervised DR quality assessment, the KNN gain G_{NN}(K) can be displayed with respect to K, with a log scale for K. 
    A global score summarizing the resulting curve is provided by its area (AUC). 
    In: 
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    - labels: 1-D numpy array with N elements, containing integers indicating the class labels of the data points. 
    Out: 
    A tuple with:
    - a 1-D numpy array of floats with N-1 elements, storing the KNN gain for K=1 to N-1. 
    - the AUC of the KNN gain curve, with a log scale for K.
    """
    # Number of data points
    N = d_hd.shape[0]
    N_1 = N-1
    k_hd = np.zeros(shape=N_1, dtype=np.int64)
    k_ld = np.zeros(shape=N_1, dtype=np.int64)
    # For each data point
    for i in range(N):
        c_i = labels[i]
        di_hd = d_hd[i,:].argsort(kind='mergesort')
        di_ld = d_ld[i,:].argsort(kind='mergesort')
        # Making sure that i is first in di_hd and di_ld
        for arr in [di_hd, di_ld]:
            for idj, j in enumerate(arr):
                if j == i:
                    idi = idj
                    break
            if idi != 0:
                arr[idi] = arr[0]
            arr = arr[1:]
        for k in range(N_1):
            if c_i == labels[di_hd[k]]:
                k_hd[k] += 1
            if c_i == labels[di_ld[k]]:
                k_ld[k] += 1
    # Computing the KNN gain
    gn = (k_ld.cumsum() - k_hd.cumsum()).astype(np.float64)/((1.0+np.arange(N_1))*N)
    # Returning the KNN gain and its AUC
    return gn, eval_auc(gn)

##############################
############################## 
# Plot functions reproducing some of the figures in [1, 7]. 
# The main functions are 'viz_2d_emb' and 'viz_qa'.
# Their documentations detail their parameters. 
# The demo at the end of this file presents how to use these functions. 
####################

def rstr(v, d=2):
    """
    Rounds v with d digits and returns it as a string. If it starts with 0, it is omitted. 
    In:
    - v: a number. 
    - d: number of digits to keep.
    Out:
    A string representing v rounded with d digits. If it starts with 0, it is omitted. 
    """
    p = 10.0**d
    v = str(int(round(v*p))/p)
    if v[0] == '0':
        v = v[1:]
    elif (len(v) > 3) and (v[:3] == '-0.'):
        v = "-.{a}".format(a=v[3:])
    return v

def check_create_dir(path):
    """
    Create a directory at the specified path only if it does not already exist.
    """
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def save_show_fig(fname=None, f_format=None, dpi=300):
    """
    Save or show a figure.
    In:
    - fname: filename to save the figure, without the file extension. If None, the figure is shown.
    - f_format: format to save the figure. If None, set to pdf. 
    - dpi: DPI to save the figure.
    Out: 
    A figure is shown if fname is None, and saved otherwise.
    """
    if fname is None:
        plt.show()
    else:
        if f_format is None:
            f_format = 'pdf'
        # Checking whether a folder needs to be created
        check_create_dir(fname)
        # Saving the figure
        plt.savefig("{fname}.{f_format}".format(fname=fname, f_format=f_format), format=f_format, dpi=dpi, bbox_inches='tight', facecolor='w', edgecolor='w', orientation='portrait', papertype='a4', transparent=False, pad_inches=0.1, frameon=None)

def viz_2d_emb(X, vcol, tit='', fname=None, f_format=None, cmap='rainbow', sdot=20, marker='o', a_scat=0.8, edcol_scat='face', stit=15, lw=2.0):
    """
    Plot a 2-D embedding of a data set.
    In:
    - X: a 2-D numpy array with shape (N, 2), where N is the number of data points to represent in the 2-D embedding.
    - vcol: a 1-D numpy array with N elements, indicating the colors of the data points in the colormap.
    - tit: title of the figure.
    - fname, f_format: path. Same as in save_show_fig.
    - cmap: colormap.
    - sdot: size of the dots.
    - marker: marker.
    - a_scat: alpha used to plot the data points.
    - edcol_scat: edge color for the points of the scatter plot. From the official documentation: "If None, defaults to (patch.edgecolor). If 'face', the edge color will always be the same as the face color. If it is 'none', the patch boundary will not be drawn. For non-filled markers, the edgecolors kwarg is ignored; color is determined by c.".
    - stit: fontsize of the title of the figure.
    - lw: linewidth for the scatter plot.
    Out:
    Same as save_show_fig.
    """  
    global module_name
    
    # Checking X
    if X.ndim != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must be a numpy array with shape (N, 2), where N is the number of data points to plot in the 2-D embedding.".format(module_name=module_name))
    if X.shape[1] != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must have 2 columns.".format(module_name=module_name))
    
    # Computing the limits of the axes
    xmin = X[:,0].min()
    xmax = X[:,0].max()
    ev = (xmax-xmin)*0.05
    x_lim = np.asarray([xmin-ev, xmax+ev])
    
    ymin = X[:,1].min()
    ymax = X[:,1].max()
    ev = (ymax-ymin)*0.05
    y_lim = np.asarray([ymin-ev, ymax+ev])
    
    vmc = vcol.min()
    vMc = vcol.max()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Setting the limits of the axes
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    # Plotting the data points
    ax.scatter(X[:,0], X[:,1], c=vcol, cmap=cmap, s=sdot, marker=marker, alpha=a_scat, edgecolors=edcol_scat, vmin=vmc, vmax=vMc, linewidths=lw)
    
    # Removing the ticks
    ax.set_xticks([], minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([], minor=False)
    ax.set_yticks([], minor=False)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=False)
    
    ax.set_title(tit, fontsize=stit)
    plt.tight_layout()
    
    # Saving or showing the figure, and closing
    save_show_fig(fname=fname, f_format=f_format)
    plt.close()

def viz_qa(Ly, fname=None, f_format=None, ymin=None, ymax=None, Lmarkers=None, Lcols=None, Lleg=None, Lls=None, Lmedw=None, Lsdots=None, lw=2, markevery=0.1, tit='', xlabel='', ylabel='', alpha_plot=0.9, alpha_leg=0.8, stit=25, sax=20, sleg=15, zleg=1, loc_leg='best', ncol_leg=1, lMticks=10, lmticks=5, wMticks=2, wmticks=1, nyMticks=11, mymticks=4, grid=True, grid_ls='solid', grid_col='lightgrey', grid_alpha=0.7, xlog=True):
    """
    Plot the DR quality criteria curves. 
    In: 
    - Ly: list of 1-D numpy arrays. The i^th array gathers the y-axis values of a curve from x=1 to x=Ly[i].size, with steps of 1. 
    - fname, f_format: path. Same as in save_show_fig.
    - ymin, ymax: minimum and maximum values of the y-axis. If None, ymin (resp. ymax) is set to the smallest (resp. greatest) value among [y.min() for y in Ly] (resp. [y.max() for y in Ly]).
    - Lmarkers: list with the markers for each curve. If None, some pre-defined markers are used.
    - Lcols: list with the colors of the curves. If None, some pre-defined colors are used.
    - Lleg: list of strings, containing the legend entries for each curve. If None, no legend is shown.
    - Lls: list of the linestyles ('solid', 'dashed', ...) of the curves. If None, 'solid' style is employed for all curves. 
    - Lmedw: list with the markeredgewidths of the curves. If None, some pre-defined value is employed. 
    - Lsdots: list with the sizes of the markers. If None, some pre-defined value is employed.
    - lw: linewidth for all the curves. 
    - markevery: approximately 1/markevery markers are displayed for each curve. Set to None to mark every dot.
    - tit: title of the plot.
    - xlabel, ylabel: labels for the x- and y-axes.
    - alpha_plot: alpha for the curves.
    - alpha_leg: alpha for the legend.
    - stit: fontsize for the title.
    - sax: fontsize for the labels of the axes. 
    - sleg: fontsize for the legend.
    - zleg: zorder for the legend. Set to 1 to plot the legend behind the data, and to None to keep the default value.
    - loc_leg: location of the legend ('best', 'upper left', ...).
    - ncol_leg: number of columns to use in the legend.
    - lMticks: length of the major ticks on the axes.
    - lmticks: length of the minor ticks on the axes.
    - wMticks: width of the major ticks on the axes.
    - wmticks: width of the minor ticks on the axes.
    - nyMticks: number of major ticks on the y-axis (counting ymin and ymax).
    - mymticks: there are 1+mymticks*(nyMticks-1) minor ticks on the y axis.
    - grid: True to add a grid, False otherwise.
    - grid_ls: linestyle of the grid.
    - grid_col: color of the grid.
    - grid_alpha: alpha of the grid.
    - xlog: True to produce a semilogx plot and False to produce a plot. 
    Out:
    A figure is shown. 
    """
    # Number of curves
    nc = len(Ly)
    # Checking the parameters
    if ymin is None:
        ymin = np.min(np.asarray([arr.min() for arr in Ly]))
    if ymax is None:
        ymax = np.max(np.asarray([arr.max() for arr in Ly]))
    if Lmarkers is None:
        Lmarkers = ['x']*nc
    if Lcols is None:
        Lcols = ['blue']*nc
    if Lleg is None:
        Lleg = [None]*nc
        add_leg = False
    else:
        add_leg = True
    if Lls is None:
        Lls = ['solid']*nc
    if Lmedw is None:
        Lmedw = [float(lw)/2.0]*nc
    if Lsdots is None:
        Lsdots = [12]*nc
    
    # Setting the limits of the y-axis
    y_lim = [ymin, ymax]
    
    # Defining the ticks on the y-axis
    yMticks = np.linspace(start=ymin, stop=ymax, num=nyMticks, endpoint=True, retstep=False)
    ymticks = np.linspace(start=ymin, stop=ymax, num=1+mymticks*(nyMticks-1), endpoint=True, retstep=False)
    yMticksLab = [rstr(v) for v in yMticks]
    
    # Initial values for xmin and xmax
    xmin, xmax = 1, -np.inf
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if xlog:
        fplot = ax.semilogx
    else:
        fplot = ax.plot
    
    # Plotting the data
    for id, y in enumerate(Ly):
        x = np.arange(start=1, step=1, stop=y.size+0.5, dtype=np.int64)
        xmax = max(xmax, x[-1])
        fplot(x, y, label=Lleg[id], alpha=alpha_plot, color=Lcols[id], linestyle=Lls[id], lw=lw, marker=Lmarkers[id], markeredgecolor=Lcols[id], markeredgewidth=Lmedw[id], markersize=Lsdots[id], dash_capstyle='round', solid_capstyle='round', dash_joinstyle='round', solid_joinstyle='round', markerfacecolor=Lcols[id], markevery=markevery)
    
    # Setting the limits of the axes
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(y_lim)
    
    # Setting the major and minor ticks on the y-axis 
    ax.set_yticks(yMticks, minor=False)
    ax.set_yticks(ymticks, minor=True)
    ax.set_yticklabels(yMticksLab, minor=False, fontsize=sax)
    
    # Defining the legend
    if add_leg:
        leg = ax.legend(loc=loc_leg, fontsize=sleg, markerfirst=True, fancybox=True, framealpha=alpha_leg, ncol=ncol_leg)
        if zleg is not None:
            leg.set_zorder(zleg)
    
    # Setting the size of the ticks labels on the x axis
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(sax)
    
    # Setting ticks length and width
    ax.tick_params(axis='both', length=lMticks, width=wMticks, which='major')
    ax.tick_params(axis='both', length=lmticks, width=wmticks, which='minor')
    
    # Setting the positions of the labels
    ax.xaxis.set_tick_params(labelright=False, labelleft=True)
    ax.yaxis.set_tick_params(labelright=False, labelleft=True)
    
    # Adding the grids
    if grid:
        ax.xaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
        ax.yaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
    ax.set_axisbelow(True)
    
    ax.set_title(tit, fontsize=stit)
    ax.set_xlabel(xlabel, fontsize=sax)
    ax.set_ylabel(ylabel, fontsize=sax)
    plt.tight_layout()
    
    # Saving or showing the figure, and closing
    save_show_fig(fname=fname, f_format=f_format)
    plt.close()

##############################
############################## 
# Demo presenting how to use the main functions of this file.
####################

if __name__ == '__main__':
    print("============================================")
    print("===== Starting the demo of mstSNE.py =====")
    print("============================================")
    
    print('%')
    print('% The demo takes a few minutes.')
    print('%')
    
    ###
    ###
    ###
    print('- Loading the Digits data set')
    # TIP: to change the employed data set, you just need to modify the next line to provide different values for X_hd and labels.
    X_hd, labels = sklearn.datasets.load_digits(return_X_y=True)
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    print('- Checking if there are identical examples in the data set')
    if contains_ident_ex(X_hd):
        print(' *** !!! Warning !!! *** The data set contains duplicated examples, which is not recommended when employing neighbor embedding methods.')
    else:
        print('Great, there are no identical examples.')
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    # Function to compute a 2-D numpy array containing the pairwise distances in a data set. This function is used to compute the HD distances used both in perplexity-free multi-scale t-SNE and for the DR quality assessment.
    compute_dist_HD = eucl_dist_matr
    # Function to compute a 2-D numpy array containing the pairwise distances in a data set. This function is used to compute the LD distances for the DR quality assessment. Note that in multi-scale t-SNE, the LD embedding is computed using Euclidean distances in the LD space, independently of the value of compute_dist_LD_qa.
    compute_dist_LD_qa = eucl_dist_matr
    # Lists to provide as parameters to viz_qa, to visualize the DR quality assessment as conducted in [1, 7].
    L_rnx, Lmarkers, Lcols, Lleg_rnx, Lls, Lmedw, Lsdots = [], [], [], [], [], [], []
    
    ###
    ###
    ###
    print('- Computing the pairwise HD distances in the data set')
    t0 = time.time()
    dm_hd = compute_dist_HD(X_hd)
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    
    ###
    ###
    ###
    # Targeted dimension of the LD embedding
    dim_LDS = 2
    # Initialization for multi-scale t-SNE. Check the 'init' parameter of the 'mstsne' function for a description. In particular, init_mstsne can be equal to 'pca', 'random' or a 2-D numpy array containing the initial LD coordinates of the data points. 
    init_mstsne = 'pca'
    # Random seed for multi-scale t-SNE. This ensures that the same LD initialization is used for all applications of multi-scale t-SNE. Check the 'seed_mstsne' parameter of the 'mstsne' function for a description. 
    seed_mstsne = 40
    
    ###
    ###
    ###
    print('- Applying multi-scale t-SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
    t0 = time.time()
    # TIP: you could set dm_hds to None in the following line if you do not want to precompute the pairwise HD distances dm_hd; in this case, pairwise HD Euclidean distances would then be computed based on X_hd in the 'mstsne' function. You can also use other HD distances than the Euclidean one for the dm_hds parameter: you just need to modify the above compute_dist_HD function to compute the 2-D numpy array storing the pairwise distances of your choice. Note that you can provide the LD coordinates to use for the initialization of multi-scale t-SNE by setting init_mstsne to a 2-D numpy.ndarray containing the initial LD positions, with one example per row and one LD dimension per column, init_mstsne[i,:] containing the initial LD coordinates related to the HD sample X_hd[i,:].
    X_ld_mstsne = mstsne(X_hds=X_hd, init=init_mstsne, n_components=dim_LDS, dm_hds=dm_hd, seed_mstsne=seed_mstsne)
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    
    ###
    ###
    ###
    print('- Evaluating the DR quality of the LD embedding obtained by multi-scale t-SNE')
    t0 = time.time()
    rnx_mstsne, auc_mstsne = eval_dr_quality(d_hd=dm_hd, d_ld=compute_dist_LD_qa(X_ld_mstsne))
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    print('AUC: {v}'.format(v=rstr(auc_mstsne, 4)))
    
    # Updating the lists for viz_qa
    L_rnx.append(rnx_mstsne)
    Lmarkers.append('x')
    Lcols.append('blue')
    Lleg_rnx.append('Ms $t$-SNE')
    Lls.append('solid')
    Lmedw.append(0.5)
    Lsdots.append(10)
    
    ###
    ###
    ###
    print('- Plotting the LD embedding obtained by multi-scale t-SNE')
    print('If a figure is shown, close it to continue.')
    # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information. 
    viz_2d_emb(X=X_ld_mstsne, vcol=labels, tit='LD embedding Ms $t$-SNE', fname=None, f_format=None)
    print('===')
    print('===')
    print('===')
    
    ###
    ### Applying (single-scale) t-SNE with several perplexities, for comparison with multi-scale t-SNE. 
    ###
    # Perplexity employed in t-SNE. 40.0 is the value employed in 'van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579-2605.'. 10.0 is a value more oriented towards the preservation of small neighborhoods, while 100.0 is more oriented towards preserving larger neighborhoods. 
    for id_perp_tsne, perp_tsne in enumerate([10.0, 40.0, 100.0]):
        print('- Applying t-SNE on the data set to obtain a {dim_LDS}-D embedding, using a perplexity = {perp_tsne}'.format(dim_LDS=dim_LDS, perp_tsne=perp_tsne))
        t0 = time.time()
        # The same 'early_exaggeration' is used as in 'van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579-2605.', and the same initialization is employed as when applying multi-scale t-SNE. 
        X_ld_tsne = sklearn.manifold.TSNE(n_components=dim_LDS, perplexity=perp_tsne, early_exaggeration=4.0, metric='euclidean', init=init_mstsne, random_state=seed_mstsne, method='exact').fit_transform(X_hd)
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=rstr(t)))
        
        ###
        ###
        ###
        print('- Evaluating the DR quality of the LD embedding obtained by t-SNE using a perplexity = {perp_tsne}'.format(perp_tsne=perp_tsne))
        t0 = time.time()
        rnx_tsne, auc_tsne = eval_dr_quality(d_hd=dm_hd, d_ld=compute_dist_LD_qa(X_ld_tsne))
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=rstr(t)))
        print('AUC: {v}'.format(v=rstr(auc_tsne, 4)))
        
        # Updating the lists for viz_qa
        L_rnx.append(rnx_tsne)
        Lmarkers.append('o' if id_perp_tsne == 0 else ('s' if id_perp_tsne == 1 else '^'))
        Lcols.append('red' if id_perp_tsne == 0 else ('green' if id_perp_tsne == 1 else 'magenta'))
        Lleg_rnx.append('$t$-SNE (perp={p})'.format(p=int(perp_tsne)))
        Lls.append('solid')
        Lmedw.append(0.3)
        Lsdots.append(10)
    
    ###
    ###
    ###
    print('- Plotting the results of the DR quality assessment, as reported in (de Bodt et al, ESANN, 2018) [1] and (de Bodt et al, IEEE TNNLS, 2020) [7].')
    print('If a figure is shown, close it to continue.')
    # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information. 
    viz_qa(Ly=L_rnx, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=Lleg_rnx, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='DR quality', xlabel='Neighborhood size $K$', ylabel=r'$R_{\mathrm{NX}}(K)$', fname=None, f_format=None, ncol_leg=1)
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    print('*********************')
    print('***** Done! :-) *****')
    print('*********************')

