"""General functions for Hartree-Fock

"""
import logging

import numpy as np
from scipy.linalg import solve, svd
from . import absil

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

fmt_HF_header = '{0:<5s}  {1:<16s}  {2:<16s}  {3:s}\n'
fmt_HF_iter = '{0:<5d}  {1:<16.12f}  {2:<16.12f}  {3:s}\n'

write_header = \
    'it. \t E \t\t\t |Gradient| \t\t |Restr.| \t step \t\t time in iteration\n'
#    12345678901234567890123456789012345678901234567890123456789012345678901234567890123456

fmt_HF_iter_general = '{0:<5d} \t {1:<16.12f} \t {2:<16.12f} \t          \t {4:6s} \t {5:s}\n'
fmt_HF_iter_gen_lag = '{0:<5d} \t {1:<16.12f} \t {2:<16.12f} \t {3:<.4f} \t {4:6s} \t {5:s}\n'


def calculate_DIIS(Dmat, grad, cur_n_DIIS, i_DIIS):
    """Calculate a DIIS step
        
    Parameters:
    -----------
    
    Dmat (np.ndarray)
        The density matrix
    
    grad (np.ndarray)
        Hartree-Fock Gradient
    
    cur_n_DIIS (int)
        The current dimension of the iterative subspace
    
    i_DIIS (int)
        index of Dmat of current iteration
    
    Returns:
    --------

    Does not return anything, but the density matrix
    is updated.
    
    TODO:
    -----

    This does not give the same result for equivalent
    restricted and unrestricted calculations.
    Investigate why.
    
    This implementation is rather simple, and is not sufficient
    to achieve convergence for benzene. Improve this.
    
    """
    B = np.zeros((cur_n_DIIS + 1, cur_n_DIIS + 1))
    B[:cur_n_DIIS, :cur_n_DIIS] = np.einsum('iap,iaq->pq',
                                            grad[:, :, :cur_n_DIIS],
                                            grad[:, :, :cur_n_DIIS])
    for i in range(cur_n_DIIS):
        B[i, cur_n_DIIS] = B[cur_n_DIIS, i] = -1.0
    B[cur_n_DIIS, cur_n_DIIS] = 0.0
    logger.debug('DIIS B matrix:\n%r', B)
    a = np.zeros((cur_n_DIIS + 1, 1))
    a[cur_n_DIIS] = -1.0
    w = solve(B, a)
    logger.debug('DIIS w vector:\n%r', w)
    logger.debug('w entry associated to the current step: %f',
                 w[i_DIIS])
    Dmat[:, :, i_DIIS] *= w[i_DIIS]
    for k in range(cur_n_DIIS):
        if k == i_DIIS:
            continue
        Dmat[:, :, i_DIIS] += w[k] * Dmat[:, :, k]
    logger.debug('Density matrix (after DIIS):\n%r',
                 Dmat[:, :, i_DIIS])


def geodesic(C, eta, S, Sqrt, invSqrt, t=1):
    """Calculates the geodesic at the Grassmannian
    
    
    Parameters:
    -----------
    C (ndarray, of shape (n,N))
        The orbital coefficients (that is, an element at the Stiefel)
    
    eta (ndarray, of shape (n,N))
        The direction at the horizontal space:
        
        C.T @ S @ eta = 0
        
    S (ndarray, shape (n,n))
        The overlap matrix of the basis set    
    
    Sqrt (ndarray, shape (n,n))
        The square root of the overlap matrix S. It is the X matrix of Szabo
    
    invSqrt (ndarray, shape (n,n))
        The inverse of Sqrt
    
    t (float, optional, default=1)
        The step to calculate the geodesic

    
    Returns:
    --------
    A np.array of shape (n, N), with the orbital coefficients
    
    
    """
    u, s, v = svd(invSqrt @ eta, full_matrices=False)
    sin, cos = np.diag(np.sin(t*s)), np.diag(np.cos(t*s))
    temp = (C @ v.T @ cos + Sqrt @ u @ sin) @ v
    return absil.gram_schmidt(temp, S)
