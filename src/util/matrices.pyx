"""Some useful functions for matrices



"""
cimport cython

import numpy as np


def normalize(double[:] v, double[:, :] G):
    """Normalize a vector with respect to an arbitrary basis.

    Parameters:
    -----------
    - v (nx1 ndarray): the vector to be normalized.
    - G (nxn ndarray): the Gram matrix associated with the basis of interest.

    Returns:
    --------
    nx1 ndarray with the orthonormalized 'v'.
    """
    return v / np.sqrt(np.transpose(v).T @ G @ v)



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def gram_schmidt(M, double[:, :] G):
    """Gram-Schmidt algorithm in an arbitrary basis.

    Parameters:
    -----------
    - M (nxk ndarray): matrix containing the k vectors to be orthonormalized.
        Each vector is a column of the matrix.
    - G (nxn ndarray): the Gram matrix associated with the basis of interest.

    Returns:
    --------
    A nxk ndarray containing the orthonormalized vectors.
    """
    cdef int i, j

    M[:, 0] = normalize(M[:, 0], G)
    for i in range(1, M.shape[1]):
        Mi = M[:, i]
        for j in range(i):
            Mj = M[:, j]
            t = Mi.T @ G @ Mj
            Mi = Mi - t * Mj
            M[:, i] = normalize(Mi, G)

    return M

