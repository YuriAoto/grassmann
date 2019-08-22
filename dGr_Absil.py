"""Some functions related to the Newton-Grassmann method of Absil

Here, we consider the function f is given by 

f(x) = <0|ext>

where |0> is a Slater determinant and |ext> a correlated wave function

"""
import numpy as np
from numpy import linalg


def calc_fI(U, det_indices):
    """Calculate the contribution of U[det_indices,:] to f

    Parameters:
    U (numpy.ndarray)     the coefficients matrix
    det_indices (list)    a list of subindices

    Return:
    det(U[det_indices, :])
    """
    return linalg.det(U[det_indices, :])

def calc_G(U, det_indices, i, j):
    """Calculate the element ij of matrix G
    
    Behaviour:
    Calculates the following the determinant:
    det(U[det_indices, :] <-j- e_i )
    See _calc_H for details.
    
    Parameters:
    U (numpy.ndarray)     the coefficients matrix
    det_indices (list)    a list of subindices
    i,j (int)             the indices

    Return:
    det(U[det_indices, :] <-j- e_i )
    """
    if i not in det_indices:
        return 0.0
    if U.shape[1] == 1:
        return 1.0
    sign = 1 if (j + det_indices.index(i))%2 == 0 else -1
    row_ind = np.array([x for x in det_indices       if x!=i], dtype=int)
    col_ind = np.array([x for x in range(U.shape[1]) if x!=j], dtype=int)
    return sign*linalg.det(U[row_ind[:,None],col_ind])

def calc_H(U, det_indices, i, j, k, l):
    """Calculate the element ijkl of matrix H
    
    Behaviour:
    Calculates the following the determinant:
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    where <-j- e_i means that the j-th column
    of the matrix is replaced by the vector e_i
    (with all zeros except at i, that has a 1).
    i is between 0 and U.shape[0], and by expansion
    on the replaced jth column, i+1 must be in det_indices
    for non vanishing determinant.
    Analogous for <-l- e_k.
    if j == l (trying to replace the same column)
    gives 0 (irrespective of i and k!)
    
    
    Parameters:
    U (numpy.ndarray)     the coefficients matrix
    det_indices (list)    a list of subindices
    i,j,k,l (int)         the indices

    Return:
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    if j == l or i == k:
        return 0.0
    if i not in det_indices or k not in det_indices:
        return 0.0
    sign = 1.0 if ((i<k) == (j<l)) else -1.0
    if U.shape == 2:
        return sign
    if (j + det_indices.index(i) + l + det_indices.index(k))%2 == 1:
        sign = -sign
    row_ind = np.array([x for x in det_indices       if (x!=i and x!=k)], dtype=int)
    col_ind = np.array([x for x in range(U.shape[1]) if (x!=j and x!=l)], dtype=int)
    return sign*linalg.det(U[row_ind[:,None],col_ind])
