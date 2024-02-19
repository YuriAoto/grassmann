"""Functions to deal with the Grassmann manifold


History:
  Forked from what was initially done in util

"""
import logging

import numpy as np
from scipy import linalg

from . import absil


def proj_tgspace(Y, Z, S=None):
    """Calculates the projection of Z into the tangent space of the Grassmannian
    
    Similar to Equation (3.41) of Absil book, but with Gram matrix.
    
    Parameters:
    -----------
    Y (2D array of shape d,N) - the point of the Grassmann manifold
    Z (2D array of shape d,N) - the matrix to be projected into the tangent space
    S (2D array of dimension n,n) - the Gram matrix
    """
    return (np.eye(Y.shape[0]) - (Y @ Y.T
                                  if S is None else
                                  Y @ Y.T @ S)) @ Z


def geodesic(C, DD, X, invX, t=1, gs=False):
    """Computes the geodesic of the Grassmannian


    Parameters:
    -----------
    - C (nxN ndarray): orbitals' coefficients (aka an element in the Stiefel).
    - DD (nxN ndarray): the descent direction in the horizontal space. It must
        satisfy C.T @ S @ DD = 0.
    - S (nxn ndarray): the overlap matrix of the basis set.
    - X (nxn ndarray): a matrix that satisfies X.T @ S @ X = Id.
    - invX (nxn ndarray): the inverse of X.
    - t (float, optional, default=1): the size of the step of the geodesic.

    Returns:
    --------
    A nxN ndarray with the new orbitals' coefficients.
    """
    U, D, V = linalg.svd(invX @ DD, full_matrices=False)
    sin, cos = np.diag(np.sin(t * D)), np.diag(np.cos(t * D))
    return (C @ V.T @ cos + X @ U @ sin) @ V


def pt(vector, direction, point, Id, X, invX, t):
    """Computes the parallel transport along a geodesic in the Grassmannian.

    Parameters:
    -----------
    - vector (nxN ndarray): the vector to be transported.
    - direction (nxN ndarray): the initial direction of the geodesic g'(0).
    - point (nxN ndarray): the starting point of the geodesic g(0).
    - Id (nxn ndarray): identity matrix (just to save memory).
    - X (nxn ndarray): a matrix that satisfies X.T @ S @ X = Id.
    - invX (nxn ndarray): the inverse of X.

    Returns:
    --------
    An nxN ndarray that represents the parallel transport of 'vector'.
    """
    U, D, V = linalg.svd(invX @ direction, full_matrices=False)
    sin, cos = np.diag(np.sin(t * D)), np.diag(np.cos(t * D))
    return ((-point @ V.T @ sin + X @ U @ cos) @ U.T @ invX
            + Id - X @ U @ U.T @ invX) @ vector


def RBLS(C_a, C_b, DD_a, DD_b, norm_DD, E_cur, E_prev, S, X, invX, hf):
    """Riemannian Backtracking Line Search.

    Parameters:
    -----------
    - C_a (nxN_a ndarray): spin alpha orbitals' coefficients.
    - C_b (nxN_b ndarray): spin beta orbitals' coefficients.
    - DD_a (nxN_a ndarray): descent direction for spin alpha.
    - DD_b (nxN_b ndarray): descent direction for spin beta.
    - norm_DD (float): the norm of the descent direction (DD_a, DD_b) squared.
    - E_cur (float): the current value of the energy.
    - E_prev (float): the value of the energy in the previous iteration.
    - S (nxn ndarray): the overlap matrix.
    - X (nxn ndarray): a matrix that satisfies X.T @ S @ X = Id.
    - invX (nxn ndarray): the inverse of X.
    - hf (class): the HartreeFockStep class in order to call the _energy method.
                  probably not optimal.

    Returns:
    --------
    A triple (auxC_a, auxC_b, t_n*Xi) with the new point satisfying the Armijo
    condition and the size of the step t_n*Xi.
    """
    r, t, t_n, max_iter, n = 1e-4, 0.5, 0.5, 25, 0
    auxC_a, auxC_b = np.copy(C_a), np.copy(C_b)

    if E_prev == float('inf'):
        Xi = 1 / np.sqrt(norm_DD)
    else:
        Xi = 4 * (E_prev - E_cur) / norm_DD

    U_a, D_a, V_a = linalg.svd(invX @ DD_a, full_matrices=False)
    U_b, D_b, V_b = linalg.svd(invX @ DD_b, full_matrices=False)
    auxV_a, auxV_b = C_a @ V_a.T, C_b @ V_b.T
    auxU_a, auxU_b = X @ U_a, X @ U_b
    sin_a, cos_a = np.diag(np.sin(D_a)), np.diag(np.cos(D_a))
    sin_b, cos_b = np.diag(np.sin(D_b)), np.diag(np.cos(D_b))
    ener = E_cur - r * t_n * Xi * norm_DD + 1

    while E_cur - ener < r * t_n * Xi * norm_DD and n < max_iter:
        n += 1
        t_n *= t
        sin_a, cos_a = np.sin(t_n * Xi * D_a, out=sin_a), np.cos(t_n * Xi * D_a, out=cos_a)
        sin_b, cos_b = np.sin(t_n * Xi * D_b, out=sin_b), np.cos(t_n * Xi * D_b, out=cos_b)
        auxC_a = (auxV_a @ cos_a + auxU_a @ sin_a) @ V_a
        auxC_b = (auxV_b @ cos_b + auxU_b @ sin_b) @ V_b
        ener = hf._energy(auxC_a, auxC_b)

    return absil.gram_schmidt(auxC_a, S), absil.gram_schmidt(auxC_b, S), t_n * Xi


def rip(first_vector, second_vector, S):
    """Riemannian Inner Product in the Grassmannian.
    
    Parameters:
    -----------
    first_vector and second_vector:
        list of vectors where each vector is in the tangent space of a Grassmannian.
    S: overlap matrix.
    
    Example:
    --------
    If we are working with, say, Gr(2, 5) x Gr(3, 5), then S is a 5x5 matrix,
    first_vector = [x_1, x_2], second_vector = [y_1, y_2],
    where x_1, y_1 \in T_{C_1}Gr(2, 5) and x_2, y_2 \in T_{C_2}Gr(3, 5).
    Observe that we can have as many Grassmannians as we want as long as S
    (which forces 5 to be fixed) stays the same.
    Then it's just a matter of basic linear algebra:
    The inner product in the product of vector spaces is the sum of the inner products.
    
    Notes:
    ------
    S could also be a list with different overlap matrices.
    The modification that should be made in this case is also zipping the list S.

    """
    return sum(
        np.trace(x.T @ S @ y) for x, y in zip(first_vector, second_vector)
    )


def Riem_norm(vector, S):
    """Riemannian Norm in the Grassmannian.
    
    Parameters:
    -----------
    - vector: list of vectors where each vector is in the tangent space of a
        Grassmannian.
    - S: overlap matrix.
    
    Notes:
    ------
    S could also be a list with different overlap matrices.
    Nothing changes in this function, just in the inner product.
    See documentation of the `rip` function.

    """
    return np.sqrt(rip(vector, vector, S))
