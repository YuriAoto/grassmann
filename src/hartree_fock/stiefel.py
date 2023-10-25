"""Functions to deal with the Stiefel manifold

Main reference:
Edelman

Notation: We will use Edelman's notation.
In particular:

Y    a point in the Stiefel, i.e, a n x p orthogonal matrix
n    the dimension of the space (number of rows of Y)
d    the number of columns of Y


"""
import logging

import numpy as np


logger = logging.getLogger(__name__)


def metric(Y, v1, v2, S=None):
    """Calculates the canonical metric in the Stiefel.
    
    Equation  (2.39) of Edelman 
    
    Parameters:
    -----------
    Y (2D arrays of shape n,p) - the point of the Stiefel manifold
    v1 and v2 (2D arrays of shape n,p) - the tangent vectors
    S (2D array of dimension n,n) - the Gram matrix
    
    """
    if S is not None: raise ValueError("S != None is not implemented.")
    return np.trace(v1.T @ (np.eye(Y.shape[0]) - Y @ Y.T / 2) @ v2)


def proj_tgspace(Y, Z, S=None, times2=False):
    """Calculates the projection of Z into the tangent space of the Stiefel
    
    Equation (2.4) of Edelman
    
    Parameters:
    -----------
    Y (2D arrays of shape d,N) - the point of the Stiefel manifold
    Z (2D array of shape d,N) - the matrix to be projected into the tangent space
    S (2D array of dimension n,n) - the Gram matrix
    """
    if S is not None: raise ValueError("S != None is not implemented.")
    return (Y @ (Y.T @ Z - Z.T @ Y) / (1 if times2 else 2)
            + (np.eye(Y.shape[0]) - Y @ Y.T) @ Z)


def check_in_tgspace(Y, Z, tol=1.0E-8, S=None):
    """Check if Z is in the tangent space of Stiefel at Y"""
#    logger.debug('checking F_YY: Z.T@Y:\n%s', Z.T @ Y)
#    logger.debug('checking F_YY: Y.T@Z:\n%s', -Y.T @ Z)
    if S is not None: raise ValueError("S != None is not implemented.")
    return np.allclose(Y.T @ Z, -Z.T @ Y, atol=tol)


def tg_vecs(Y, S=None):
    """Yield tangent vectors according to Eq. (2.6) of Edelman
    
    TODO: (?) It would be nice to use (2.5), that gives a basis
    
    """
    if S is not None: raise ValueError("S != None is not implemented.")
    n, p = Y.shape
    for i in range(p):
        for j in range(i):
            A = np.zeros((p, p))
            A[i,j] = 1
            A[j,i] = -1
            yield f'A_{i}_{j}', Y @ A
    I_YYt = np.eye(n) - Y @ Y.T
    for _mu in range(n):
        for i in range(p):
            C = np.zeros((n, p))
            C[_mu,i] = 1
            yield f'C_{_mu}_{i}', I_YYt @ C


def test_FYY(Y, FYY, S=None):
    """This is a test funtion to see the solution of Equation (2.60) of Edelman
    
    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    """
    if S is not None: raise ValueError("S != None is not implemented.")
    logger.debug('Testing finding unique F_YY(D) in tangent space')
    n, p = Y.shape
    for nameD, D in tg_vecs(Y):
#        logger.debug('checking F_YY: D = %s:\n%s', nameD, D)
        logger.debug('checking F_YY: D = %s is in tg space: %s',
                     nameD, check_in_tgspace(Y, D))
        FYYD = FYY @ np.reshape(D, n*p, order='F')
        FYYD = np.reshape(FYYD, (n, p), order='F')
        FYYDproj = proj_tgspace(Y, FYYD, times2=True)
        logger.debug('checking F_YY: FYYDproj is in tg space: %s',
                     check_in_tgspace(Y, FYYDproj))
        for nameX, X in tg_vecs(Y):
            prod1 = np.trace(FYYD.T @ X)
            prod2 = metric(Y, FYYDproj, X)
            logger.debug('  F_YY(D, %s) = %s\n'
                         '<F_YY(D), %s> = %s\n'
                         'Are they equal??  %s',
                         nameX, prod1, nameX, prod2,
                         abs(prod1-prod2)<1.0E-8)
