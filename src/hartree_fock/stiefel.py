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


def tg_vecs(Y, S=None, in_Gr_tgspace=True, in_Gr_tgspace_orth=True):
    """Yield tangent vectors according to Eq. (2.6) of Edelman
    
    TODO: (?) It would be nice to use (2.5), that gives a basis
    
    """
    n, p = Y.shape
    if in_Gr_tgspace_orth:
        for i in range(p):
            for j in range(i):
                A = np.zeros((p, p))
                A[i,j] = 1
                A[j,i] = -1
                yield f'A_{i}_{j}', Y @ A
    if in_Gr_tgspace:
        I_YYt = np.eye(n) - (Y @ Y.T
                             if S is None else
                             Y @ Y.T @ S)
        for _mu in range(n):
            for i in range(p):
                C = np.zeros((n, p))
                C[_mu,i] = 1
                yield f'C_{_mu}_{i}', I_YYt @ C


def test_HessFDX_with_eta(Y, FYY, eta, F, S=None):
    """This is a test to check if equation below is zero
    
    F_YY(D, X) + tr(Y.T F D Y.T X)/2

    where D=eta from the prblem in the grassmannian,
    X = YN (N symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix
    

    """
    logger.debug('Testing FYY_D_X + tr_YtFDX = F_YY(D, X) + tr(Y.T F D Y.T X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False, S=S):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        FYY_D_X = np.trace(eta.T @ FYY_X)
        tr_YtFDX = np.trace(Y.T @ F @ eta @ YtX)
        logger.debug('For X=%s\n'
                     'FYY_D_X = %s\n'
                     'tr_YtFDX = %s\n'
                     'FYY_D_X + tr_YtFDX = %s',
                     nameX,
                     FYY_D_X,
                     tr_YtFDX,
                     FYY_D_X + tr_YtFDX)


def test_FYY_0(Y, FYY, F, S=None):
    """This is a test to check if equation below is zero
    
    F_YY(D, X) + tr(Y.T F D Y.T X)/2

    where D=eta from the prblem in the grassmannian,
    X = YN (N symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix
    

    """
    logger.debug('Testing FYY_D_X = F_YY(D, X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        FYY_D_X = np.trace(eta.T @ FYY_X)
        tr_YtFDX = np.trace(Y.T @ F @ eta @ YtX)
        logger.debug('For X=%s\n'
                     'FYY_D_X = %s\n'
                     'tr_YtFDX = %s\n'
                     'FYY_D_X + tr_YtFDX = %s',
                     nameX,
                     FYY_D_X,
                     tr_YtFDX,
                     FYY_D_X + tr_YtFDX)


def test_HessFDX(Y, FYY, FY, S=None):
    """This is a test to check if equation below is zero
    
    F_YY(D, X) + tr(Y.T F D Y.T X)/2

    where D=is in the tangent space of Gr from the prblem in the grassmannian,
    X = YN (N skew symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix
    

    """
    logger.debug('Testing FYY_D_X + tr_YtFDX = F_YY(D, X) + tr(Y.T F D Y.T X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y):  #, in_Gr_tgspace=False):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        for nameD, D in tg_vecs(Y):  #, in_Gr_tgspace_orth=False):
            FYY_D_X = np.trace(D.T @ FYY_X)
            tr_YtFDX = np.trace(FY.T @ D @ YtX)
            logger.debug('For X=%s and D=%s: FYY_D_X + tr_YtFDX = %s\n'
                         'FYY_D_X = %s\n'
                         'tr_YtFDX = %s\n',
                         nameX, nameD, FYY_D_X + tr_YtFDX,
                         FYY_D_X,
                         tr_YtFDX
                         )
        


def test_FYY_0(Y, FYY, FY, S=None):
    """This is a test to check if equation below is zero
    
    F_YY(D, X)

    where D=is in the tangent space of Gr from the prblem in the grassmannian,
    X = YN (N skew symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix
    

    """
    logger.debug('Testing FYY_D_X = F_YY(D, X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        for nameD, D in tg_vecs(Y, in_Gr_tgspace_orth=False):
            FYY_D_X = np.trace(D.T @ FYY_X)
            logger.debug('For X=%s and D=%s: FYY_D_X = %s\n',
                         nameX, nameD, FYY_D_X
                         )
        


def test_FYY_eta_0(Y, FYY, FY, eta, S=None):
    """This is a test to check if equation below is zero
    
    F_YY(eta, X)

    where eta=is the solution of the problem in the Grassmannian
    X = YN (N skew symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix
eta
    

    """
    logger.debug('Testing FYY_D_X = F_YY(D, X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        for nameD, D in tg_vecs(Y, in_Gr_tgspace_orth=False):
            FYY_D_X = np.trace(D.T @ FYY_X)
            logger.debug('For X=%s and D=%s: FYY_D_X = %s\n',
                         nameX, nameD, FYY_D_X
                         )
        



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
