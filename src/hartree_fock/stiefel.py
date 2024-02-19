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
from scipy import linalg

logger = logging.getLogger(__name__)

from orbitals import orbitals
from util import array_indices


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


def proj_tgspace(Y, Z, S=None):
    """Calculates the projection of Z into the tangent space of the Stiefel
    
    Similar to the modified vertion of Equation (2.4) of Edelman, but with Gram matrix.
    
    Parameters:
    -----------
    Y (2D array of shape d,N) - the point of the Stiefel manifold
    Z (2D array of shape d,N) - the matrix to be projected into the tangent space
    S (2D array of dimension n,n) - the Gram matrix
    """
    YtZ = (Y.T @ Z
           if S is None else
           Y.T @ S @ Z)
    return Z - Y @ sym(YtZ)


def check_in_tgspace(Y, v, tol=1.0E-8, S=None):
    """Check if v is in the tangent space of Stiefel at Y"""
    Ytv = (Y.T @ v
           if S is None else
           Y.T @ S @ v)
    is_allclose = np.allclose(Ytv, -Ytv.T, atol=tol)
    logger.info('T(St): ||Y.T @ v + (Y.T @ v).T|| = %s  (Is v in T_Y St: %s)',
                linalg.norm(Ytv + Ytv.T),
                is_allclose)
    return is_allclose


def tg_vecs(Y,
            mode='2.5',
            S=None,
            in_Gr_tgspace=True,
            in_Gr_tgspace_orth=True):
    """Yield tangent vectors according to Eq. (2.5) or (2.6) of Edelman
    
    """
    n, p = Y.shape
    if in_Gr_tgspace_orth:
        for j in range(p):
            for i in range(j):
                A = np.zeros((p, p))
                A[i,j] = 1
                A[j,i] = -1
                yield f'A_{i}_{j}', Y @ A
    if in_Gr_tgspace:
        if mode == '2.5':
            Y_perp = orbitals.complete_orb_space([Y], [n], S=S)[0][:,p:]
            for _mu in range(n-p):
                for i in range(p):
                    B = np.zeros((n-p, p))
                    B[_mu, i] = 1
                    yield f'B_{_mu}_{i}', Y_perp @ B
        elif mode == '2.6':
            I_YYt = np.eye(n) - (Y @ Y.T
                                 if S is None else
                                 Y @ Y.T @ S)
            for _mu in range(n):
                for i in range(p):
                    C = np.zeros((n, p))
                    C[_mu, i] = 1
                    yield f'C_{_mu}_{i}', I_YYt @ C
        else:
            raise ValueError('Unknown mode to generate tangent vectors')

def ith_tg_vecs(Y,
                i,
                mode='2.5',
                S=None):
    """Return the i-th tangent vector that is yield by tg_vecs"""
    n, p = Y.shape
    if i < array_indices.triangular(p-1):
        ii, j = array_indices.ij_from_triang(i)
        A = np.zeros((p, p))
        A[ii,j] = 1
        A[j,ii] = -1
        return Y @ A
    i -= array_indices.triangular(p-1)
    if mode == '2.5':
        if i >= (n-p) * p:
            raise ValueError('Too large index for tan vector in mode 2.5')
        Y_perp = orbitals.complete_orb_space([Y], [n], S=S)[0][:,p:]
        _mu, ii = array_indices.ij_from_rect(i, p)
        B = np.zeros((n-p, p))
        B[_mu, ii] = 1
        return Y_perp @ B
    elif mode == '2.6':
        if i >= n * p:
            raise ValueError('Too large index for tan vector in mode 2.6')
        I_YYt = np.eye(n) - (Y @ Y.T
                             if S is None else
                             Y @ Y.T @ S)
        _mu, ii = array_indices.ij_from_rect(i, p)
        C = np.zeros((n, p))
        C[_mu, ii] = 1
        return I_YYt @ C
    else:
        raise ValueError('Unknown mode to generate tangent vectors')
    



def sym(X):
    return (X + X.T)/2


def geodesic(C, v, S=None, t=1):
    """Computes the geodesic on the Stiefel
    
    
    Parameters:
    -----------
    - C (nxN ndarray): orbitals' coefficients (aka an element in the Stiefel).
    - v (nxN ndarray): the vector in the tangent space of Stiefel.
      It must satisfy
          C.T @ S @ v + (C.T @ S @ v).T = 0.
    - S (nxn ndarray): the overlap matrix of the basis set.
    - t (float, optional, default=1): the size of the step of the geodesic.

    Returns:
    --------
    A nxN ndarray with the new orbitals' coefficients.
    """
    d, N = C.shape
    if S is None:
        S = np.eye(d)
    CtSv = C.T @ S @ v
    exp_central = np.empty((2*N, 2*N))
    exp_central[:N, :N] = CtSv
    exp_central[N:, N:] = CtSv
    exp_central[:N, N:] = -v.T @ S @ v
    exp_central[N:, :N] = np.eye(N)
    exp_central = linalg.expm(t * exp_central)
    return (C @ exp_central[:N, :N] + v @ exp_central[N:, :N]) @ linalg.expm(- t * CtSv)


def make_Hess(Y, euc_grad, euc_Hess, S, invS):
    """make the Hessian at Stiefel
    
    Parameters:
    -----------
    Y (2D array) - point at the Stiefel
    euc_grad (2D array) - Euclidean gradient
    euc_Hess (2D array) - Euclidean Hessian
    S (2D array) - overlap matrix
    invS (2D array) - inverse of overlap matrix
    
    """
    d, N = Y.shape
    Id_d, Id_N = np.eye(d), np.eye(N)
    Riem_Hess = np.kron(Id_N, invS) @ euc_Hess
    Riem_Hess -= np.kron(sym(euc_grad.T @ Y), Id_d)
    proj = np.kron(Id_N, (2*Id_d - Y @ Y.T @ S))
    Yt_kron_Y = np.kron(Y.T @ S, Y)
    # loops below make perm(Yt_kron_Y)
    for i in range(N):
        for p in range(d):
            proj[:, i*d + p] -= Yt_kron_Y[p*N + i]
    return proj @ Riem_Hess / 2


def make_Hess_v2(Y, euc_grad, euc_Hess, S, invS):
    """make the Hessian at Stiefel
    
    Using the fact that
      self.Euc_grad.T @ Y = Y.T @ self.Euc_grad.T
    that is, matrix above is symmetric
    
    Parameters:
    -----------
    Y (2D array) - point at the Stiefel
    euc_grad (2D array) - Euclidean gradient
    euc_Hess (2D array) - Euclidean Hessian
    S (2D array) - overlap matrix
    invS (2D array) - inverse of overlap matrix

    """
    d, N = Y.shape   
    Id_d, Id_N = np.eye(d), np.eye(N)
    Riem_Hess = np.kron(Id_N, invS) @ euc_Hess
    Riem_Hess -= np.kron(sym(euc_grad.T @ Y), Id_d)
    proj = np.kron(Id_N, Y @ Y.T @ S)
    Yt_kron_Y = np.kron(Y.T @ S, Y)
    # loops below make perm(Yt_kron_Y)
    for i in range(N):
        for p in range(d):
            proj[:, i*d + p] -= Yt_kron_Y[p*N + i]
    Riem_Hess = proj @ Riem_Hess / 2
    proj = np.kron(Id_N, (Id_d - Y @ Y.T @ S))
    Riem_Hess += proj @ (np.kron(Id_N, invS) @ euc_Hess
                         - np.kron(sym(euc_grad.T @ Y), Id_d))
    return Riem_Hess


def make_augmentation(Y, S):
    """Create the augmentation to impose a solution in tg space of Stiefel
    
    Parameters:
    -----------
    Y (2D array) : the element of the Stiefel manifold
    S (2D array) : the overlap matrix
    
    Return:
    -------
    a N² x Nd matrix A such that
    
    A @ vec(v) = 0
    
    if v belongs to the tangent Space of the Stiefel at Y
    
    """
    d, N = Y.shape
    YtS = Y.T @ S
    augmentation = np.kron(np.eye(N), YtS)
    # loops below make perm(Y.T @ S \otimes Id)
    for p in range(N):
        s_line = np.s_[p*d: (p+1)*d]
        for i in range(N):
            augmentation[i*N + p, s_line] += YtS[i, :]
    return augmentation



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
        FYY_D_X = np.trace(eta.T @ FYY_X
                           if S is None else
                           eta.T @ S @ FYY_X)
        tr_YtFDX = np.trace(Y.T @ F @ eta @ YtX)/2
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


def test_FYY_0(Y, FYY, S=None):
    """Test to check if equation below is zero
    
    F_YY(D, X)

    where D is in the tangent space of Gr,
    X is in its orthogonal complement (within the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    FY (2D array of shape n,n) - the euclidean gradient
    

    """
    logger.debug('Testing FYY_D_X = F_YY(D, X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False, S=S):
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        for nameD, D in tg_vecs(Y, in_Gr_tgspace_orth=False, S=S):
            FYY_D_X = np.trace(D.T @ FYY_X
                               if S is None else
                               D.T @ S @ FYY_X)
            logger.debug('For X=%s and D=%s: FYY_D_X = %s\n',
                         nameX, nameD, FYY_D_X
                         )
        


def log_eta_FYY_X(Y, FYY, eta, S=None):
    """This logs F_YY(eta, X) = eta.T @ S @ F_YY @ X

    for all X in the orthogonal complement of T_Y Gr
    (within the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    eta (2D array of shape n,n) - the eta, a vector in T_Y St
    S (2D array of shape n,n) - the overlap matrix
    

    """
    logger.debug('Calculating X_FYY_eta = F_YY(X, eta)')
    n, p = Y.shape
    FYY_eta = FYY @ np.reshape(eta, n*p, order='F')
    FYY_eta = np.reshape(FYY_eta, (n, p), order='F')
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False, S=S):
        X_FYY_eta = np.trace(X.T @ FYY_eta
                             if S is None else
                             X.T @ S @ FYY_eta)
        logger.debug('For X=%s: X_FYY_eta = %s\n', nameX, X_FYY_eta)



def log_eta_Fyy_Dn__tr_DnYFy(Y, Fy, riemGrad, Fyy, eta, S=None, restricted=None):
    """This logs F_YY(D, eta) - tr(D.T eta X.T Fy)

    for all D in T_St
    (within the tangent space of Stiefel)
    
    In the nonorthogonal basis:
    
    D.T @ S @ Fyy[eta] - tr(D.T S n X.T Fy)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FY (2D array of shape n,p) - the euclidean Gradient
    Fyy (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    eta (2D array of shape n,n) - the eta, a vector in T_Y St
    S (2D array of shape n,n) - the overlap matrix
    

    """
    logger.debug('In log_eta_Fyy_Dn__tr_DnYFy')
    n, p = Y.shape
    Fyy_eta = Fyy @ np.reshape(eta, n*p, order='F')
    Fyy_eta__eta_Y_Fy = np.reshape(Fyy_eta, (n, p), order='F')
##    Fyy_eta__eta_Y_Fy -= eta @ Y.T @ Fy
    for nameD, D in tg_vecs(Y, S=S):
        D_Fyy_eta__eta_Y_Fy = np.trace(D.T @ Fyy_eta__eta_Y_Fy
                                       if S is None else
                                       D.T @ S @ Fyy_eta__eta_Y_Fy)
        logger.debug('For D=%s: D_Fyy_eta = %s\n'
                     '          D_grad      = %s\n',
                     nameD,
                     D_Fyy_eta__eta_Y_Fy,
                     np.trace(D.T @ S @ riemGrad))
        


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
