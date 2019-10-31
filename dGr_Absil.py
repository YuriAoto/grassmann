"""Functions related to the Newton-Grassmann method of Absil

Here, we consider the function f given by

f(x) = <0|ext>

where |0> is a Slater determinant and |ext> a correlated wave function.
The functions here are used to search critical points of this function
by the Newton method on the Grassmannian, as described in:

P-A Absil, R Mahony, and R Sepulchre, "Riemannian Geometry of Grassmann
Manifolds with a View on Algorithmic Computation", Acta App. Math. 80,
1999-220, 2004

Functions:
----------

calc_all_F
overlap_to_det
generate_lin_system
check_Newton_Absil_eq

"""
import math
import logging
import numpy as np
from scipy import linalg

import dGr_general_WF as genWF
from dGr_util import get_I, logtime

logger = logging.getLogger(__name__)

def _calc_fI(U, det_indices):
    """Calculate the contribution of U[det_indices,:] to f
    
    Parameters:
    -----------
    
    U (numpy.ndarray)
        The coefficients matrix
    
    det_indices (ndarray if int8)
        A list of subindices for the columns of U
    
    Return:
    -------
    
    det(U[det_indices, :])
    """
    return linalg.det(U[det_indices, :])

def _calc_G(U, det_indices, i, j):
    """Calculate the element ij of matrix G
    
    Behaviour:
    ----------
    
    Calculates the following determinant:
    det(U[det_indices, :] <-j- e_i )
    See _calc_H for details.
    
    Parameters:
    -----------
    
    U (numpy.ndarray)
        The coefficients matrix
    
    det_indices (ndarray if int8)
        A list of subindices
    
    i,j (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i )
    """
    if i not in det_indices:
        return 0.0
    if U.shape[1] == 1:
        return 1.0
    sign = 1 if (j + np.where(det_indices == i)[0][0]) % 2 == 0 else -1
    row_ind = np.array([x for x in det_indices       if x!=i], dtype=int)
    col_ind = np.array([x for x in range(U.shape[1]) if x!=j], dtype=int)
    return sign * linalg.det(U[row_ind[:,None],col_ind])

def _calc_H(U, det_indices, i, j, k, l):
    """Calculate the element ijkl of matrix H
    
    Behaviour:
    ----------
    
    Calculates the following determinant:
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
    -----------
    
    U (numpy.ndarray)
        The coefficients matrix
    
    det_indices (ndarray if int8)
        A list of subindices
    
    i,j,k,l (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    if j == l or i == k:
        return 0.0
    if i not in det_indices or k not in det_indices:
        return 0.0
    sign = 1.0 if ((i<k) == (j<l)) else -1.0
    if U.shape[1] == 2:
        return sign
    if (j + np.where(det_indices == i)[0][0] +
        l + np.where(det_indices == k)[0][0]) % 2 == 1:
        sign = -sign
    row_ind = np.array([x for x in det_indices       if (x!=i and x!=k)], dtype=int)
    col_ind = np.array([x for x in range(U.shape[1]) if (x!=j and x!=l)], dtype=int)
    return sign * linalg.det(U[row_ind[:,None],col_ind])

def calc_all_F(wf, U):
    """Calculate all F needed in an iteration
    
    Behaviour:
    ----------
    
    Calculate all possible _calc_fI for the spirreps
    and string indices.
    
    Parameters:
    -----------
    wf (dGr_general_WF.Wave_Function)
    
    U (list of np.ndarray)
        See overlap_to_det for the details
    
    Return:
    -------
    
    A wf.n_spirrep list of 1D np.ndarray, in the order of
    spirrep for the list and string_indices for the array.
    """
    F = []
    for spirrep in wf.spirrep_blocks(restricted = False):
        F.append(np.zeros(wf.n_strings(spirrep, U[spirrep].shape[1])))
        for I in wf.string_indices(spirrep=spirrep,
                                   only_this_occ = U[spirrep].shape[1]):
            F[-1][int(I)] = _calc_fI(U[spirrep], I.occ_orb)
    return F

def overlap_to_det(wf, U, F=None, assume_orth=True):
    """Calculate the overlap between wf and the determinant U
    
    Behaviour:
    ----------
    
    Calculates f(x) = <wf|U>, where wf is a normalised wave function
    and U is a Slater determinant, not necessarily on the same orbital
    basis of wf.
    
    Limitations:
    ------------
    
    Only for unrestricted cases (well, restricted cases work, if
    the parameters are given in a redundant unrestricted format)
    
    Parameters:
    -----------
    
    wf (dGr_general_WF.Wave_Function)
        The external wave function
    
    U (list of np.ndarray)
        An element of the Stiefel manifold that represents a
        Slater determinant. It should be a list such as
        [U_a^1, ..., U_a^g, U_b^1, ..., U_b^g]
        where U_sigma^i is the U for spin sigma (alpha=a or beta=b) and irrep i.
    
    F (list of np.ndarray, default = None)
        The result of calc_all_F. Calculate if None.
    
    assume_orth (bool, default = True)
        If not True, the result is divided by the normalisation of U.
        Remember that wf is assumed to be normalised already!
        
    Return:
    -------
    
    The float <wf|U>
    """
    if F is None:
        F = calc_all_F(wf, U)
    f = 0.0
    for I in wf.string_indices(no_occ_orb=True,
                               only_this_occ=genWF.Orbitals_Sets(
                                   list(map(lambda U_i: U_i.shape[1], U)))):
        f_contr = 1.0
        for spirrep, I_spirrep in enumerate(I):
            if len(I_spirrep) == 0:
                continue
            I_spirrep.wf = wf
            I_spirrep.spirrep = spirrep
            f_contr *= F[spirrep][int(I_spirrep)]
        f += wf[I] * f_contr
    if not assume_orth:
        for U_spirrep in U:
            f /= math.sqrt(linalg.det(np.matmul(U_spirrep.T, U_spirrep)))
    return f

def generate_lin_system(wf, U, lim_XC, F=None, with_full_H=True):
    """Generate the linear system for Absil's method
    
    Behaviour:
    ----------
    
    Calculate the matrices that define the main linear system of
    equations in Absil's method: X @ eta = C
    
    Limitations:
    ------------
    
    Only for unrestricted calculations
    
    Parameters:
    -----------
    
    U (list of np.ndarray)
    
    wf (dGr_general_WF.Wave_Function)
    
    F (list of np.ndarray, default=None)
    
        See overlap_to_det for the details of the above parameters
    
    lim_XC (list of int)
        Limits of each spirrep in the blocks of matrices X and C
    
    with_full_H (bool, default=True)
        If True, calculates and store the full matrix H
        If False, store only row-wise.
        Although storing the full matrix uses more memory,
        numpy's broadicasting should speed up the calculation
    
    Return:
    -------
    
    The tuple (X, C), such that eta satisfies X @ eta = C
    """
    if not with_full_H:
        raise NotImplementedError('with_full_H = False is not Implemented')
    if F is None:
        F = calc_all_F(U, wf)
    K = [U_spirrep.shape[0] for U_spirrep in U]
    n = [U_spirrep.shape[1] for U_spirrep in U]
    sum_Kn = sum([K[i] * n[i] for i in range(len(K))])
    sum_nn = sum([n[i]**2 for i in range(len(K))])
    X = np.zeros((sum_Kn + sum_nn, sum_Kn))
    C = np.zeros(sum_Kn + sum_nn)
    for spirrep_1 in wf.spirrep_blocks(restricted=False):
        logger.info('Starting spirrep_1 = %d', spirrep_1)
        logger.info('n_irrep = %d', wf.n_irrep)
        Pi = np.identity(K[spirrep_1]) - U[spirrep_1] @ U[spirrep_1].T
        logger.debug('Pi = %s', Pi)
        for I_1 in wf.string_indices(spirrep=spirrep_1):
            with logtime('Calc H, G') as T_1:
                logger.debug('At I_1 = %s', I_1)
                if len(I_1) != n[spirrep_1]:
                    continue
                H = np.zeros((K[spirrep_1], n[spirrep_1],
                              K[spirrep_1], n[spirrep_1]))
                G_1 = np.zeros((K[spirrep_1], n[spirrep_1]))
                for i in range(K[spirrep_1]):
                    for j in range(n[spirrep_1]):
                        H[i,j,i,j] = -F[spirrep_1][int(I_1)]
                        for k in range(i):
                            for l in range(j):
                                H[k,l,i,j] = _calc_H(U[spirrep_1],
                                                     I_1.occ_orb,
                                                     k, l, i, j)
                                H[k,j,i,l] = H[i,l,k,j] = -H[k,l,i,j]
                                H[i,j,k,l] = H[k,l,i,j]
                        #  G_1[i,j] = np.dot(H[:,0,i,j], U[spirrep_1][:,0])
                        G_1[i,j] = _calc_G(U[spirrep_1],
                                           I_1.occ_orb,
                                           i, j)
                logger.debug('current H:\n%s', H)
                logger.debug('current G:\n%s', G_1)
                H = Pi @ (np.multiply.outer(U[spirrep_1], G_1) - H)
                logger.debug('Pi (U G - H):\n%s', H)
            with logtime('Calc S') as T_2:
                S = 0.0
                logger.info('spirrep_1 = %d; I_1 = %s', spirrep_1, I_1)
                for I_full in wf.string_indices(
                        coupled_to=(genWF.Spirrep_Index(spirrep=spirrep_1,
                                                        I=I_1),)):
                    if list(map(len, I_full)) != list(map(lambda x: x.shape[1], U)):
                        continue
                    F_contr = 1.0
                    for spirrep_other, I_other in enumerate(I_full):
                        if spirrep_other != spirrep_1 and len(I_other) > 0:
                            F_contr *= F[spirrep_other][int(I_other)]
                    S += wf[I_full] * F_contr
                logger.debug('S = %s; H:\n%s', S, H)
            with logtime('Calc Xdiag, C') as T_3:
                X[lim_XC[spirrep_1]:lim_XC[spirrep_1 + 1],
                  lim_XC[spirrep_1]:lim_XC[spirrep_1 + 1]] += S * np.reshape(
                      H,
                      (K[spirrep_1] * n[spirrep_1],
                       K[spirrep_1] * n[spirrep_1]),
                      order='F').T
                G_1 -= F[spirrep_1][int(I_1)] * U[spirrep_1]
                logger.debug('S = %s; G_1:\n %s', S, G_1)
                C[lim_XC[spirrep_1]:lim_XC[spirrep_1 + 1]] += S * np.reshape(
                    G_1,
                    (K[spirrep_1] * n[spirrep_1],),
                    order='F')
            for spirrep_2 in wf.spirrep_blocks(restricted=False):
                if spirrep_2 <= spirrep_1:
                    continue
                logger.info('At spirrep_2 = %d', spirrep_2)
                with logtime('spirrep_2'):
                    G_2 = np.zeros((K[spirrep_2], n[spirrep_2]))
                    for I_2 in wf.string_indices(
                            spirrep=spirrep_2,
                            coupled_to=(genWF.Spirrep_Index(spirrep=spirrep_1,
                                                            I=I_1),)):
                        if len(I_2) != n[spirrep_2]:
                            continue
                        logger.debug('I_2 = %s', I_2)
                        for k in range(K[spirrep_2]):
                            for l in range(n[spirrep_2]):
                                G_2[k,l] = _calc_G(U[spirrep_2],
                                                   I_2.occ_orb,
                                                   k, l)
                        G_2 -= F[spirrep_2][int(I_2)] * U[spirrep_2]
                        S = 0.0
                        for I_full in wf.string_indices(
                                coupled_to=(genWF.Spirrep_Index(spirrep=spirrep_1,
                                                                I=I_1),
                                            genWF.Spirrep_Index(spirrep=spirrep_2,
                                                                I=I_2))):
                            if list(map(len, I_full)) != list(map(lambda x: x.shape[1], U)):
                                continue
                            F_contr = 1.0
                            for spirrep_other, I_other in enumerate(I_full):
                                if (wf.ref_occ[spirrep_other] > 0
                                    and spirrep_other != spirrep_1
                                    and spirrep_other != spirrep_2):
                                    F_contr *= F[spirrep_other][int(I_other)]
                            S += wf[I_full] * F_contr
                        logger.debug('G_1:\n%s', G_1)
                        logger.debug('G_2:\n%s', G_2)
                        logger.debug('S = %s', S)
                        X[lim_XC[spirrep_1]:lim_XC[spirrep_1 + 1],
                          lim_XC[spirrep_2]:lim_XC[spirrep_2 + 1]] -= S * np.reshape(
                              np.multiply.outer(G_1, G_2),
                              (K[spirrep_1] * n[spirrep_1],
                               K[spirrep_2] * n[spirrep_2]),
                              order='F')
            for spirrep_2 in wf.spirrep_blocks(restricted=False):
                if spirrep_1 > spirrep_2:
                    X[lim_XC[spirrep_1]:lim_XC[spirrep_1 + 1],
                      lim_XC[spirrep_2]:lim_XC[spirrep_2 + 1]] = \
                    X[lim_XC[spirrep_2]:lim_XC[spirrep_2 + 1],
                      lim_XC[spirrep_1]:lim_XC[spirrep_1 + 1]].T
    # Terms to guarantee orthogonality to U:
    prev_ij = lim_XC[-1]
    prev_kl = 0
    for spirrep, U_spirrep in enumerate(U):
        for i in range(n[spirrep]):
            logger.debug('Adding U^T:\n%s', U_spirrep.T)
            logger.debug('Position of U^T = [%s: %s + %s, %s: %s + %s',
                         prev_ij, prev_ij, n[spirrep],
                         prev_kl, prev_kl, K[spirrep])
            X[prev_ij: prev_ij + n[spirrep],
              prev_kl: prev_kl + K[spirrep]] = U_spirrep.T
            prev_ij += n[spirrep]
            prev_kl += K[spirrep]
    return X, C


def check_Newton_Absil_eq(wf, U, eta, eps = 0.001):
    """Check, numerically, if eta satisfies Absil equation
    
    Parameters:
    -----------
    
    wf (dGr_general_WF.Wave_Function)
    
    U (list of np.ndarray)

        See overlap_to_det for the details of the above parameters

    eta (list of np.ndarray)
        Possible solution of Absil equation, with same structure of U
    
    eps (float)
        Step size to calculate numerical derivatives (default = 0.001)
    
    Behaviour:
    ----------
    
    Print in the log (info and/or warnings) the main elements of
    Absil equation for the Newton step on the Grassmannian.
    """
    restricted = not isinstance(U, tuple)
    def get_orig_U(shift = None):
        if restricted:
            Ua = np.copy(U)
            Ub = np.copy(U)
        else:
            Ua, Ub = np.copy(U[0]), np.copy(U[1])
        if shift is not None:
            Ua = Ua + shift[0]
            Ub = Ub + shift[1]
        return Ua, Ub
    def calc_grad(this_U):
        do_orth = False
        if restricted:
            this_Ua = this_Ub = this_U
        else:
            this_Ua, this_Ub = this_U
        grad_a = np.zeros(pt_a.shape if restricted else this_U[0].shape)
        for i in range(grad_a.shape[0]):
            for j in range(grad_a.shape[1]):
                Ua = np.copy(this_Ua)
                Ua[i,j] = Ua[i,j] + eps
                ## If restricted, should it be Ua, Ua??:
                Ua_Ub = (orth(Ua), orth(this_Ub)) if do_orth else (Ua, this_Ub)
                grad_a[i,j] = distance_to_det(wf, Ua_Ub)
                Ua = np.copy(this_Ua)
                Ua[i,j] = Ua[i,j] - eps
                Ua_Ub = (orth(Ua), orth(this_Ub)) if do_orth else (Ua, this_Ub)
                grad_a[i,j] = (grad_a[i,j] - distance_to_det(wf, Ua_Ub))/(2*eps)
        if not restricted:
            grad_b = np.zeros(this_U[1].shape)
            for i in range(grad_b.shape[0]):
                for j in range(grad_b.shape[1]):
                    Ub = np.copy(this_Ub)
                    Ub[i,j] = Ub[i,j] + eps
                    ## If restricted, should it be Ua, Ua??
                    Ua_Ub = (orth(this_Ua), orth(Ub)) if do_orth else (this_Ua, Ub)
                    grad_b[i,j] = distance_to_det(wf, Ua_Ub)
                    Ub = np.copy(this_Ub)
                    Ub[i,j] = Ub[i,j] - eps
                    Ua_Ub = (orth(this_Ua), orth(Ub)) if do_orth else (this_Ua, Ub)
                    grad_b[i,j] = (grad_b[i,j] - distance_to_det(wf, Ua_Ub))/(2*eps)
        if restricted:
            return grad_a
        else:
            return grad_a, grad_b
    if restricted:
        eta_a = eta_b = eta
    else:
        eta_a, eta_b = eta
    logger.info('eta_a:\n%s', eta_a)
    logger.info('eta_b:\n%s', eta_b)
    Ua, Ub = get_orig_U()
    logger.info('Ua:\n%s', Ua)
    logger.info('Ub:\n%s', Ub)
    Proj_a = np.identity(Ua.shape[0]) - Ua @ Ua.T
    if not restricted:
        Proj_b = np.identity(Ub.shape[0]) - Ub @ Ub.T
    logger.info('Proj_a:\n%s', Proj_a)
    if not restricted:
        logger.info('Proj_b:\n%s', Proj_b)
    grad_a = calc_grad((Ua, Ub))
    if not restricted:
        grad_a, grad_b = grad_a
    logger.info('grad_a:\n%s', grad_a)
    if not restricted:
        logger.info('grad_b:\n%s', grad_b)
    RHS_a = Proj_a @ grad_a
    if not restricted:
        RHS_b = Proj_b @ grad_b
    logger.info('RHS of Absil equation for alpha (without the minus):\n%s', RHS_a)
    if not restricted:
        logger.info('RHS of Absil equation for beta (without the minus):\n%s', RHS_b)
    Ua, Ub = get_orig_U(shift = (eps*eta_a, eps*eta_b))
    Proj_plus_a = np.identity(Ua.shape[0]) - Ua @ linalg.inv(Ua.T @ Ua) @ Ua.T
    Dgrad_plus_a = calc_grad((Ua, Ub))
    if not restricted:
        Proj_plus_b = np.identity(Ub.shape[0]) - Ub @ linalg.inv(Ub.T @ Ub) @ Ub.T
        Dgrad_plus_a, Dgrad_plus_b = Dgrad_plus_a
    Ua, Ub = get_orig_U(shift = (-eps*eta_a, -eps*eta_b))
    Proj_minus_a = np.identity(Ua.shape[0]) - Ua @ linalg.inv(Ua.T @ Ua) @ Ua.T
    Dgrad_minus_a = calc_grad((Ua, Ub))
    if not restricted:
        Proj_minus_b = np.identity(Ub.shape[0]) - Ub @ linalg.inv(Ub.T @ Ub) @ Ub.T
        Dgrad_minus_a, Dgrad_minus_b = Dgrad_minus_a
    Dgrad_a = (Proj_plus_a @ Dgrad_plus_a - Proj_minus_a @ Dgrad_minus_a)/(2*eps)
    if not restricted:
        Dgrad_b = (Proj_plus_b @ Dgrad_plus_b - Proj_minus_b @ Dgrad_minus_b)/(2*eps)
    hess_a = (Dgrad_plus_a - Dgrad_minus_a)/(2*eps)
    if not restricted:
        hess_b = (Dgrad_plus_b - Dgrad_minus_b)/(2*eps)
    logger.info('U.T @ eta (alpha):\n%s', Ua.T @ eta_a)
    if not restricted:
        logger.info('U.T @ eta (beta):\n%s', Ub.T @ eta_b)
    logger.info('D(Pi grad f)(U)[eta], alpha:\n%s', Dgrad_a)
    if not restricted:
        logger.info('D(Pi grad f)(U)[eta], beta:\n%s', Dgrad_b)
    logger.info('LHS of Absil, alpha:\n%s', Proj_a @ Dgrad_a)
    if not restricted:
        logger.info('LHS of Absil, beta:\n%s', Proj_b @ Dgrad_b)
    logger.info('hess (d/dt grad f(y + eta)), alpha:\n%s', hess_a)
    if not restricted:
        logger.info('hess (d/dt grad f(y + eta)), beta:\n%s', hess_b)
    logger.info('LHS of Absil (using modified equation, '
                + 'without projector), alpha:\n%s',
                hess_a - eta_a @ Ua.T @ grad_a)
    if not restricted:
        logger.info('LHS of Absil (using modified equation, '
                    + 'without projector), beta:\n%s',
                    hess_b - eta_b @ Ub.T @ grad_b)
    logger.info('LHS of Absil (using modified equation), alpha:\n%s',
                Proj_a @ hess_a - Proj_a @ eta_a @ Ua.T @ grad_a)
    if not restricted:
        logger.info('LHS of Absil (using modified equation), beta:\n%s',
                    Proj_b @ hess_b - Proj_b @ eta_b @ Ub.T @ grad_b)
