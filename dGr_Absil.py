"""Functions related to the Newton-Grassmann method of Absil

Here, we consider the function f is given by 

f(x) = <0|ext>

where |0> is a Slater determinant and |ext> a correlated wave function

"""
import math
import logging
import numpy as np
from numpy import linalg

import dGr_general_WF as genWF
from dGr_util import get_I

logger = logging.getLogger(__name__)

def _calc_fI(U, det_indices):
    """Calculate the contribution of U[det_indices,:] to f

    Parameters:
    U (numpy.ndarray)     the coefficients matrix
    det_indices (list)    a list of subindices

    Return:
    det(U[det_indices, :])
    """
    return linalg.det(U[det_indices, :])

def _calc_G(U, det_indices, i, j):
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

def _calc_H(U, det_indices, i, j, k, l):
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

def distance_to_det(wf, U, thresh_cI=1E-10, assume_orth = False):
    """Calculates the distance to the determinant U

    See dGr_FCI_Molpro.Molpro_FCI_Wave_Function.distance_to_det
    """
    if isinstance(U, tuple):
        Ua, Ub = U
    else:
        Ua = Ub = U
    for det in wf.all_dets():
        if abs(det.c) < thresh_cI:
            continue
        if isinstance(det, genWF.Ref_Det):
            f0_a = _calc_fI(Ua, get_I(wf.n_alpha))
            f0_b = _calc_fI(Ub, get_I(wf.n_beta))
            f = det.c * f0_a * f0_b
        else:
            try:
                f
            except NameError:
                raise NameError('First determinant has to be genWF.Ref_Det!')
        if isinstance(det, genWF.Singly_Exc_Det):
            if det.spin > 0:
                f += det.c * _calc_fI(Ua, get_I(wf.n_alpha, det.i, det.a)) * f0_b
            else:
                f += det.c * _calc_fI(Ub, get_I(wf.n_beta, det.i, det.a)) * f0_a
        elif isinstance(det, genWF.Doubly_Exc_Det):
            if det.spin_ia * det.spin_jb < 0:
                f += (det.c
                      * _calc_fI(Ua, get_I(wf.n_alpha, det.i, det.a))
                      * _calc_fI(Ub, get_I(wf.n_beta,  det.j, det.b)))
            elif det.spin_ia > 0:
                f += (det.c * f0_b
                      * _calc_fI(Ua, get_I(wf.n_alpha,
                                                [det.i, det.j],
                                                sorted([det.a, det.b]))))
            else:
                f += (det.c * f0_a
                      * _calc_fI(Ub, get_I(wf.n_beta,
                                                [det.i, det.j],
                                                sorted([det.a, det.b]))))
    if not assume_orth:
        Da = linalg.det(np.matmul(Ua.T, Ua))
        Db = linalg.det(np.matmul(Ub.T, Ub))
        f /= math.sqrt(Da * Db)
    return f


def get_ABC_matrices(wf, U, thresh_cI=1E-10):
    """Calculates the arrays A,B,C needed for Absil's algorithm

    See dGr_FCI_Molpro.Molpro_FCI_Wave_Function.distance_to_det
    """
    if isinstance(U, tuple):
        Ua, Ub = U
        restricted = False
    else:
        Ua = Ub = U
        restricted = True
    K, na = Ua.shape
    nb = Ub.shape[1]
    if na != nb:
        raise NotImplementedError('We need both Ua and Ub with same shape!')
    if K != wf.orb_dim:
        raise ValueError('First dimension of U must match orb_dim!')
    if na != wf.n_alpha:
        raise ValueError('Second dimension of Ua must match the n_alpha!')
    if nb != wf.n_beta:
        raise ValueError('Second dimension of Ua must match the n_beta!')
    ABshape = Ua.shape*2
    Aa_a = np.zeros(ABshape)
    Aa_b = np.zeros(ABshape)
    Ba = np.zeros(ABshape)
    Ca = np.zeros(Ua.shape)
    if not restricted:
        Ab_a = np.zeros(ABshape)
        Ab_b = np.zeros(ABshape)
        Bb = np.zeros(ABshape)
        Cb = np.zeros(Ub.shape)
    for det in wf.all_dets():
        if abs(det.c) < thresh_cI:
            continue
        if isinstance(det, genWF.Ref_Det):
            Ia = range(wf.n_alpha)
            Ib = range(wf.n_beta)
        elif isinstance(det, genWF.Singly_Exc_Det):
            if det.spin > 0:
                Ia = get_I(wf.n_alpha, det.i, det.a)
                Ib = range(wf.n_beta)
            else:
                Ia = range(wf.n_alpha)
                Ib = get_I(wf.n_beta, det.i, det.a)
        elif isinstance(det, genWF.Doubly_Exc_Det):
            if det.spin_ia * det.spin_jb < 0:
                Ia = get_I(wf.n_alpha, det.i, det.a)
                Ib = get_I(wf.n_beta,  det.j, det.b)
            elif det.spin_ia > 0:
                Ia = get_I(wf.n_alpha,
                           [det.i, det.j],
                           sorted([det.a, det.b]))
                Ib = range(wf.n_beta)
            else:
                Ia = range(wf.n_alpha)
                Ib = get_I(wf.n_beta,
                           [det.i, det.j],
                           sorted([det.a, det.b]))
        Fa = _calc_fI(Ua, Ia)
        Fb = _calc_fI(Ub, Ib)
        Proj_a = np.identity(K)
        Ga = np.zeros(Ua.shape)
        if not restricted:
            Gb = np.zeros(Ub.shape)
            Proj_b = np.identity(K)
        for k in range(K):
            for l in range(na):
                Hkl_a = np.zeros(Ua.shape)
                if not restricted:
                    Hkl_b = np.zeros(Ub.shape)
                for i in range(K):
                    Proj_a[i,k] -= np.dot(Ua[i,:], Ua[k,:])
                    if not restricted:
                        Proj_b[i,k] -= np.dot(Ub[i,:], Ub[k,:])
                    for j in range(na):
                        if j != l:
                            Hkl_a[i,j] = _calc_H(Ua, Ia, i, j, k, l)
                            if not restricted:
                                Hkl_b[i,j] = _calc_H(Ub, Ib, i, j, k, l)
                        else:
                            Ba[i,j,k,l] += det[0] * Fa * Fb * Proj_a[i,k]
                            if not restricted:
                                Bb[i,j,k,l] += det[0] * Fa * Fb * Proj_b[i,k]
                Ga[k,l] = _calc_G(Ua, Ia, k, l)
                Gb[k,l] = _calc_G(Ub, Ib, k, l)
#                 Ga[k,l] += np.dot(Hkl_a[:,na-1], Ua[:,na-1])
#                 if restricted:
#                     Gb[k,l] = _calc_G(Ub, Ib, k, l)
#                 else:
#                     Gb[k,l] += np.dot(Hkl_b[i,nb-1], Ub[i,nb-1])
                Aa_a[k,l,:,:] += det[0] * Fb * np.matmul(Proj_a, Hkl_a)
                if not restricted:
                    Ab_b[k,l,:,:] += det[0] * Fa * np.matmul(Proj_b, Hkl_b)
        det_G_FU = det[0] * (Ga - Fa * Ua)
        Ca += Fb * det_G_FU
        Aa_b += np.multiply.outer(det_G_FU, Gb)
        if not restricted:
            det_G_FU = det[0] * (Gb - Fb * Ub)
            Cb += Fa * det_G_FU
            Ab_a += np.multiply.outer(det_G_FU, Ga)
    if restricted:
        return (Aa_a, Aa_b), Ba, Ca
    else:
        return ((Aa_a, Aa_b),
                (Ab_a, Ab_b)), (Ba, Bb), (Ca, Cb)

def generate_lin_system(A, B, C, U):
    """Given the matrices A, B, C, reshape to get the linear system
    
    Behaviour:
    
    From the matrices A, B, C, reshape them to get the
    matrix (a 2d array) B_minus_A and the vector (a 1D array) C,
    such that
    
    B_minus_A @ eta = C
    
    is the linear system to calculate eta, in the step of
    Absil's Newton-Grassmann optimisation
    
    Limitations:
    
    It assumes that the number of alpha and beta electrons are the same
    
    Parameters:
    
    A   (2-tuple of 4D array, for rescricted)
        (A^a_a, A^a_b) = (A_same, A_mix)
        
        (2-tuple of 2-tuples of 4D arrays, for unrestricted)
        ((A^a_a, A^a_b),
         (A^b_a, A^b_b))
    
    B   (4D array, for rescricted)
        B^a
        (2-tuple of 4D array, for unrescricted)
        (B^a, B^b)
    
    C   (2D array, for rescricted)
        C^a
        (2-tuple of 2D array, for unrescricted)
        (C^a, C^b)
    
    U   (2D array, for rescricted)
        Ua
        (2-tuple of 2D array, for unrescricted)
        (Ua, Ub)
        The transformation matrix in this iteration
        
    Return:
    
    The 2D array A_minus_B and the 1D array Cn
    """
    restricted = not isinstance(C, tuple)
    # n = nubmer of electrons
    # K = nubmer of orbitals
    # nK = nubmer of electrons times the number of spatial orbitals
    if restricted:
        K = A[0].shape[0]
        n = A[0].shape[1]
    else:
        K = A[0][0].shape[0]
        n = A[0][0].shape[1]
    nK = n*K
    # test all entries and shapes?
    if restricted:
        Cn = np.zeros(nK + n)
        Cn[:nK] = np.ravel(C,order='C')
    else:
        Cn = np.zeros(2*(nK + n))
        Cn[:2*nK] = np.concatenate((np.ravel(C[0], order='C'),
                                    np.ravel(C[1], order='C')))
    if restricted:
        B_minus_A = np.zeros((nK + n, nK))
        B_minus_A[:nK,:] = np.reshape(B, (nK, nK), order='C')
        B_minus_A[:nK,:] -= np.reshape(A[0], (nK, nK), order='C')
        B_minus_A[:nK,:] -= np.reshape(A[1], (nK, nK), order='C')
        # --> Extra term due to normalisation
        B_minus_A[:nK,:] += 2*np.multiply.outer(Cn[:nK], np.ravel(U, order='C'))
        # --> Terms to guarantee orthogonality to U
        B_minus_A[nK:,:] += U.T
    else:
        B_minus_A = np.zeros((2*(nK + n), 2*nK))
        B_minus_A[:nK, :nK] = np.reshape(B[0],
                                         (nK, nK),
                                         order='C')
        B_minus_A[:nK, :nK] -= np.reshape(A[0][0],
                                          (nK, nK),
                                          order='C')
        B_minus_A[nK:2*nK, nK:] = np.reshape(B[1],
                                         (nK, nK),
                                         order='C')
        B_minus_A[nK:2*nK, nK:] -= np.reshape(A[1][1],
                                          (nK, nK),
                                          order='C')
        B_minus_A[:nK, nK:] -= np.reshape(A[0][1],
                                          (nK, nK),
                                          order='C')
        B_minus_A[nK:2*nK, :nK] -= np.reshape(A[1][0],
                                          (nK, nK),
                                          order='C')
        # --> Extra term due to normalisation
        B_minus_A[:nK, :nK] += np.multiply.outer(Cn[:nK],
                                                 np.ravel(U[0], order='C'))
        B_minus_A[:nK, nK:] += np.multiply.outer(Cn[:nK],
                                                 np.ravel(U[1], order='C'))
        B_minus_A[nK:2*nK, :nK] += np.multiply.outer(Cn[nK:2*nK],
                                                 np.ravel(U[0], order='C'))
        B_minus_A[nK:2*nK, nK:] += np.multiply.outer(Cn[nK:2*nK],
                                                 np.ravel(U[1], order='C'))
        # --> Terms to guarantee orthogonality to U
        ## Can be made more efficiente if order = 'F' is used!!!
        for iel in range(n):
            for iorb in range(K):
                B_minus_A[2*nK     + iel,      iel + n*iorb] = U[0][iorb,iel]
                B_minus_A[2*nK + n + iel, nK + iel + n*iorb] = U[1][iorb,iel]
    return B_minus_A, Cn

    

def check_Newton_Absil_eq(wf, U, eta, eps = 0.001):
    """Check, numerically, if eta satisfies Absil equation

    Parameters:
    U (2D numpy.array or 2-tuple of 2D numpy.array)
        Current Slater determinant. See distance_to_det.
    eta (2D numpy.array or 2-tuple of 2D numpy.array)
        Possible solution of Absil equation

    Behaviour:
    Print in the log (info and/or warnings) main elements of
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
    logger.info('Proj_a:\n%s', str(Proj_a))
    if not restricted:
        logger.info('Proj_b:\n%s', str(Proj_b))
    grad_a = calc_grad((Ua, Ub))
    if not restricted:
        grad_a, grad_b = grad_a
    logger.info('grad_a:\n%s', str(grad_a))
    if not restricted:
        logger.info('grad_b:\n%s', str(grad_b))
    RHS_a = Proj_a @ grad_a
    if not restricted:
        RHS_b = Proj_b @ grad_b
    logger.info('RHS of Absil equation for alpha (without the minus):\n%s', str(RHS_a))
    if not restricted:
        logger.info('RHS of Absil equation for beta (without the minus):\n%s', str(RHS_b))
    Ua, Ub = get_orig_U(shift = (eps*eta_a, eps*eta_b))
    Proj_plus_a = np.identity(Ua.shape[0]) - Ua @ np.linalg.inv(Ua.T @ Ua) @ Ua.T
    Dgrad_plus_a = calc_grad((Ua, Ub))
    if not restricted:
        Proj_plus_b = np.identity(Ub.shape[0]) - Ub @ np.linalg.inv(Ub.T @ Ub) @ Ub.T
        Dgrad_plus_a, Dgrad_plus_b = Dgrad_plus_a
    Ua, Ub = get_orig_U(shift = (-eps*eta_a, -eps*eta_b))
    Proj_minus_a = np.identity(Ua.shape[0]) - Ua @ np.linalg.inv(Ua.T @ Ua) @ Ua.T
    Dgrad_minus_a = calc_grad((Ua, Ub))
    if not restricted:
        Proj_minus_b = np.identity(Ub.shape[0]) - Ub @ np.linalg.inv(Ub.T @ Ub) @ Ub.T
        Dgrad_minus_a, Dgrad_minus_b = Dgrad_minus_a
    Dgrad_a = (Proj_plus_a @ Dgrad_plus_a - Proj_minus_a @ Dgrad_minus_a)/(2*eps)
    if not restricted:
        Dgrad_b = (Proj_plus_b @ Dgrad_plus_b - Proj_minus_b @ Dgrad_minus_b)/(2*eps)
    hess_a = (Dgrad_plus_a - Dgrad_minus_a)/(2*eps)
    if not restricted:
        hess_b = (Dgrad_plus_b - Dgrad_minus_b)/(2*eps)
    logger.info('U.T @ eta (alpha):\n%s', str(Ua.T @ eta_a))
    if not restricted:
        logger.info('U.T @ eta (beta):\n%s', str(Ub.T @ eta_b))
    logger.info('D(Pi grad f)(U)[eta], alpha:\n%s', str(Dgrad_a))
    if not restricted:
        logger.info('D(Pi grad f)(U)[eta], beta:\n%s', str(Dgrad_b))
    logger.info('LHS of Absil, alpha:\n%s', str(Proj_a @ Dgrad_a))
    if not restricted:
        logger.info('LHS of Absil, beta:\n%s', str(Proj_b @ Dgrad_b))
    logger.info('hess (d/dt grad f(y + eta)), alpha:\n%s', str(hess_a))
    if not restricted:
        logger.info('hess (d/dt grad f(y + eta)), beta:\n%s', str(hess_b))
    logger.info('LHS of Absil (using modified equation, '
                + 'without projector), alpha:\n%s',
                str(hess_a - eta_a @ Ua.T @ grad_a))
    if not restricted:
        logger.info('LHS of Absil (using modified equation, '
                    + 'without projector), beta:\n%s',
                    str(hess_b - eta_b @ Ub.T @ grad_b))
    logger.info('LHS of Absil (using modified equation), alpha:\n%s',
                str(Proj_a @ hess_a - Proj_a @ eta_a @ Ua.T @ grad_a))
    if not restricted:
        logger.info('LHS of Absil (using modified equation), beta:\n%s',
                    str(Proj_b @ hess_b - Proj_b @ eta_b @ Ub.T @ grad_b))
