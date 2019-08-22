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

def check_Newton_Absil_eq(dist_to_det, U, eta, eps = 0.001):
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
                grad_a[i,j] = dist_to_det(Ua_Ub)
                Ua = np.copy(this_Ua)
                Ua[i,j] = Ua[i,j] - eps
                Ua_Ub = (orth(Ua), orth(this_Ub)) if do_orth else (Ua, this_Ub)
                grad_a[i,j] = (grad_a[i,j] - dist_to_det(Ua_Ub))/(2*eps)
        if not restricted:
            grad_b = np.zeros(this_U[1].shape)
            for i in range(grad_b.shape[0]):
                for j in range(grad_b.shape[1]):
                    Ub = np.copy(this_Ub)
                    Ub[i,j] = Ub[i,j] + eps
                    ## If restricted, should it be Ua, Ua??
                    Ua_Ub = (orth(this_Ua), orth(Ub)) if do_orth else (this_Ua, Ub)
                    grad_b[i,j] = dist_to_det(Ua_Ub)
                    Ub = np.copy(this_Ub)
                    Ub[i,j] = Ub[i,j] - eps
                    Ua_Ub = (orth(this_Ua), orth(Ub)) if do_orth else (this_Ua, Ub)
                    grad_b[i,j] = (grad_b[i,j] - dist_to_det(Ua_Ub))/(2*eps)
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
