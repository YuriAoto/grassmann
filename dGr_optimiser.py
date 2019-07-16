"""Optimiser of the distance to an external wave function

History
    Aug 2018 - Start
    Mar 2019 - Organise and comment the code
               Add to git

Yuri
"""
import copy
import math
import sys
import logging

from collections import namedtuple
import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


class Results(namedtuple('Results',
                         ['f',
                          'U',
                          'norm',
                          'last_iteration',
                          'converged',
                          'n_pos_H_eigVal'])):
    """A namedtuple for the results of optimisation
    
    Attributes:
    f                  The value of the function (the overlap) in the end of the procedure
    U                  (Ua, Ub), the transformation matrices for the optimised orbitals
    norm               norm of vectors that should vanish at convergence
    last_iteration     number of iterations
    converged          True or False, to indicate convergence
    n_pos_HeigVal      number of positive Hessian eigenvalues
    """
    __slots__ = ()

def optimise_distance_to_FCI(fci_wf,
                             construct_Jac_Hess,
                             str_Jac_Hess,
                             calc_wf_from_z,
                             transform_wf,
                             max_iter = 20,
                             f_out = sys.stdout,
                             ini_U = None,
                             thrsh_Z = 1.0E-8,
                             thrsh_J = 1.0E-8):
    """Find a single Slater determinant that minimise the distance to fci_wf
    
    Behaviour:
    
    This function uses the Newton-Raphson method for the optimisation,
    using the exponential parametrisation.
    Because the Jacobian and Hessian are simple to calculate only when
    the FCI wave function is on the same orbital basis of the slater determinant
    of the optimisation step, a full transformation of wave function fci_wf is
    done in every step.
    Thus, the initial wave function fci_wf has to have all determinants of the
    full CI wave function, even if the coefficients are zero.
    
    See Molpro_FCI_Wave_Function for an example of the classes and
    functions used.
    
    The argument fci_wf is not changed.
    
    Parameters:
    
    fci_wf   an instance of a class that represents the fci wave function 
             (or a wave function with the structure of a FCI wave function).
             
    
    construct_Jac_Hess   a method that receives fci_wf and contructs the
                         Jacobian and the Hessian of the function
                         f(x) = <x,fci_wf>, where x is a Slater determinant
    
    str_Jac_Hess       a method that returns str of the Jacobian and the Hessian
    
    calc_wf_from_z     a method that calculates the wave function from 
                       a parametrisation z
    
    transform_wf      a method to transforma the wave function to a new basis
    
    max_iter   the maximum number of iterations in the optimisation
               (default = 20)
    
    f_out   the output stream (default = sys.stdout)
    
    ini_U   if not None, it should be a two elements tuple with
            initial transformation for alpha and beta orbitals.
            if None, Identity is used as initial transformation.
            (default = None)
    
    thrsh_Z    Convergence threshold for Z vector (defult = 1.0E-8)
    thrsh_J    Convergence threshold for Jacobian (defult = 1.0E-8)
    
    Returns:
    
    The namedtuple Results. Some particularities are:
    
    If the first determinant is not the one with largest coefficients,
    f is a tuple (f, det_maxC), with the first entry as the coefficient of
    the first determinant (that is what we are optimising), and det_maxC is
    the (complete) determinant with the largest coefficient.
    
    
    """
    def get_i_max_coef(wf):
        max_coef = 0.0
        i_max_coef = -1
        for i,c in enumerate(wf.determinants):
            if abs(c[0]) > max_coef:
                max_coef = abs(c[0])
                i_max_coef = i
        return i_max_coef

    converged = False
    try_uphill = False
    i_pos_eigVal = -1
    i_iteration = 0
    if ini_U is None:
        Ua = np.identity(fci_wf.orb_dim)
        Ub = np.identity(fci_wf.orb_dim)
        cur_wf = copy.copy(fci_wf)
    else:
        Ua = ini_U[0]
        Ub = ini_U[1]
        cur_wf = transform_wf(fci_wf, Ua, Ub)

    f_out.write('{0:<5s}  {1:<23s}  {2:<11s}  {3:<11s}  {4:<11s}\n'.\
                format('it.', 'f', '|z|', '|Jac|', '[det. largest coef.]'))

    for i_iteration in range(max_iter):
        logger.info('Starting iteration %d',
                    i_iteration)
        if  logger.level <= logging.DEBUG:
            logger.debug('Wave function:\n' + str(cur_wf))

        Jac, Hess = construct_Jac_Hess(cur_wf)
        logger.info('Hessian and Jacobian are built\n')
        if logger.level <= 1:
            logger.log(1, str_Jac_Hess(Jac, Hess, cur_wf))

        if False: # Check Jac and Hessian
            num_Jac, num_Hess = construct_Jac_Hess(cur_wf,
                                                   analytic=False)
            diff = 0.0
            for i,x in enumerate(num_Jac):
                diff += abs(x-Jac[i])
            logger.debug('Sum abs(diff(Jac, numJac)) = {0:10.6e}'.\
                         format(diff))
            diff = 0.0
            for i1, x1 in enumerate(num_Hess):
                for i2, x2 in enumerate(x1):
                    diff += abs(x2-Hess[i1][i2])
            logger.debug('Sum abs(diff(Hess, numHess)) = {0:10.6e}'.\
                         format(diff))
            logger.debug('Analytic:\n%s',
                         str_Jac_Hess(Jac, Hess, cur_wf))
            logger.debug('Numeric:\n%s',
                         str_Jac_Hess(num_Jac, num_Hess, cur_wf))

        eig_val, eig_vec = linalg.eigh(Hess)
        Hess_has_pos_eig = False
        Hess_dir_with_posEvec = None
        n_pos_eigV = 0
        for i,eigVal in enumerate(eig_val):
            if eigVal > 0.0:
                n_pos_eigV += 1
                Hess_has_pos_eig = True
                Hess_dir_with_posEvec = np.array(eig_vec[:,i])
        if Hess_has_pos_eig:
            logger.warning('Hessian has positive eigenvalue.')

        normJ = 0.0
        for i in Jac:
            normJ += i**2
        normJ = math.sqrt(normJ)

        if not try_uphill:
            # Newton step
            Hess_inv = linalg.inv(Hess)
            logger.info('Hessian is inverted.')
            z = -np.matmul(Hess_inv, Jac)
            logger.info('z vector has been calculated by Newton step.')
        else:
            # Look maximum in the direction of eigenvector of Hess with pos
            #
            # An attempt to automatically set gamma had failed tests...
            #den = np.matmul(np.matmul(Jac.transpose(), Hess), Jac)
            #gamma = 0.0
            #for j in Jac:
            #    gamma += j**2
            #gamma = -gamma/den
            gamma = 0.5
            logger.info('Calculating z vector by Gradient descent;\n'
                        + 'gamma = {0:.5f}\n'.format(gamma))
            max_c0 = cur_wf.determinants[0][0]
            max_i0 = 0
            if logger.level <= 15:
                logger.log(15, 'Current C0: {0:.4f}'.format(max_c0))
            for i in range(6):
                tmp_wf, tmp_Ua, tmp_Ub = calc_wf_from_z(i*gamma*Hess_dir_with_posEvec, cur_wf,
                                                        just_C0=True)
                this_c0 = tmp_wf.determinants[0][0]
                logger.debug('Attempting for i=%d: C0 = %f', i, this_c0)
                if abs(max_c0) < abs(this_c0):
                    max_c0 = this_c0
                    max_i0 = i
            z = max_i0*gamma*Hess_dir_with_posEvec
            logger.info('z vector obtained: %d*gamma*Hess_dir_with_posEvec',
                        max_i0)
            try_uphill = False
        normZ = 0.0
        for i in z:
            normZ += i**2
        normZ = math.sqrt(normZ)
        i_max_coef = get_i_max_coef(cur_wf)
        if i_max_coef > 0:
            f_maxC = '({1:<9.7f})'.format(cur_wf.determinants[i_max_coef][0])
            det_maxC = str(cur_wf.determinants[i_max_coef])
        else:
            f_maxC = det_maxC = ''
        f_out.write('{0:<5d}  {1:<10.8f} {2:<12s}  {3:<11.8f}  {4:<11.8f}  {5:<s}\n'.\
                    format(i_iteration,
                           cur_wf.determinants[0][0],
                           f_maxC,
                           math.sqrt(normZ),
                           math.sqrt(normJ),
                           det_maxC))
        f_out.flush()
        logger.info('Norm of z vector: {0:8.5e}'.\
                    format(normZ))
        logger.info('Norm of J vector: {0:8.5e}'.\
                    format(normJ))
        if (not try_uphill
            and normJ < thrsh_J
            and normZ < thrsh_Z):
            if not Hess_has_pos_eig:
                converged = True
                break
            else:
                try_uphill = True
        logger.info('Calculating new basis and transforming wave function.')
        cur_wf, cur_Ua, cur_Ub = calc_wf_from_z(z, cur_wf)
        logger.info('New basis and wave function have been calculated!')
        Ua = np.matmul(Ua, cur_Ua)
        Ub = np.matmul(Ub, cur_Ub)

    if i_max_coef > 0:
        f_final = (cur_wf.determinants[0][0], cur_wf.determinants[i_max_coef])
    else:
        f_final = cur_wf.determinants[0][0]
    return Results(f = f_final,
                   U = (Ua, Ub),
                   norm = (normZ, normJ),
                   last_iteration = i_iteration,
                   converged = converged,
                   n_pos_H_eigVal = n_pos_eigV)


def optimise_distance_to_CI(ci_wf
                            max_iter = 20,
                            f_out = sys.stdout,
                            ini_U = None,
                            thrsh_eta = 1.0E-8,
                            thrsh_C = 1.0E-8,
                            only_C = False,
                            only_eta = False):
    """Find a single Slater determinant that maximises the overlap to ci_wf
    
    Behaviour:
    
    This function uses the Newton method on the Grassmannian, as discussed in
    
    P-A Absil, R Mahony, and R Sepulchre, "Riemannian Geometry of Grassmann
    Manifolds with a View on Algorithmic Computation", Acta App. Math. 80,
    1999-220, 2004
    
    In this method we do not change the representation of the external wave
    function (ci_wf), given with a (eventually truncated) configurations
    interaction parametrisation.
    
    One of the steps is to solve the linear system (A-B) @ eta = C.
    The convergence is obtained checkin the norm of eta and C
    
    Parameters:
    
    ci_wf      an instance of a class that represents the external wave function.
               Such class must have some attributes, that are properly explained below.
    
    max_iter   the maximum number of iterations in the optimisation
               (default = 20)
    
    f_out   the output stream (default = sys.stdout)
    
    ini_U   if not None, it should be a two elements tuple with
            initial transformation for alpha and beta orbitals,
            from the basis of the ci_wf to the basis of the slater
            determinant
            if None, Identity is used as initial transformation.
            (default = None)
    
    thrsh_eta      Convergence threshold for eta vector (default = 1.0E-8)
    thrsh_C        Convergence threshold for C vector (default = 1.0E-8)
    only_C         If True, stops iterations if C vector passes in the convergence
                   test, irrespective of the norm of eta (and does not go further
                   in the iteration) (default = False)
    only_eta       If True, stops iterations if eta vector passes in the convergence
                   test, irrespective of the norm of C (default = False)
    
    Attributes that ci_wf must have:
    n_alpha, n_beta (int)              the number of alha ana beta electrons
    dim_orb_space (int)                dimension of the orbital space
    get_ABC_matrices (method)          given a transformation U to another orbital basis,
                                       return the matrices A, B and C, that contain
                                       the elements to calculate the Newton step
    distance_to_det (method)           given a transformation U to another orbital basis,
                                       calculates the distance between the external
                                       wave function to the slater determinant associated to
                                       U
    
    Returns:
    
    The namedtuple Results. Some particularities are:
    
    norm is a 2-tuple with norm_eta and norm_C.
    converged is also a 2-tuple, showing convergence for
    C and eta.
    n_pos_H_eigVal is not set yet!! We have to know how to calculate this...
    
    """
    n_pos_eigV = None
    converged_eta = False
    converged_C = False
    f = None
    f_out.write('{0:<5s}  {1:<11s}  {2:<11s}  {3:<11s}\n'.\
                format('it.', 'f', '|eta|', '|C|'))
    if only_C and only_eta:
        raise ValueError('Do not set both only_C and only_eta to True!')
    if ini_U is not None:
        U = ini_U
        if ini_U[0].shape[0] != ci_wf.dim_orb_space:
            raise ValueError ('ini_U alpha has shape {0:} but dim of orbital space of ci_wf is {1:}'.\
                              format(ini_U[0].shape[0], ci_wf.dim_orb_space))
        if ini_U[1].shape[0] != ci_wf.dim_orb_space:
            raise ValueError ('ini_U beta has shape {0:} but dim of orbital space of ci_wf is {1:}'.\
                              format(ini_U[1].shape[0], ci_wf.dim_orb_space))
        if ini_U[0].shape[1] != ci_wf.n_alpha:
            raise ValueError ('ini_U alpha has shape {0:} but n_alpha in ci_wf is {1:}'.\
                              format(ini_U[0].shape[1], ci_wf.n_alpha))
        if ini_U[1].shape[1] != ci_wf.n_beta:
            raise ValueError ('ini_U beta has shape {0:} but n_beta in ci_wf is {1:}'.\
                              format(ini_U[1].shape[1], ci_wf.n_beta))
    else:
        U = (identity(ci_wf.dim_orb_space)[:,:ci_wf.n_alpha],
             identity(ci_wf.dim_orb_space)[:,:ci_wf.n_beta])
    for i_iteration in range(max_iter):
        A, B, C = ci_wf.get_ABC_matrices(U)
        if logger.level <= logging.DEBUG:
            logger.debug('matrix A(alpha, alpha):\n' + str_matrix(A[0][0]))
            logger.debug('matrix A(alpha, beta):\n' + str_matrix(A[0][1]))
            logger.debug('matrix A(beta , alpha):\n' + str_matrix(A[1][0]))
            logger.debug('matrix A(beta , beta):\n' + str_matrix(A[1][1]))
            logger.debug('matrix B(alpha):\n' + str_matrix(B[0]))
            logger.debug('matrix B(beta):\n' + str_matrix(B[1]))
            logger.debug('matrix C(alpha):\n' + str_matrix(C[0]))
            logger.debug('matrix C(beta):\n' + str_matrix(C[1]))
        norm_C = 0.0
        for line in C[0]:
            for C_ij in line:
                norm_C += C_ij**2
        for line in C[1]:
            for C_ij in line:
                norm_C += C_ij**2
        norm_C = math.sqrt(norm_C)
        logger.info('norm of C matrix: {.5e}'.format(norm_C))
        if norm_C < thrsh_C:
            converged_C = True
            if only_C:
                break
        else:
            converged_C = False
        B_minus_A, C = _generates_Absil_lin_system(A, B, C)
        if logger.level <= logging.DEBUG:
            logger.debug('lin_system, B-A:\n' + str_matrix(B_minus_A))
            logger.debug('lin_system, C:\n' + str_matrix(C))
        eta = linalg.solve(B_minus_A, C)
        if logger.level <= logging.DEBUG:
            logger.debug('lin_system (solution), eta:\n' + str_matrix(eta))
        Usvd_a, SGMsvd_a, VTsvd_a = linalg.svd(eta[:ci_wf.n_alpha])
        Usvd_b, SGMsvd_b, VTsvd_b = linalg.svd(eta[ci_wf.n_alpha:])
        if logger.level <= logging.DEBUG:
            logger.debug('SVD results, Usvd_a:\n'   + str_matrix(Usvd_a))
            logger.debug('SVD results, SGMsvd_a:\n' + str_matrix(SGMsvd_a))
            logger.debug('SVD results, VTsvd_a:\n'  + str_matrix(VTsvd_a))
            logger.debug('SVD results, Usvd_b:\n'   + str_matrix(Usvd_b))
            logger.debug('SVD results, SGMsvd_b:\n' + str_matrix(SGMsvd_b))
            logger.debug('SVD results, VTsvd_b:\n'  + str_matrix(VTsvd_b))
        U = (U[0] @ VTsvd_a @ np.cos(SGMsvd_a) + Usvd_a @ np.sin(SGMsvd_a),
             U[1] @ VTsvd_b @ np.cos(SGMsvd_b) + Usvd_b @ np.sin(SGMsvd_b))
        if logger.level <= logging.DEBUG:
            logger.debug('new U_a:\n' + str_matrix(U[0]))
            logger.debug('new U_b:\n' + str_matrix(U[1]))
        U = tuple(map(linalg.orth, U))
        if logger.level <= logging.DEBUG:
            logger.debug('new U_a, after orthogonalisation:\n' + str_matrix(U[0]))
            logger.debug('new U_b, after orthogonalisation:\n' + str_matrix(U[1]))
        f = ci_wf.distance_to_det(U)
        norm_eta = 0.0
        for icol in eta:
            for eta_ij in icol:
                norm_eta += eta_ij**2
        norm_eta = math.sqrt(norm_eta)
        logger.info('norm of eta matrix: {.5e}'.format(norm_eta))
        f_out.write('{0:<5d}  {1:<11.8f}  {2:<11.8f}  {3:<11.8f}\n'.\
                    format(i_iteration,
                           f,
                           norm_eta,
                           norm_C))
        if norm_eta < thrsh_eta:
            converged_eta = True
            if only_eta:
                break
        else:
            converged_eta = False
        if converged_C and converged_eta:
            break
    return Results(f = f,
                   U = U,
                   norm = (norm_eta, norm_C)
                   last_iteration = i_iteration,
                   converged = (converged_eta, converged_C),
                   n_pos_H_eigVal = n_pos_eigV)

