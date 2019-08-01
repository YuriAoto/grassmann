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
import datetime
import time

from collections import namedtuple
import numpy as np
from scipy import linalg

from dGr_util import str_matrix

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

def _get_norm_of_matrix(M):
    """Return norm of M
    
    If M = array,              returns sqrt(sum Mij)
    If M = (array_a, array_b), returns sqrt((sum M[0]ij + M[1]ij)/2)
    """
    norm = 0.0
    if isinstance(M, tuple):
        norm += _get_norm_of_matrix(M[0])
        norm += _get_norm_of_matrix(M[1])
        norm = norm/2
    else:
        for line in M:
            for M_ij in line:
                norm += M_ij**2
    return math.sqrt(norm)

def _generate_Absil_lin_system(A, B, C, U):
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

def optimise_distance_to_CI(ci_wf,
                            max_iter = 20,
                            f_out = sys.stdout,
                            restricted = True,
                            ini_U = None,
                            thrsh_eta = 1.0E-6,
                            thrsh_C = 1.0E-6,
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
    
    Limitations:
    
    The system is assumed to have the same number of alpha and beta electrons.
    
    
    Parameters:
    
    ci_wf      an instance of a class that represents the external wave function.
               Such class must have some attributes, that are explained below.
    
    max_iter   the maximum number of iterations in the optimisation
               (default = 20)
    
    f_out      the output stream (default = sys.stdout)
    
    restricted  (bool, optional, default = True)
                Optimise the spatial part of both alpha and beta equally.
                Assume that the reference orbital is also restricted
                and that the ci_wf is symmetric in alpha and beta.
    
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
    orb_dim (int)                      dimension of the orbital space
    get_ABC_matrices (method)          given a transformation U to another orbital basis,
                                       return the matrices A, B and C, that contain
                                       the elements to calculate the Newton step.
                                       See the documentation of _generate_Absil_lin_system
                                       for a description of shapes of these matrices.
    distance_to_det (method)           given a transformation U to another orbital basis,
                                       calculates the distance between the external
                                       wave function to the slater determinant associated to
                                       U.
    
    Returns:
    
    The namedtuple Results. Some particularities are:
    
    norm is a 2-tuple with norm_eta and norm_C.
    converged is also a 2-tuple, showing convergence for
    C and eta.
    n_pos_H_eigVal is not set yet!! We have to know how to calculate this...
    
    """
    check_equations = True
    n_pos_eigV = None
    converged_eta = False
    converged_C = False
    f = None
    f_out.write('{0:<5s}  {1:<11s}  {2:<11s}  {3:<11s}\n'.\
                format('it.', 'f', '|eta|', '|C|'))
    if only_C and only_eta:
        raise ValueError('Do not set both only_C and only_eta to True!')
    if restricted and ci_wf.n_alpha != ci_wf.n_beta:
        raise ValueError('For restricted calculation, CI wf must have n_alpha = n_beta')
    if ini_U is not None:
        if restricted:
            if not isinstance(ini_U, np.array):
                raise ValueError('For restricted calculations, ini_U must be a numpy.array')
            if ini_U.shape[0] != ci_wf.orb_dim:
                raise ValueError ('ini_U has shape[0] {0:} but dim of orbital space of ci_wf is {1:}'.\
                                  format(ini_U.shape[0], ci_wf.orb_dim))
            if ini_U.shape[1] != ci_wf.n_alpha:
                raise ValueError ('ini_U alpha has shape[1] {0:} but n_alpha in ci_wf is {1:}'.\
                                  format(ini_U.shape[1], ci_wf.n_alpha))
        else:
            if not isinstance(ini_U, tuple):
                raise ValueError('For unrestricted calculations,'
                                 + ' ini_U must be a 2-tuple of numpy.array')
            if ini_U[0].shape[0] != ci_wf.orb_dim:
                raise ValueError (('ini_U alpha has shape[0] {0:}'
                                   + ' but dim of orbital space of ci_wf is {1:}').\
                                  format(ini_U[0].shape[0], ci_wf.orb_dim))
            if ini_U[1].shape[0] != ci_wf.orb_dim:
                raise ValueError (('ini_U beta has shape[0] {0:}'
                                   + ' but dim of orbital space of ci_wf is {1:}').\
                                  format(ini_U[1].shape[0], ci_wf.orb_dim))
            if ini_U[0].shape[1] != ci_wf.n_alpha:
                raise ValueError (('ini_U alpha has shape[1] {0:}'
                                   + ' but n_alpha in ci_wf is {1:}').\
                                  format(ini_U[0].shape[1], ci_wf.n_alpha))
            if ini_U[1].shape[1] != ci_wf.n_beta:
                raise ValueError (('ini_U beta has shape[1] {0:}'
                                   + ' but n_beta in ci_wf is {1:}').\
                                  format(ini_U[1].shape[1], ci_wf.n_beta))
        U = ini_U
    else:
        if restricted:
            U = np.identity(ci_wf.orb_dim)[:,:ci_wf.n_alpha]
        else:
            U = (np.identity(ci_wf.orb_dim)[:,:ci_wf.n_alpha],
                 np.identity(ci_wf.orb_dim)[:,:ci_wf.n_beta])
    for i_iteration in range(max_iter):
        logger.info('Calculating matrices A, B, C...')
        ini_time = time.time()

        ## testing f
        # Ua, Ub = U
        # Ua[2,2] += 0.1
        # print(ci_wf.distance_to_det((Ua, Ub)))
        # print(ci_wf.distance_to_det((linalg.orth(Ua), Ub)))
        # Ua[4,1] -= 0.5
        # print(ci_wf.distance_to_det((Ua, Ub)))
        # print(ci_wf.distance_to_det((linalg.orth(Ua), Ub)))
        # Ub[4,1] += 0.3
        # print(ci_wf.distance_to_det((Ua, Ub)))
        # print(ci_wf.distance_to_det((linalg.orth(Ua), Ub)))
        # exit()

        A, B, C = ci_wf.get_ABC_matrices(U)
        end_time = time.time()
        elapsed_time = str(datetime.timedelta(seconds=(end_time - ini_time)))
        logger.info('Total time to calculate matrices A, B, C: {}'.\
                    format(elapsed_time))
        if logger.level <= logging.DEBUG:
            if restricted:
                logger.debug('matrix A:\n' + str_matrix(A))
                logger.debug('matrix B:\n' + str_matrix(B))
                logger.debug('matrix C:\n' + str_matrix(C))
            else:
                logger.debug('matrix A(alpha, alpha):\n' + str(A[0][0]))
                logger.debug('matrix A(alpha, beta):\n' + str(A[0][1]))
                logger.debug('matrix A(beta , alpha):\n' + str(A[1][0]))
                logger.debug('matrix A(beta , beta):\n' + str(A[1][1]))
                logger.debug('matrix B(alpha):\n' + str(B[0]))
                logger.debug('matrix B(beta):\n' + str(B[1]))
                logger.debug('matrix C(alpha):\n' + str_matrix(C[0]))
                logger.debug('matrix C(beta):\n' + str_matrix(C[1]))
        norm_C = _get_norm_of_matrix(C)
        logger.info('norm of matrix C: {:.5e}'.format(norm_C))
        if norm_C < thrsh_C:
            converged_C = True
            if only_C:
                break
        else:
            converged_C = False
        logger.info('Generating linear system...')
        ini_time = time.time()
        B_minus_A, C = _generate_Absil_lin_system(A, B, C, U)
        end_time = time.time()
        elapsed_time = str(datetime.timedelta(seconds=(end_time - ini_time)))
        logger.info('Total time to generate linear system: {}'.\
                    format(elapsed_time))
        if logger.level <= logging.DEBUG:
            logger.debug('lin_system, B-A:\n' + str_matrix(B_minus_A))
            logger.debug('lin_system, C:\n' + str(C))
##            logger.debug('determinant of B-A:\n' + str(linalg.det(B_minus_A)))
        logger.info('Solving linear system...')
        ini_time = time.time()
        ###        eta = linalg.solve(B_minus_A, C)
        eta = linalg.lstsq(B_minus_A, C, cond=None)[0]
##        print (results) check for effective rank?
##        exit()
        end_time = time.time()
        elapsed_time = str(datetime.timedelta(seconds=(end_time - ini_time)))
        logger.info('Total time to solve the linear system: {}'.\
                    format(elapsed_time))
        if logger.level <= logging.DEBUG:
            logger.debug('lin_system (solution), eta:\n' + str(eta))
        if restricted:
            eta = np.reshape(eta, C.shape, order='C')
            Usvd_a, SGMsvd_a, VTsvd_a = linalg.svd(eta,
                                                   full_matrices=False)
        else:
            eta = (np.reshape(eta[:ci_wf.n_alpha*ci_wf.orb_dim],
                              U[0].shape, order='C'),
                   np.reshape(eta[ci_wf.n_alpha*ci_wf.orb_dim:],
                              U[1].shape, order='C'))
            # testing: projecting eta
##            eta = (np.matmul((np.identity(U[0].shape[0]) - np.matmul(U[0], U[0].T)), eta[0]),
##                   np.matmul((np.identity(U[1].shape[0]) - np.matmul(U[1], U[1].T)), eta[1]))
            Usvd_a, SGMsvd_a, VTsvd_a = linalg.svd(eta[0],
                                                   full_matrices=False)
            Usvd_b, SGMsvd_b, VTsvd_b = linalg.svd(eta[1],
                                                   full_matrices=False)
        if check_equations:
            ci_wf.check_Newton_Absil_eq(U, eta, eps = 0.0000001)
        if logger.level <= logging.DEBUG:
            logger.debug('SVD results, Usvd_a:\n'   + str_matrix(Usvd_a))
            logger.debug('SVD results, SGMsvd_a:\n' + str(SGMsvd_a))
            logger.debug('SVD results, VTsvd_a:\n'  + str_matrix(VTsvd_a))
            if not restricted:
                logger.debug('SVD results, Usvd_a:\n'   + str_matrix(Usvd_a))
                logger.debug('SVD results, SGMsvd_a:\n' + str(SGMsvd_a))
                logger.debug('SVD results, VTsvd_a:\n'  + str_matrix(VTsvd_a))
        if restricted:
            U = np.matmul(U, VTsvd_a.T * np.cos(SGMsvd_a))
            U +=  Usvd_a * np.sin(SGMsvd_a)
        else:
            U = (np.matmul(U[0], VTsvd_a.T * np.cos(SGMsvd_a))
                 + Usvd_a * np.sin(SGMsvd_a),
                 np.matmul(U[1], VTsvd_b.T * np.cos(SGMsvd_b))
                 + Usvd_b * np.sin(SGMsvd_b))
        if logger.level <= logging.DEBUG:
            if restricted:
                logger.debug('new U:\n' + str_matrix(U))
            else:
                logger.debug('new U_a:\n' + str_matrix(U[0]))
                logger.debug('new U_b:\n' + str_matrix(U[1]))
        if restricted:
            U = linalg.orth(U)
        else:
            U = tuple(map(linalg.orth, U))
        if logger.level <= logging.DEBUG:
            logger.debug('new U_a, after orthogonalisation:\n' + str_matrix(U[0]))
            logger.debug('new U_b, after orthogonalisation:\n' + str_matrix(U[1]))
        logger.info('Calculating f...')
        ini_time = time.time()
        ### Maybe use f from ci_wf.get_ABC_matrices(U), that is of previous iteration?
        f = ci_wf.distance_to_det(U)
        end_time = time.time()
        elapsed_time = str(datetime.timedelta(seconds=(end_time - ini_time)))
        logger.info('Total time to calculate f: {}'.\
                    format(elapsed_time))
        norm_eta = _get_norm_of_matrix(eta)
        logger.info('norm of eta matrix: {:.5e}'.format(norm_eta))
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
                   norm = (norm_eta, norm_C),
                   last_iteration = i_iteration,
                   converged = (converged_eta, converged_C),
                   n_pos_H_eigVal = n_pos_eigV)
