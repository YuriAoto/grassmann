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
from datetime import timedelta

from collections import namedtuple
import numpy as np
from scipy import linalg

from dGr_util import str_matrix, logtime
from dGr_general_WF import Wave_Function
import dGr_Absil as Absil

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
    U                  [U_sigma^i], the transformation matrices for the optimised orbitals
    norm               norm of vectors that should vanish at convergence
    last_iteration     number of iterations
    converged          True or False, indicating convergence
    n_pos_HeigVal      number of positive eigenvalues of the Hessian
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


def optimise_distance_to_CI(ci_wf,
                            max_iter = 20,
                            f_out = sys.stdout,
                            restricted = True,
                            ini_U = None,
                            occupation = None,
                            thrsh_eta = 1.0E-5,
                            thrsh_C = 1.0E-5,
                            only_C = False,
                            only_eta = False,
                            check_equations = False):
    """Find the single Slater determinant that maximises the overlap to ci_wf
    
    Behaviour:
    
    This function uses the Newton method on the Grassmannian, as discussed in
    
    P-A Absil, R Mahony, and R Sepulchre, "Riemannian Geometry of Grassmann
    Manifolds with a View on Algorithmic Computation", Acta App. Math. 80,
    1999-220, 2004
    
    In this method we do not change the representation of the external wave
    function (ci_wf), given with a (eventually truncated) configurations
    interaction parametrisation.
    
    One of the steps is to solve the linear system X @ eta = C.
    The convergence is obtained by checking the norm of eta and/or C
    
    Important convention:
    For the spin/symmetry adaption, the code assumes that there is a
    standard order for spin/irreducible representations, that is
    followed everywhere.
    For example, an unrestricted calculation with point group of order 2
    has 4 possible combinations:
    alpha/irrep_1, alpha/irrep_2, beta/irrep_1, beta/irrep_2
    If ordered in this way, this leads to 0, 1, 2, 3, that will index
    such combination.
    Each one of these is a spirrep, and is what is yield by the function
    ci_wf.spirrep_blocks.
    The convention will be:
    # Molpro's order for irreps, for restricted cases
    # alpha first, beta later, with Molpro's order for irreps in each
      block of spin, for unrestricted cases
    
    Limitations:
    Only for unrestricted optimisations in the moment
    Does not check if we are in a maximum or saddle point
    
    Parameters:
    
    ci_wf (dGr_general_WF.Wave_Function)
        The external eave function
    
    max_iter (int, default = 20)
        The maximum number of iterations in the optimisation
    
    f_out (file object, default = sys.stdout)
        The output
    
    restricted (bool, default = False)
        Optimise the spatial part of both alpha and beta equally.
        It is not implemented yet!
    
    ini_U (list of np.ndarray, default=None)
        if not None, it should have the initial transformation
        of orbitals from the basis of the ci_wf to the basis of the initial
        Slater determinant.
        The list is like:
        [U_a^1, ..., U_a^g, U_b^1, ..., U_b^g]
        where U_sigma^i is the U for spin sigma (alpha=a or beta=b) and irrep i.
        If it is a restricted calculation, only one part should be given:
        [U^1, ..., U^g]
        If None, a column-truncated Identity is used as initial transformation.
        (default = None)
    
    occupation (tuple of int, default=None)
        The occupation of each spin-irrep block to be used in the optimization:
        (n_a^1, ..., n_a^g, n_b^1, ..., n_b^g)
        This should be consistent to the spin and symmetry of the external wave function
        If None is given, uses ci_wf.ref_occ.
        If ini_U is given, this occupation is not considered, and the
        implicit occupation given by the number of columns of U is used.
    
    thrsh_eta (float, optional, default = 1.0E-5)
        Convergence threshold for the eta vector
    
    thrsh_C (float, optional, default = 1.0E-5)
        Convergence threshold for the C vector
    
    only_C (bool, default = False)
        If True, stops the iterations if the C vector passes in the convergence
        test, irrespective of the norm of eta (and does not go further
        in the iteration)
    
    only_eta (bool, default = False)
        If True, stops the iterations if the eta vector passes in the convergence
        test, irrespective of the norm of C (and does not go further
        in the iteration)
    
    check_equations (bool, default = False)
        If True, checks numerically if the Absil equation is satisfied.
        It is slow and for testing purposes.
        
    Return:
    
    The namedtuple Results. Some particularities are:
    
    norm is a 2-tuple with norm_eta and norm_C.
    converged is also a 2-tuple, showing convergence for C and eta.
    n_pos_H_eigVal is not set yet!! We have to discover how to calculate this...
    
    TODO:
    implement restricted calculations
    calculate n_pos_H_eigVal
    """
    if not isinstance(ci_wf, Wave_Function):
        raise ValueError('ci_wf should be an instance of dGr_general_WF.Wave_Function.')
    n_pos_eigV = None
    converged_eta = False
    converged_C = False
    f = None
    if only_C and only_eta:
        raise ValueError('Do not set both only_C and only_eta to True!')
    if restricted:
        raise NotImplementedError('Restricted calculation is not working yet.')
    if ini_U is None:
        U = []
        ini_occ = occupation if occupation is not None else ci_wf.ref_occ
        for i in range(2 * ci_wf.n_irrep):
            U.append(np.identity(ci_wf.orb_dim[i % ci_wf.n_irrep])[:,:(ini_occ[i])])
    else:
        if ((not isinstance(ini_U, list))
            or len(ini_U) != 2 * ci_wf.n_irrep):
            raise ValueError('ini_U must be a list,'
                             +' of lenght 2 * ci_wf.n_irrep of numpy.array.')
        sum_n_a = sum_n_b = 0
        for i in range(2 * ci_wf.n_irrep):
            i_irrep =  i % ci_wf.n_irrep
            if ini_U[i].shape[0] != ci_wf.orb_dim[i_irrep]:
                raise ValueError (('Shape error in ini_U {0:} for irrep {1:}:'
                                   + ' U.shape[0] = {2:} != {3:} = ci_wf.orb_dim').\
                                  format('alpha' if i < ci_wf.n_irrep else 'beta',
                                         i_irrep,
                                         ini_U[i].shape[0],
                                         ci_wf.orb_dim[i_irrep]))
            if i < ci_wf.n_irrep:
                sum_n_a += ini_U[i].shape[1]
            else:
                sum_n_b += ini_U[i].shape[1]
        for sum_n, n, spin in [(sum_n_a, ci_wf.n_alpha, 'alpha'),
                               (sum_n_b, ci_wf.n_beta,  'beta')]:
            if sum_n != n:
                raise ValueError (('Shape error in ini_U {0:}:'
                                   + ' sum U.shape[1] = {1:} != {2:} = ci_wf.n_{0:}').\
                                  format(spin, sum_n, n))
        U = ini_U
    lim_XC = [0]
    for i in range(2 * ci_wf.n_irrep):
        lim_XC.append(lim_XC[-1] + U[i].shape[0] * U[i].shape[1])
    norm_C = norm_eta = elapsed_time = '---'
    converged_eta = converged_C = False
    fmt_full =  '{0:<5d}  {1:<11.8f}  {2:<11.8f}  {3:<11.8f}  {4:s}\n'
    fmt_ini =   '{0:<5d}  {1:<11.8f}  {2:<11s}  {3:<11s}  {4:s}\n'
    f_out.write('{0:<5s}  {1:<11s}  {2:<11s}  {3:<11s}  {4:s}\n'.\
                format('it.', 'f', '|eta|', '|C|', 'time in iteration'))
    for i_iteration in range(max_iter):
        with logtime('Calculating f') as T_calc_f:
            all_F = Absil.calc_all_F(ci_wf, U)
            f = Absil.distance_to_det(ci_wf, U, F=allF)
        f_out.write((fmt_ini if i_iteration == 0 else fmt_full).\
                    format(i_iteration,
                           f,
                           norm_eta,
                           norm_C,
                           elapsed_time))
        if converged_C and converged_eta:
            break
        with logtime('Generating linear system') as T_gen_lin_system:
            X, C = Absil.generate_lin_system(ci_wf, U, lim_XC, F=allF)
        if logger.level <= logging.DEBUG:
            logger.debug('matrix X:\n' + str(X))
            logger.debug('matrix C:\n' + str_matrix(C))
        norm_C = linalg.norm(C)
        logger.info('norm of matrix C: {:.5e}'.format(norm_C))
        if norm_C < thrsh_C:
            converged_C = True
            if only_C:
                break
        else:
            converged_C = False
        with logtime('Solving linear system') as T_solve_lin_system:
            lin_sys_solution = linalg.lstsq(X, C, cond=None)
        if logger.level <= logging.DEBUG:
            logger.debug('Solution of the linear system, eta:\n'
                         + str(lin_sys_solution[0]))
        logger.info('Rank of matrix X: ' +  str(lin_sys_solution[2]))
        norm_eta = linalg.norm(lin_sys_solution[0])
        if norm_eta < thrsh_eta:
            converged_eta = True
            if only_eta:
                break
            else:
                converged_eta = False
        logger.info('Norm of matrix eta: {:.5e}'.format(norm_eta))
        eta = []
        svd_res = []
        with logtime('Singular value decomposition of eta') as T_svd:
            for i in range(2 * ci_wf.n_irrep):
                eta.append(np.reshape(
                    lin_sys_solution[0][lim_XC[i]:lim_XC[i+1]],
                    U[i].shape, order='F'))
                svd_res.append(linalg.svd(eta[-1],
                                          full_matrices=False))
        if check_equations:
            with logtime('Cheking equations') as T_check_eq:
                Absil.check_Newton_Absil_eq(ci_wf, U, eta, eps = 0.0000001)
        if logger.level <= logging.DEBUG:
            for i in range(2 * ci_wf.n_irrep):
                logger.debug('SVD results, Usvd_a:\n'   + str_matrix(svd_res[i][0]))
                logger.debug('SVD results, SGMsvd_a:\n' + str(       svd_res[i][1]))
                logger.debug('SVD results, VTsvd_a:\n'  + str_matrix(svd_res[i][2]))
        for i in range(2 * ci_wf.n_irrep):
            U[i]  = np.matmul(U[i], svd_res[i][2].T * np.cos(svd_res[i][1]))
            U[i] += svd_res[i][0] * np.sin(svd_res[i][1])
        if logger.level <= logging.DEBUG:
            for i in range(2 * ci_wf.n_irrep):
                logger.debug('new U for {} and irrep {}:\n'.\
                             format('alpha' if i < ci_wf.n_irrep else 'beta',
                                    i % ci_wf.n_irrep) + str_matrix(U[i]))
        with logtime('Orthogonalisation of U') as T_orth_U:
            U = map(linalg.orth, U)
        if logger.level <= logging.DEBUG:
            for i in range(2 * ci_wf.n_irrep):
                logger.debug('new U for {} and irrep {}, after orthogonalisation:\n'.\
                             format('alpha' if i < ci_wf.n_irrep else 'beta',
                                    i % ci_wf.n_irrep) + str_matrix(U[i]))
        elapsed_time = str(timedelta(seconds=(T_orth_U.end_time
                                              - T_calc_f.ini_time)))
    return Results(f = f,
                   U = U,
                   norm = (norm_eta, norm_C),
                   last_iteration = i_iteration,
                   converged = (converged_eta, converged_C),
                   n_pos_H_eigVal = n_pos_eigV)
