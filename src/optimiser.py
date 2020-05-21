"""Optimisers for the overlap between the Grassmannian and a wave function

Here, we consider the function f given by

f(x) = <0|ext>

where |0> is a Slater determinant and |ext> a correlated wave function.
The functions here are the optimisers.

Classes:
--------
Results

Functions:
----------
optimise_distance_to_FCI
optimise_distance_to_CI

"""
import copy
import sys
import logging
from datetime import timedelta

from collections import namedtuple
import numpy as np
from scipy import linalg

from util import logtime
from wave_functions.general import Wave_Function
from wave_functions.cisd import Wave_Function_CISD
import absil
import orbitals as orb
from exceptions import dGrValueError

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

np.set_printoptions(linewidth=150)


class Results(namedtuple('Results',
                         ['f',
                          'U',
                          'norm',
                          'last_iteration',
                          'converged',
                          'n_pos_H_eigVal'])):
    """A namedtuple for the results of optimisation
    
    Attributes:
    -----------
    f (float)
        The value of the function (the overlap) in the end of the procedure
    
    U (list of 2d np.arrays)
        the transformation matrices for the optimised orbitals
    
    norm (float or tuple of floats)
        norm of vectors that should vanish at convergence
    
    last_iteration (int)
        number of iterations
    
    converged (bool)
        Indicates convergence
    
    n_pos_HeigVal (int)
        number of positive eigenvalues of the Hessian
    """
    __slots__ = ()


def optimise_overlap_orbRot(wf,
                            max_iter=20,
                            f_out=sys.stdout,
                            restricted=True,
                            ini_U=None,
                            occupation=None,
                            thrsh_Z=1.0E-8,
                            thrsh_J=1.0E-8,
                            enable_uphill=True):
    """Find a single Slater determinant that minimise the distance to wf
    
    Behaviour:
    ----------
    
    This function uses the Newton-Raphson method for the optimisation,
    using the exponential parametrisation.
    Because the Jacobian and Hessian are simple to calculate only when
    the FCI wave function is on the same orbital basis of the slater
    determinant of the optimisation step, a full transformation of
    wave function wf is done in every step.
    Such transformation uses the structure of wf, and thus
    the initial wave function wf has to have all determinants of the
    full CI wave function, even if the coefficients are zero.
    
    The argument wf is not changed.
    
    Parameters:
    -----------
    
    wf (dGr_general_WF.Wave_Function)
        The external wave function
    
    max_iter (int, default = 20)
        The maximum number of iterations in the optimisation
    
    f_out (file object, default = sys.stdout)
        The output
    
    restricted (bool, default = False)
        Optimises the spatial part of both alpha and beta equally.
        It is not explicitly used!
        The code decides when use restricted calculation
    
    ini_U (list of np.ndarray, default=None)
        if not None, it should have the initial transformation
        of orbitals from the basis of the ci_wf to the basis of the initial
        Slater determinant.
        The list is like:
        [U_a^1, ..., U_a^g, U_b^1, ..., U_b^g]
        where U_sigma^i is the U for spin sigma (alpha=a or beta=b)
        and irrep i.
        If it is a restricted calculation, only one part should be given:
        [U^1, ..., U^g]
        If None, the Identity for each irrep is used as initial
        transformation, that is, the occupied orbitals are the first ones
        in the MO basis of wf.
    
    occupation (tuple of int, default=None)
        The occupation of each spirrep block to be used in the optimization:
        (n_a^1, ..., n_a^g, n_b^1, ..., n_b^g)
        This should be consistent to the spin and symmetry of the external
        wave function.
        If None is given, uses wf.ref_occ.
        If ini_U is given, this occupation is not considered, and the
        implicit occupation given by the number of columns of U is used.
        Currently: not implemented!
    
    thrsh_Z (float, optional, default=1.0E-8)
        Convergence threshold for the Z vector
    
    thrsh_J (float, optional, default=1.0E-8)
        Convergence threshold for the Jacobian
    
    enable_uphill (bool, optional, default=True)
        If True, try algorithm to go uphill in case of positive eigenvalues
        of the Hessian
        
    Returns:
    --------
    
    The namedtuple Results. Some particularities are:
    
    If the first determinant is not the one with largest coefficients,
    f is a tuple (f, det_maxC), with the first entry as the coefficient of
    the first determinant (that is what we are optimising), and det_maxC is
    the (complete) determinant with the largest coefficient.
    
    TODO:
    -----
    
    Implement calculation for occupation other than ref_occ. See
    commented line below

    """
    def get_i_max_coef(wf):
        max_coef = 0.0
        i_max_coef = -1
        for i, det in enumerate(wf):
            if abs(det.c) > max_coef:
                max_coef = abs(det.c)
                i_max_coef = i
        return i_max_coef
    converged = False
    try_uphill = False
    i_iteration = 0
    if ini_U is None:
        U = []
        # ini_occ = occupation if occupation is not None else wf.ref_occ
        for i in wf.spirrep_blocks(restricted=restricted):
            U.append(np.identity(wf.orb_dim[i % wf.n_irrep]))
        cur_wf = copy.copy(wf)
    else:
        U = orb.complete_orb_space(ini_U, wf.orb_dim)
        cur_wf = wf.change_orb_basis(U)
    fmt_full = '{0:<5d}  {1:<11.8f}  {2:<11.8f}  {3:<11.8f}  {4:s}\n'
    f_out.write('{0:<5s}  {1:<11s}  {2:<11s}  {3:<11s}  {4:s}\n'.
                format('it.', 'f', '|z|', '|Jac|',
                       'time in iteration [det. largest coef.]'))
    for i_iteration in range(max_iter):
        with logtime('Starting iteration {}'.format(i_iteration)) as T_start:
            pass
        if i_iteration > 0:
            if not try_uphill and normJ < thrsh_J and normZ < thrsh_Z:
                if not Hess_has_pos_eig or not enable_uphill:
                    converged = True
                    break
                else:
                    try_uphill = True
            with logtime(
                    'Calculating new basis and transforming wave function.'):
                cur_wf, cur_U = cur_wf.calc_wf_from_z(z)
            for i in range(len(U)):
                U[i] = U[i] @ cur_U[i]
            if loglevel <= logging.DEBUG:
                logger.debug('Wave function:\n%s', cur_wf)
        with logtime('Making Jacobian and Hessian'):
            Jac, Hess = cur_wf.make_Jac_Hess_overlap()
        logger.log(1, 'Jacobian:\n%r', Jac)
        logger.log(1, 'Hessian:\n%r', Hess)
        with logtime('Eigenvectors of Hessian'):
            eig_val, eig_vec = linalg.eigh(Hess)
        Hess_has_pos_eig = False
        Hess_dir_with_posEvec = None
        n_pos_eigV = 0
        for i, eigVal in enumerate(eig_val):
            if eigVal > 0.0:
                n_pos_eigV += 1
                Hess_has_pos_eig = True
                Hess_dir_with_posEvec = np.array(eig_vec[:, i])
        if Hess_has_pos_eig:
            logger.warning('Hessian has positive eigenvalue.')
        normJ = linalg.norm(Jac)
        if not try_uphill:
            # Newton step
            with logtime('Solving lin system for vector z.'):
                z = -linalg.solve(Hess, Jac)
        else:
            # Look maximum in the direction of eigenvector of Hess with pos
            #
            # An attempt to automatically set gamma had failed tests...
            # den = np.matmul(np.matmul(Jac.transpose(), Hess), Jac)
            # gamma = 0.0
            # for j in Jac:
            #    gamma += j**2
            # gamma = -gamma/den
            with logtime('Trying uphill'):
                gamma = 0.5
                logger.info('Calculating z vector by Gradient descent;'
                            + '\n gamma = %s', gamma)
                max_c0 = cur_wf.C0
                max_i0 = 0
                logger.log(15, 'Current C0: %.4f', max_c0)
                for i in range(6):
                    tmp_wf, tmp_U = cur_wf.calc_wf_from_z(
                        i * gamma * Hess_dir_with_posEvec,
                        just_C0=True)
                    this_c0 = tmp_wf.C0
                    logger.debug('Attempting for i=%d: C0 = %.4f',
                                 i, this_c0)
                    if abs(max_c0) < abs(this_c0):
                        max_c0 = this_c0
                        max_i0 = i
                z = max_i0 * gamma * Hess_dir_with_posEvec
                logger.info(
                    'z vector obtained: %d * gamma * Hess_dir_with_posEvec',
                    max_i0)
                try_uphill = False
        logger.debug('z vector:\n%r', z)
        with logtime('Calculating norm of Z') as T_norm_Z:
            normZ = linalg.norm(z)
        elapsed_time = str(timedelta(seconds=(T_norm_Z.end_time
                                              - T_start.ini_time)))
        f_out.write(fmt_full.
                    format(i_iteration,
                           cur_wf.C0,
                           normZ,
                           normJ,
                           elapsed_time))
        i_max_coef = get_i_max_coef(cur_wf)
        if i_max_coef > 0:
            f_out.write('   ^ Max coefficient: {}\n'.
                        format(str(cur_wf[i_max_coef])))
        f_out.flush()
    if i_max_coef > 0:
        f_final = (cur_wf.C0, cur_wf[i_max_coef])
    else:
        f_final = cur_wf.C0
    return Results(f=f_final,
                   U=U,
                   norm=(normZ, normJ),
                   last_iteration=i_iteration,
                   converged=converged,
                   n_pos_H_eigVal=n_pos_eigV)


def optimise_overlap_Absil(ci_wf,
                           max_iter=20,
                           f_out=sys.stdout,
                           restricted=True,
                           ini_U=None,
                           occupation=None,
                           thrsh_eta=1.0E-5,
                           thrsh_C=1.0E-5,
                           only_C=False,
                           only_eta=False,
                           check_equations=False):
    """Find the single Slater determinant that maximises the overlap to ci_wf
    
    Behaviour:
    ----------
    
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
    ------------
    
    Only for unrestricted optimisations in the moment
    Does not check if we are in a maximum or saddle point
    
    Parameters:
    -----------
    
    ci_wf (dGr_general_WF.Wave_Function)
        The external wave function
    
    max_iter (int, default = 20)
        The maximum number of iterations in the optimisation
    
    f_out (file object, default = sys.stdout)
        The output
    
    restricted (bool, default = False)
        Optimise the spatial part of both alpha and beta equally.
        It is not explicitly used!
        The code decides when use restricted calculation
    
    ini_U (list of np.ndarray, default=None)
        if not None, it should have the initial transformation
        of orbitals from the basis of the ci_wf to the basis of the initial
        Slater determinant.
        The list is like:
        [U_a^1, ..., U_a^g, U_b^1, ..., U_b^g]
        where U_sigma^i is the U for spin sigma (alpha=a or beta=b)
        and irrep i.
        If it is a restricted calculation, only one part should be given:
        [U^1, ..., U^g]
        If None, a column-truncated Identity for each irrep is used as
        initial transformation, that is, the occupied orbitals are the
        first ones in the MO basis of ci_wf.
    
    occupation (tuple of int, default=None)
        The occupation of each spirrep block to be used in the optimization:
        (n_a^1, ..., n_a^g, n_b^1, ..., n_b^g)
        This should be consistent to the spin and symmetry of the external
        wave function.
        If None is given, uses ci_wf.ref_occ.
        If ini_U is given, this occupation is not considered, and the
        implicit occupation given by the number of columns of U is used.
        Currently: not implemented!
    
    thrsh_eta (float, optional, default = 1.0E-5)
        Convergence threshold for the eta vector
    
    thrsh_C (float, optional, default = 1.0E-5)
        Convergence threshold for the C vector
    
    only_C (bool, default = False)
        If True, stops the iterations if the C vector passes
        in the convergence test, irrespective of the norm of eta
        (and does not go further in the iteration)
    
    only_eta (bool, default = False)
        If True, stops the iterations if the eta vector passes
        in the convergence test, irrespective of the norm of C
        (and does not go further in the iteration)
    
    check_equations (bool, default = False)
        If True, checks numerically if the Absil equation is satisfied.
        It is slow and for testing purposes.
        
    Return:
    -------
    
    The namedtuple Results. Some particularities are:
    
    norm is a 2-tuple with norm_eta and norm_C.
    converged is also a 2-tuple, showing convergence for C and eta.
    n_pos_H_eigVal is not set yet!!
    We have to discover how to calculate this...
    
    TODO:
    -----

    implement restricted calculations
    calculate n_pos_H_eigVal
    """
    if not isinstance(ci_wf, Wave_Function):
        raise dGrValueError(
            'ci_wf should be an instance of dGr_general_WF.Wave_Function.')
    n_pos_eigV = None
    converged_eta = False
    converged_C = False
    zero_skip_linalg = 1.0E-8
    f = None
    if only_C and only_eta:
        raise dGrValueError('Do not set both only_C and only_eta to True!')
    restricted = isinstance(ci_wf, Wave_Function_CISD)
    if ini_U is None:
        U = []
        ini_occ = occupation if occupation is not None else ci_wf.ref_occ
        for i in ci_wf.spirrep_blocks(restricted=restricted):
            U.append(np.identity(
                ci_wf.orb_dim[i % ci_wf.n_irrep])[:, :(ini_occ[i])])
    else:
        if not isinstance(ini_U, list):
            raise dGrValueError('ini_U must be a list of numpy.array.')
        if restricted:
            if len(ini_U) != ci_wf.n_irrep:
                raise dGrValueError('ini_U must be a list,'
                                    + ' of lenght ci_wf.n_irrep of numpy.array'
                                    + ' (for restricted calculations).')
        else:
            if len(ini_U) != 2 * ci_wf.n_irrep:
                raise dGrValueError('ini_U must be a list,'
                                    + ' of lenght ci_wf.n_irrep of numpy.array'
                                    + ' (for unrestricted calculations).')
        sum_n_a = sum_n_b = 0
        for i in ci_wf.spirrep_blocks(restricted=restricted):
            i_irrep = i % ci_wf.n_irrep
            if ini_U[i].shape[0] != ci_wf.orb_dim[i_irrep]:
                raise dGrValueError(
                    ('Shape error in ini_U {0:} for irrep {1:}:'
                     + ' U.shape[0] = {2:} != {3:} = ci_wf.orb_dim').
                    format('alpha' if i < ci_wf.n_irrep else 'beta',
                           i_irrep,
                           ini_U[i].shape[0],
                           ci_wf.orb_dim[i_irrep]))
            if i < ci_wf.n_irrep:
                sum_n_a += ini_U[i].shape[1]
            else:
                sum_n_b += ini_U[i].shape[1]
        for sum_n, n, spin in [(sum_n_a, ci_wf.n_alpha, (''
                                                         if restricted else
                                                         ' alpha')),
                               (sum_n_b, ci_wf.n_beta, ' beta')]:
            if sum_n != n:
                raise dGrValueError(('Shape error in ini_U{0:}:'
                                     + ' sum U.shape[1] = '
                                     + ' {1:} != {2:} = ci_wf.n_{0:}').
                                    format(spin, sum_n, n))
            if restricted:
                break
        U = ini_U
    slice_XC = absil.make_slice_XC(U)
    norm_C = norm_eta = elapsed_time = '---'
    converged_eta = converged_C = False
    fmt_full = '{0:<5d}  {1:<11.8f}  {2:<11.8f}  {3:<11.8f}  {4:s}\n'
    f_out.write('{0:<5s}  {1:<11s}  {2:<11s}  {3:<11s}  {4:s}\n'.
                format('it.', 'f', '|eta|', '|C|', 'time in iteration'))
    for i_iteration in range(max_iter):
        with logtime('Generating linear system') as T_gen_lin_system:
            f, X, C = absil.generate_lin_system(ci_wf, U, slice_XC)
        if loglevel <= logging.DEBUG:
            np.save('X_matrix-{}.npy'.format(i_iteration), X)
            np.save('C_matrix-{}.npy'.format(i_iteration), C)
        logger.debug('f: %.5f', f)
        logger.debug('matrix X:\n%s', X)
        logger.debug('matrix C:\n%s', C)
        norm_C = linalg.norm(C)
        logger.info('norm of matrix C: %.5e', norm_C)
        if norm_C < thrsh_C:
            converged_C = True
            if only_C:
                break
        else:
            converged_C = False
        with logtime('Solving linear system'):
            lin_sys_solution = linalg.lstsq(X, C, cond=None)
        logger.debug('Solution of the linear system, eta:\n%s',
                     lin_sys_solution[0])
        logger.info('Rank of matrix X: %d', lin_sys_solution[2])
        norm_eta = linalg.norm(lin_sys_solution[0])
        if norm_eta < thrsh_eta:
            converged_eta = True
            if only_eta:
                break
        else:
            converged_eta = False
        logger.info('Norm of matrix eta: %.5e', norm_eta)
        eta = []
        svd_res = []
        with logtime('Singular value decomposition of eta'):
            for i in ci_wf.spirrep_blocks(restricted=restricted):
                eta.append(np.reshape(
                    lin_sys_solution[0][slice_XC[i]],
                    U[i].shape, order=('C'
                                       if isinstance(ci_wf,
                                                     Wave_Function_CISD) else
                                       'F')))
                norm_eta_i = linalg.norm(eta[-1])
                if norm_eta_i < zero_skip_linalg:
                    svd_res.append((np.zeros(eta[-1].shape),
                                    np.zeros((eta[-1].shape[1],)),
                                    np.identity(eta[-1].shape[1])))
                    logger.info(
                        'Skipping svd for spirrep block %d.'
                        + ' Norm of eta[%d] = %.8f',
                        i, i, norm_eta_i)
                else:
                    svd_res.append(linalg.svd(eta[-1],
                                              full_matrices=False))
        if check_equations:
            with logtime('Cheking equations'):
                chk_eq = absil.check_Newton_eq(ci_wf, U, eta,
                                               restricted, eps=0.001)
                f_out.write('Cheking equations: ' + 'OK' if chk_eq else 'FAIL')
        if loglevel <= logging.DEBUG:
            for i in ci_wf.spirrep_blocks(restricted=restricted):
                logger.debug('SVD results, Usvd_a:\n%s', svd_res[i][0])
                logger.debug('SVD results, SGMsvd_a:\n%s', svd_res[i][1])
                logger.debug('SVD results, VTsvd_a:\n%s', svd_res[i][2])
        for i in ci_wf.spirrep_blocks(restricted=restricted):
            U[i] = np.matmul(U[i], svd_res[i][2].T * np.cos(svd_res[i][1]))
            U[i] += svd_res[i][0] * np.sin(svd_res[i][1])
        if loglevel <= logging.DEBUG:
            for i in ci_wf.spirrep_blocks(restricted=restricted):
                logger.debug('new U for %s and irrep %s:\n%s',
                             'alpha' if i < ci_wf.n_irrep else 'beta',
                             i % ci_wf.n_irrep,
                             U[i])
        with logtime('Orthogonalisation of U') as T_orth_U:
            for i, Ui in enumerate(U):
                norm_Ui = linalg.norm(Ui)
                if norm_Ui > zero_skip_linalg:
                    U[i] = linalg.orth(Ui)
        if loglevel <= logging.DEBUG:
            for i in ci_wf.spirrep_blocks(restricted=restricted):
                logger.debug(
                    'new U for %s and irrep %s, after orthogonalisation:\n%s',
                    'alpha' if i < ci_wf.n_irrep else 'beta',
                    i % ci_wf.n_irrep,
                    U[i])
        elapsed_time = str(timedelta(seconds=(T_orth_U.end_time
                                              - T_gen_lin_system.ini_time)))
        f_out.write(fmt_full.
                    format(i_iteration,
                           f,
                           norm_eta,
                           norm_C,
                           elapsed_time))
        f_out.flush()
        if converged_C and converged_eta:
            break
    return Results(f=f,
                   U=U,
                   norm=(norm_eta, norm_C),
                   last_iteration=i_iteration,
                   converged=(converged_eta, converged_C),
                   n_pos_H_eigVal=n_pos_eigV)
