"""The distance between the CC manifold and and arbitrary (FCI) wave function

This module contains functions to calculate the distance among a number of
points in the space of wave functions.


"""
import math
import logging

import numpy as np
from numpy import linalg

from input_output.log import logtime
from util.results import Results, OptResults, inside_box
from coupled_cluster import manifold as cc_manifold
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.fci import FCIWaveFunction, contribution_from_clusters

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

_str_excitation_list = ['R',
                        'S',
                        'D',
                        'T',
                        'Q']


def _str_excitation(x):
    return _str_excitation_list[x] if x < 5 else str(x)


class DistResults(Results):
    """Results of calculations of distances to the CC manifold
    
    Extra attributes:
    -----------------
    distance (float)
        The distance from the wave function to the CC manifold,
        in the metric induced by intermediate normalisation.
    
    wave_function_as_fci (FCIWaveFunction)
        The wave function as FCIWaveFunction. It is a property, that is
        calculated if requested.
    
    level (str, 'D' or 'SD')
        The coupled cluster level.
    
    """
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._wave_function_as_fci = None
    
    @inside_box
    def __str__(self):
        x = [super().__str__()]
        x.append(f'D(FCI, CC{self.level} manifold) = {self.distance:.8f}\n')
        return '\n'.join(x)
    
    @property
    def wave_function_as_fci(self):
        if self._wave_function_as_fci is None:
            self._wave_function_as_fci = FCIWaveFunction.from_int_norm(self.wave_function)
        return self._wave_function_as_fci



class VertDistResults(DistResults):
    """Store the results from vertical_dist_to_cc_manifold
    
    Extra attributes:
    -----------------    
    wave_function (IntermNormWaveFunction)
        the wave function at the CC manifold. It is obtained by
        IntermNormWaveFunction.from_projected_fci
    
    distance_ci (float)
        The verical distance from the wave function to the CI manifold,
        in the metric induced by intermediate normalisation.
    
    right_dir (dictionary)
        Indicates how many coefficients are in the "right direction" compared
        to the CC manifold. The keys are the ranks, and the values are 2-tuples,
        that show at [0] how many directions are curved towards the wave
        function and at [1] the total number of such directions. Example:
        
        {'T': (10,12),
        'Q': (5,5),
        '5': (1,1)}
        
        Means:
        10 out of 12 triples are in the right direction
        5 out of 5 quadruples are in the right direction
        1 out of 1 quintuples are in the right direction
    
    """
    @inside_box
    def __str__(self):
        x = [super().__str__().replace('D(FCI', 'D_vert(FCI')]
        x.append('Number of excitations where the CC manifold\n'
                 + '   curves towards the wave function:')
        for rank, n in self.right_dir.items():
            x.append(f'{rank}: {n[0]} of {n[1]}')
        return '\n'.join(x)


def vertical_dist_to_cc_manifold(wf,
                                 level='SD',
                                 recipes_f=None,
                                 coeff_thr=1.0E-10):
    """Calculate the vertical distance on the CC manifold
    
    This function is intended to compare the wave function to the
    CCD or CCSD manifold by a "vertical "projection". It writes several
    information to the logfile (at info level). These are:
    
    For every triple, quadruple, ..., and octuple excitation,
    compares the coefficient of the excitation to the sum of product of
    the coefficients given by the cluster decomposition of that excitation.
    It checks if the coefficient has the "correct" sign to be in the
    "right direction" to which the CC manifold curves to.
    A general information is printed to log in the end of the function.
    
    Behaviour:
    ----------
    It will print to the logs all tests, if loglevel is larger or equal
    20 (INFO)
    
    Limitation:
    -----------
    It can handle only up to n-fold excitation, where n is the largest
    case that has a file with the cluster decomposition, as given through
    recipes_f.
    
    Side Effects:
    ------------
    The wave function wf is put in the intermediate normalisation
    and the sign of the coefficients are associated to the determinants
    in the maximum coincidence order.
    
    Parameters:
    -----------
    wf (FCIWaveFunction)
        The wave function to be analysed
    
    level (str, optional, default='D'; possible values: 'D', 'SD')
        The level of CC theory to compare.
    
    recipes_f (str, a file name, optional, default=None)
        The files that describe the cluster decomposition.
        See decompose for the details
    
    coeff_thr (float, optional, default=1.0E-10)
        Slater determinants with coefficients lower than this
        threshold value are ignored in the analysis
    
    Return:
    -------
    An instance of VertDistResults.    
    """
    logfmt = '\n'.join(['det = %s',
                        'C dec = %s',
                        'C dec/C = %s',
                        '(C dec - C)^2 = %s',
                        'same sign = %s'])
    if level not in ['D', 'SD']:
        raise ValueError('Possible values for level are "S" and "SD"')
    level_is_sd = level == 'SD'
    wf.normalise(mode='intermediate')
    wf.set_max_coincidence_orbitals()
    cc_wf = IntermNormWaveFunction.from_projected_fci(wf, 'CC' + level)
    norm_ci = 0.0
    norm = 0.0
    right_dir = {}
    for det in wf:
        if not wf.symmetry_allowed_det(det):
            continue
        rank, alpha_hp, beta_hp = wf.get_exc_info(det)
        if rank == 1 and level_is_sd:
            if cc_wf[rank, alpha_hp, beta_hp] != det.c:
                raise Exception('Not consistent!!')
        if rank > 2 and (level_is_sd or rank % 2 == 0):
            C = contribution_from_clusters(alpha_hp, beta_hp, cc_wf, level)
            norm_contribution = (det.c - C)**2
            norm_ci += det.c**2
            cc_towards_wf = det.c * C >= 0
            norm += norm_contribution
            if abs(det.c) > coeff_thr:
                rank = _str_excitation(rank)
                if rank not in right_dir:
                    right_dir[rank] = [0, 1]
                else:
                    right_dir[rank][1] += 1
                if cc_towards_wf:
                    right_dir[rank][0] += 1
            logger.info(logfmt,
                        det,
                        C,
                        C/det.c if (abs(det.c) > coeff_thr) else f'{C}/{det.c}',
                        norm_contribution,
                        cc_towards_wf)
        elif (not level_is_sd) and rank % 2 == 1:
            #  rank > 2 or (rank, level) == (1, 'D'):
            norm += det.c**2
            norm_ci += det.c**2
            logger.info('Adding .c² to the norm: det = %s', det)
    tolog = ['Number of excitations where the CC manifold\n'
             + '   curves towards the wave function:']
    for rank, n in right_dir.items():
        tolog.append(f'{rank}: {n[0]} of {n[1]}')
    logger.info('\n'.join(tolog))
    res = VertDistResults(f'Vertical distance to CC{level} manifold')
    res.success = True
    res.level = level
    res.distance = math.sqrt(norm)
    res.distance_ci = math.sqrt(norm_ci)
    res.wave_function = cc_wf
    res.right_dir = right_dir
    return res


class MinDistResults(OptResults, DistResults):
    """Store the results from calc_dist_to_cc_manifold"""
    pass


def calc_dist_to_cc_manifold(wf,
                             level='SD',
                             maxiter=10,
                             f_out=None,
                             diag_hess=True,
                             save_as_fci_wf=False,
                             thrsh_Z=1.0E-8,
                             thrsh_J=1.0E-8,
                             ini_wf=None,
                             use_FCI_directly=False,
                             recipes_f=None,
                             coeff_thr=1.0E-10):
    """Calculate the distance to the coupled cluster manifold

    Side Effects:
    -------------
    The wave function wf is put in the intermediate normalisation and
    with the convention of ordered orbitals for the signs
    of the determinants

    Parameters:
    -----------
    wf (FCIWaveFunction)
        The wave function to be analysed
    
    level (str, optional, default='D'; possible values: 'D', 'SD')
        The level of coupled cluster theory to compare.

    maxiter (int, optional, default=10)
        The maximum number of iterations

    f_out (File object, optional, default=None)
        The output to print the iterations.
        If None the iterations are not printed.

    diag_hess (bool, optional, default=True)
        Use approximate Hessian (only diagonal elements)

    save_as_fci_wf (bool, optional, default=False)
        If True, save the final cc_wf as a FCI wave function too,
        in the attribute wave_function_as_fci of the result

    thrsh_Z (float, optional, default=1.0E-8)
        Convergence threshold for z

    thrsh_J (float, optional, default=1.0E-8)
        Convergence threshold for the Jacobian

    ini_wf (None or an instance of WaveFunction)
        The initial wave function.
        If None or an instance of FCIWaveFunction,
        the "vertical" projection of wf or of ini_wf is used.

    use_FCI_directly (bool, optional, default=False)
        It is meaningful only if ini_wf is passed and is an instance
        of FCIWaveFunction.
        If True, uses this wave function directly in the optimisation
        procedure, instead getting the vertical projection to the CC
        manifold first. In this case, the first step of the optimisation
        procedure use a "CC wave function" that does not belong to the
        CC manifold.
        If False, uses the vertical projection to the CC manifold.

    recipes_f (str, a file name, optional, default=None)
        The files that describe the cluster decomposition.
        See decompose for the details

    coeff_thr (float, optional, default=1.0E-10)
        Slater determinants with coefficients lower than this
        threshold value are ignored in the analysis
    
    Return:
    -------
    An instance of MinDistResults.
    
    """
    wf.normalise(mode='intermediate')
    converged = False
    normZ = normJ = 1.0
    if ini_wf is None:
        cc_wf = IntermNormWaveFunction.from_projected_fci(wf, 'CC' + level)
    elif isinstance(ini_wf, FCIWaveFunction) and not use_FCI_directly:
        cc_wf = IntermNormWaveFunction.from_projected_fci(ini_wf, 'CC' + level)
    elif isinstance(ini_wf, FCIWaveFunction) and not use_FCI_directly:
        cc_wf = IntermNormWaveFunction.similar_to(wf, 'CC' + level, restricted=False)
    elif isinstance(ini_wf, IntermNormWaveFunction):
        cc_wf = IntermNormWaveFunction.unrestrict(ini_wf)
    else:
        raise ValueError('Unknown type of initial wave function')
    wf.set_ordered_orbitals()
    logger.debug('The FCI Wave Function:\n%s', wf)
    Jac = np.empty(cc_wf.n_indep_ampl)
    if diag_hess:
        Hess = np.empty((1, 1))
    else:
        Hess = np.empty((cc_wf.n_indep_ampl, cc_wf.n_indep_ampl))
    corr_orb = wf.corr_orb.as_array()
    virt_orb = wf.virt_orb.as_array()
    cc_wf_as_fci = FCIWaveFunction.similar_to(wf, restricted=False)
    if f_out is not None:
        f_out.write(
            '------------------------------------------------------------\n'
            '         Optimising the distance to the CC manifold\n'
            ' it  distance    |Z|         |J|          time in iteration\n')
    for i_iteration in range(maxiter):
        with logtime(f'Starting iteration {i_iteration}') as T_iter:
            if (i_iteration == 0
                and isinstance(ini_wf, FCIWaveFunction)
                    and use_FCI_directly):
                cc_wf_as_fci._coefficients[:] = ini_wf._coefficients
            else:
                with logtime('Transforming CC wave function to FCI-like'):
                    cc_wf_as_fci.get_coefficients_from_int_norm_wf(cc_wf,
                                                                   ordered_orbitals=True)
            logger.debug('Wave Function, at iteration %d:\n%s',
                         i_iteration,
                         cc_wf_as_fci)
            with logtime('Distance to current CC wave function'):
                dist = wf.dist_to(cc_wf_as_fci, metric='IN')
            if (i_iteration > 0
                and normJ < thrsh_J
                    and normZ < thrsh_Z):
                converged = True
                break
            with logtime('Making Jacobian and approximate Hessian'):
                cc_manifold.min_dist_jac_hess(
                    wf._coefficients,
                    cc_wf_as_fci._coefficients,
                    Jac,
                    Hess,
                    wf.orbs_before,
                    corr_orb,
                    virt_orb,
                    wf._alpha_string_graph,
                    wf._beta_string_graph,
                    diag_hess,
                    level=level)
            if diag_hess:
                z = Jac
                normJ = Hess[0, 0]
            else:
                if loglevel <= 1:
                    to_log = ['Jacobian:\n']
                    for iH in range(Jac.shape[0]):
                        to_log.append(f'{iH:3d} {Jac[iH]:8.5f}\n')
                    to_log.append('\n\nHessian:\n')
                    to_log.append('    ')
                    for jH in range(Hess.shape[1]):
                        to_log.append(f' {jH:8d} ')
                    to_log.append('\n')
                    for iH in range(Hess.shape[0]):
                        to_log.append(f'{iH:3d} ')
                        for jH in range(Hess.shape[1]):
                            to_log.append(f' {Hess[iH, jH]:8.5f} ')
                        to_log.append('\n')
                    to_log.append('\n')
                    logger.log(1, '%s', ''.join(to_log))
                with logtime('Calculating z: Solving linear system.'):
                    z = -linalg.solve(Hess, Jac)
                normJ = linalg.norm(Jac)
            logger.log(1, 'Update vector z:\n%r', z)
            normZ = linalg.norm(z)
            cc_wf.update_amplitudes(z, mode='indep ampl')
        if f_out is not None:
            f_out.write(f' {i_iteration:>2d}  {dist:<9.6f}   {normZ:<10.3e}'
                        f'  {normJ:<10.3e}   {T_iter.elapsed_time}\n')
            f_out.flush()
    if f_out is not None:
        f_out.write('-----------------------------------------------------------\n\n')
    res = MinDistResults(f'Minimun distance to CC{level} manifold')
    res.level = level
    res.success = converged
    if not converged:
        res.error = 'Optimisation did not converge'
    res.distance = dist
    res.wave_function = cc_wf
    if save_as_fci_wf:
        res.wave_function_as_fci = cc_wf_as_fci
    res.norm = (normZ, normJ)
    res.n_iter = i_iteration
    return res


_wf_space_graph_full = """

                          CC manifold
                            /
                    FCI    /
                          minD CC   CC wave function
                         /          closest to FCI
                        /
                       CC    regular CC
                      /      wave function
                     /
                    vert CC  the vertical projection
                   /         into the CC manifold
                  /
                 /
                /        the regular CI wave function
 -------------------x---CI---------------
CI manifold         ^
                 vert CI   the vertical projection
                           into the CI manifold
"""

_wf_space_graph_no_ci = """

                          CC manifold
                            /
                    FCI    /
                          minD_CC   CC wave function
                         /          closest to FCI
                        /
                       CC    regular CC
                      /      wave function
                     /
                    vert_CC  the vertical projection
                   /         into the CC manifold
                  /
                 /
                /
 -------------------x--------------------
CI manifold         ^
                 vert CI   the vertical projection
                           into the CI manifold
"""

_wf_space_graph_no_cc = """

                          CC manifold
                            /
                    FCI    /
                          minD_CC   CC wave function
                         /          closest to FCI
                        /
                       /
                      /
                     /
                    vert_CC  the vertical projection
                   /         into the CC manifold
                  /
                 /
                /        the regular CI wave function
 -------------------x---CI---------------
CI manifold         ^
                 vert_CI   the vertical projection
                           into the CI manifold
"""

_wf_space_graph_no_cc_ci = """

                          CC manifold
                            /
                    FCI    /
                          minD_CC   CC wave function
                         /          closest to FCI
                        /
                       /
                      /
                     /
                    vert_CC  the vertical projection
                   /         into the CC manifold
                  /
                 /
                /
 -------------------x--------------------
CI manifold         ^
                 vert_CI   the vertical projection
                           into the CI manifold
"""


class AllDistResults(Results):
    """Store the results from calc_all_distances
    
    Extra attributes:
    -----------------
    full_names (bool)
        If True, print the full(er) names of points
    
    Several floats that store the distances among main points at the
    CI and CC manifolds. These are XX__YY and XX__YY_ampl, with the
    distance between XX and YY, in the metric of the intermediate
    normalisation (in the full space) and in the parameter space of
    the amplitudes (whenever possible). These attributes are:
    
    fci__min_d
    fci__vert
    fci__cc
    fci__vert_ci
    fci__ci
    min_d__vert
    min_d__vert_ampl
    cc__vert
    cc__vert_ampl
    cc__min_d
    cc__min_d_ampl
    
    
    """
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.has_ci = False
        self.has_cc = False
        
    @inside_box
    def __str__(self):
        x = []
        x.append(f'D(FCI, minD_CC) = {self.fci__min_d:.5f}')
        try:
            x.append(f'D(FCI, minD_CC) = {self.fci__min_d_expl:.5f} (explicit calculation)')
        except AttributeError:
            pass
        x.append(f'D(FCI, vert_CC) = {self.fci__vert:.5f}')
        try:
            x.append(f'D(FCI, vert_CC) = {self.fci__vert_expl:.5f} (explicit calculation)')
        except AttributeError:
            pass
        x.append(f'D(FCI, vert_CI) = {self.fci__vert_ci:.5f}')
        if self.has_cc:
            x.append(f'D(FCI, CC)      = {self.fci__cc:.5f}')
        if self.has_ci:
            x.append(f'D(FCI, CI)      = {self.fci__ci:.5f}')
        x.append('')
        if self.has_ci and self.has_cc:
            x.append(f'D(CC, CI)           = {self.ci__cc:.5f} ')
        x.append('')
        x.append(f'D(minD_CC, vert_CC) = {self.min_d__vert:.5f} '
                 + f'({self.min_d__vert_ampl:.5f} in ampl space)')
        if self.has_cc:
            x.append(f'D(CC, vert_CC)      = {self.cc__vert:.5f} '
                     + f'({self.cc__vert_ampl:.5f} in ampl space)')
            x.append(f'D(CC, minD_CC)      = {self.cc__min_d:.5f} '
                     + f'({self.cc__min_d_ampl:.5f} in ampl space)')

        return (super().__str__() + '\n'
                + (_wf_space_graph_full
                   if (self.has_ci and self.has_cc) else
                   (_wf_space_graph_no_cc
                    if self.has_ci else
                    (_wf_space_graph_no_ci
                     if self.has_cc else
                     _wf_space_graph_no_cc_ci))) + '\n'
                + '\n'.join(x))


def calc_all_distances(fci_wf, res_vert, res_min_d, cc_wf, ci_wf, level,
                       explicit_calcs=False):
    """Calculate all distances among main points in the CC and CI manifolds
    
    
    Parameters:
    -----------
    fci_wf (FCIWaveFunction)
        The FCI wave function
    
    res_vert (VertDistResults)
        The results of the vertical_dist_to_cc_manifold
    
    res_min_d (MinDistResults)
        The results of the calc_dist_to_cc_manifold
    
    cc_wf (IntermNormWaveFunction)
        The CC wave function (optimised, that solves the CC equations)
        If None distances to it are not calculated
    
    ci_wf (IntermNormWaveFunction)
        The CI wave function (optimised, that solves the CI equations)
        If None distances to it are not calculated
    
    level (str, 'D' or 'SD')
        The level of excitations used in the CC wave functions
    
    explicit_calcs (bool, optional, default=False)
        Explicitly calculate the distances between fci and the wave functions
        in res_min_d and res_vert. This should be the same as stored
        in these variables.
    
    Results:
    --------
    An instance of AllDistResults
    
    TODO:
    -----
    calculate distance from vert(CI) to vert(CC)
    
    """
    res = AllDistResults(f'Distances among CC{level}/CI{level} wave functions')
    res.fci__min_d = res_min_d.distance
    if explicit_calcs:
        res.fci__min_d_expl = res_min_d.wave_function_as_fci.dist_to(fci_wf)
    res.fci__vert = res_vert.distance
    if explicit_calcs:
        res.fci__vert_expl = res_vert.wave_function_as_fci.dist_to(fci_wf)
    res.fci__vert_ci = res_vert.distance_ci
    res.min_d__vert_ampl = res_min_d.wave_function.dist_to(res_vert.wave_function)
    res.min_d__vert = res_min_d.wave_function_as_fci.dist_to(
        res_vert.wave_function_as_fci)
    if cc_wf is not None:
        res.has_cc = True
        cc_as_fci = FCIWaveFunction.from_int_norm(cc_wf)
        res.fci__cc = cc_as_fci.dist_to(fci_wf)
        res.cc__min_d_ampl = cc_wf.dist_to(res_min_d.wave_function)
        res.cc__min_d = cc_as_fci.dist_to(res_min_d.wave_function_as_fci)
        res.cc__vert_ampl = cc_wf.dist_to(res_vert.wave_function)
        res.cc__vert = cc_as_fci.dist_to(res_vert.wave_function_as_fci)
    if ci_wf is not None:
        res.has_ci = True
        ci_as_fci = FCIWaveFunction.from_int_norm(ci_wf)
        res.fci__ci = ci_as_fci.dist_to(fci_wf)
        if cc_wf is not None:
            res.ci__cc = ci_as_fci.dist_to(cc_as_fci)
    res.success = True
    return res
