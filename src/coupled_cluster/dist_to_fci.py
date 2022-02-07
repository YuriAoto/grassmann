"""The distance between the CC manifold and and arbitrary (FCI) wave function

This module contains functions to calculate the distance among a number of
points in the space of wave functions.


"""
import math
import logging

import numpy as np
from numpy import linalg
import numpy as np

from input_output.log import logtime
from util.results import Results, OptResults, inside_box
from coupled_cluster import manifold as cc_manifold
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.fci import FCIWaveFunction, contribution_from_clusters
from wave_functions.slater_det import SlaterDet
import wave_functions.strings_rev_lexical_order as str_order
from util.other import int_array
from util.variables import int_dtype

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

_str_excitation_list = ['R',
                        'S',
                        'D',
                        'T',
                        'Q']


def _str_excitation(x):
    return _str_excitation_list[x] if x < 5 else str(x)

def _str_by_rank(by_rank):
    x = []
    x.append(f' by rank:          S          D          T          Q')
    x.append(f'        : '
             + ' '.join([f'{d:10.8f}' for d in by_rank[1:5]]))
    x.append(f'                   5          6          7          8')
    x.append(f'        : '
             + ' '.join([f'{d:10.8f}' for d in by_rank[5:]]))
    return '\n'.join(x)


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
    
    norm_rank (9-len np.ndarray of int)
        The contribution by rank for the distance. The following must hold:
        distance == sqrt(sum[i**2 for i in norm_rank])
    
    """
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._wave_function_as_fci = None
    
    def check_dist_by_rank(self):
        d = math.sqrt(sum([i**2 for i in self.norm_rank]))
        return (f'dist:       {self.distance:.8}\n'
                f'from rank:  {d:.8}\n'
                f'difference: {self.distance-d:.8}\n')
    
    @inside_box
    def __str__(self):
        x = [super().__str__()]
        x.append(f'D(FCI, CC{self.level} manifold) = {self.distance:.8f}')
        x.append(_str_by_rank(self.norm_rank))
#        x.append(self.check_dist_by_rank())
        return '\n'.join(x)
    
    @property
    def wave_function_as_fci(self):
        if self._wave_function_as_fci is None:
            self._wave_function_as_fci = FCIWaveFunction.from_interm_norm(self.wave_function)
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
    
    norm_ci_rank (9-len np.ndarray of int)
        The contribution by rank for the distance_ci. The following must hold:
        distance_ci == sqrt(sum[i**2 for i in norm_ci_rank])
    
    coeff_thr (list of floats)
        The parameter coeff_thr passed to vertical_dist_to_cc_manifold.
    
    right_dir (list of dictionary)
        Indicates how many coefficients are in the "right direction" compared
        to the CC manifold.
        The list is ordered as coeff_thr, and each entry is a dictionary,
        that indicates how many coefficients are in the right direction
        when considering the corresponding threshold for the coefficient
        of the list coeff_thr.
        The keys are the ranks, and the values are 2-tuples,
        that show at [0] how many directions are curved towards the wave
        function and at [1] the total number of such directions. Example:
        
        for coeff_thr = [1e-6, 1e-3]
        
        [{'T': (10,12),
        'Q': (5,5),
        '5': (1,1)},
        {'T': (8,10),
        'Q': (2,4),
        '5': (1,1)}]
        
        Means:
        When considering only coefficients larger than 1e-6:
        10 out of 12 triples are in the right direction
        5 out of 5 quadruples are in the right direction
        1 out of 1 quintuples are in the right direction
        
        When considering only coefficients larger than 1e-3:
        8 out of 10 triples are in the right direction
        2 out of 4 quadruples are in the right direction
        1 out of 1 quintuples are in the right direction]
    
    """
    def check_dist_by_rank_ci(self):
        d = math.sqrt(sum([i**2 for i in self.norm_ci_rank]))
        return (f'dist:       {self.distance_ci:.8}\n'
                f'from rank:  {d:.8}\n'
                f'difference: {self.distance_ci-d:.8}\n')
    
    @inside_box
    def __str__(self):
        x = [super().__str__().replace('D(FCI', 'D_vert(FCI')]
        x.append(f'D(FCI, CI{self.level} manifold) = {self.distance_ci:.8f}')
        x.append(_str_by_rank(self.norm_ci_rank))
#        x.append(self.check_dist_by_rank_ci())
        x.append('Number of excitations where the CC manifold\n'
                 + '   curves towards the wave function:')
        rank_info = []
        len_info = 0
        for rank in self.right_dir[0]:
            rank_info.append([f'{rank}: '])
            for r_dir in self.right_dir:
                rkinfo = (f'{r_dir[rank][0]} of {r_dir[rank][1]}'
                          if r_dir[rank][1] else '')
                rank_info[-1].append(rkinfo)
                if len(rkinfo) > len_info:
                    len_info = len(rkinfo)
        len_info = str(len_info)
        fmt_h = '{:>' + len_info + '.1e}'
        fmt = '{:>' + len_info + '}'
        x.append('   ' + '  '.join(map(fmt_h.format, self.coeff_thr)))
        for rkinfo in rank_info:
            x.append(rkinfo[0] + '  '.join(map(fmt.format, rkinfo[1:])))
        return '\n'.join(x)


def _check_right_direction(cc_towards_wf, rank, right_dir, coeff_thr, abs_c):
    """Check if in the right direction, and update right_dir
    
    Helper to vertical_dist_to_cc_manifold, to update
    the list of dicts right_dir.
    
    """
    rank = _str_excitation(rank)
    if rank not in right_dir[0]:
        for r_dir in right_dir:
            r_dir[rank] = [0, 0]
    for i, check in enumerate(map(lambda c_thrs: abs_c > c_thrs, coeff_thr)):
        if check:
            right_dir[i][rank][1] += 1
            if cc_towards_wf:
                right_dir[i][rank][0] += 1


def vertical_dist_to_cc_manifold(wf,
                                 level='SD',
                                 recipes_f=None,
                                 coeff_thr=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
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
    
    coeff_thr (list of floats, optional, default=[1e-6, 1e-5, ... 1e-1])
        Slater determinants with coefficients lower than this
        thresholds value are ignored in the analysis.
        See VertDistResults for a proper description.
    
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
    norm_rank = np.zeros(9)
    norm_ci_rank = np.zeros(9)
    right_dir = [{} for i in range(len(coeff_thr))]
    min_coeff_thr = min(coeff_thr)
    for det in wf:
        if not wf.symmetry_allowed_det(det):
            continue
        rank, alpha_hp, beta_hp = wf.get_exc_info(det)
        c2 = det.c**2
# ----------
# This test was done to check consistency of the implementation
# Now, after changing __getitem__ of IntermNormWaveFunction,
# this does not wark, as it requires an SDExcitation.
# FCIWaveFunction.get_exc_info, however, returns
# rank, alpha_hp, beta_hp, separated, as it can be a higher order
# excitation... We will disable this test, although one can create
# an SDExcitation.  
#
#
#        if rank == 1 and level_is_sd:
#            if cc_wf[rank, alpha_hp, beta_hp] != det.c:
#                raise Exception('Not consistent!!')
# ----------
        if rank > 2 and (level_is_sd or rank % 2 == 0):
            C = contribution_from_clusters(alpha_hp, beta_hp, cc_wf, level)
            norm_contribution = (det.c - C)**2
            norm_ci += c2
            cc_towards_wf = det.c * C >= 0
            norm += norm_contribution
            norm_rank[rank] += norm_contribution
            norm_ci_rank[rank] += c2
            abs_c = abs(det.c)
            if abs_c > min_coeff_thr:
                _check_right_direction(cc_towards_wf, rank, right_dir, coeff_thr, abs_c)
            logger.info(logfmt,
                        det,
                        C,
                        C/det.c if (abs(det.c) > min_coeff_thr) else f'{C}/{det.c}',
                        norm_contribution,
                        cc_towards_wf)
        elif (not level_is_sd) and rank % 2 == 1:
            #  rank > 2 or (rank, level) == (1, 'D'):
            norm += c2
            norm_ci += c2
            norm_rank[rank] += c2
            norm_ci_rank[rank] += c2
            logger.info('Adding .c² to the norm: det = %s', det)
    if loglevel <= logging.INFO:
        tolog = ['Number of excitations where the CC manifold\n'
                 + '   curves towards the wave function:']
        for i, c_thr in enumerate(coeff_thr):
            tolog.append(f'For coeff threshold of {c_thr}')
            for rank, n in right_dir[i].items():
                tolog.append(f'{rank}: {n[0]} of {n[1]}')
        logger.info('\n'.join(tolog))
    res = VertDistResults(f'Vertical distance to CC{level} manifold')
    res.success = True
    res.level = level
    res.distance = math.sqrt(norm)
    res.distance_ci = math.sqrt(norm_ci)
    res.norm_ci_rank = np.sqrt(norm_ci_rank)
    res.norm_rank = np.sqrt(norm_rank)
    res.wave_function = cc_wf
    res.right_dir = right_dir
    res.coeff_thr = coeff_thr
    return res


class MinDistResults(OptResults, DistResults):
    """Store the results from calc_dist_to_cc_manifold"""
    pass


def calc_dist_to_cc_manifold(wf,
                             level='SD',
                             max_iter=10,
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

    max_iter (int, optional, default=10)
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
    cc_wf_as_fci = FCIWaveFunction.similar_to(wf, restricted=False)
    if f_out is not None:
        f_out.write(
            '------------------------------------------------------------\n'
            '         Optimising the distance to the CC manifold\n'
            ' it  distance    |Z|         |J|          time in iteration\n')
    for i_iteration in range(max_iter):
        with logtime(f'Starting iteration {i_iteration}') as T_iter:
            if (i_iteration == 0
                and isinstance(ini_wf, FCIWaveFunction)
                    and use_FCI_directly):
                cc_wf_as_fci._coefficients[:] = ini_wf._coefficients
            else:
                with logtime('Transforming CC wave function to FCI-like'):
                    cc_wf_as_fci.get_coefficients_from_interm_norm_wf(cc_wf,
                                                                      ordered_orbitals=True)
            logger.debug('Wave Function, at iteration %d:\n%r',
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
                cc_manifold.min_dist_jac_hess(wf,
                                              cc_wf_as_fci,
                                              Jac,
                                              Hess,
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
    res = MinDistResults(f'Minimum distance to CC{level} manifold')
    res.level = level
    res.success = converged
    if not converged:
        res.error = 'Optimisation did not converge'
    res.distance = dist
    res.wave_function = cc_wf
    if save_as_fci_wf:
        res.wave_function_as_fci = cc_wf_as_fci
    res.norm = (normZ, normJ)
    res.norm_rank = wf.dist_by_rank(cc_wf_as_fci, metric='IN')
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
            ref /        the regular CI wave function
 --------------X----x---CI---------------
CI manifold   HF    ^
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
            ref /
 --------------X----x--------------------
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
            ref /        the regular CI wave function
 --------------X----x---CI---------------
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
            ref /
 --------------X----x--------------------
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
    the amplitudes (whenever possible). In this case the attribute should
    come with "_ampl".
    
    """
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.has_ci = False
        self.has_cc = False
        
    @inside_box
    def __str__(self):
        x = []
        def dist_str(key):
            to_replace = [('__', ', '),
                          ('_expl', ''),
                          ('_ampl', '')
            ]
            after_str = ''
            if '_ampl' in key: after_str += '_ampl'
            if '_expl' in key: after_str += '_expl'
            for pattern, repl in to_replace:
                key = key.replace(pattern, repl)
            return f'D({key}){after_str}'

        the_distances = [(dist_str(k), k) for k in sorted(self.__dict__) if '__' in k]
        the_distances.reverse()
        maxlen = max(list(map(lambda x: len(x[0]), the_distances)))
        for fmt_k, k in the_distances:
            spaces = ' ' * (maxlen - len(fmt_k))
            x.append(f'{fmt_k}{spaces}  {self.__dict__[k]:.5f}')
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
    vertCI_wf = IntermNormWaveFunction.from_projected_fci(fci_wf, 'CI' + level)
    res.ref__FCI = fci_wf.dist_to_ref()
    res.ref__minD = res_min_d.wave_function_as_fci.dist_to_ref()
    res.ref__vertCC = res_vert.wave_function_as_fci.dist_to_ref()
    res.ref__vertCI = vertCI_wf.dist_to_ref()
    res.FCI__minD = res_min_d.distance
    if explicit_calcs:
        res.FCI__minD_expl = res_min_d.wave_function_as_fci.dist_to(fci_wf)
    res.FCI__vertCC = res_vert.distance
    if explicit_calcs:
        res.FCI__vertCC_expl = res_vert.wave_function_as_fci.dist_to(fci_wf)
    res.FCI__vertCI = res_vert.distance_ci
    if explicit_calcs:
        res.FCI__vertCI_expl = FCIWaveFunction.from_interm_norm(vertCI_wf).dist_to(fci_wf)
    res.minD__vertCC_ampl = res_min_d.wave_function.dist_to(res_vert.wave_function)
    res.minD__vertCC = res_min_d.wave_function_as_fci.dist_to(
        res_vert.wave_function_as_fci)
    if cc_wf is not None:
        res.has_cc = True
        cc_as_fci = FCIWaveFunction.from_interm_norm(cc_wf)
        res.ref__CC = cc_as_fci.dist_to_ref()
        res.FCI__CC = cc_as_fci.dist_to(fci_wf)
        res.CC__minD = cc_as_fci.dist_to(res_min_d.wave_function_as_fci)
        res.CC__minD_ampl = cc_wf.dist_to(res_min_d.wave_function)
        res.CC__vertCC = cc_as_fci.dist_to(res_vert.wave_function_as_fci)
        res.CC__vertCC_ampl = cc_wf.dist_to(res_vert.wave_function)
    if ci_wf is not None:
        res.has_ci = True
        res.ref__CI = ci_wf.dist_to_ref()
        ci_as_fci = FCIWaveFunction.from_interm_norm(ci_wf)
        res.FCI__CI = ci_as_fci.dist_to(fci_wf)
        if cc_wf is not None:
            res.CC__CI = ci_as_fci.dist_to(cc_as_fci)
    res.success = True
    return res
