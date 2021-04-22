"""The distance between the CC manifold and and arbitrary (FCI) wave function




"""
import copy
import math
import logging

import numpy as np
from numpy import linalg

from input_output.log import logtime
from util.results import Results, OptResults
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
    and the sign of the coefficients should be associated to the determinants
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
    An instance of Results.
    Some attributes:
    distance: the "vertical" distance between wf and the CC manifold,
              in the metric induced by intermediate normalisation.
    wave_function: the wave function at the CC manifold, as an instace
                   of IntermNormWaveFunction. It is obtained by
                   IntermNormWaveFunction.from_projected_fci
    right_dir: a dictionary, with keys as ranks, and values as 2-tuples,
               that show [0] how many directions are curved towards wf
               and [1] the total number of such directions. Example:
               {'T': (10,12),
                'Q': (5,5),
                '5': (1,1)}
                10 out of 12 triples are in the right direction
                5 out of 5 quadruples are in the right direction
                1 out of 1 quintuples are in the right direction
    
    """
    logfmt = '\n'.join(['det = %s',
                        'C dec = %s',
                        'C dec/C = %s',
                        '(C dec - C)^2 = %s',
                        'same sign = %s'])
    if level not in ['D', 'SD']:
        raise ValueError('Possible values for level are "S" and "SD"')
    wf.normalise(mode='intermediate')
    wf.set_max_coincidence_orbitals()
    cc_wf = IntermNormWaveFunction.from_projected_fci(wf, 'CC' + level)
    norm = 0.0
    right_dir = {}
    for det in wf:
        if not wf.symmetry_allowed_det(det):
            continue
        rank, alpha_hp, beta_hp = wf.get_exc_info(det)
        if rank == 1 and level == 'SD':
            if cc_wf[rank, alpha_hp, beta_hp] != det.c:
                raise Exception('Not consistent!!')
        if (abs(det.c) > coeff_thr
            and rank > 2
                and (level == 'SD' or rank % 2 == 0)):
            C = contribution_from_clusters(alpha_hp, beta_hp, cc_wf, level)
            norm_contribution = (det.c - C)**2
            cc_towards_wf = det.c * C >= 0
            norm += norm_contribution
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
                        C/det.c,
                        norm_contribution,
                        cc_towards_wf)
        elif rank > 2 or (rank, level) == (1, 'D'):
            norm += det.c**2
            logger.info('Adding .c² to the norm: det = %s', det)

    tolog = ['Number of excitations where the CC manifold\n'
             + '   curves towards the wave function:']
    for rank, n in right_dir.items():
        tolog.append(f'{rank}: {n[0]} of {n[1]}')
    logger.info('\n'.join(tolog))
    res = Results('Vertical distance to CC manifold')
    res.distance = math.sqrt(norm)
    res.wave_function = cc_wf
    res.right_dir = right_dir
    return res


def calc_dist_to_cc_manifold(wf,
                             level='SD',
                             maxiter=10,
                             f_out=None,
                             diag_hess=True,
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
    with the convention of orbitals with maximum coincidence for the signs
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
    An instance of OptResults.

    """
    wf.normalise(mode='intermediate')
    wf.set_max_coincidence_orbitals()
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
    n_ampl = len(cc_wf)
    corr_orb = wf.corr_orb.as_array()
    virt_orb = wf.virt_orb.as_array()
    cc_wf_as_fci = FCIWaveFunction.similar_to(wf, restricted=False)
#    logger.info('Wave Function, before iterations:\n%s', wf)
    if f_out is not None:
        f_out.write(
            'it.   dist      |Z|       |J|        time in iteration\n')
    for i_iteration in range(maxiter):
        with logtime(f'Starting iteration {i_iteration}') as T_iter:
            if (i_iteration == 0
                and isinstance(ini_wf, FCIWaveFunction)
                    and use_FCI_directly):
                cc_wf_as_fci._coefficients[:] = ini_wf._coefficients
            else:
                with logtime('Transforming CC wave function to FCI-like'):
                    cc_wf_as_fci.get_coefficients_from_int_norm_wf(cc_wf)
#            logger.info('Wave Function, at iteration %d:\n%s',
#                        i_iteration,
#                        cc_wf_as_fci)
            with logtime('Distance to current CC wave function'):
                dist = wf.dist_to(cc_wf_as_fci, metric='IN')
            if (i_iteration > 0
                and normJ < thrsh_J
                    and normZ < thrsh_Z):
                converged = True
                break
            with logtime('Making Jacobian and approximate Hessian'):
                Jac, Hess = cc_manifold.min_dist_jac_hess(
                    wf._coefficients,
                    cc_wf_as_fci._coefficients,
                    n_ampl,
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
                    logger.log(1, 'Jacobian:\n%r', np.array(Jac))
                    logger.log(1, 'Hessian:\n%r', np.array(Hess))
                with logtime('Calculating z: Solving linear system.'):
                    z = -linalg.solve(Hess, Jac)
                normJ = linalg.norm(Jac)
            logger.log(1, 'Update vector z:\n%r', z)
            normZ = linalg.norm(z)
            cc_wf.update_amplitudes(z)
        if f_out is not None:
            f_out.write(f' {i_iteration:<4} {dist:7.5}  {normZ:6.4}'
                        f'  {normJ:6.4}   {T_iter.elapsed_time}\n')
            f_out.flush()
    res = OptResults('Minimun distance to CC manifold')
    res.success = converged
    if not converged:
        res.error = 'Optimisation did not converge'
    res.distance = dist
    res.wave_function = cc_wf
    res.norm = (normZ, normJ)
    res.n_iter = i_iteration
    return res
