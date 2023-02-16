"""A general optimiser for Hartree-Fock

"""
import sys
import copy
import logging
from collections import namedtuple

import numpy as np

from util.results import OptResults
from input_output.log import logtime
from . import util
from .iteration_step import HartreeFockStep
from orbitals.orbitals import MolecularOrbitals

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


def _check_nelec_ms(n_elec, restricted, ms2):
    if restricted is None:
        restricted = n_elec % 2 == 0 and ms2 == 0
    if ms2 is None:
        ms2 = n_elec % 2
    if (n_elec - ms2) % 2 != 0:
        raise ValueError(f'Number of electrons, {n_elec} not compatible with 2MS, {ms2}.')
    if restricted and not(n_elec % 2 == 0 and ms2 == 0):
        raise ValueError('Restricted calculation requires'
                         ' an even number of electrons and ms=0')
    return restricted, ms2


DiisInfo = namedtuple('DiisInfo',
                      ['n', 'at_F', 'at_P'])


def hartree_fock(integrals,
		 nucl_rep,
		 n_elec,
                 ini_orb,
                 ms2=None,
                 restricted=None,
                 max_iter=20,
                 grad_thresh=1.0E-5,
                 grad_type='F_occ_virt',
                 energy_thresh=1.0E-10,
                 diis_info=None,
                 f_out=sys.stdout,
                 conjugacy=None,
                 step_size=1.0,
                 HF_step_type=lambda **x: "RH-SCF"):
    """A general Hartree-Fock procedure

    
    integrals (Integrals)
        The molecular integrals in the basis

    nucl_rep (float)
        The nuclear repulsion

    n_elec (int)
    	The number of electrons (must be even)
    
    ini_orb (MolecularOrbitals)
        Initial orbitals for the HF procedure

    ms2 (int)
        Two times MS (n alpha minus n beta).
        If None, use the lowest value compatible with the number of electrons

    restricted (bool)
        Restricted calculation. If None use True for systems with even
        number of electrons, False for odd number of electrons

    max_iter (int, optional, default=20)
	Maximum number of iterations
    
    grad_thresh (float, optional, default=1.0E-5)
	Threshold for the norn of the gradient

    grad_type (str, optional, default='F_occ_virt')
        Gradient type for SCF. Possible valies are:
            'F_occ_virt' - occupied-virtual block of Fock matrix
            'F_asym' - anti-symmetric part of generalized Fock matrix
    
    diis_info (int, optional, default=None)
        Information regarding DIIS
    
    f_out (File object, optional, default=sys.stdout)
        The output to print the iterations.
        If None the iterations are not printed.

    conjugacy (string, optional, default=None)
        Only makes sense in the Conjugate Gradient Method. Options: "PR" and "FR"
        to use the Polak--Ribière or the Fletcher--Reeves formula in the conjugacy
        coefficient.

    HF_step_type (callable, optional, default=lambda **x:"RH-SCF")
        A callable that receives the following arguments:
            i_SCF (int)
            grad (np.ndarray, optional)
        and return one of the following strings:
            "RH-SCF", "densMat-SCF", "RRN", "orb_rot-Newton"
        that will dictate which method should be used in the
        Hartree-Fock optimisation, in that iteration
    """
    restricted, ms2 = _check_nelec_ms(n_elec, restricted, ms2)
    kind_of_calc = 'closed-shell RHF' if restricted else 'UHF'
    converged = False
    hf_step = HartreeFockStep()
    hf_step.restricted = restricted
    hf_step.integrals = integrals
    hf_step.N_a = (n_elec + ms2) // 2
    hf_step.N_b = (n_elec - ms2) // 2
    hf_step.n_occ = n_elec
    hf_step.diis_info = (DiisInfo(n=0, at_F=False, at_P=False)
                         if diis_info is None else
                         diis_info)
    hf_step.grad_type = grad_type
    hf_step.conjugacy = conjugacy
    hf_step.step_size = step_size
    logger.info('Starting Hartree-Fock calculation. Type:%s\n', kind_of_calc)
    logger.info('Nuclear repulsion energy: %f\n'
                'Number of  electrons: %d\n'
                'DIIS: %s\n'
                'Restricted: %s\n'
                '2ms: %s\n'
                'grad type: %s\n',
                nucl_rep,
                n_elec,
                hf_step.diis_info,
                restricted,
                ms2,
                grad_type
    )
    if restricted:
        logger.info('Number of occupied orbitals: %d', hf_step.N_a)
    else:
        logger.info('Number of occupied orbitals: %d (alpha), %d (beta)',
                    hf_step.N_a,
                    hf_step.N_b)
    hf_step.orb = MolecularOrbitals(ini_orb)

    logger.debug('Initial molecular orbitals:\n %s', hf_step.orb)
    assert hf_step.orb.is_orthonormal(integrals.S,
                                      N=(hf_step.N_a
                                         if restricted else
                                        (hf_step.N_a, hf_step.N_b))), \
        "Initial orbitals are not orthonormal"
    hf_step.initialise(HF_step_type(i_SCF=0, grad_norm=hf_step.grad_norm))

    if f_out is not None:
        f_out.write(util.write_header)

    for i_SCF in range(max_iter):
        step_type = HF_step_type(i_SCF=i_SCF, grad_norm=hf_step.grad_norm)
        logger.info('Starting HF iteration %d', i_SCF)
        with logtime('HF iteration') as T:
            if step_type == 'RH-SCF':
                hf_step.roothan_hall(i_SCF)

            elif step_type == 'densMat-SCF':
                hf_step.density_matrix_scf(i_SCF)

            elif step_type == 'RNR':
                hf_step.RNR(i_SCF)

            elif step_type == 'orb_rot-Newton':
                hf_step.newton_orb_rot(i_SCF)

            elif step_type == 'NRLM':
                hf_step.NRLM(i_SCF)

            elif step_type == 'RGD':
                hf_step.RGD(i_SCF)

            elif step_type == 'GDLM':
                hf_step.GDLM(i_SCF)

            elif step_type == 'RCG':
                hf_step.RCG(i_SCF)

            else:
                raise ValueError("Unknown type of Hartree-Fock step: "
                                 + step_type)

        if f_out is not None:
            f_out.write((util.fmt_HF_iter_gen_lag
                         if step_type in {'NRLM', 'GDLM'} else
                         util.fmt_HF_iter_general).format(
                             i_SCF,
                             nucl_rep + hf_step.energy,
                             hf_step.grad_norm,
                             hf_step.restriction_norm,
                             step_type,
                             T.elapsed_time))
            f_out.flush()

        if hf_step.grad_norm < grad_thresh or abs(hf_step.diff_energy) < energy_thresh:
            logger.info('Convergence reached in %d iterations.', i_SCF)
            converged = True
            break

    assert hf_step.orb.is_orthonormal(integrals.S,
                                      hf_step.N_a
                                      if restricted else
                                      (hf_step.N_a, hf_step.N_b)), \
                                      "Final orbitals are not orthonormal"
    res = OptResults(kind_of_calc)
    res.energy = nucl_rep + hf_step.energy
    res.orbitals = copy.deepcopy(hf_step.orb)
    res.orbitals.name = 'RHF' if restricted else 'UHF'
    logger.debug('Final orbitals:\n%s', res.orbitals)
    hf_step.calc_density_matrix()
    if restricted:
        res.density = hf_step.P_a
    else:
        res.density = hf_step.P_a, hf_step.P_b
    res.success = converged
    res.n_iter = i_SCF
    if hf_step.large_cond_number:
        res.warning = 'Large conditioning number at iterations: '
        if len(hf_step.large_cond_number) == 1:
            res.warning += f'{hf_step.large_cond_number[0]}'
        if len(hf_step.large_cond_number) == 2:
            res.warning += f'{hf_step.large_cond_number[0]}, {hf_step.large_cond_number[-1]}'
        if len(hf_step.large_cond_number) > 3:
            res.warning += f'{hf_step.large_cond_number[0]}, ..., {hf_step.large_cond_number[-1]}'
            res.warning += \
                '\n          (look for "Large conditioning number" at the log file)'

    if not converged:
        logger.info('End of Hartree-Fock calculation')
    return res
