"""A general optimiser for Hartree-Fock

"""
import sys
import logging

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

def hf_initial_orbitals(ini_orb, integrals, restricted):
    if ini_orb is None:
        logger.info('Using eigenvectors of orthogonalised h as starting guess')
        return MolecularOrbitals.from_eig_h(
            integrals,
            integrals.basis_set + '(AO)',
            restricted=restricted)
    else:
        logger.info('Using orbitals given by the user as initial guess.')
        if ini_orb.restricted and not restricted:
            raise ValueError('Initial orbitals should be of unrestricted type.')
        orb = MolecularOrbitals(ini_orb)
        return orb.orthogonalise(X=integrals.X)


def hartree_fock(integrals,
		 nucl_rep,
		 n_elec,
                 ms2=None,
                 restricted=None,
                 max_iter=20,
                 grad_thresh=1.0E-5,
                 f_out=sys.stdout,
                 n_DIIS=0,
                 HF_step_type=lambda **x: "RH-SCF",
                 ini_orb=None):
    """A general Hartree-Fock procedure

    
    integrals (Integrals)
        The molecular integrals in the basis

    nucl_rep (float)
        The nuclear repulsion

    n_elec (int)
    	The number of electrons (must be even)

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
    
    f_out (File object, optional, default=sys.stdout)
        The output to print the iterations.
        If None the iterations are not printed.
    
    n_DIIS (int, optional, default=0)
        maximum diminsion of the iterative subspace in DIIS
        0 means that DIIS is not used
    
    HF_step_type (callable, optional, default=lambda **x:"RH-SCF")
        A callable that receives the following arguments:
            i_SCF (int)
            grad (np.ndarray, optional)
        and return one of the following strings:
            "RH-SCF", "densMat-SCF", "Absil", "orb_rot-Newton"
        that will dictate which method should be used in the
        Hartree-Fock optimisation, in that iteration
    
    ini_orb (MolecularOrbitals, optional, default=None)
        Initial orbitals for the HF procedure
    """
    restricted, ms2 = _check_nelec_ms(n_elec, restricted, ms2)
    kind_of_calc = 'closed-shell RHF' if restricted else 'UHF'
    converged = False
    hf_step = HartreeFockStep()
    hf_step.restricted = restricted
    hf_step.integrals = integrals
    hf_step.n_occ_alpha = (n_elec + ms2) // 2
    hf_step.n_occ_beta = (n_elec - ms2) // 2
    hf_step.n_occ = n_elec
    hf_step.n_DIIS = n_DIIS
    logger.info('Starting Hartree-Fock calculation. Type:%s\n', kind_of_calc)
    logger.info('Nuclear repulsion energy: %f\n'
                'Number of  electrons: %d\n'
                'n DIIS: %d\n',
                nucl_rep,
                n_elec,
                hf_step.n_DIIS)
    if restricted:
        logger.info('Number of occupied orbitals: %d', hf_step.n_occ_alpha)
    else:
        logger.info('Number of occupied orbitals: %d (alpha), %d (beta)',
                    hf_step.n_occ_alpha,
                    hf_step.n_occ_beta)
    hf_step.orb = hf_initial_orbitals(ini_orb, integrals, restricted)
    logger.debug('Initial molecular orbitals:\n %s', hf_step.orb)
    assert hf_step.orb.is_orthonormal(integrals.S), "Orbitals are not orthonormal"
    hf_step.initialise(HF_step_type(i_SCF=0, grad_norm=100.0), True)
    
    if f_out is not None:
        f_out.write(util.fmt_HF_header_general.format('it.', 'E',
                                                      '|Gradient|',
                                                      'step',
                                                      'time in iteration'))
        
    for i_SCF in range(max_iter):
        logger.info('Starting HF iteration %d', i_SCF)
        step_type = HF_step_type(i_SCF=i_SCF, grad_norm=hf_step.grad_norm)
        with logtime('HF iteration') as T:
            if step_type == 'RH-SCF':
                hf_step.roothan_hall(i_SCF)
                
            elif step_type == 'densMat-SCF':
                hf_step.density_matrix_scf(i_SCF)
                
            elif step_type == 'Absil':
                hf_step.newton_absil(i_SCF)
                
            elif step_type == 'orb_rot-Newton':
                hf_step.newton_orb_rot(i_SCF)
                
            else:
                raise ValueError("Unknown type of Hartree-Fock step: "
                                 + step_type)
        
        if f_out is not None:
            f_out.write(util.fmt_HF_iter_general.format(
                i_SCF, nucl_rep + hf_step.energy,
                hf_step.grad_norm, step_type, T.elapsed_time))
            f_out.flush()
            
        if hf_step.grad_norm < grad_thresh:
            logger.info('Convergence reached in %d iterations.', i_SCF)
            converged = True
            break

    hf_step.orb.name = 'RHF' if restricted else 'UHF'
    res = OptResults(kind_of_calc)
    res.energy = nucl_rep + hf_step.energy
    res.orbitals = hf_step.orb
    res.success = converged
    res.n_iter = i_SCF
    if not converged:
        res.warning = 'No convergence was obtained'
        logger.info('End of Hartree-Fock calculation')
    return res
