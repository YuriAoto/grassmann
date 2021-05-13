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


def Restricted_Closed_Shell_SCF(integrals,
				nucl_rep,
				n_elec,
                                max_iter=20,
                                grad_thresh=1.0E-5,
                                f_out=sys.stdout,
                                n_DIIS=0,
                                HF_step_type=lambda **x: "RH-SCF",
                                ini_orb=None):
    """A Restricted Closed Shell SCF Hartree-Fock procedure
    

    
    integrals (Integrals)
        The molecular integrals in the basis

    nucl_rep (float)
        The nuclear repulsion

    n_elec (int)
    	The number of electrons (must be even)

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
    
    HF_step_type (callable, optional, default=lambda x:"RH-SCF")
        A callable that receives the following arguments:
            i_SCF (int)
            grad (np.ndarray, optional)
        and return one of the following strings:
            "RH-SCF", "densMat-SCF", "Absil", "orb_rot-Newton"
        that will dictate which method should be used in the
        Hartree-Fock optimisation, in that iteration
    
    ini_orb (MolecularOrbitals, optional, default=None)
        Initial orbitals for the SCF procedure
    """
    if n_elec % 2 != 0:
        raise ValueError(
            'Closed-shell RHF requires system with even number of electrons.')
    kind_of_calc = 'closed-shell RHF'
    converged = False
    hf_step = HartreeFockStep()
    hf_step.integrals = integrals
    hf_step.n_occ = n_elec // 2
    hf_step.n_DIIS = n_DIIS
    logger.info('Starting Closed-Shell Restricted Hartree-Fock calculation\n'
                + 'Nuclear repulsion energy: %f\n'
                + 'n electrons: %d\n'
                + 'n occupied orbitals: %d\n'
                + 'n DIIS: %d\n',
                nucl_rep, n_elec,
                hf_step.n_occ, hf_step.n_DIIS)
    
    if ini_orb is None:
        logger.info('Using eigenvectors of orthogonalised h as starting guess')
        hf_step.orb = MolecularOrbitals.from_eig_h(
            hf_step.integrals,
            integrals.basis_set + '(AO)')
    else:
        logger.info('Using orbitals given by the user as initial guess.')
        hf_step.orb = MolecularOrbitals(ini_orb)
        hf_step.orb.orthogonalise(X=integrals.X)
    if loglevel <= logging.DEBUG:
        assert hf_step.orb.is_orthonormal(
            integrals.S), "Orbitals are not orthonormal"
        
    hf_step.i_DIIS = -1  # does this have to be inside hf_step
    hf_step.grad = np.zeros((hf_step.n_occ, len(hf_step.orb) - hf_step.n_occ,
                             max(n_DIIS, 1)))
    hf_step.Dmat = np.zeros((len(hf_step.orb),
                             len(hf_step.orb),
                             max(n_DIIS, 1)))
    
    if f_out is not None:
        f_out.write(util.fmt_HF_header_general.format('it.', 'E',
                                                      '|Gradient|',
                                                      'step',
                                                      'time in iteration'))
        
    for i_SCF in range(max_iter):
        logger.info('Starting SCF iteration %d', i_SCF)
        step_type = HF_step_type(i_SCF=i_SCF, grad=hf_step.grad)
        with logtime('SCF iteration') as T:
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
                hf_step.gradNorm, step_type, T.elapsed_time))
            f_out.flush()
            
        if hf_step.gradNorm < grad_thresh:
            logger.info('Convergence reached in %d iterations.', i_SCF)
            converged = True
            break

    hf_step.orb.name = 'RHF'
    res = OptResults(kind_of_calc)
    res.energy = nucl_rep + hf_step.energy
    res.orbitals = hf_step.orb
    res.success = converged
    res.n_iter = i_SCF
    if not converged:
        res.warning = 'No convergence was obtained'
        logger.info('End of Closed-Shell Restricted Hartree-Fock calculation')
    return res


def Unrestricted_SCF(integrals,
		     nucl_rep,
		     n_elec,
                     ms2,
                     max_iter=2,
                     grad_thresh=1.0E-5,
                     f_out=sys.stdout,
                     n_DIIS=0,
                     HF_step_type=lambda **x: "Absil",
                     ini_orb=None):
    """An Unrestricted Closed Shell SCF Hartree-Fock procedure
    

    
    integrals (Integrals)
      	The molecular integrals in the basis

    nucl_rep (float)
	The nuclear repulsion

    n_elec (int)
	The number of electrons

    ms2 (int)
        Two times MS (n alpha minus n beta)
    
    max_iter (int, optional, default=20)
	Maximum number of iterations
    
    grad_thresh (float, optional, default=1.0E-5)
	Threshold for the norm of the gradient
    
    f_out (File object, optional, default=sys.stdout)
        The output to print the iterations.
        If None the iterations are not printed.
    
    n_DIIS (int, optional, default=0)
        Maximum dimension of the iterative subspace in DIIS
        0 means that DIIS is not used
    
    HF_step_type (callable, optional, default=lambda x:"RH-SCF")
        A callable that receives the following arguments:
            i_SCF (int)
            grad (np.ndarray, optional)
        and return one of the following strings:
            "RH-SCF", "densMat-SCF", "Absil", "orb_rot-Newton"
        that will dictate which method should be used in the
        Hartree-Fock optimisation, in that iteration
    
    ini_orb (MolecularOrbitals, optional, default=None)
        Initial orbitals for the SCF procedure
    """
    if (n_elec - ms2) % 2 != 0:
        raise ValueError(
            'Number of electrons not compatible with MS.')
    kind_of_calc = 'UHF'
    converged = False
    hf_step = HartreeFockStep()
    hf_step.integrals = integrals
    hf_step.n_occ_alpha = (n_elec + ms2) // 2
    hf_step.n_occ_beta = (n_elec - ms2) // 2
    hf_step.n_occ = n_elec
    if hf_step.n_occ_alpha < 0 or hf_step.n_occ_beta < 0:
        raise ValueError(
            'Number of electrons not compatible with MS.')
    hf_step.n_DIIS = n_DIIS
    logger.info('Starting Unrestricted Hartree-Fock calculation\n'
                + 'Nuclear repulsion energy: %f\n'
                + 'n electrons alpha: %d\n'
                + 'n electrons beta: %d\n'
                + 'n DIIS: %d\n',
                nucl_rep, hf_step.n_occ_alpha,
                hf_step.n_occ_beta, hf_step.n_DIIS)
    
    if ini_orb is None:
        logger.info('Using eigenvectors of orthogonalised h as starting guess')
        hf_step.orb = MolecularOrbitals.from_eig_h(
            hf_step.integrals,
            integrals.basis_set + '(AO)',
            restricted=False)
    else:
        logger.info('Using orbitals given by the user as initial guess.')
        if ini_orb.restricted:
            raise ValueError('Initial orbitals should be of unrestricted type.')
        hf_step.orb = MolecularOrbitals(ini_orb)
        hf_step.orb.orthogonalise(X=integrals.X)
        logger.debug('Initial molecular orbitals:\n %s', hf_step.orb)
    if loglevel <= logging.DEBUG:
        assert hf_step.orb.is_orthonormal(
            integrals.S), "Orbitals are not orthonormal"

    hf_step.initialise(HF_step_type(i_SCF=0))
    
    if f_out is not None:
        f_out.write(util.fmt_HF_header_general.format('it.', 'E',
                                                      '|Gradient|',
                                                      'step',
                                                      'time in iteration'))
        
    for i_SCF in range(max_iter):
        logger.info('Starting SCF iteration %d', i_SCF)
        step_type = HF_step_type(i_SCF=i_SCF, grad=hf_step.grad)
        with logtime('SCF iteration') as T:
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
                hf_step.gradNorm, step_type, T.elapsed_time))
            f_out.flush()
            
        if hf_step.gradNorm < grad_thresh:
            logger.info('Convergence reached in %d iterations.', i_SCF)
            converged = True
            break

    hf_step.orb.name = 'UHF'
    res = OptResults(kind_of_calc)
    res.energy = nucl_rep + hf_step.energy
    res.orbitals = hf_step.orb
    res.success = converged
    res.n_iter = i_SCF
    if not converged:
        res.warning = 'No convergence was obtained'
        logger.info('End of Closed-Shell Restricted Hartree-Fock calculation')
    return res
