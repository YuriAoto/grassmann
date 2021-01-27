"""Naive subroutines for SCF

These SCF implementations are very inefficient, as
they construct matrices in Python loops (??),
and do not exploit smarter and faster integrals usage.
These are however clean and direct codes,
to be used as reference for more elaborated implementations

History:
 Jul 2018 - Start: closed shell RHF and basic DIIS
 Aug 2020 - Merge with Grassmann; put in naive.py
"""
import sys
import math
import logging

import numpy as np
from scipy import linalg

from input_output.log import logtime
from orbitals.orbitals import MolecularOrbitals
from . import util

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

sqrt2 = np.sqrt(2.0)


def Restricted_Closed_Shell_SCF(mol_geom,
                                max_iter=50,
                                grad_thresh=1.0E-5,
                                f_out=sys.stdout,
                                n_DIIS=10,
                                ini_orb=None):
    """A Restricted Closed Shell SCF Roothan-Hall procedure
    
    Parameters:
    -----------
    
    mol_geom (MolecularGeometry)
        The molecular geometry, with all integrals already calculated
    
    max_iter (int, optional, default=20)
        Maximum number of iterations
    
    grad_thresh (float, optional, default=20)
        Threshold for the norn of the gradient
    
    f_out (File object, optional, default=sys.stdout)
        The output to print the iterations.
        If None the iterations are not printed.
    
    n_DIIS (int, optional, default=10)
        maximum diminsion of the iterative subspace in DIIS
        0 means that DIIS is not used
    
    ini_orb (MolecularOrbitals, optional, default=None)
        Initial orbitals for the SCF procedure
    
    Returns:
    --------
    The result, as instance of HFResult
    
    """
    kind_of_calc = 'closed-shell RHF'
    converged = False
    intgrls = mol_geom.integrals
    if mol_geom.n_elec % 2 != 0:
        raise ValueError(
            'Closed-shell RHF requires system with even number of electrons.')
    n_occ = mol_geom.n_elec // 2
    logger.info('Starting Closed-Shell Restricted Hartree-Fock calculation\n'
                + 'Nuclear repulsion energy: %f\n'
                + 'n electrons: %d\n'
                + 'n occupied orbitals: %d\n'
                + 'n DIIS: %d\n',
                mol_geom.nucl_rep, mol_geom.n_elec, n_occ, n_DIIS)
    if ini_orb is None:
        h_orth = intgrls.X.T @ intgrls.h @ intgrls.X
        logger.info('Using eigenvectors of orthogonalised h as starting guess')
        logger.debug('h_orth:\n%r', h_orth)
        e, C = linalg.eigh(h_orth)
        orb = MolecularOrbitals.from_array(
            intgrls.X @ C, 1,
            in_the_basis=mol_geom.atomic_basis_set + ' (AO)')
    else:
        logger.info('Using orbitals given by the user as initial guess.')
        raise NotImplementedError(
            'Check when and whether we need to multiply by X')
        # Always need to multiply by X??
        orb = MolecularOrbitals(ini_orb)
        # orb = intgrls.X @ orb
    logger.debug('Initial orbitals:\n%s', orb)
    if loglevel <= logging.DEBUG:
        logger.debug('C^+ S C (should be the unity matrix):\n',
                     orb[0].T @ intgrls.S @ orb[0])
    if False:
        D = Determinant(n_elec=mol_geom.n_elec, n_basis=len(orb))
        Ham = Hamiltonian(mol_geo=mol_geom, alpha_orbitals=orb)
        logger.debug('Initial energy from fully transformed MO:\n%f',
                     D.Energy(Ham))
    
    i_DIIS = -1
    grad = np.zeros((n_occ, len(orb) - n_occ, max(n_DIIS, 1)))
    Dmat = np.zeros((len(orb), len(orb), max(n_DIIS, 1)))
    Fock = np.zeros((len(orb), len(orb)))
    if f_out is not None:
        f_out.write(util.fmt_HF_header.format('it.', 'E',
                                              '|Gradient|',
                                              'time in iteration'))
    
    for i_SCF in range(max_iter):
        logger.info('At SCF iteration: %d', i_SCF)
        if i_SCF < n_DIIS:
            cur_n_DIIS = i_SCF
        else:
            cur_n_DIIS = n_DIIS - 1
        if n_DIIS > 0:
            logger.info('current n DIIS: %d', cur_n_DIIS)
        
        with logtime('Form the density matrix') as Tdens:
            for p in range(len(orb)):
                for q in range(len(orb)):
                    Dmat[p, q, i_DIIS] = 0.0
                    for i in range(n_occ):
                        Dmat[p, q, i_DIIS] += orb[0][p, i] * orb[0][q, i]
                    Dmat[p, q, i_DIIS] *= 2
        logger.debug('Density matrix:\n%r', Dmat[:, :, i_DIIS])
        
        if n_DIIS > 0 and cur_n_DIIS > 0:
            with logtime('DIIS step'):
                util.calculate_DIIS(Dmat, grad, cur_n_DIIS, i_DIIS)
        
        with logtime('Form Fock matrix'):
            for m in range(len(orb)):
                for n in range(len(orb)):
                    Fock[m, n] = intgrls.h[m, n]
                    if n >= m:
                        for r in range(len(orb)):
                            for s in range(len(orb)):
                                Fock[m, n] += (Dmat[r, s, i_DIIS]
                                               * (intgrls.g[m, n, r, s] -
                                                  intgrls.g[m, s, r, n] / 2))
                    else:
                        Fock[m, n] = Fock[n, m]
        logger.debug('Fock matrix:\n%r', Fock)
        
        with logtime('Calculate Energy'):
            E = mol_geom.nucl_rep
            E_one_elec = E_two_elec = 0.0
            for p in range(len(orb)):
                for q in range(len(orb)):
                    E += Dmat[p, q, i_DIIS] * (intgrls.h[p, q]
                                               + Fock[p, q]) / 2
                    E_one_elec += Dmat[p, q, i_DIIS] * intgrls.h[p, q]
                    E_two_elec += Dmat[p, q, i_DIIS] * (Fock[p, q]
                                                        - intgrls.h[p, q]) / 2
        logger.info(
            'Energy: %f\nOne-electron energy: %f\nTwo-electron energy: %f',
            E, E_one_elec, E_two_elec)
        
        i_DIIS += 1
        if i_DIIS >= n_DIIS:
            i_DIIS = 0
        if n_DIIS > 0:
            logger.info('current i_DIIS: %d', i_DIIS)
        
        with logtime('Fock matrix in the MO basis'):
            F_MO = orb[0].T @ Fock @ orb[0]
            grad[:, :, i_DIIS] = F_MO[:n_occ, n_occ:]
            gradNorm = linalg.norm(grad[:, :, i_DIIS]) * sqrt2
            logger.info('Gradient norm = %f', gradNorm)
            logger.debug('Gradient:\n%r', grad[:, :, i_DIIS])
            orb.energies = np.array([F_MO[i, i] for i in range(len(orb))])
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r', orb.energies)
        
        if False:
            D = Determinant(n_elec=mol_geom.n_elec, n_basis=len(orb))
            Ham = Hamiltonian(mol_geo=mol_geom, alpha_orbitals=orb)
            logger.debug('Checking energy from fully transformed MO: %f',
                         D.Energy(Ham))
        
        with logtime('Orbitals back in AO basis') as TorbAO:
            Fock = intgrls.X.T @ Fock @ intgrls.X
            logger.debug('Transformed Fock matrix:\n%s', Fock)
            e, C = linalg.eig(Fock)
            e_sort_index = np.argsort(e)
            e = e[e_sort_index]
            C = C[:, e_sort_index]
            orb[0][:, :] = intgrls.X @ C
            logger.debug('New orbitals:\n%s', orb)
        
        if f_out is not None:
            f_out.write(util.fmt_HF_iter.format(
                i_SCF, E, gradNorm, TorbAO.relative_to(Tdens)))
            f_out.flush()
        
        if gradNorm < grad_thresh:
            logger.info('Convergence reached in %d iterations.', i_SCF)
            converged = True
            break
    
    orb.name = 'RHF'
    res = util.HFResult(E, orb, converged, i_SCF)
    res.kind = kind_of_calc
    if not converged:
        res.warning = 'No convergence was obtained'
    logger.info('End of Closed-Shell Restricted Hartree-Fock calculation')
    return res


def Unrestricted_SCF(mol_geom,
                     max_iter=50,
                     grad_thresh=1.0E-5,
                     unpaired_elec=None,
                     f_out=sys.stdout,
                     n_DIIS=10,
                     ini_orb=None,
                     apply_initial_rotation=True):
    """An Unrestricted SCF Roothan-Hall procedure
    
    Parameters:
    -----------
    
    mol_geom (MolecularGeometry)
        The molecular geometry, with all integrals already calculated
    
    max_iter (int, optional, default=20)
        Maximum number of iterations
    
    unpaired_elec (int or None, optional, default=None)
        Number of unpaired electrons. This is equal 2*Ms.
        If None, the 0 is used for even number of electrons
        and 1 for odd number of electrons
    
    grad_thresh (float, optional, default=20)
        Threshold for the norn of the gradient
    
    f_out (File object or None, optional, default=sys.stdout)
        The output to print the iterations.
        If None the iterations are not printed.
    
    n_DIIS (int, optional, default=0)
        maximum diminsion of the iterative subspace in DIIS
        0 means that DIIS is not used
    
    ini_orb (2-tuple of MolecularOrbitals, optional, default=None)
        Alpha and beta initial orbitals for the SCF procedure
    
    apply_initial_rotation (bool, optional, default=True)
        Used when ini_orb is None. In such case, if True,
        initial beta orbitals are obtained from alpha by
        mixing adjacent pairs of orbitals. This is used to avoid
        convergence to the RHF solution

    """
    kind_of_calc = 'UHF'
    converged = False
    intgrls = mol_geom.integrals
    if unpaired_elec is None:
        unpaired_elec = mol_geom.n_elec % 2
    else:
        if unpaired_elec % 2 != mol_geom.n_elec % 2:
            raise ValueError(
                'Number of unpaired electrons (' + str(unpaired_elec)
                + ') not compatible with total number of electrons ('
                + str(mol_geom.n_elec) + ')')
    n_occ_b = (mol_geom.n_elec - unpaired_elec) // 2
    n_occ_a = mol_geom.n_elec - n_occ_b
    logger.info('Starting Unrestricted Hartree-Fock calculation\n'
                + 'Nuclear repulsion energy: %f\n'
                + 'n electrons: %d\n'
                + 'n occupied orbitals: (alpha = %d, beta = %d)\n'
                + 'n DIIS: %d\n',
                mol_geom.nucl_rep, mol_geom.n_elec, n_occ_a, n_occ_b, n_DIIS)
    if ini_orb is None:
        h_orth = intgrls.X.T @ intgrls.h @ intgrls.X
        logger.info('Using eigenvectors of orthogonalised h as starting guess')
        logger.debug('h_orth:\n%r', h_orth)
        e, C = linalg.eigh(h_orth)
        orb_a = MolecularOrbitals.from_array(
            intgrls.X @ C, 1,
            in_the_basis=mol_geom.atomic_basis_set + ' (AO)')
        if apply_initial_rotation:
            rot = np.identity(len(orb_a))
            sqrt2_2 = math.sqrt(2.0) / 2
            for i in range(len(orb_a) // 2):
                rot[2*i, 2*i] = sqrt2_2
                rot[2*i, 2*i+1] = sqrt2_2
                rot[2*i+1, 2*i] = -sqrt2_2
                rot[2*i+1, 2*i+1] = sqrt2_2
            orb_b = MolecularOrbitals.from_array(orb_a._coefficients @ rot)
        else:
            orb_b = MolecularOrbitals(orb_a)
    else:
        logger.info('Using orbitals given by the user as initial guess.')
        raise NotImplementedError(
            'Check when and whether we need to multiply by X')
        # Always need to multiply by X??
        orb_a = MolecularOrbitals(ini_orb[0])
        orb_b = MolecularOrbitals(ini_orb[1])
        # orb = intgrls.X @ orb
        orb_a.basis_coef = MolecularOrbitals.from_array(
            intgrls.X @ ini_orb[0].basis_coef)
        orb_b.basis_coef = MolecularOrbitals.from_array(
            intgrls.X @ ini_orb[1].basis_coef)
    logger.debug('Alpha initial orbitals:\n%s', orb_a)
    logger.debug('Beta initial orbitals:\n%s', orb_b)
    if loglevel <= logging.DEBUG:
        logger.debug('C^+ S C (alpha; should be the unity matrix):\n',
                     orb_a[0].T @ intgrls.S @ orb_a[0])
        logger.debug('C^+ S C (beta; should be the unity matrix):\n',
                     orb_b[0].T @ intgrls.S @ orb_b[0])
    if False:
        D = Determinant(n_elec=mol_geom.n_elec, n_basis=len(orb_a))
        Ham = Hamiltonian(mol_geo=mol_geom,
                          alpha_orbitals=orb_a, beta_orbitals=orb_b,
                          restricted=False)
        logger.debug('Initial energy from fully transformed MO:\n%f',
                     D.Energy(Ham))
    
    i_DIIS = -1
    grad_a = np.zeros((n_occ_a, len(orb_a) - n_occ_a,
                       max(n_DIIS, 1)))
    grad_b = np.zeros((n_occ_b, len(orb_b) - n_occ_b,
                       max(n_DIIS, 1)))
    Dmat_a = np.zeros((len(orb_a), len(orb_a), max(n_DIIS, 1)))
    Dmat_b = np.zeros((len(orb_b), len(orb_b), max(n_DIIS, 1)))
    Fock_a = np.zeros((len(orb_a), len(orb_a)))
    Fock_b = np.zeros((len(orb_b), len(orb_b)))
    if f_out is not None:
        f_out.write(util.fmt_HF_header.format('it.', 'E',
                                              '|Gradient|',
                                              'time in iteration'))
    
    for i_SCF in range(max_iter):
        logger.info('At SCF iteration: %d', i_SCF)
        if i_SCF < n_DIIS:
            cur_n_DIIS = i_SCF
        else:
            cur_n_DIIS = n_DIIS - 1
        if n_DIIS > 0:
            logger.info('current n DIIS: %i', cur_n_DIIS)

        with logtime('Form the density matrices') as Tdens:
            for p in range(len(orb_a)):
                for q in range(len(orb_a)):
                    Dmat_a[p, q, i_DIIS] = 0.0
                    for i in range(n_occ_a):
                        Dmat_a[p, q, i_DIIS] += orb_a[0][p, i] * orb_a[0][q, i]
            for p in range(len(orb_b)):
                for q in range(len(orb_b)):
                    Dmat_b[p, q, i_DIIS] = 0.0
                    for i in range(n_occ_b):
                        Dmat_b[p, q, i_DIIS] += orb_b[0][p, i] * orb_b[0][q, i]
        logger.debug('Density matrix (alpha):\n%r', Dmat_a[:, :, i_DIIS])
        logger.debug('Density matrix (beta):\n%r', Dmat_b[:, :, i_DIIS])
        
        if n_DIIS > 0 and cur_n_DIIS > 0:
            with logtime('DIIS step'):
                util.calculate_DIIS(Dmat_a, grad_a, cur_n_DIIS, i_DIIS)
                util.calculate_DIIS(Dmat_b, grad_b, cur_n_DIIS, i_DIIS)
        
        with logtime('Form Fock matrix'):
            for m in range(len(orb_a)):
                for n in range(len(orb_a)):
                    Fock_a[m, n] = Fock_b[m, n] = intgrls.h[m, n]
                    if n >= m:
                        for r in range(len(orb_a)):
                            for s in range(len(orb_a)):
                                tot = (Dmat_a[r, s, i_DIIS]
                                       + Dmat_b[r, s, i_DIIS]) * \
                                       intgrls.g[m, n, r, s]
                                Fock_a[m, n] += (tot
                                                 - Dmat_a[r, s, i_DIIS]
                                                 * intgrls.g[m, s, r, n])
                                Fock_b[m, n] += (tot
                                                 - Dmat_b[r, s, i_DIIS]
                                                 * intgrls.g[m, s, r, n])
                    else:
                        Fock_a[m, n] = Fock_a[n, m]
                        Fock_b[m, n] = Fock_b[n, m]
        logger.debug('Fock matrix (alpha):\n%r', Fock_a)
        logger.debug('Fock matrix (beta):\n%r', Fock_b)
        
        with logtime('Calculate Energy'):
            E = E_one_elec = E_two_elec = 0.0
            for p in range(len(orb_a)):
                for q in range(len(orb_a)):
                    E += Dmat_a[p, q, i_DIIS] * (intgrls.h[p, q]
                                                 + Fock_a[p, q])
                    E += Dmat_b[p, q, i_DIIS] * (intgrls.h[p, q]
                                                 + Fock_b[p, q])
                    E_one_elec += (Dmat_a[p, q, i_DIIS]
                                   + Dmat_b[p, q, i_DIIS]) * intgrls.h[p, q]
                    E_two_elec += Dmat_a[p, q, i_DIIS] * (
                        Fock_a[p, q] - intgrls.h[p, q])
                    E_two_elec += Dmat_b[p, q, i_DIIS] * (
                        Fock_b[p, q] - intgrls.h[p, q])
            E_two_elec /= 2
            E = mol_geom.nucl_rep + E/2
        logger.info(
            'Energy: %f\nOne-electron energy: %f\nTwo-electron energy: %f',
            E, E_one_elec, E_two_elec)
        
        i_DIIS += 1
        if i_DIIS >= n_DIIS:
            i_DIIS = 0
        if n_DIIS > 0:
            logger.info('current i_DIIS: %i', i_DIIS)
            
        with logtime('Fock matrix in the MO basis'):
            F_MO_a = orb_a[0].T @ Fock_a @ orb_a[0]
            F_MO_b = orb_b[0].T @ Fock_b @ orb_b[0]
            grad_a[:, :, i_DIIS] = F_MO_a[:n_occ_a, n_occ_a:]
            gradNorm = linalg.norm(grad_a[:, :, i_DIIS])
            grad_b[:, :, i_DIIS] = F_MO_b[:n_occ_b, n_occ_b:]
            gradNorm = np.sqrt(gradNorm**2
                               + linalg.norm(grad_b[:, :, i_DIIS])**2)
            logger.info('Gradient norm = %f', gradNorm)
            logger.debug('Gradient (alpha):\n%r', grad_a[:, :, i_DIIS])
            logger.debug('Gradient (beta):\n%r', grad_b[:, :, i_DIIS])
            orb_a.energies = np.array([F_MO_a[i, i]
                                       for i in range(len(orb_a))])
            orb_b.energies = np.array([F_MO_b[i, i]
                                       for i in range(len(orb_b))])
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies (alpha):\n%r',
                            orb_a.energies)
                logger.info('Current orbital energies (beta):\n%r',
                            orb_b.energies)
        
        if False:
            D = Determinant(n_elec=mol_geom.n_elec, n_basis=len(orb_a))
            Ham = Hamiltonian(mol_geo=mol_geom,
                              alpha_orbitals=orb_a, beta_orbitals=orb_b,
                              restricted=False)
            logger.debug('Checking energy from fully transformed MO: %f',
                         D.Energy(Ham))
        
        with logtime('Orbitals back in AO basis') as TorbAO:
            Fock_a = intgrls.X.T @ Fock_a @ intgrls.X
            Fock_b = intgrls.X.T @ Fock_b @ intgrls.X
            logger.debug('Transformed Fock matrix (alpha):\n%s', Fock_a)
            logger.debug('Transformed Fock matrix (beta):\n%s', Fock_b)
            e_a, C_a = linalg.eig(Fock_a)
            e_b, C_b = linalg.eig(Fock_b)
            e_sort_index = np.argsort(e_a)
            e_a = e_a[e_sort_index]
            C_a = C_a[:, e_sort_index]
            e_sort_index = np.argsort(e_b)
            e_b = e_b[e_sort_index]
            C_b = C_b[:, e_sort_index]
            orb_a[0][:, :] = intgrls.X @ C_a
            orb_b[0][:, :] = intgrls.X @ C_b
            logger.debug('New orbitals (alpha):\n%s', orb_a)
            logger.debug('New orbitals (beta):\n%s', orb_b)
        
        if f_out is not None:
            f_out.write(util.fmt_HF_iter.format(
                i_SCF, E, gradNorm, TorbAO.relative_to(Tdens)))
            f_out.flush()
        
        if gradNorm < grad_thresh:
            logger.info('Convergence reached in %d iterations.', i_SCF)
            converged = True
            break
        
    orb_a.name = 'UHF (alpha)'
    orb_b.name = 'UHF (beta)'
    res = util.HFResult(E, (orb_a, orb_b), converged, i_SCF)
    res.kind = kind_of_calc
    if not converged:
        res.warning = 'No convergence was obtained'
    logger.info('End of Unestricted Hartree-Fock calculation')
    return res
