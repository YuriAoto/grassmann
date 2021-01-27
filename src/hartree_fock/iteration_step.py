"""Class to run a iteration step in a Hartree Fock optimisation


"""
import logging

import numpy as np
from scipy import linalg

from util.variables import sqrt2
from input_output.log import logtime
from . import util

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


class HartreeFockStep():
    """The iteration step in a Hartree-Fock optimisation
    
    """
    def __init__(self):
        self.n_DIIS = 0
        self.n_occ = 0
        self.intgrls = None
        self.orb = None
        self.energy = None
        self.grad = None
        self.Dmat = None
        self.Fock = None
        self.gradNorm = None
    
    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo
        
        """
        if i_SCF < self.n_DIIS:
            cur_n_DIIS = i_SCF
        else:
            cur_n_DIIS = self.n_DIIS
        if self.n_DIIS > 0:
            logger.info('current n DIIS:', cur_n_DIIS)

        with logtime('Form the density matrix'):
            self.Dmat[:, :, self.i_DIIS] = 2 * np.einsum(
                'pi,qi->pq',
                self.orb[0][:, :self.n_occ],
                self.orb[0][:, :self.n_occ])
        logger.debug('Density matrix:\n%r', self.Dmat[:, :, self.i_DIIS])

        if self.n_DIIS > 0 and cur_n_DIIS > 0:
            with logtime('DIIS step'):
                util.calculate_DIIS(
                    self.Dmat, self.grad, cur_n_DIIS, self.i_DIIS)

        with logtime('Form Fock matrix'):
            # F[mn] = h[mn] + Dmat[rs]*(g[mnrs] - g[mrsn]/2)
            # The way that einsum is made matters a lot in the time
            Fock = np.array(self.intgrls.h)
            tmp = np.einsum('rs,Frs->F',
                            self.Dmat[:, :, self.i_DIIS],
                            self.intgrls.g._integrals)
            Fock += np.einsum('F,Fmn->mn',
                              tmp,
                              self.intgrls.g._integrals)
            tmp = np.einsum('rs,Fms->Frm',
                            self.Dmat[:, :, self.i_DIIS],
                            self.intgrls.g._integrals)
            Fock -= np.einsum('Frm,Frn->mn',
                              tmp,
                              self.intgrls.g._integrals) / 2
        logger.debug('Fock matrix:\n%r', Fock)

        with logtime('Calculate Energy'):
            self.energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                       self.intgrls.h + Fock) / 2
            self.one_el_energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                              self.intgrls.h)
            self.two_el_energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                              Fock - self.intgrls.h) / 2
        logger.info(
            'Electronic energy: %f\nOne-electron energy: %f'
            + '\nTwo-electron energy: %f',
            self.energy, self.one_el_energy, self.two_el_energy)

        self.i_DIIS += 1
        if self.i_DIIS >= self.n_DIIS:
            self.i_DIIS = 0
        if self.n_DIIS > 0:
            logger.info('current self.i_DIIS:', self.i_DIIS)
        with logtime('Fock matrix in the MO basis'):
            F_MO = self.orb[0].T @ Fock @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.n_occ, self.n_occ:]
            self.gradNorm = linalg.norm(self.grad[:, :, self.i_DIIS]) * sqrt2
            logger.info('Gradient norm = %f', self.gradNorm)
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r', self.orb.energies)

        with logtime('Fock matrix in the orthogonal basis'):
            Fock = self.intgrls.X.T @ Fock @ self.intgrls.X
            e, C = linalg.eigh(Fock)
    #        e, C = linalg.eig(Fock)
    #        e_sort_index = np.argsort(e)
    #        e = e[e_sort_index]
    #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.intgrls.X @ C

    def density_matrix_scf(i_SCF):
        raise NotImplementedError("Density matrix based SCF")

    def newton_absil(i_SCF):
        raise NotImplementedError("Caio, this is up to you!")
    
    def newton_orb_rot(i_SCF):
        raise NotImplementedError("As described in Helgaker's book")
