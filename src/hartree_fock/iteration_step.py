"""Class to run an iteration step in Hartree Fock optimisation


"""
import logging
import time

import numpy as np
from scipy import linalg

from util.variables import sqrt2
from input_output.log import logtime
from . import util
from . import absil

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()



class HartreeFockStep():
    """The iteration step in a Hartree-Fock optimisation
    
    """
    def __init__(self):
        self.n_DIIS = 0
        self.n_occ = 0
        self.n_occ_alpha = 0
        self.n_occ_beta = 0
        self.integrals = None
        self.orb = None
        self.energy = None
        self.grad = None
        self.Dmat = None
        self.Fock = None
        self.gradNorm = None
        self.invS = None
        self.invSqrt = None

    def initialise(self, step_type):
        if step_type == 'RH-SCF':
            self.i_DIIS = -1
            self.grad = np.zeros((self.n_occ_alpha,
                                  len(self.orb) - self.n_occ_alpha,
                                  max(self.n_DIIS, 1)))
            self.Dmat = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
        elif step_type == 'densMat-SCF':
            pass
        elif step_type == 'Absil':
            self.invS = linalg.inv(self.integrals.S)
            self.invSqrt = linalg.inv(self.integrals.X)
            self.grad = np.zeros((len(self.orb), self.n_occ))
            self.integrals.g.transform_to_ijkl()
        elif step_type == 'orb_rot-Newton':
            pass
        else:
            raise ValueError("Unknown type of Hartree-Fock step: "
                             + step_type)
        
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
            Fock = np.array(self.integrals.h)
            tmp = np.einsum('rs,Frs->F',
                            self.Dmat[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            Fock += np.einsum('F,Fmn->mn',
                              tmp,
                              self.integrals.g._integrals)
            tmp = np.einsum('rs,Fms->Frm',
                            self.Dmat[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            Fock -= np.einsum('Frm,Frn->mn',
                              tmp,
                              self.integrals.g._integrals) / 2
        logger.debug('Fock matrix:\n%r', Fock)

        with logtime('Calculate Energy'):
            self.energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                       self.integrals.h + Fock) / 2
            self.one_el_energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                              self.integrals.h)
            self.two_el_energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                              Fock - self.integrals.h) / 2
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
            Fock = self.integrals.X.T @ Fock @ self.integrals.X
            e, C = linalg.eigh(Fock)
    #        e, C = linalg.eig(Fock)
    #        e_sort_index = np.argsort(e)
    #        e = e[e_sort_index]
    #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C

    def density_matrix_scf(self, i_SCF):
        raise NotImplementedError("Density matrix based SCF")

    def newton_absil(self, i_SCF):
        """Hartree-Fock using Newton Method as it's in Absil.
        
        """
        N_alpha, N_beta = self.n_occ_alpha, self.n_occ_beta
        N, n = N_alpha + N_beta, len(self.orb)
        X, Y = self.orb[0][:,:N_alpha], self.orb[1][:,:N_beta]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.invS, self.invSqrt
        augmentedD = np.zeros((n * N + N_alpha ** 2 + N_beta ** 2, n * N))
        augmentedR = np.zeros((n * N + N_alpha ** 2 + N_beta ** 2,))
        R, Id = np.empty((n, N)), np.eye(n)
        projX, projY = Id - (X @ X.T @ S), Id - (Y @ Y.T @ S)
        g, h = self.integrals.g._integrals, self.integrals.h
        xxt, yyt = X @ X.T, Y @ Y.T
        gradX, gradY = self.grad[:,:N_alpha], self.grad[:,N_alpha:]

        with logtime("final energy"):
            self.energy = absil.energyfinal(X, Y, xxt, yyt, h, g)
        
        with logtime("final gradient"):
            gradX = absil.gradfinal(X, Y, xxt, yyt, h, g)
            gradY = absil.gradfinal(Y, X, yyt, xxt, h, g)
        D = absil.direc_derivative(X, Y, xxt, yyt, projX, projY, gradX, gradY,
                                   invS, h, g)
        augmentedD[:n*N,:] = D
        gradX, gradY = invS @ gradX, invS @ gradY
        R[:,:N_alpha], R[:,N_alpha:] = -projX @ gradX, -projY @ gradY
        R = np.reshape(R, (n * N,), 'F')
        augmentedR[:n*N] = R
        if N_alpha != 0:
            augmentedD[n*N:n*N+N_alpha**2,:n*N_alpha] = np.kron(np.eye(N_alpha),
                                                                X.T @ S)
        if N_beta != 0:
            augmentedD[n*N+N_alpha**2:,n*N_alpha:] = np.kron(np.eye(N_beta),
                                                               Y.T @ S)
        eta = np.reshape(np.linalg.lstsq(augmentedD, augmentedR, rcond=None)[0],
                         (n, N),
                         'F')
        if N_alpha != 0:
            u, s, v = linalg.svd(Sqrt @ eta[:,:N_alpha], full_matrices=False)
            sin, cos = np.diag(np.sin(s)), np.diag(np.cos(s))
            X = X @ v.T @ cos + invSqrt @ u @ sin
            self.orb[0][:,:N_alpha] = absil.gs(X,S)
        if N_beta != 0:
            u, s, v = linalg.svd(Sqrt @ eta[:,N_alpha:], full_matrices=False)
            sin, cos = np.diag(np.sin(s)), np.diag(np.cos(s))
            Y = Y @ v.T @ cos + invSqrt @ u @ sin
            self.orb[1][:,:N_beta] = absil.gs(Y, S)
        self.gradNorm = linalg.norm(R)
        
    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")
