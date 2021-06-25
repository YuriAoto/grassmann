"""Class to run an iteration step in Hartree Fock optimisation


"""
import logging

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
            self.grad = np.zeros((2 * len(self.orb),
                                  self.n_occ))
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
        N = N_alpha + N_beta
        n = len(self.orb)
        D = np.zeros((2*n*N, 2*n*N))
        X = self.orb[0][:,:self.n_occ_alpha]
        Y = self.orb[1][:,:self.n_occ_beta]
        Z = np.zeros((2*n, N))
        Z[:n,:N_alpha], Z[n:,N_alpha:] = X, Y
        overlap = np.zeros((2*n, 2*n))
        overlap[:n,:n], overlap[n:,n:] = self.integrals.S, self.integrals.S
        Sqrt = np.zeros((2*n, 2*n))
        Sqrt[:n,:n], Sqrt[n:,n:] = self.integrals.X, self.integrals.X
        
        self.energy = absil.energy(N_alpha, N_beta, n, Z,
                                   self.integrals.g._integrals,
                                   self.integrals.h)
        self.grad = absil.gradone(N_alpha, N_beta, n, X, Y,
                                  self.integrals.h)
        # self.grad *= np.trace(Z.T @ overlap @ Z) # provavelmente desnecessário
        # self.grad -= self.energy * (overlap @ Z + overlap.T @ Z)
        # self.grad /= np.trace(Z.T @ overlap @ Z) ** 2
        teste = absil.verificagrad(n, N_alpha, N_beta,
                                   self.integrals.g._integrals,
                                   self.integrals.h,
                                   self.grad,
                                   Z,
                                   overlap)
        print(teste)
        # print(self.grad)
        # print(Z)
        R = - (np.identity(2*n) - (Z @ Z.T @ overlap)) @ self.grad
        # print(R)
        print(R.T @ overlap @ Z)
        self.gradNorm = linalg.norm(R)
        # R = np.reshape(R, (2*n*N, 1), 'F')
        # D = absil.directionalderivative(n, N_alpha, N_beta,
        #                                 self.integrals.g._integrals,
        #                                 self.integrals.h,
        #                                 Z, overlap, self.grad)
        # eta = np.linalg.solve(D, R)
        # eta = np.reshape(eta, (2*n, N), 'F')
        u, s, v = linalg.svd(R, full_matrices=False)
        s = np.diag(s)
        Z = Z @ v.T @ np.cos(0.5 * s) + u @ np.sin(0.5 * s)
        self.orb[0][:,:self.n_occ_alpha] = Z[:n,:N_alpha] # ponto inicial
        self.orb[1][:,:self.n_occ_beta] = Z[n:,N_alpha:]
        
    
    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")
