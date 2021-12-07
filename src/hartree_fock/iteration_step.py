"""Class to run an iteration step in Hartree Fock optimisation


"""
import logging
import time
import copy
from math import sqrt

import numpy as np
from scipy import linalg

from util.variables import sqrt2
from input_output.log import logtime
from . import util
from . import absil
from . import absilnp

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


class HartreeFockStep():
    """The iteration step in a Hartree-Fock optimisation
    
    """
    def __init__(self):
        self.restricted = None
        self.n_DIIS = 0
        self.n_occ = 0
        self.N_a = 0
        self.N_b = 0
        self.integrals = None
        self.orb = None
        self.energy = None
        self.P_a = None
        self.P_b = None
        self.grad = None
        self.Fock_a = None
        self.Fock_b = None
        self.one_el_energy = None
        self.two_el_energy = None
        self.grad_norm = 100.0
        self.g = None
        self.blocks = None

    def initialise(self, step_type):
        if step_type == 'RH-SCF':
            self.i_DIIS = -1
            self.grad = np.zeros((self.N_a,
                                  len(self.orb) - self.N_a,
                                  max(self.n_DIIS, 1)))
            self.P_a = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
            if not self.restricted:
                self.grad_beta = np.zeros((self.N_b,
                                           len(self.orb) - self.N_b,
                                           max(self.n_DIIS, 1)))
                self.P_b = np.zeros((len(self.orb),
                                           len(self.orb),
                                           max(self.n_DIIS, 1)))

        elif step_type == 'densMat-SCF':
            pass
        elif step_type == 'Absil':
            pass
        elif step_type == 'orb_rot-Newton':
            pass
        elif step_type == 'gradient':
            pass
        else:
            raise ValueError("Unknown type of Hartree-Fock step: "
                             + step_type)

    def calc_density_matrix(self):
        """Calculate the density matrix (or matrices)

        P = C @ C.T
        """
        self.P_a[:, :, self.i_DIIS] = np.einsum(
            'pi,qi->pq',
            self.orb[0][:, :self.N_a],
            self.orb[0][:, :self.N_a])
        if self.restricted:
            self.P_a[:, :, self.i_DIIS] = 2 * self.P_a[:, :, self.i_DIIS]
            logger.debug('Density matrix:\n%r', self.P_a[:, :, self.i_DIIS])
        else:
            self.P_b[:, :, self.i_DIIS] = np.einsum(
                'pi,qi->pq',
                self.orb[1][:, :self.N_b],
                self.orb[1][:, :self.N_b])
            logger.debug('Alpha density matrix:\n%r', self.P_a[:, :, self.i_DIIS])
            logger.debug('Beta density matrix:\n%r', self.P_b[:, :, self.i_DIIS])

    def calc_diis(self, i_SCF):
        """Calculate DIIS step"""
        if i_SCF < self.n_DIIS:
            cur_n_DIIS = i_SCF
        else:
            cur_n_DIIS = self.n_DIIS
        if self.n_DIIS > 0:
            logger.info('current n DIIS: %d', cur_n_DIIS)
        if self.n_DIIS > 0 and cur_n_DIIS > 0:
            util.calculate_DIIS(self.P_a, self.grad, cur_n_DIIS, self.i_DIIS)
            if not self.restricted:
                util.calculate_DIIS(self.P_b, self.grad_beta, cur_n_DIIS, self.i_DIIS)

    def calc_fock_matrix(self):
        """Calculate the Fock matrix
        
        F[mn] = h[mn] + P_a[rs]*(g[mnrs] - g[mrsn]/2)
        The way that einsum is made matters a lot in the time
        
        TODO: integrate restricted and unrestricted codes
        """
        if self.restricted:
            self.Fock_a = np.array(self.integrals.h)
            tmp = np.einsum('rs,Frs->F',
                            self.P_a[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            self.Fock_a += np.einsum('F,Fmn->mn',
                                     tmp,
                                     self.integrals.g._integrals)
            tmp = np.einsum('rs,Fms->Frm',
                            self.P_a[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            self.Fock_a -= np.einsum('Frm,Frn->mn',
                                     tmp,
                                     self.integrals.g._integrals) / 2
        else:            
            self.Fock_a = absil.fock(self.blocks[0], self.blocks[1],
                                     self.blocks[2], self.blocks[3],
                                     self.integrals.h, self.integrals.g._integrals)
            self.Fock_b = absil.fock(self.blocks[1], self.blocks[0],
                                     self.blocks[3], self.blocks[2],
                                     self.integrals.h, self.integrals.g._integrals)

    def calc_energy(self):
        """Calculate the energy
        
        TODO: integrate restricted and unrestricted codes
        """
        if self.restricted:
            self.energy = np.tensordot(self.P_a[:, :, self.i_DIIS],
                                       self.integrals.h + self.Fock_a) / 2
            self.one_el_energy = np.tensordot(self.P_a[:, :, self.i_DIIS],
                                              self.integrals.h)
            self.two_el_energy = np.tensordot(self.P_a[:, :, self.i_DIIS],
                                              self.Fock_a - self.integrals.h) / 2
        else:
            self.one_el_energy = ((self.P_a[:, :, self.i_DIIS]
                                   + self.P_b[:, :, self.i_DIIS])* self.integrals.h).sum()
            self.energy = 0.5*((self.P_a[:, :, self.i_DIIS]*self.Fock_a
                                + self.P_b[:, :, self.i_DIIS]*self.Fock_b).sum()
                               + self.one_el_energy)
            self.two_el_energy = self.energy - self.one_el_energy
        logger.info(
            'Electronic energy: %f\nOne-electron energy: %f'
            + '\nTwo-electron energy: %f',
            self.energy, self.one_el_energy, self.two_el_energy)
        self.i_DIIS += 1
        if self.i_DIIS >= self.n_DIIS:
            self.i_DIIS = 0
        if self.n_DIIS > 0:
            logger.info('current self.i_DIIS: %d', self.i_DIIS)

    def calc_mo_fock(self):
        if self.restricted:
            F_MO = self.orb[0].T @ self.Fock_a @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.N_a, self.N_a:]
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            self.grad_norm = linalg.norm(self.grad[:, :, self.i_DIIS]) * sqrt2
            logger.info('Gradient norm = %f', self.grad_norm)
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies)
        else:
            F_MO = self.orb[0].T @ self.Fock_a @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.N_a, self.N_a:]
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            F_MO = self.orb[1].T @ self.Fock_b @ self.orb[1]
            self.grad_beta[:, :, self.i_DIIS] = F_MO[:self.N_b, self.N_b:]
            self.orb.energies_beta = np.array([F_MO[i, i]
                                               for i in range(len(self.orb))])
            self.grad_norm = sqrt(linalg.norm(self.grad[:, :, self.i_DIIS])**2
                                 + linalg.norm(self.grad_beta[:, :, self.i_DIIS])**2)
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies)
                logger.info('Current beta orbital energies:\n%r',
                            self.orb.energies_beta)

    def diag_fock(self):
        if self.restricted:
            self.Fock_a = self.integrals.X.T @ self.Fock_a @ self.integrals.X
            e, C = linalg.eigh(self.Fock_a)
            #        e, C = linalg.eig(self.Fock)
            #        e_sort_index = np.argsort(e)
            #        e = e[e_sort_index]
            #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
        else:
            self.Fock_a = self.integrals.X.T @ self.Fock_a @ self.integrals.X
            self.Fock_b = self.integrals.X.T @ self.Fock_b @ self.integrals.X
            e, C = linalg.eigh(self.Fock_a)
            eb, Cb = linalg.eigh(self.Fock_b)
            #        e, C = linalg.eig(Fock)
            #        e_sort_index = np.argsort(e)
            #        e = e[e_sort_index]
            #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
            self.orb[1][:, :] = self.integrals.X @ Cb


    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo"""
        with logtime('Form the density matrix'):
            self.calc_density_matrix()
        with logtime('DIIS step'):
            self.calc_diis(i_SCF)
        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(self.orb[0][:, :self.N_a],
                                              self.orb[1][:, :self.N_b],
                                              self.P_a[:, :, self.i_DIIS],
                                              self.P_b[:, :, self.i_DIIS],
                                              self.integrals.g._integrals)
        with logtime('Form Fock matrix'):
            self.calc_fock_matrix()
        with logtime('Calculate Energy'):
            self.calc_energy()
        with logtime('Fock matrix in the MO basis'):
            self.calc_mo_fock()
        with logtime('Fock matrix in the orthogonal basis'):
            self.diag_fock()

    def density_matrix_scf(self, i_SCF):
        raise NotImplementedError("Density matrix based SCF")

    def newton_absil(self, i_SCF):
        """Hartree-Fock using Newton Method as it's in Absil."""
        if self.restricted:
            raise ValueError('Restricted version of Absil not implemented')
        N_a, N_b = self.N_a, self.N_b
        N, n = N_a + N_b, len(self.orb)
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        h, g = self.integrals.h, self.integrals.g._integrals

        self.calc_density_matrix()
        P_a = self.P_a[:, :, self.i_DIIS]
        P_b = self.P_b[:, :, self.i_DIIS]
        L = np.zeros((n*N + N_a**2 + N_b**2, n*N))
        R = np.zeros((n*N + N_a**2 + N_b**2,))
        Id = np.eye(n)
        proj_a, proj_b = Id - P_a @ S, Id - P_b @ S

        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()

        with logtime('computing the energy'):
            self.calc_energy()

        with logtime('computing the gradient'):
            grad_a = self.Fock_a @ C_a
            logger.debug('gradient spin alpha:\n%r', grad_a)
            grad_b = self.Fock_b @ C_b
            logger.debug('gradient spin beta:\n%r', grad_b)

        with logtime('computing the hessian'):
            hess = absil.hessian(C_a, C_b, self.Fock_a, self.Fock_b,
                                 self.blocks[2], self.blocks[3],
                                 self.blocks[4], self.blocks[5], g)
            logger.debug('hessian:\n%r', hess)
            hess = np.kron(np.eye(N), invS) @ hess
            hess[: n*N_a, : n*N_a] -= absil.dir_proj(C_a, grad_a)
            hess[n*N_a :, n*N_a :] -= absil.dir_proj(C_b, grad_b)
            if N_a != 0:
                tmp_a = np.kron(np.eye(N_a), proj_a)
                tmp_b = np.kron(np.eye(N_a), proj_b)
                hess[: n*N_a, : n*N_a] = tmp_a @ hess[: n*N_a, : n*N_a]
                hess[: n*N_a, n*N_a :] = tmp_b @ hess[: n*N_a, n*N_a :]
            if N_b != 0:
                tmp_a = np.kron(np.eye(N_b), proj_a)
                tmp_b = np.kron(np.eye(N_b), proj_b)
                hess[n*N_a :, n*N_a :] = tmp_b @ hess[n*N_a :, n*N_a :]
                hess[n*N_a :, : n*N_a] = tmp_a @ hess[n*N_a :, : n*N_a]

        L[: n*N, :] = hess
        R[: n*N_a] -= np.reshape(proj_a @ invS @ grad_a, (n*N_a,), 'F')
        R[n*N_a : n*N] -= np.reshape(proj_b @ invS @ grad_b, (n*N_b,), 'F')
        self.grad_norm = linalg.norm(R)

        if N_a != 0:
            L[n*N : (n*N + N_a**2), : n*N_a] = np.kron(np.eye(N_a),
                                                       C_a.T @ S)
        if N_b != 0:
            L[(n*N + N_a**2) :, n*N_a :] = np.kron(np.eye(N_b),
                                                   C_b.T @ S)

        with logtime('solving the main equation'):
            eta = np.linalg.lstsq(L, R, rcond=None)

        if eta[1].size < 1:
            logger.warning('Hessian does not have full rank')
        elif eta[1] > 1e-10:
            logger.warning('Large conditioning number: %.5e', eta[1])

        eta = np.reshape(eta[0], (n, N), 'F')
        logger.debug('eta: \n%s', eta)

        with logtime('updating the point'):
            if N_a != 0:
                u, s, v = linalg.svd(Sqrt @ eta[:, :N_a], full_matrices=False)
                sin, cos = np.diag(np.sin(s)), np.diag(np.cos(s))
                self.orb[0][:, :N_a] = absil.gram_schmidt(C_a @ v.T @ cos
                                                           + invSqrt @ u @ sin,
                                                          S)

            if N_b != 0:
                u, s, v = linalg.svd(Sqrt @ eta[:, N_a:], full_matrices=False)
                sin, cos = np.diag(np.sin(s)), np.diag(np.cos(s))
                self.orb[1][:, :N_b] = absil.gram_schmidt(C_b @ v.T @ cos
                                                           + invSqrt @ u @ sin,
                                                          S)
        
    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")

    def gradient_descent(self, i_SCF):
        """Hartree-Fock using Gradient Descent Method."""
        N_a, N_b = self.N_a, self.N_b
        N, n = N_a + N_b, len(self.orb)
        X, Y = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        augD = np.zeros((n*N + N_a**2 + N_b**2, n * N))
        augR = np.zeros((n*N + N_a**2 + N_b**2,))
        R, Id = np.empty((n, N)), np.eye(n)
        projX, projY = Id - (X @ X.T @ S), Id - (Y @ Y.T @ S)
        g, h = self.integrals.g._integrals, self.integrals.h
        xxt, yyt = X @ X.T, Y @ Y.T
        gradX, gradY = self.grad[:, :N_a], self.grad[:, N_a:]

        with logtime("computing Fock matrix using three indices in cython."):
            fock_a = absil.fock_three_3(xxt, yyt, h, g)
            fock_b = absil.fock_three_3(yyt, xxt, h, g)

        with logtime("computing the energy."):
            self.one_el_energy = ((xxt + yyt)*h).sum()
            self.energy = 0.5*((xxt*fock_a + yyt*fock_b).sum()
                               + self.one_el_energy)
            self.two_el_energy = self.energy - self.one_el_energy

        with logtime("computing the gradient."):
            gradX = 2 * fock_a @ X
            gradY = 2 * fock_b @ Y

        gradX, gradY = invS @ gradX, invS @ gradY
        R[:, :N_a], R[:, N_a:] = -projX @ gradX, -projY @ gradY
        self.grad_norm = linalg.norm(R)

        if N_a != 0:
            u, s, v = linalg.svd(Sqrt @ gradX, full_matrices=False)
            sin, cos = np.diag(np.sin(0.1 * s)), np.diag(np.cos(0.1 * s))
            X = X @ v.T @ cos + invSqrt @ u @ sin
            self.orb[0][:, :N_a] = absil.gram_schmidt(X, S)
            
        if N_b != 0:
            u, s, v = linalg.svd(Sqrt @ gradY, full_matrices=False)
            sin, cos = np.diag(np.sin(0.1 * s)), np.diag(np.cos(0.1 * s))
            Y = Y @ v.T @ cos + invSqrt @ u @ sin
            self.orb[1][:, :N_b] = absil.gram_schmidt(Y, S)
