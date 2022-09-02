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

np.set_printoptions(linewidth=250)
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
        self.norm_restriction = 100.0
        self.large_cond_number = []
        self.energies = None
        self.energies_beta = None

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
            self.i_DIIS = -1
            self.P_a = np.zeros((len(self.orb),
                                 len(self.orb),
                                 max(self.n_DIIS, 1)))
            if not self.restricted:
                self.P_b = np.zeros((len(self.orb),
                                           len(self.orb),
                                     max(self.n_DIIS, 1)))

        elif step_type == 'orb_rot-Newton':
            pass

        elif step_type == 'gradient':
            self.i_DIIS = -1
            self.P_a = np.zeros((len(self.orb),
                                 len(self.orb),
                                 max(self.n_DIIS, 1)))
            if not self.restricted:
                self.P_b = np.zeros((len(self.orb),
                                     len(self.orb),
                                     max(self.n_DIIS, 1)))

        elif step_type == 'lagrange':
            self.i_DIIS = -1
            self.energies = np.diag(self.orb.energies[:self.N_a])
            self.P_a = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
            if not self.restricted:
                self.P_b = np.zeros((len(self.orb),
                                     len(self.orb),
                                     max(self.n_DIIS, 1)))
                self.energies_beta = np.diag(self.orb.energies_beta[:self.N_b])

        elif step_type == 'gradient-lagrange':
            self.i_DIIS = -1
            self.energies = np.diag(self.orb.energies[:self.N_a])
            self.P_a = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
            if not self.restricted:
                self.P_b = np.zeros((len(self.orb),
                                     len(self.orb),
                                     max(self.n_DIIS, 1)))
                self.energies_beta = np.diag(self.orb.energies_beta[:self.N_b])

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
            self.energies, C = linalg.eigh(self.Fock_a)
            self.energies = np.reshape(np.diag(self.energies[:self.N_a]),
                                       (self.N_a**2,))
            self.energies_beta, Cb = linalg.eigh(self.Fock_b)
            self.energies_beta = np.reshape(np.diag(self.energies_beta[:self.N_b]),
                                            (self.N_b**2,))
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
            self.large_cond_number.append(i_SCF)

        eta = np.reshape(eta[0], (n, N), 'F')
        logger.debug('eta: \n%s', eta)
        logger.info('Is C.T @ S @ eta close to zero: %s (alpha); %s (beta)',
                    np.allclose(C_a.T @ S @ eta[:, :N_a], np.zeros((N_a, N_a))),
                    np.allclose(C_b.T @ S @ eta[:, N_a:], np.zeros((N_b, N_b))))
        with logtime('updating the point'):
            if N_a != 0:
                self.orb[0][:, :N_a] = util.geodesic(C_a, eta[:, :N_a], S,
                                                     Sqrt, invSqrt)
            if N_b != 0:
                self.orb[1][:, :N_b] = util.geodesic(C_b, eta[:, N_a:], S,
                                                     Sqrt, invSqrt)
        logger.info('Is C.T @ S @ C close to the identity: %s (alpha); %s (beta)',
                    np.allclose(C_a.T @ S @ C_a, np.eye(N_a)),
                    np.allclose(C_b.T @ S @ C_b, np.eye(N_b)))
        
    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")

    def gradient_descent(self, i_SCF):
        """Hartree-Fock using Riemannian Gradient Descent Method."""
        N_a, N_b = self.N_a, self.N_b
        N, n = N_a + N_b, len(self.orb)
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        h, g = self.integrals.h, self.integrals.g._integrals

        self.calc_density_matrix()
        P_a = self.P_a[:, :, self.i_DIIS]
        P_b = self.P_b[:, :, self.i_DIIS]
        Id = np.eye(n)
        proj_a, proj_b = Id - P_a @ S, Id - P_b @ S

        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()

        with logtime('computing the energy'):
            self.calc_energy()
            initial_energy = self.energy

        with logtime('computing the gradient'):
            if N_a:
                grad_a = self.Fock_a @ C_a
                logger.debug('gradient spin alpha:\n%r', grad_a)
            if N_b:
                grad_b = self.Fock_b @ C_b
                logger.debug('gradient spin beta:\n%r', grad_b)

        grad_a, grad_b = 2*proj_a @ invS @ grad_a, 2*proj_b @ invS @ grad_b
        self.grad_norm = linalg.norm(np.hstack((grad_a, grad_b)))
        norm_a, norm_b = linalg.norm(grad_a), linalg.norm(grad_b)
        tau, r = 0.5, 1e-4

        if N_a:
            if i_SCF == 0:
                lr_a = 1e-1 / norm_a
            else:
                lr_a = max(4*(self.prev_energy - initial_energy)/norm_a**2, 1e-1)
            old_C_a = np.copy(C_a)
            for _ in range(10):
                self.orb[0][:, :N_a] = util.geodesic(old_C_a, -grad_a, S,
                                                     Sqrt, invSqrt, t=lr_a)
                self.calc_density_matrix()
                self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)
                self.calc_fock_matrix()
                self.calc_energy()
                if initial_energy - self.energy >= r*lr_a*norm_a**2:
                    break
                lr_a *= tau
        if N_b:
            if i_SCF == 0:
                lr_b = 1e-1 / norm_b
            else:
                lr_b = max(4*(self.prev_energy - initial_energy)/norm_b**2, 1e-1)
            old_C_b = np.copy(C_b)
            for _ in range(10):
                self.orb[1][:, :N_b] = util.geodesic(old_C_b, -grad_b, S,
                                                     Sqrt, invSqrt, t=lr_b)
                self.calc_density_matrix()
                self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)
                self.calc_fock_matrix()
                self.calc_energy()
                if initial_energy - self.energy >= r*lr_b*norm_b**2:
                    break
                lr_b *= tau

        self.prev_energy = initial_energy

    def newton_lagrange(self, i_SCF):
        """Hartree-Fock using Newton's Method with Lagrange multipliers."""
        if self.restricted:
            raise ValueError('Restricted version of Lagrange Newton\'s not\
            implemented')
        N_a, N_b, N, n = self.N_a, self.N_b, self.N_a + self.N_b, len(self.orb)
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        h, g = self.integrals.h, self.integrals.g._integrals
        hess_Lag = np.zeros((n*N + N_a**2 + N_b**2, n*N + N_a**2 + N_b**2))
        grad_Lag = np.empty((n*N + N_a**2 + N_b**2,))

        with logtime('Orthogonalizing coefficients and computing the energy'):
            temp_a = np.copy(self.orb[0][:, :N_a])
            temp_b = np.copy(self.orb[1][:, :N_b])
            if N_a:
                self.orb[0][:, :N_a] = absil.gram_schmidt(self.orb[0][:, :N_a], S)
            if N_b:
                self.orb[1][:, :N_b] = absil.gram_schmidt(self.orb[1][:, :N_b], S)
            self.calc_density_matrix()
            self.blocks = absil.common_blocks(self.orb[0][:, :N_a],
                                              self.orb[1][:, :N_b],
                                              self.P_a[:, :, self.i_DIIS],
                                              self.P_b[:, :, self.i_DIIS],
                                              g)
            with logtime('computing Fock matrix'):
                self.calc_fock_matrix()
            with logtime('computing the energy'):
                self.calc_energy()

        C_a = self.orb[0][:, :N_a] = temp_a
        C_b = self.orb[1][:, :N_b] = temp_b
        aux_a = C_a.T @ S; aux_b = C_b.T @ S
        self.calc_density_matrix()

        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C_a, C_b,
                                              self.P_a[:, :, self.i_DIIS],
                                              self.P_b[:, :, self.i_DIIS],
                                              g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()

        with logtime('computing the gradient'):
            if N_a:
                grad_energy_a = np.reshape(self.Fock_a @ C_a, (n*N_a,), 'F')
                jacob_restr_a = np.empty((N_a**2, N_a*n))
                e_j = np.zeros((N_a, 1))
                for j in range(N_a):
                    e_j[j] = 1.0
                    jacob_restr_a[:, n*j : n*(j+1)] = np.kron(aux_a, e_j)
                    jacob_restr_a[N_a*j : N_a*(j+1), n*j : n*(j+1)] += aux_a
                    e_j[j] = 0.0
                jacob_restr_a = 0.5*jacob_restr_a
                logger.debug('gradient spin alpha:\n%r', grad_energy_a)
            if N_b:
                grad_energy_b = np.reshape(self.Fock_b @ C_b, (n*N_b,), 'F')
                jacob_restr_b = np.empty((N_b**2, N_b*n))
                e_j = np.zeros((N_b, 1))
                for j in range(N_b):
                    e_j[j] = 1.0
                    jacob_restr_b[:, n*j : n*(j+1)] = np.kron(aux_b, e_j)
                    jacob_restr_b[N_b*j : N_b*(j+1), n*j : n*(j+1)] += aux_b
                    e_j[j] = 0.0
                jacob_restr_b = 0.5*jacob_restr_b
                logger.debug('gradient spin beta:\n%r', grad_energy_b)

        with logtime('computing the hessian'):
            hess = absil.hessian(C_a, C_b, self.Fock_a, self.Fock_b,
                                 self.blocks[2], self.blocks[3],
                                 self.blocks[4], self.blocks[5], g)
            if N_a:
                hess[:n*N_a, :n*N_a] -= np.kron(self.energies.T, S)
            if N_b:
                hess[n*N_a:n*N, n*N_a:n*N] -= np.kron(self.energies_beta.T, S)
            logger.debug('hessian:\n%r', hess)

        hess_Lag[: n*N, : n*N] = hess

        if N_a:
            hess_Lag[n*N : n*N + N_a**2, : n*N_a] = jacob_restr_a
            hess_Lag[: n*N_a, n*N : n*N + N_a**2] = -jacob_restr_a.T
            jacob_restr_a = jacob_restr_a.T @ np.reshape(self.energies,
                                                         (N_a**2,), 'F')
            grad_Lag[: n*N_a] = grad_energy_a - jacob_restr_a
            grad_Lag[n*N : n*N + N_a**2] = 0.5*np.reshape(aux_a @ C_a
                                                          - np.eye(N_a),
                                                          (N_a**2,), 'F')
        if N_b:
            hess_Lag[n*N + N_a**2 :, n*N_a : n*N] = jacob_restr_b
            hess_Lag[n*N_a : n*N, n*N + N_a**2 :] = -jacob_restr_b.T
            jacob_restr_b = jacob_restr_b.T @ np.reshape(self.energies_beta,
                                                         (N_b**2,), 'F')
            grad_Lag[n*N_a : n*N] = grad_energy_b - jacob_restr_b
            grad_Lag[n*N + N_a**2:] = 0.5*np.reshape(aux_b @ C_b - np.eye(N_b),
                                                     (N_b**2,) , 'F')

        self.grad_norm = linalg.norm(grad_Lag[:n*N])
        self.norm_restriction = linalg.norm(grad_Lag[n*N:])

        with logtime('solving the main equation'):
            eta = np.linalg.lstsq(hess_Lag, -grad_Lag, rcond=None)
        if eta[1].size < 1:
            logger.warning('Hessian does not have full rank')
        elif eta[1] > 1e-10:
            logger.warning('Large conditioning number: %.5e', eta[1])

        eta_C = np.reshape(eta[0][: n*N], (n, N), 'F')
        self.energies += np.reshape(eta[0][n*N : n*N + N_a**2], (N_a, N_a), 'F')
        logger.debug('eta: \n%s', eta_C)
        logger.debug('energies alpha: \n%s', self.energies)
        if N_b:
            self.energies_beta += np.reshape(eta[0][n*N + N_a**2:],
                                             (N_b, N_b), 'F')
            logger.debug('energies beta: \n%s', self.energies_beta)

        with logtime('updating the point'):
            if N_a:
                self.orb[0][:, :N_a] += eta_C[:, :N_a]
            if N_b:
                self.orb[1][:, :N_b] += eta_C[:, N_a:]

    def gradient_descent_lagrange(self, i_SCF):
        """Hartree-Fock using usual Gradient Descent in the Lagrangian."""
        if self.restricted:
            raise ValueError('Restricted version of Lagrange Newton\'s not\
            implemented')
        N_a, N_b, N, n = self.N_a, self.N_b, self.N_a + self.N_b, len(self.orb)
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        h, g = self.integrals.h, self.integrals.g._integrals
        P_a = self.P_a[:, :, self.i_DIIS]
        P_b = self.P_b[:, :, self.i_DIIS]
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        r, tau = 1e-4, 0.5
        Id_a, Id_b = np.eye(N_a), np.eye(N_b)
        aux_a = C_a.T @ S; aux_b = C_b.T @ S
        restr_a = aux_a @ C_a - Id_a
        restr_b = aux_b @ C_b - Id_b

        self.calc_density_matrix()
        self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
            
        with logtime('computing the energy'):
            self.calc_energy()
            initial_energy = (self.energy - np.trace(self.energies.T @ restr_a)
                              - np.trace(self.energies_beta.T @ restr_b))

        with logtime('computing the gradient'):
            if N_a:
                grad_energy_a = self.Fock_a @ (2*C_a)
                jacob_restr_a = 2*S @ C_a @ self.energies
                norm_a = linalg.norm(np.vstack((grad_energy_a - jacob_restr_a,
                                                -restr_a)))
                logger.debug('gradient spin alpha:\n%r', grad_energy_a)
            if N_b:
                grad_energy_b = self.Fock_b @ (2*C_b)
                jacob_restr_b = 2*S @ C_b @ self.energies_beta
                norm_b = linalg.norm(np.vstack((grad_energy_b - jacob_restr_b,
                                                -restr_b)))
                logger.debug('gradient spin beta:\n%r', grad_energy_b)

        self.grad_norm = norm_a + norm_b

        if N_a:
            if i_SCF == 0:
                lr_a = 1e-1 / norm_a
            else:
                lr_a = max(4*(self.prev_energy - initial_energy)/norm_a**2, 1e-1)
            old_C_a = np.copy(C_a)
            old_energies_a = np.copy(self.energies)
            temp_b = np.trace(self.energies_beta.T @ restr_b)
            for _ in range(10):
                self.orb[0][:, :N_a] = old_C_a - lr_a*(grad_energy_a - jacob_restr_a)
                self.energies = old_energies_a + lr_a*restr_a
                self.calc_density_matrix()
                self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)
                self.calc_fock_matrix()
                self.calc_energy()
                new_restr_a = C_a.T @ S @ C_a - Id_a
                new_energy = (self.energy - np.trace(self.energies.T @ new_restr_a)
                              - temp_b)
                if initial_energy - new_energy >= r*lr_a*norm_a**2:
                    break
                lr_a *= tau
        if N_b:
            if i_SCF == 0:
                lr_b = 1e-1 / norm_b
            else:
                lr_b = max(4*(self.prev_energy - initial_energy)/norm_b**2, 1e-1)
            old_C_b = np.copy(C_b)
            old_energies_b = np.copy(self.energies_beta)
            temp_a = np.trace(self.energies.T @ restr_a)
            for _ in range(10):
                self.orb[1][:, :N_b] = old_C_b - lr_b*(grad_energy_b - jacob_restr_b)
                self.energies_beta = old_energies_b + lr_b*restr_b
                self.calc_density_matrix()
                self.blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)
                self.calc_fock_matrix()
                self.calc_energy()
                new_restr_b = C_b.T @ S @ C_b - Id_b
                new_energy = (self.energy - temp_a
                              - np.trace(self.energies_beta.T @ new_restr_b))
                if initial_energy - new_energy >= r*lr_b*norm_b**2:
                    break
                lr_b *= tau

        self.prev_energy = initial_energy
