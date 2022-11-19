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
        self.n_occ = 0
        self.N_a = 0
        self.N_b = 0
        self.integrals = None
        self.orb = None
        self.energy = None
        self.P_a = None
        self.P_b = None
        self.grad_type = None
        self.grad_occvirtF = None # Do ew really need both? Unify with grad from other methods
        self.grad = None
        self.grad_b = None
        self.Fock_a = None
        self.Fock_b = None
        self.diis_info = None
        self.diis_a = None
        self.diis_b = None
        self.one_el_energy = None
        self.two_el_energy = None
        self.grad_norm = 100.0
        self.g = None
        self.blocks_a = None
        self.blocks_b = None
        self.norm_restriction = 100.0
        self.large_cond_number = []
        self.energies = None
        self.energies_b = None
        self.old_energy = None

    @property
    def n(self):
        """The basis set size: the dimension of spatial orbitals space"""
        return len(self.orb)

    @property
    def grad_shape(self):
        """The shape of gradient. It depends on grad_occvirtF"""
        return ((self.N_a, self.n - self.N_a)
                if self.grad_occvirtF else
                (self.n, self.n))

    @property
    def grad_b_shape(self):
        """The shape of beta gradient. It depends on grad_occvirtF"""
        return ((self.N_b, self.n - self.N_b)
                if self.grad_occvirtF else
                (self.n, self.n))

    def _init_P(self):
        self.P_a = np.zeros((self.n, self.n))
        if not self.restricted:
            self.P_b = np.zeros((self.n, self.n))

    def initialise(self, step_type):
        """Initialize matrices"""
        if step_type == 'RH-SCF':
            self.grad_occvirtF = True if self.grad_type == 'F_occ_virt' else False
            self.diis_a = util.Diis(self.diis_info.n)
            if not self.restricted:
                self.diis_b = util.Diis(self.diis_info.n)

        elif step_type == 'densMat-SCF':
            pass

        elif step_type == 'RRN':
            self._init_P()
        
        elif step_type == 'orb_rot-Newton':
            pass
        
        elif step_type == 'RGD':
            self._init_P()

        elif step_type == 'NMLM':
            self._init_P()
            self.energies = np.diag(self.orb.energies[:self.N_a])
            if not self.restricted:
                self.energies_b = np.diag(self.orb.energies_b[:self.N_b])

        elif step_type == 'GDLM':
            self._init_P()
            self.energies = np.diag(self.orb.energies[:self.N_a])
            if not self.restricted:
                self.energies_b = np.diag(self.orb.energies_b[:self.N_b])

        else:
            raise ValueError("Unknown type of Hartree-Fock step: "
                             + step_type)

    def calc_density_matrix(self):
        """Calculate the density matrix (or matrices)
        
        P = C @ C.T
        """
        self.P_a = self.orb[0][:, :self.N_a] @ self.orb[0][:, :self.N_a].T
        if self.restricted:
            self.P_a = 2 * self.P_a
            logger.debug('Density matrix:\n%r', self.P_a)
        else:
            self.P_b = self.orb[1][:, :self.N_b] @ self.orb[1][:, :self.N_b].T
            logger.debug('Alpha density matrix:\n%r', self.P_a)
            logger.debug('Beta density matrix:\n%r', self.P_b)

    def calc_fock_matrix(self):
        """Calculate the Fock matrix
        
        F[mn] = h[mn] + P_a[rs]*(g[mnrs] - g[mrsn]/2)
        The way that einsum is made matters a lot in the time
        
        TODO: integrate restricted and unrestricted codes
        """
        self.Fock_a = absil.fock(self.blocks_a[0], self.blocks_b[0],
                                 self.blocks_a[1], self.blocks_b[1],
                                 self.integrals.h,
                                 self.integrals.g._integrals)
        self.Fock_a = np.asarray(self.Fock_a)
        logger.debug('Fock matrix (alpha):\n%s', self.Fock_a)
        if not self.restricted:
            self.Fock_b = absil.fock(self.blocks_b[0], self.blocks_a[0],
                                     self.blocks_b[1], self.blocks_a[1],
                                     self.integrals.h,
                                     self.integrals.g._integrals)
            self.Fock_b = np.asarray(self.Fock_b)
            logger.debug('Fock matrix (beta):\n%s', self.Fock_b)

    def calc_energy(self):
        """Calculate the energy
        
        TODO: integrate restricted and unrestricted codes
        """
        if self.restricted:
            self.energy = np.tensordot(self.P_a, self.integrals.h + self.Fock_a)/2
            self.one_el_energy = np.tensordot(self.P_a, self.integrals.h)
            self.two_el_energy = np.tensordot(self.P_a, self.Fock_a - self.integrals.h)/2
        else:
            self.one_el_energy = ((self.P_a + self.P_b)*self.integrals.h).sum()
            self.energy = 0.5*((self.P_a*self.Fock_a
                                + self.P_b*self.Fock_b).sum()
                               + self.one_el_energy)
            self.two_el_energy = self.energy - self.one_el_energy
        logger.info('Electronic energy: %f\nOne-electron energy: %f'
                    '\nTwo-electron energy: %f',
                    self.energy, self.one_el_energy, self.two_el_energy)

    def calc_SCF_grad(self):
        if self.grad_occvirtF:
            F_MO = self.orb[0].T @ self.Fock_a @ self.orb[0]
            self.grad = F_MO[:self.N_a, self.N_a:]
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            if not self.restricted:
                F_MO = self.orb[1].T @ self.Fock_b @ self.orb[1]
                self.grad_b = F_MO[:self.N_b, self.N_b:]
                self.orb.energies_b = np.array([F_MO[i, i]
                                                   for i in range(len(self.orb))])
        else:
            tmp = self.Fock_a @ self.P_a @ self.integrals.S
            self.grad = tmp - tmp.T
            if not self.restricted:
                tmp = self.Fock_b @ self.P_b @ self.integrals.S
                self.grad_b = tmp - tmp.T
        if self.restricted:
            self.grad_norm = linalg.norm(self.grad) / sqrt2
        else:
            self.grad_norm = sqrt(linalg.norm(self.grad)**2
                                  + linalg.norm(self.grad_b)**2)
            
        logger.info('Gradient norm = %f', self.grad_norm)
        if self.restricted:
            logger.debug('Gradient:\n%s', self.grad)
            if loglevel <= logging.INFO and self.grad_occvirtF:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies)
        else:
            logger.debug('Gradient (alpha):\n%s', self.grad)
            logger.debug('Gradient (beta):\n%s', self.grad_b)
            if self.grad_occvirtF:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies_b)
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies_b)

    def diag_fock(self):
        if self.restricted:
            Fock_orth = self.integrals.X.T @ self.Fock_a @ self.integrals.X
            e, C = linalg.eigh(Fock_orth)
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
        else:
            self.Fock_a = self.integrals.X.T @ self.Fock_a @ self.integrals.X
            self.Fock_b = self.integrals.X.T @ self.Fock_b @ self.integrals.X
            self.energies, C = linalg.eigh(self.Fock_a)
            self.energies = np.reshape(np.diag(self.energies[:self.N_a]),
                                       (self.N_a**2,))
            self.energies_b, Cb = linalg.eigh(self.Fock_b)
            self.energies_b = np.reshape(np.diag(self.energies_b[:self.N_b]),
                                            (self.N_b**2,))
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
            self.orb[1][:, :] = self.integrals.X @ Cb


    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo"""
        with logtime('Form the density matrix'):
            self.calc_density_matrix()

        if self.diis_info.at_P:
            with logtime('DIIS step'):
                if i_SCF:
                    self.P_a = self.diis_a.calculate(self.grad, self.P_a)
                    if not self.restricted:
                        self.P_b = self.diis_b.calculate(self.grad_b, self.P_b)

        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(self.orb[0][:, :self.N_a],
                                                self.P_a,
                                                self.integrals.g._integrals)
            self.blocks_b = absil.common_blocks(self.orb[1][:, :self.N_b],
                                                self.P_b,
                                                self.integrals.g._integrals)

        with logtime('Form Fock matrix'):
            self.calc_fock_matrix()
        with logtime('Calculate Energy'):
            self.calc_energy()
        with logtime('Calculate Gradient'):
            self.calc_SCF_grad()
        if self.diis_info.at_F:
            with logtime('DIIS step'):
                self.Fock_a = self.diis_a.calculate(self.grad, self.Fock_a)
                if not self.restricted:
                    self.Fock_b = self.diis_b.calculate(self.grad_b, self.Fock_b)
        with logtime('Fock matrix in the orthogonal basis'):
            self.diag_fock()

    def density_matrix_scf(self, i_SCF):
        raise NotImplementedError("Density matrix based SCF")

    def RRN(self, i_SCF):
        """Hartree--Fock using Riemannian Raphson--Newton Method.

        References: Edelman and Absil's papers.
        """
        N_a, N_b = self.N_a, self.N_b
        N, n = N_a + N_b, len(self.orb)
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        h, g = self.integrals.h, self.integrals.g._integrals
        L = np.zeros((n*N + N_a**2 + N_b**2, n*N))
        R = np.zeros((n*N + N_a**2 + N_b**2,))
        aux = np.zeros((n*N, n*N))
        Id = np.eye(n)

        self.calc_density_matrix()
        proj_a, proj_b = Id - self.P_a @ S, Id - self.P_b @ S

        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)

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
                                 self.blocks_a[1], self.blocks_b[1],
                                 self.blocks_a[2], self.blocks_b[2], g)
            logger.debug('euclidean hessian:\n%r', hess)
            hess = np.kron(np.eye(N), invS) @ hess
            if N_a != 0:
                tmp = np.kron(np.eye(N_a), proj_a)
                hess[: n*N_a, : n*N_a] = tmp @ hess[: n*N_a, : n*N_a]
                hess[: n*N_a, n*N_a :] = tmp @ hess[: n*N_a, n*N_a :]
            if N_b != 0:
                tmp = np.kron(np.eye(N_b), proj_b)
                hess[n*N_a :, n*N_a :] = tmp @ hess[n*N_a :, n*N_a :]
                hess[n*N_a :, : n*N_a] = tmp @ hess[n*N_a :, : n*N_a]
            aux[:n*N_a, :n*N_a] = np.kron(grad_a.T @ C_a, np.eye(n))
            aux[n*N_a:, n*N_a:] = np.kron(grad_b.T @ C_b, np.eye(n))
            hess -= aux

        L[:n*N, :] = hess
        R[:n*N_a] -= np.reshape(proj_a @ invS @ grad_a, (n*N_a,), 'F')
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
            if N_a:
                self.orb[0][:, :N_a] = util.geodesic(C_a, eta[:, :N_a], S,
                                                     Sqrt, invSqrt)
            if N_b:
                self.orb[1][:, :N_b] = util.geodesic(C_b, eta[:, N_a:], S,
                                                     Sqrt, invSqrt)
        logger.info('Is C.T @ S @ C close to the identity: %s (alpha); %s (beta)',
                    np.allclose(C_a.T @ S @ C_a, np.eye(N_a)),
                    np.allclose(C_b.T @ S @ C_b, np.eye(N_b)))

    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")

    def RGD(self, i_SCF):
        """Hartree--Fock using Riemannian Gradient Descent Method.

        Update method: Backtracking Line Search.
        References: Boumal's book and Pymanopt.
        """
        def line_search(C_a, C_b, ini_energy, direc_a, direc_b,
                        S, Sqrt, invSqrt, g):
            """Run Line Searching as it's in Pymanopt.

            r: constrols the decrease.
            gamma: learning rate
            """
            norm_ini_grad = linalg.norm(direc_a) + linalg.norm(direc_b)
            if self.old_energy is not None:
                gamma = 4*(self.old_energy - ini_energy)/norm_ini_grad**2
            else:
                gamma = 1 / norm_ini_grad
            counter, max_iter, r, contraction = 0, 25, 1e-4, 0.5
            prev_C_a, prev_C_b = np.copy(C_a), np.copy(C_b)
            C_a = util.geodesic(prev_C_a, direc_a, S, Sqrt, invSqrt, t=gamma)
            C_b = util.geodesic(prev_C_b, direc_b, S, Sqrt, invSqrt, t=gamma)
            P_a, P_b = C_a @ C_a.T, C_b @ C_b.T
            self.blocks_a = absil.common_blocks(C_a, P_a, g)
            self.blocks_b = absil.common_blocks(C_b, P_b, g)
            self.Fock_a = absil.fock(self.blocks_a[0], self.blocks_b[0],
                                     self.blocks_a[1], self.blocks_b[1],
                                     self.integrals.h,
                                     self.integrals.g._integrals)
            self.Fock_a = np.asarray(self.Fock_a)
            self.Fock_b = absil.fock(self.blocks_b[0], self.blocks_a[0],
                                     self.blocks_b[1], self.blocks_a[1],
                                     self.integrals.h,
                                     self.integrals.g._integrals)
            self.Fock_b = np.asarray(self.Fock_b)
            self.one_el_energy = ((P_a + P_b)*self.integrals.h).sum()
            self.energy = 0.5*((P_a*self.Fock_a + P_b*self.Fock_b).sum()
                               + self.one_el_energy)
            self.two_el_energy = self.energy - self.one_el_energy
            new_energy = self.energy

            while ((ini_energy - new_energy) < r*gamma*norm_ini_grad**2 and
                   counter < max_iter):
                gamma *= contraction
                C_a = util.geodesic(prev_C_a, direc_a, S, Sqrt, invSqrt, t=gamma)
                C_b = util.geodesic(prev_C_b, direc_b, S, Sqrt, invSqrt, t=gamma)
                P_a, P_b = C_a @ C_a.T, C_b @ C_b.T
                self.blocks_a = absil.common_blocks(C_a, P_a, g)
                self.blocks_b = absil.common_blocks(C_b, P_b, g)
                self.Fock_a = absil.fock(self.blocks_a[0], self.blocks_b[0],
                                         self.blocks_a[1], self.blocks_b[1],
                                         self.integrals.h,
                                         self.integrals.g._integrals)
                self.Fock_a = np.asarray(self.Fock_a)
                self.Fock_b = absil.fock(self.blocks_b[0], self.blocks_a[0],
                                         self.blocks_b[1], self.blocks_a[1],
                                         self.integrals.h,
                                         self.integrals.g._integrals)
                self.Fock_b = np.asarray(self.Fock_b)
                self.one_el_energy = ((P_a + P_b)* self.integrals.h).sum()
                self.energy = 0.5*((P_a*self.Fock_a + P_b*self.Fock_b).sum()
                                   + self.one_el_energy)
                self.two_el_energy = self.energy - self.one_el_energy
                new_energy = self.energy
                counter += 1

            self.old_energy = ini_energy
            return C_a, C_b

        N_a, N_b = self.N_a, self.N_b
        N, n = N_a + N_b, len(self.orb)
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        h, g = self.integrals.h, self.integrals.g._integrals

        self.calc_density_matrix()

        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()

        with logtime('computing the energy'):
            self.calc_energy()

        with logtime('computing the gradient'):
            if N_a:
                grad_a = 2*(np.eye(n) - C_a @ C_a.T @ S) @ invS @ self.Fock_a @ C_a
                logger.debug('gradient spin alpha:\n%r', grad_a)
                self.grad_norm = linalg.norm(grad_a)
            if N_b:
                grad_b = 2*(np.eye(n) - C_b @ C_b.T @ S) @ invS @ self.Fock_b @ C_b
                logger.debug('gradient spin beta:\n%r', grad_b)
                self.grad_norm += linalg.norm(grad_b)

        # with logtime('updating the point'):
        #     if N_a:
        #         self.orb[0][:, :N_a] = util.geodesic(C_a, -grad_a, S,
        #                                              Sqrt, invSqrt, t=1e-2)
        #     if N_b:
        #         self.orb[1][:, :N_b] = util.geodesic(C_b, -grad_b, S,
        #                                              Sqrt, invSqrt, t=1e-2)

        with logtime('updating the point'):
            C_a, C_b = line_search(C_a, C_b, self.energy, -grad_a, -grad_b,
                                   S, Sqrt, invSqrt, g)
            self.orb[0][:, :N_a], self.orb[1][:, :N_b] = C_a, C_b

    def NMLM(self, i_SCF):
        """Hartree--Fock using Newton's Method with Lagrange multipliers."""
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
                self.orb[0][:, :N_a] = absil.gram_schmidt(self.orb[0][:, :N_a],
                                                          S)
            if N_b:
                self.orb[1][:, :N_b] = absil.gram_schmidt(self.orb[1][:, :N_b],
                                                          S)
            self.calc_density_matrix()
            self.blocks_a = absil.common_blocks(self.orb[0][:, :N_a], self.P_a, g)
            self.blocks_b = absil.common_blocks(self.orb[0][:, :N_b], self.P_b, g)
            with logtime('computing Fock matrix'):
                self.calc_fock_matrix()
            with logtime('computing the energy'):
                self.calc_energy()

        C_a = self.orb[0][:, :N_a] = temp_a
        C_b = self.orb[1][:, :N_b] = temp_b
        aux_a = C_a.T @ S; aux_b = C_b.T @ S
        self.calc_density_matrix()

        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)

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
            hess_Lag[:n*N, :n*N] = absil.hessian(C_a, C_b, self.Fock_a, self.Fock_b,
                                                   self.blocks_a[1], self.blocks_b[1],
                                                   self.blocks_a[2], self.blocks_b[2],
                                                   g)
            if N_a:
                if len(self.energies.shape) < 2:
                    self.energies = np.diag(self.energies[:N_a])
                hess_Lag[:n*N_a, :n*N_a] -= np.kron(self.energies.T, S)
                hess_Lag[n*N : n*N + N_a**2, : n*N_a] = jacob_restr_a
                hess_Lag[: n*N_a, n*N : n*N + N_a**2] = -jacob_restr_a.T
                jacob_restr_a = jacob_restr_a.T @ np.reshape(self.energies,
                                                             (N_a**2,), 'F')
                grad_Lag[: n*N_a] = grad_energy_a - jacob_restr_a
                grad_Lag[n*N : n*N + N_a**2] = 0.5*np.reshape(aux_a @ C_a
                                                              - np.eye(N_a),
                                                              (N_a**2,), 'F')
            if N_b:
                if len(self.energies_b.shape) < 2:
                    self.energies_b = np.diag(self.energies_b[:N_b])
                hess_Lag[n*N_a:n*N, n*N_a:n*N] -= np.kron(self.energies_b.T, S)
                hess_Lag[n*N + N_a**2 :, n*N_a : n*N] = jacob_restr_b
                hess_Lag[n*N_a : n*N, n*N + N_a**2 :] = -jacob_restr_b.T
                jacob_restr_b = jacob_restr_b.T @ np.reshape(self.energies_b,
                                                             (N_b**2,), 'F')
                grad_Lag[n*N_a : n*N] = grad_energy_b - jacob_restr_b
                grad_Lag[n*N + N_a**2:] = 0.5*np.reshape(aux_b @ C_b
                                                         - np.eye(N_b),
                                                         (N_b**2,) , 'F')
                logger.debug('hessian:\n%r', hess_Lag)

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
            self.energies_b += np.reshape(eta[0][n*N + N_a**2:],
                                          (N_b, N_b), 'F')
            logger.debug('energies beta: \n%s', self.energies_b)

        with logtime('updating the point'):
            if N_a:
                self.orb[0][:, :N_a] += eta_C[:, :N_a]
            if N_b:
                self.orb[1][:, :N_b] += eta_C[:, N_a:]

    def GDLM(self, i_SCF):
        """Hartree--Fock using Gradient Descent in the Lagrangian.

        Update method: Backtracking Line Search.
        """
