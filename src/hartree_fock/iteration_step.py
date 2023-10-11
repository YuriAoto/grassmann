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
        self.energy_prev = float('inf')
        self.diff_energy = 1.0
        self.P = None
        self.P_b = None
        self.grad_type = None
        self.grad_occvirtF = None # Do we really need both? Unify with grad from other methods
        self.fock = None
        self.fock_b = None
        self.diis_info = None
        self.diis = None
        self.diis_b = None
        self.one_el_energy = None
        self.two_el_energy = None
        self.grad_norm = None
        self.g = None
        self.blocks = None
        self.blocks_b = None
        self.restriction_norm = None
        self.large_cond_number = []
        self.energies = None
        self.energies_b = None
        self.conjugacy = None
        self.step_size = 1e-3

    @property
    def d(self):
        """The basis set size: the dimension of spatial orbitals space"""
        return len(self.orb)

    @property
    def grad_shape(self):
        """The shape of gradient. It depends on grad_occvirtF"""
        return ((self.N_a, self.d - self.N_a)
                if self.grad_occvirtF else
                (self.d, self.d))

    @property
    def grad_b_shape(self):
        """The shape of beta gradient. It depends on grad_occvirtF"""
        return ((self.N_b, self.d - self.N_b)
                if self.grad_occvirtF else
                (self.d, self.d))

    @property
    def zeros(self):
        return np.zeros(self.integrals.g._integrals.shape[0])
    
    def _init_P(self):
        self.P = np.zeros((self.d, self.d))
        if not self.restricted:
            self.P_b = np.zeros((self.d, self.d))

    def initialise(self, step_type):
        """Initialize matrices"""
        if step_type == 'RH-SCF':
            self.grad_occvirtF = True if self.grad_type == 'F_occ_virt' else False
            self.diis = util.Diis(self.diis_info.n)
            if not self.restricted:
                self.diis_b = util.Diis(self.diis_info.n)

        elif step_type == 'densMat-SCF':
            pass

        elif step_type == 'RNR':
            self._init_P()
            self.Id_d = np.eye(self.d)
            self.Id_N_a = np.eye(self.N_a)
            if not self.restricted:
                self.Id_N_b = np.eye(self.N_b)

        elif step_type == 'orb_rot-Newton':
            pass

        elif step_type == 'RGD':
            self._init_P()
            self.Id_d = np.eye(self.d)
            self.Id_N_a = np.eye(self.N_a)
            if not self.restricted:
                self.Id_N_b = np.eye(self.N_b)

        elif step_type == 'NRLM':
            self._init_P()
            self.Id_N_a = np.eye(self.N_a)
            self.energies = np.diag(self.orb.energies[:self.N_a])
            if not self.restricted:
                self.Id_N_b = np.eye(self.N_b)
                self.energies_b = np.diag(self.orb.energies_b[:self.N_b])

        elif step_type == 'RCG':
            self._init_P()
            self.gamma = 0
            self.eta = 0
            self.Id_d = np.eye(self.d)
            self.Id_N_a = np.eye(self.N_a)
            if not self.restricted:
                self.eta_b = 0
                self.Id_N_b = np.eye(self.N_b)

        else:
            raise ValueError("Unknown type of Hartree-Fock step: "
                             + step_type)

    def calc_density_matrix(self):
        """Calculate the density matrix (or matrices)
        
        P = C @ C.T
        """
        self.P = self.orb[0][:, :self.N_a] @ self.orb[0][:, :self.N_a].T
        if self.restricted:
            self.P = 2 * self.P
            logger.debug('Density matrix:\n%r', self.P)
        else:
            self.P_b = self.orb[1][:, :self.N_b] @ self.orb[1][:, :self.N_b].T
            logger.debug('Alpha density matrix:\n%r', self.P)
            logger.debug('Beta density matrix:\n%r', self.P_b)

    def calc_fock_matrix(self):
        """Calculate the Fock matrix
        
        F[ij] = h[ij] + \sum_{k,l} P[kl]*(g[ijkl] - g[ikjl]/2)
        The way that einsum is made matters a lot in the time
        
        TODO: integrate restricted and unrestricted codes
        """
        if self.restricted:
            self.fock = absil.fock(
                self.blocks[0],
                self.zeros,
                self.blocks[1],
                self.integrals.h,
                self.integrals.g._integrals
            )
            self.fock = np.asarray(self.fock)
            logger.debug('Fock matrix:\n%s', self.fock)
        else:
            self.fock = absil.fock(
                self.blocks[0],
                self.blocks_b[0],
                self.blocks[1],
                self.integrals.h,
                self.integrals.g._integrals
            )
            self.fock = np.asarray(self.fock)
            logger.debug('Fock matrix (alpha):\n%s', self.fock)
            self.fock_b = absil.fock(
                self.blocks_b[0],
                self.blocks[0],
                self.blocks_b[1],
                self.integrals.h,
                self.integrals.g._integrals
            )
            self.fock_b = np.asarray(self.fock_b)
            logger.debug('Fock matrix (beta):\n%s', self.fock_b)

    def calc_energy(self):
        """Calculate the energy
        
        TODO: integrate restricted and unrestricted codes
        """
        P_fock = np.tensordot(self.P, self.fock)
        self.one_el_energy = np.tensordot(self.P, self.integrals.h)
        self.energy = (self.one_el_energy + P_fock) / 2

        if not self.restricted:
            P_fock_b = np.tensordot(self.P_b, self.fock_b)
            self.one_el_energy += np.tensordot(self.P_b, self.integrals.h)
            self.energy = (self.one_el_energy + P_fock + P_fock_b) / 2

        self.two_el_energy = self.energy - self.one_el_energy
        logger.info('Electronic energy: %f\nOne-electron energy: %f'
                    '\nTwo-electron energy: %f',
                    self.energy, self.one_el_energy, self.two_el_energy)

    def calc_SCF_grad(self):
        if self.grad_occvirtF:
            F_MO = self.orb[0].T @ self.fock @ self.orb[0]
            self.grad_SCF = F_MO[:self.N_a, self.N_a:]
            self.orb.energies = np.array(
                [F_MO[i, i] for i in range(self.d)]
            )
            if not self.restricted:
                F_MO = self.orb[1].T @ self.fock_b @ self.orb[1]
                self.grad_SCF_b = F_MO[:self.N_b, self.N_b:]
                self.orb.energies_b = np.array(
                    [F_MO[i, i] for i in range(self.d)]
                )
        else:
            tmp = self.fock @ self.P @ self.integrals.S
            self.grad_SCF = tmp - tmp.T
            if not self.restricted:
                tmp = self.fock_b @ self.P_b @ self.integrals.S
                self.grad_SCF_b = tmp - tmp.T

        if self.restricted:
            self.grad_norm = linalg.norm(self.grad_SCF) / sqrt2
        else:
            self.grad_norm = sqrt(
                linalg.norm(self.grad_SCF)**2 + linalg.norm(self.grad_SCF_b)**2
            )

        logger.info('Gradient norm = %f', self.grad_norm)

        if self.restricted:
            logger.debug('Gradient:\n%s', self.grad_SCF)
            if loglevel <= logging.INFO and self.grad_occvirtF:
                logger.info(
                    'Current orbital energies:\n%r', self.orb.energies
                )
        else:
            logger.debug('Gradient (alpha):\n%s', self.grad_SCF)
            logger.debug('Gradient (beta):\n%s', self.grad_SCF_b)
            if self.grad_occvirtF:
                logger.info(
                    'Current orbital energies (alpha):\n%r', self.orb.energies
                )
                logger.info(
                    'Current orbital energies (beta):\n%r', self.orb.energies_b
                )

    def diag_fock(self):
        if self.restricted:
            fock_orth = self.integrals.X.T @ self.fock @ self.integrals.X
            e, C = linalg.eigh(fock_orth)
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
        else:
            self.fock = self.integrals.X.T @ self.fock @ self.integrals.X
            self.fock_b = self.integrals.X.T @ self.fock_b @ self.integrals.X
            self.energies, C = linalg.eigh(self.fock)
            self.energies = np.reshape(
                np.diag(self.energies[:self.N_a]), (self.N_a**2,)
            )
            self.energies_b, C_b = linalg.eigh(self.fock_b)
            self.energies_b = np.reshape(
                np.diag(self.energies_b[:self.N_b]), (self.N_b**2,)
            )
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
            self.orb[1][:, :] = self.integrals.X @ C_b

    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo"""
        with logtime('Form the density matrix'):
            self.calc_density_matrix()

        if self.diis_info.at_P:
            with logtime('DIIS step'):
                if i_SCF:
                    self.P = self.diis.calculate(self.grad_SCF, self.P)
                    if not self.restricted:
                        self.P_b = self.diis_b.calculate(
                            self.grad_SCF_b,
                            self.P_b
                        )

        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(
                self.orb[0][:, :self.N_a],
                self.P,
                self.integrals.g._integrals
            )
            if not self.restricted:
                self.blocks_b = absil.common_blocks(
                    self.orb[1][:, :self.N_b],
                    self.P_b,
                    self.integrals.g._integrals
                )

        with logtime('Form Fock matrix'):
            self.calc_fock_matrix()
        with logtime('Calculate Energy'):
            self.calc_energy()
        with logtime('Calculate Gradient'):
            self.calc_SCF_grad()
        if self.diis_info.at_F:
            with logtime('DIIS step'):
                self.fock = self.diis.calculate(self.grad_SCF, self.fock)
                if not self.restricted:
                    self.fock_b = self.diis_b.calculate(
                        self.grad_SCF_b,
                        self.fock_b
                    )
        with logtime('Fock matrix in the orthogonal basis'):
            self.diag_fock()

    def RGD(self, i_SCF):
        """Hartree--Fock using Riemannian Gradient Descent Method.

        References: Boumal's book, Pymanopt and Caio's dissertation.
        """
        C = self.orb[0][:, :self.N_a]
        if not self.restricted:
            C_b = self.orb[1][:, :self.N_b]
        S, invS = self.integrals.S, self.integrals.invS
        X, invX = self.integrals.X, self.integrals.invX
        g = self.integrals.g._integrals

        self.calc_density_matrix()
        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C, self.P, g)
            if not self.restricted:
                self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
        with logtime('computing the energy'):
            self.calc_energy()
        with logtime('computing the gradient'):
            if self.restricted:
                proj = self.Id_d - 0.5 * self.P @ S
                Euc_grad = 4 * self.fock @ C
                Riem_grad = proj @ invS @ Euc_grad
                logger.debug('Euclidean gradient:\n%r', Euc_grad)
                logger.debug('Riemannian gradient:\n%r', Riem_grad)
                self.grad_norm = util.Riem_norm([Riem_grad], S)
            else:
                proj = self.Id_d - self.P @ S
                Euc_grad = 2 * self.fock @ C
                Riem_grad = proj @ invS @ Euc_grad
                logger.debug('Euclidean gradient (alpha):\n%r', Euc_grad)
                logger.debug('Riemannian gradient (alpha):\n%r', Riem_grad)
                proj_b = self.Id_d - self.P_b @ S
                Euc_grad_b = 2 * self.fock_b @ C_b
                Riem_grad_b = proj_b @ invS @ Euc_grad_b
                logger.debug('Euclidean gradient (beta):\n%r', Euc_grad_b)
                logger.debug('Riemannian gradient (beta):\n%r', Riem_grad_b)
                self.grad_norm = util.Riem_norm(
                    [Riem_grad, Riem_grad_b],
                    S
                )

        with logtime('updating the point'):
            self.orb[0][:, :self.N_a] = util.geodesic(
                C, -Riem_grad, X, invX, t=self.step_size
            )
            if not self.restricted:
                self.orb[1][:, :self.N_b] = util.geodesic(
                    C_b, -Riem_grad_b, X, invX, t=self.step_size
                )            

        self.diff_energy = self.energy_prev - self.energy
        self.energy_prev = self.energy


    def RCG(self, i_SCF):
        """Hartree--Fock using Riemannian Conjugate Gradient."""
        N_a, d = self.N_a, self.d
        C = self.orb[0][:, :N_a]
        if not self.restricted:
            N_b, N = self.N_b, self.N_a + self.N_b
            C_b = self.orb[1][:, :N_b]
        S, invS = self.integrals.S, self.integrals.invS
        X, invX = self.integrals.X, self.integrals.invX
        g = self.integrals.g._integrals

        # todo: this doesn't work if we start with another method because i_SCF
        # will be greater than 0.
        with logtime('computing direction to transport (eta)'):
            if self.restricted:
                if i_SCF == 0:
                    self.calc_density_matrix()
                    self.blocks = absil.common_blocks(C, self.P, g)
                    self.calc_fock_matrix()
                    proj = self.Id_d - 0.5 * self.P @ S
                    Euc_grad = 4 * self.fock @ C
                    self.Riem_grad = proj @ invS @ Euc_grad
                    self.eta = -np.copy(self.Riem_grad)
                elif i_SCF % (N_a * (d - N_a)) == 0:
                    self.eta = -np.copy(self.Riem_grad)
            else:
                if i_SCF == 0:
                    self.calc_density_matrix()
                    self.blocks = absil.common_blocks(C, self.P, g)
                    self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
                    self.calc_fock_matrix()
                    proj = self.Id_d - self.P @ S
                    Euc_grad = 2 * self.fock @ C
                    self.Riem_grad = proj @ invS @ Euc_grad
                    self.eta = -np.copy(self.Riem_grad)
                    proj_b = self.Id_d - self.P_b @ S
                    Euc_grad_b = 2 * self.fock_b @ C_b
                    self.Riem_grad_b = proj_b @ invS @ Euc_grad_b
                    self.eta_b = -np.copy(self.Riem_grad_b)
                elif i_SCF % (N_a * (d - N_a) + N_b * (d - N_b)) == 0:
                    self.eta = -np.copy(self.Riem_grad)
                    self.eta_b = -np.copy(self.Riem_grad_b)

        self.calc_energy()
        if self.restricted:
            old_C = np.copy(C)
            aux = util.rip([self.Riem_grad], [self.Riem_grad], S)
        else:
            old_C = np.copy(C)
            old_C_b = np.copy(C_b)
            aux = util.rip(
                [self.Riem_grad, self.Riem_grad_b],
                [self.Riem_grad, self.Riem_grad_b],
                S
            )

        with logtime('updating the point'):
            self.orb[0][:, :N_a] = util.geodesic(
                C, self.eta, X, invX, self.step_size
            )
            if not self.restricted:
                self.orb[1][:, :N_b] = util.geodesic(
                    C_b, self.eta_b, X, invX, self.step_size
                )

        self.diff_energy = self.energy_prev - self.energy
        self.energy_prev = self.energy

        self.calc_density_matrix()
        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C, self.P, g)
            if not self.restricted:
                self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
        with logtime('computing the energy'):
            self.calc_energy()
        with logtime('computing the gradient'):
            if self.restricted:
                proj = self.Id_d - 0.5 * self.P @ S
                Euc_grad = 4 * self.fock @ C
                self.Riem_grad = proj @ invS @ Euc_grad
                logger.debug('Euclidean gradient:\n%r', Euc_grad)
                logger.debug('Riemannian gradient:\n%r', self.Riem_grad)
                self.grad_norm = util.Riem_norm([self.Riem_grad], S)
            else:
                proj = self.Id_d - self.P @ S
                Euc_grad = 2 * self.fock @ C
                self.Riem_grad = proj @ invS @ Euc_grad
                logger.debug('Euclidean gradient (alpha):\n%r', Euc_grad)
                logger.debug('Riemannian gradient (alpha):\n%r', self.Riem_grad)
                proj_b = self.Id_d - self.P_b @ S
                Euc_grad_b = 2 * self.fock_b @ C_b
                self.Riem_grad_b = proj_b @ invS @ Euc_grad_b
                logger.debug('Euclidean gradient (beta):\n%r', Euc_grad_b)
                logger.debug('Riemannian gradient (beta):\n%r', self.Riem_grad_b)
                self.grad_norm = util.Riem_norm(
                    [self.Riem_grad, self.Riem_grad_b], S
                )

        self.gamma = 1 / aux
        if self.restricted:
            self.gamma *= util.rip([self.Riem_grad], [self.Riem_grad], S)
        else:
            self.gamma *= util.rip(
                [self.Riem_grad, self.Riem_grad_b],
                [self.Riem_grad, self.Riem_grad_b],
                S
            )

        with logtime('parallel transporting the direction (eta)'):
            self.eta = -self.Riem_grad + self.gamma * util.pt(
                self.eta, self.eta, old_C, self.Id_d, X, invX, self.step_size
            )
            if not self.restricted:
                self.eta_b = -self.Riem_grad_b + self.gamma * util.pt(
                    self.eta_b, self.eta_b, old_C_b, self.Id_d, X, invX, self.step_size
                )


    def RNR(self, i_SCF):
        """Hartree--Fock using Riemannian Newton--Raphson Method.

        References: papers by Edelman and Absil.
        """
        N_a, d = self.N_a, self.d
        C = self.orb[0][:, :N_a]
        if not self.restricted:
            N_b, N = self.N_b, self.N_a + self.N_b
            C_b = self.orb[1][:, :N_b]
        S, invS = self.integrals.S, self.integrals.invS
        X, invX = self.integrals.X, self.integrals.invX
        g = self.integrals.g._integrals

        self.calc_density_matrix()
        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C, self.P, g)
            if not self.restricted:
                self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
        with logtime('computing the energy'):
            self.calc_energy()
        with logtime('computing the gradient'):
            if self.restricted:
                proj = self.Id_d - 0.5 * self.P @ S
                Euc_grad = 4 * self.fock @ C
                Riem_grad = proj @ invS @ Euc_grad
                logger.debug('Euclidean gradient:\n%r', Euc_grad)
                logger.debug('Riemannian gradient:\n%r', Riem_grad)
                self.grad_norm = util.Riem_norm([Riem_grad], S)
                aug_grad = np.zeros(N_a * (d + N_a))
                aug_grad[:d*N_a] = np.reshape(Riem_grad, -1, 'F')
            else:
                proj = self.Id_d - self.P @ S
                Euc_grad = 2 * self.fock @ C
                Riem_grad = proj @ invS @ Euc_grad
                logger.debug('Euclidean gradient (alpha):\n%r', Euc_grad)
                logger.debug('Riemannian gradient (alpha):\n%r', Riem_grad)
                proj_b = self.Id_d - self.P_b @ S
                Euc_grad_b = 2 * self.fock_b @ C_b
                Riem_grad_b = proj_b @ invS @ Euc_grad_b
                logger.debug('Euclidean gradient (beta):\n%r', Euc_grad_b)
                logger.debug('Riemannian gradient (beta):\n%r', Riem_grad_b)
                self.grad_norm = util.Riem_norm(
                    [Riem_grad, Riem_grad_b],
                    S
                )
                aug_grad = np.zeros((d + N_a)*N_a + (d + N_b)*N_b)
                aug_grad[:d*N_a] = np.reshape(Riem_grad, (d*N_a,), 'F')
                aug_grad[d*N_a:d*(N_a + N_b)] = np.reshape(Riem_grad_b, (d*N_b), 'F')

        with logtime('computing the hessian'):
            if self.restricted:
                Euc_Hess = absil.restricted_hessian(
                    C, self.fock, self.blocks[1], self.blocks[2], g
                )
                logger.debug('Euclidean Hessian:\n%r', Euc_Hess)
                Riem_Hess = np.kron(self.Id_N_a, proj @ invS) @ Euc_Hess
                Riem_Hess -= np.kron(Euc_grad.T @ C, self.Id_d)
                aug_Hess = np.vstack([Riem_Hess, np.kron(self.Id_N_a, C.T @ S)])
            else:
                aug_Hess = np.zeros((d*N + N_a**2 + N_b**2, d*N))
                aux_Hess = np.zeros((d*N, d*N))
                Euc_Hess = absil.hessian(
                    C,
                    C_b,
                    self.fock,
                    self.fock_b,
                    self.blocks[1],
                    self.blocks_b[1],
                    self.blocks[2],
                    self.blocks_b[2],
                    g
                )
                logger.debug('Euclidean Hessian:\n%r', Euc_Hess)
                aux_Hess[:d*N_a, :d*N_a] = np.kron(self.Id_N_a, proj @ invS)
                aux_Hess[d*N_a:, d*N_a:] = np.kron(self.Id_N_b, proj_b @ invS)
                Riem_Hess = aux_Hess @ Euc_Hess
                aux_Hess[:d*N_a, :d*N_a] = np.kron(Euc_grad.T @ C, self.Id_d)
                aux_Hess[d*N_a:, d*N_a:] = np.kron(
                    Euc_grad_b.T @ C_b, self.Id_d
                )
                Riem_Hess -= aux_Hess
                aug_Hess[:d*N, :] = Riem_Hess
                aug_Hess[d*N:(d*N + N_a**2), :d*N_a] = np.kron(
                    self.Id_N_a, C.T @ S
                )
                aug_Hess[(d*N + N_a**2):, d*N_a:] = np.kron(
                    self.Id_N_b, C_b.T @ S
                )

        with logtime('solving the main equation'):
            eta = np.linalg.lstsq(aug_Hess, -aug_grad, rcond=None)

        if eta[1].size < 1:
            logger.warning('Hessian does not have full rank')
        elif eta[1] > 1e-10:
            logger.warning('Large conditioning number: %.5e', eta[1])
            self.large_cond_number.append(i_SCF)

        if self.restricted:
            eta = np.reshape(eta[0], (d, N_a), 'F')
            logger.debug('eta: \n%s', eta)
            logger.info(
                'Is C.T @ S @ eta close to zero: %s',
                np.allclose(C.T @ S @ eta, np.zeros((N_a, N_a)))
            )
            with logtime('updating the point'):
                self.orb[0][:, :N_a] = util.geodesic(C, eta, X, invX)
        else:
            eta = np.reshape(eta[0], (d, N), 'F')
            logger.info(
                'Is C.T @ S @ eta close to zero: %s',
                np.allclose(C.T @ S @ eta[:, :N_a], np.zeros((N_a, N_a)))
            )
            logger.info(
                'Is C.T @ S @ eta close to zero (beta): %s',
                np.allclose(C_b.T @ S @ eta[:, N_a:], np.zeros((N_b, N_b)))
            )
            with logtime('updating the point'):
                self.orb[0][:, :N_a] = util.geodesic(C, eta[:, :N_a], X, invX)
                self.orb[1][:, :N_b] = util.geodesic(C_b, eta[:, N_a:], X, invX)


    def NRLM(self, i_SCF):
        """Hartree--Fock using Newton--Raphson with Lagrange multipliers."""
        N_a, N_b, N, n = self.N_a, self.N_b, self.N_a + self.N_b, self.d
        S, g = self.integrals.S, self.integrals.g._integrals
        hess_Lag = np.zeros((n*N + N_a**2 + N_b**2, n*N + N_a**2 + N_b**2))
        grad_Lag = np.empty((n*N + N_a**2 + N_b**2,))

        with logtime('Orthogonalizing coefficients and computing the energy'):
            temp = np.copy(self.orb[0][:, :N_a])
            temp_b = np.copy(self.orb[1][:, :N_b])
            self.orb[0][:, :N_a] = absil.gram_schmidt(self.orb[0][:, :N_a], S)
            self.orb[1][:, :N_b] = absil.gram_schmidt(self.orb[1][:, :N_b], S)
            self.calc_density_matrix()
            self.blocks = absil.common_blocks(self.orb[0][:, :N_a], self.P, g)
            self.blocks_b = absil.common_blocks(self.orb[1][:, :N_b], self.P_b, g)
            with logtime('computing Fock matrix'):
                self.calc_fock_matrix()
            with logtime('computing the energy'):
                self.calc_energy()

        C = self.orb[0][:, :N_a] = temp
        C_b = self.orb[1][:, :N_b] = temp_b
        eps, eps_b = self.energies, self.energies_b
        aux, aux_b = C.T @ S, C_b.T @ S

        self.calc_density_matrix()

        with logtime('computing common blocks'):
            self.blocks = absil.common_blocks(C, self.P, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()

        with logtime('computing the gradient'):
            R = aux @ C - self.Id_N_a
            B = [np.kron(aux.T, self.Id_N_a[:, i]) for i in range(N_a)]
            Jac = np.kron(self.Id_N_a, aux) + np.vstack(B).T
            grad = 2*self.fock @ C
            grad_Lag[:n*N_a] = (np.reshape(grad, (n*N_a,), 'F')
                                - Jac.T @ np.reshape(eps, (N_a**2,), 'F'))
            grad_Lag[n*N:(n*N + N_a**2)] = np.reshape(-R, (N_a**2,), 'F')
            logger.debug('gradient spin alpha:\n%r', grad)

            R_b = aux_b @ C_b - self.Id_N_b
            B = [np.kron(aux_b.T, self.Id_N_b[:, i]) for i in range(N_b)]
            Jac_b = np.kron(self.Id_N_b, aux_b) + np.vstack(B).T
            grad_b = 2*self.fock_b @ C_b
            grad_Lag[n*N_a:n*N] = (np.reshape(grad_b, (n*N_b,), 'F')
                                   - Jac_b.T @ np.reshape(eps_b, (N_b**2,), 'F'))
            grad_Lag[n*N + N_a**2:] = np.reshape(-R_b, (N_b**2,), 'F')
            logger.debug('gradient spin beta:\n%r', grad_b)

        with logtime('computing the hessian'):
            hess_Lag[:n*N, :n*N] = absil.hessian(C, C_b,
                                                 self.fock, self.fock_b,
                                                 self.blocks[1],
                                                 self.blocks_b[1],
                                                 self.blocks[2],
                                                 self.blocks_b[2],
                                                 g)
            hess_Lag[:n*N_a, :n*N_a] -= np.kron(eps + eps.T, S)
            hess_Lag[n*N:(n*N + N_a**2), :n*N_a] = -Jac
            hess_Lag[:n*N_a, n*N:(n*N + N_a**2)] = -Jac.T
            hess_Lag[n*N_a:n*N, n*N_a:n*N] -= np.kron(eps_b + eps_b.T, S)
            hess_Lag[(n*N + N_a**2):, n*N_a:n*N] = -Jac_b
            hess_Lag[n*N_a:n*N, (n*N + N_a**2):] = -Jac_b.T
            logger.debug('hessian:\n%r', hess_Lag)

        self.grad_norm = linalg.norm(grad_Lag)
        self.restriction_norm = linalg.norm(R) + linalg.norm(R_b)

        with logtime('solving the main equation'):
            eta = np.linalg.lstsq(hess_Lag, -grad_Lag, rcond=None)
        if eta[1].size < 1:
            logger.warning('Hessian does not have full rank')
        elif eta[1] > 1e-10:
            logger.warning('Large conditioning number: %.5e', eta[1])

        eta_C = np.reshape(eta[0][:n*N], (n, N), 'F')
        self.energies += np.reshape(eta[0][n*N:(n*N + N_a**2)], (N_a, N_a), 'F')
        logger.debug('eta: \n%s', eta_C)
        logger.debug('energies alpha: \n%s', self.energies)
        self.energies_b += np.reshape(eta[0][(n*N + N_a**2):],
                                      (N_b, N_b), 'F')
        logger.debug('energies beta: \n%s', self.energies_b)

        with logtime('updating the point'):
            self.orb[0][:, :N_a] += eta_C[:, :N_a]
            self.orb[1][:, :N_b] += eta_C[:, N_a:]

    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")

    def density_matrix_scf(self, i_SCF):
        raise NotImplementedError("Density matrix based SCF")
