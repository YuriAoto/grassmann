"""Class to run an iteration step in Hartree Fock optimisation


"""
import logging
import time
import copy
from math import sqrt

import numpy as np
from scipy import linalg

from util.variables import sqrt2
from util import array_indices
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
        self.step_type = None
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
        self.grad_norm = np.inf
        self.grad_type = None
        self.grad_occvirtF = None # Do we really need both? Unify with grad from other methods
        self.fock = None
        self.fock_b = None
        self.diis_info = None
        self.diis = None
        self.diis_b = None
        self.one_el_energy = None
        self.two_el_energy = None
        self.g = None
        self.blocks = None
        self.blocks_b = None
        self.restriction_norm = None
        self.large_cond_number = []
        self.conjugacy = None
        self.step_size = 1e-3
        self.lagmult = None
        self.lagmult_b = None
        self.energy_prev = float('inf')
        self.hf_nrlm_multpl_symm = None
        self.eta = None
        self.eta_b = None
        self.Riem_grad = None
        self.Riem_grad_b = None

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
            def _initialize_one_spin_NRLM(N, lagmtpl_triang, orbener):
                """Initialize variables for specific spin/restricted in NRLM
                
                N (int) - number of occupied orbitals
                lagmtpl_triang (bool) - True if only triangular part of
                                        multipliers is used
                orbener (iter of floats) - orbital energies
                
                Note that, even if lagmtpl_triang == True,
                the matrix with initial lagrange multipliers
                is a (symmetric) square matrix
                """
                energies = np.diag(orbener[:N])
                if lagmtpl_triang:
                    nlag = array_indices.triangular(N)
                else:
                    nlag = N**2
                Id_N = np.eye(N)
                logger.debug('n lagrange multipliers: %d', nlag)
                logger.debug('Initial Lagrange multipliers: %s', energies)
                return nlag, energies, Id_N

            self._init_P()
            self.lagmul_triangular = self.hf_nrlm_multpl_symm == 'triangular'
            (self.n_lagmtpl, self.lagmult, self.Id_N_a) = _initialize_one_spin_NRLM(
                self.N_a,
                self.lagmul_triangular,
                self.orb.energies)
            if not self.restricted:
                (self.n_lagmtpl_b, self.lagmult_b, self.Id_N_b) = _initialize_one_spin_NRLM(
                    self.N_b,
                    self.lagmul_triangular,
                    self.orb.energies_b)

            self.Id_N_b = np.eye(self.N_b)
            self.Id_d = np.eye(self.d)
            self.lagmult = np.diag(self.orb.energies[:self.N_a]) * (2 if self.restricted else 1)
            if not self.restricted:
                self.lagmult_b = np.diag(self.orb.energies_b[:self.N_b])

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

    def set_grad_norm(self, grad, grad_b=None, Riemannian=False):
        if Riemannian and self.restricted:
            self.grad_norm = util.Riem_norm([grad], self.integrals.S) / sqrt2
        elif Riemannian:
            self.grad_norm = util.Riem_norm([grad, grad_b], self.integrals.S)
        elif self.restricted:
            self.grad_norm = linalg.norm(grad) / sqrt2
        else:
            self.grad_norm = sqrt(linalg.norm(grad)**2 + linalg.norm(grad_b)**2)
        logger.debug('Gradient norm set to %s', self.grad_norm)
        return self.grad_norm


    def calc_density_matrix(self):
        """Calculate the density matrix (or matrices)
        
        For unrestricted:
        P = C @ C.T
        P_b = C_b @ C_b.T
        
        For restricted:
        P = 2 * C @ C.T = P_a + P_b

        """
        self.P = self.orb[0][:, :self.N_a] @ self.orb[0][:, :self.N_a].T
        if self.restricted:
            self.P = 2 * self.P
            logger.debug('Density matrix:\n%r', self.P)
        else:
            self.P_b = self.orb[1][:, :self.N_b] @ self.orb[1][:, :self.N_b].T
            logger.debug('Density matrix (alpha):\n%r', self.P)
            logger.debug('Density matrix (beta):\n%r', self.P_b)

    def calc_common_blocks(self):
        """Calculate the common blocks"""
        self.blocks = absil.common_blocks(self.orb[0][:, :self.N_a],
                                          self.P,
                                          self.integrals.g._integrals)
        if not self.restricted:
            self.blocks_b = absil.common_blocks(self.orb[1][:, :self.N_b],
                                                self.P_b,
                                                self.integrals.g._integrals)

    def calc_fock_matrix(self):
        """Calculate the Fock matrix"""
        if self.restricted:
            self.fock = absil.restricted_fock(self.blocks[0],
                                              self.blocks[1],
                                              self.integrals.h,
                                              self.integrals.g._integrals)
            self.fock = np.asarray(self.fock)
            logger.debug('Fock matrix:\n%s', self.fock)
        else:
            self.fock = absil.fock(self.blocks[0], self.blocks_b[0],
                                   self.blocks[1],
                                   self.integrals.h,
                                   self.integrals.g._integrals)
            self.fock = np.asarray(self.fock)
            logger.debug('Fock matrix (alpha):\n%s', self.fock)
            self.fock_b = absil.fock(self.blocks_b[0], self.blocks[0],
                                     self.blocks_b[1],
                                     self.integrals.h,
                                     self.integrals.g._integrals)
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

    def calc_energy_non_orth_coeff(self, save_all=False):
        """Calculate the energy associated to the coefficients
        
        Save the original coefficients, orthogonalize them,
        calculate the energy, and then put the originals back
        
        To save also density matrix, common_blocks and fock matrices,
        use save_all=True
        
        """
        temp_C = np.copy(self.orb[0][:, :self.N_a])
        if not self.restricted:
            temp_C_b = np.copy(self.orb[1][:, :self.N_b])
        if save_all:
            temp_P = np.copy(self.P)
            temp_b = np.copy(self.blocks)
            temp_F = np.copy(self.fock)
            if not self.restricted:
                temp_P_b = np.copy(self.P_b)
                temp_b_b = np.copy(self.blocks_b)
                temp_F_b = np.copy(self.fock_b)
        
        self.orb.orthogonalize(S=self.integrals.S)
        self.calc_density_matrix()
        self.calc_common_blocks()
        self.calc_fock_matrix()
        self.calc_energy()
        
        self.orb[0][:, :self.N_a] = temp_C
        if not self.restricted:
            self.orb[1][:, :self.N_b] = temp_C_b
        if save_all:
            self.P = temp_P
            self.blocks = temp_b
            self.fock = temp_F
            if not self.restricted:
                self.P_b = temp_P_b
                self.blocks_b = temp_b_b
                self.fock_b = temp_F_b

    def calc_SCF_grad(self):
        if self.grad_occvirtF:
            F_MO = self.orb[0].T @ self.fock @ self.orb[0]
            self.grad = F_MO[:self.N_a, self.N_a:]
            self.orb.energies = np.array(
                [F_MO[i, i] for i in range(self.d)]
            )
            if not self.restricted:
                F_MO = self.orb[1].T @ self.fock_b @ self.orb[1]
                self.grad_b = F_MO[:self.N_b, self.N_b:]
                self.orb.energies_b = np.array(
                    [F_MO[i, i] for i in range(self.d)]
                )
        else:
            tmp = self.fock @ self.P @ self.integrals.S
            self.grad = tmp - tmp.T
            if not self.restricted:
                tmp = self.fock_b @ self.P_b @ self.integrals.S
                self.grad_b = tmp - tmp.T

        if self.restricted:
            self.set_grad_norm(self.grad)
            logger.debug('Gradient:\n%s', self.grad)
            if loglevel <= logging.INFO and self.grad_occvirtF:
                logger.info(
                    'Current orbital energies:\n%r', self.orb.energies
                )
        else:
            self.set_grad_norm(self.grad, self.grad_b)
            logger.debug('Gradient (alpha):\n%s', self.grad)
            logger.debug('Gradient (beta):\n%s', self.grad_b)
            if self.grad_occvirtF:
                logger.info(
                    'Current orbital energies (alpha):\n%r', self.orb.energies
                )
                logger.info(
                    'Current orbital energies (beta):\n%r', self.orb.energies_b
                )

    def calc_Riem_grad(self, save_eucl_grad=False):
        if self.restricted:
            proj = self.Id_d - 0.5 * self.P @ self.integrals.S
            self.Euc_grad = 4 * self.fock @ self.orb[0][:, :self.N_a]
            self.Riem_grad = proj @ self.integrals.invS @ self.Euc_grad
            logger.debug('Euclidean gradient:\n%r', self.Euc_grad)
            logger.debug('Riemannian gradient:\n%r', self.Riem_grad)
            self.set_grad_norm(self.Riem_grad)
            if not save_eucl_grad:
                del self.Euc_grad
        else:
            proj = self.Id_d - self.P @ self.integrals.S
            self.Euc_grad = 2 * self.fock @ self.orb[0][:, :self.N_a]
            self.Riem_grad = proj @ self.integrals.invS @ self.Euc_grad
            logger.debug('Euclidean gradient (alpha):\n%r', self.Euc_grad)
            logger.debug('Riemannian gradient (alpha):\n%r', self.Riem_grad)
            proj_b = self.Id_d - self.P_b @ self.integrals.S
            self.Euc_grad_b = 2 * self.fock_b @ self.orb[1][:, :self.N_b]
            self.Riem_grad_b = proj_b @ self.integrals.invS @ self.Euc_grad_b
            logger.debug('Euclidean gradient (beta):\n%r', self.Euc_grad_b)
            logger.debug('Riemannian gradient (beta):\n%r', self.Riem_grad_b)
            self.set_grad_norm(self.Riem_grad, self.Riem_grad_b)
            if not save_eucl_grad:
                del self.Euc_grad
                del self.Euc_grad_b

    def orbitals_Riemannian_update(self, v, v_b=None, t=1.0):
        """Update the orbitals through the geodesic in the direction of v (and v_b)"""
        self.orb[0][:, :self.N_a] = util.geodesic(self.orb[0][:, :self.N_a],
                                                  v,
                                                  self.integrals.X,
                                                  self.integrals.invX,
                                                  t)
        if not self.restricted:
            self.orb[1][:, :self.N_b] = util.geodesic(self.orb[1][:, :self.N_b],
                                                      v_b,
                                                      self.integrals.X,
                                                      self.integrals.invX,
                                                      t)

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


    def make_dens_fock_energy(self, calc_energy=True, diis_at_P=False):
        with logtime('Form the density matrix'):
            self.calc_density_matrix()
        if diis_at_P and self.diis_info.at_P:
            with logtime('DIIS step'):
                self.P = self.diis.calculate(self.grad, self.P)
                if not self.restricted:
                    self.P_b = self.diis_b.calculate(
                        self.grad_b,
                        self.P_b
                    )
        with logtime('Computing common blocks'):
            self.calc_common_blocks()
        with logtime('Form Fock matrix'):
            self.calc_fock_matrix()
        if calc_energy:
            with logtime('Calculate Energy'):
                self.calc_energy()

    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo"""
        self.make_dens_fock_energy(diis_at_P=i_SCF)
        with logtime('Calculate Gradient'):
            self.calc_SCF_grad()
        if self.diis_info.at_F:
            with logtime('DIIS step'):
                self.fock = self.diis.calculate(self.grad, self.fock)
                if not self.restricted:
                    self.fock_b = self.diis_b.calculate(
                        self.grad_b,
                        self.fock_b
                    )
        with logtime('Fock matrix in the orthogonal basis'):
            self.diag_fock()

    def RGD(self, i_SCF):
        """Hartree--Fock using Riemannian Gradient Descent Method.
        
        References: Boumal's book, Pymanopt and Caio's dissertation.
        """
        self.make_dens_fock_energy()
        with logtime('computing the gradient'):
            self.calc_Riem_grad()
        with logtime('updating the point'):
            self.orbitals_Riemannian_update(-self.Riem_grad,
                                            v_b=(None
                                                 if self.restricted else
                                                 -self.Riem_grad_b),
                                            t=self.step_size)
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
                    self.set_grad_norm(self.Riem_grad)
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
                    self.set_grad_norm(self.Riem_grad, self.Riem_grad_b)
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
            self.orbitals_Riemannian_update(self.eta,
                                            v_b=(None
                                                 if self.restricted else
                                                 self.eta_b),
                                            t=self.step_size)

        self.diff_energy = self.energy_prev - self.energy
        self.energy_prev = self.energy

        self.make_dens_fock_energy()
        with logtime('computing the gradient'):
            self.calc_Riem_grad()

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

        self.make_dens_fock_energy()
            
        with logtime('computing the gradient'):
            self.calc_Riem_grad(save_eucl_grad=True)
            if self.restricted:
                aug_grad = np.zeros(N_a * (d + N_a))
                aug_grad[:d*N_a] = np.reshape(self.Riem_grad, -1, 'F')
            else:
                aug_grad = np.zeros((d + N_a)*N_a + (d + N_b)*N_b)
                aug_grad[:d*N_a] = np.reshape(self.Riem_grad, (d*N_a,), 'F')
                aug_grad[d*N_a:d*(N_a + N_b)] = np.reshape(self.Riem_grad_b, (d*N_b), 'F')

        with logtime('computing the hessian'):
            if self.restricted:
                Euc_Hess = absil.restricted_hessian(
                    C,
                    self.fock,
                    self.blocks[1],
                    self.blocks[2],
                    g
                )
            else:
                Euc_Hess = absil.hessian(
                    C, C_b,
                    self.fock, self.fock_b,
                    self.blocks[1], self.blocks_b[1],
                    self.blocks[2], self.blocks_b[2],
                    g
                )
            logger.debug('Euclidean Hessian:\n%r', Euc_Hess)
            if self.restricted:
                proj = self.Id_d - 0.5 * self.P @ self.integrals.S
                Riem_Hess = np.kron(self.Id_N_a, proj @ invS) @ Euc_Hess
                Riem_Hess -= np.kron(self.Euc_grad.T @ C, self.Id_d)
                aug_Hess = np.vstack([Riem_Hess, np.kron(self.Id_N_a, C.T @ S)])
            else:
                proj = self.Id_d - self.P @ self.integrals.S
                proj_b = self.Id_d - self.P_b @ self.integrals.S
                aux_Hess = np.zeros((d*N, d*N))
                aux_Hess[:d*N_a, :d*N_a] = np.kron(self.Id_N_a, proj @ invS)
                aux_Hess[d*N_a:, d*N_a:] = np.kron(self.Id_N_b, proj_b @ invS)
                Riem_Hess = aux_Hess @ Euc_Hess
                aux_Hess[:d*N_a, :d*N_a] = np.kron(self.Euc_grad.T @ C, self.Id_d)
                aux_Hess[d*N_a:, d*N_a:] = np.kron(self.Euc_grad_b.T @ C_b, self.Id_d)
                Riem_Hess -= aux_Hess
                aug_Hess = np.zeros((d*N + N_a**2 + N_b**2, d*N))
                aug_Hess[:d*N, :] = Riem_Hess
                aug_Hess[d*N:(d*N + N_a**2), :d*N_a] = np.kron(self.Id_N_a, C.T @ S)
                aug_Hess[(d*N + N_a**2):, d*N_a:] = np.kron(self.Id_N_b, C_b.T @ S)

        with logtime('solving the main equation'):
            eta = _solve_rnr_equation(aug_Hess, aug_grad, d,
                                      check_in_tgspace=False,
                                      C=(C
                                         if self.restricted else
                                         np.concatenate((C, C_b), 1)),
                                      S=S)
        with logtime('updating the point'):
            self.orbitals_Riemannian_update(eta[:, :N_a], v_b=eta[:, N_a:])


    def NRLM(self, i_SCF):
        """Hartree--Fock using Newton--Raphson with Lagrange multipliers."""
        N_a, N_b, n = self.N_a, self.N_b, self.d
        N = self.N_a if self.restricted else self.N_a + self.N_b
        S, g = self.integrals.S, self.integrals.g._integrals
        n_params_a = n * N_a + self.n_lagmtpl
        if self.restricted:
            n_params = n_params_a
        else:
            n_params_b = n * N_b + self.n_lagmtpl_b
            n_params = n_params_a + n_params_b
        hess_Lag = np.zeros((n_params, n_params))
        grad_Lag = np.empty((n_params,))
        """
        Note on the implementation:
        ---------------------------
        The order of the parameters in the matrices are:
        For restricted:
            C   L
        
        For unrestricted:
            C   C_b   L   L_b
        
        where C represent orbital coefficients and L Lagrange multipliers.
        For convenience, we predefine these slices below
        """
        s_Ca = np.s_[:n*N_a]
        s_Cb = np.s_[n*N_a:n*N]
        s_La = np.s_[n*N: n*N + self.n_lagmtpl]
        s_Lb = np.s_[n*N + self.n_lagmtpl:]
        s_C = s_Ca if self.restricted else np.s_[:n*N]
        s_L = s_La if self.restricted else np.s_[n*N:]
        SC = S @ self.orb[0][:, :N_a]
        if not self.restricted:
            SC_b = S @ self.orb[1][:, :N_b]

        logger.debug('Lagrange multipliers (before calc ener non orth):\n%s', self.lagmult)

        with logtime('Orthogonalizing coefficients and computing the energy'):
            # Energy is computed here because orbitals may not be orthonormal
            # Thus, they have to be orthogonalized first
            self.calc_energy_non_orth_coeff()
            
        if self.lagmul_triangular:
            # These matrices are symmetric and have diagonal elements mutiplied by 2
            # They are a easier way to store the otherwise clumsy
            # indexing of pairs: 00, 01, 11, 02, 12, 22, ...
            lagmult = self.lagmult + self.Id_N_a * self.lagmult
            if not self.restricted:
                lagmult_b = self.lagmult_b + self.Id_N_b * self.lagmult_b
        else:
            lagmult, lagmult_b = self.lagmult, self.lagmult_b
        logger.debug('Lagrange multipliers:\n%s', lagmult)
        if not self.restricted:
            logger.debug('Lagrange multipliers (beta):\n%s', lagmult_b)

        self.make_dens_fock_energy(calc_energy=False)

        with logtime('Computing the gradient'):
            Jac, Riem_grad = _nrlm_grad(
                self.orb[0][:, :N_a], self.fock, lagmult,
                self.Id_N_a, S, self.integrals.invS, SC,
                self.restricted, self.lagmul_triangular,
                grad_Lag[s_Ca], grad_Lag[s_La])
            if not self.restricted:
                Jac_b, Riem_grad_b = _nrlm_grad(
                    self.orb[1][:, :N_b], self.fock_b, lagmult_b,
                    self.Id_N_b, S, self.integrals.invS, SC_b,
                    self.restricted, self.lagmul_triangular,
                    grad_Lag[s_Cb], grad_Lag[s_Lb])
        if self.restricted:
            self.restriction_norm = self.set_grad_norm(grad_Lag[s_La])
            self.set_grad_norm(Riem_grad, Riemannian=True)
        else:
            self.restriction_norm = self.set_grad_norm(grad_Lag[s_La],grad_Lag[s_Lb])
            self.set_grad_norm(Riem_grad, Riem_grad_b, Riemannian=True)
        logger.debug('Gradient:\n%s', grad_Lag)

        with logtime('Computing the Hessian'):
            if self.restricted:
                hess_Lag[s_C, s_C] = absil.restricted_hessian(self.orb[0][:, :N_a],
                                                              self.fock,
                                                              self.blocks[1],
                                                              self.blocks[2],
                                                              g)
            else:
                hess_Lag[s_C, s_C] = absil.hessian(self.orb[0][:, :N_a],
                                                   self.orb[1][:, :N_b],
                                                   self.fock, self.fock_b,
                                                   self.blocks[1],
                                                   self.blocks_b[1],
                                                   self.blocks[2],
                                                   self.blocks_b[2],
                                                   g)
            logger.debug('Hessian (of energy only):\n%r', hess_Lag)
            
            hess_Lag[s_Ca, s_Ca] -= (np.kron(lagmult, S)
                                     if self.lagmul_triangular else
                                     np.kron(lagmult + lagmult.T, S))
            if not self.restricted:
                hess_Lag[s_Cb, s_Cb] -= (np.kron(lagmult_b, S)
                                         if self.lagmul_triangular else
                                         np.kron(lagmult_b + lagmult_b.T, S))
            logger.debug('Hessian (including lagmult x S):\n%r', hess_Lag)
            
            if self.lagmul_triangular:
                _nrlm_J_hess(SC, hess_Lag[s_Ca, s_La])
                if not self.restricted: _nrlm_J_hess(SC_b, hess_Lag[s_Cb, s_Lb])
            else:
                hess_Lag[s_Ca, s_La] = -Jac
                if not self.restricted: hess_Lag[s_Cb, s_Lb] = -Jac_b
            # Copy the symmetric block
            hess_Lag[s_La, s_Ca] = hess_Lag[s_Ca, s_La].T
            if not self.restricted: hess_Lag[s_Lb, s_Cb] = hess_Lag[s_Cb, s_Lb].T
            logger.debug('Hessian:\n%r', hess_Lag)

        with logtime('solving the main equation'):
            C_update, L_update = _solve_nrlm_equation(hess_Lag, grad_Lag, n, N)

        with logtime('Updating the Orbitals and multipliers'):
            logger.debug('C_update: \n%s', C_update)
            logger.debug('L_update: \n%s', L_update)
            _nrlm_update_lagmult(self.lagmult, L_update[:self.n_lagmtpl],
                                 self.lagmul_triangular, N_a)
            logger.debug('Lagrange multipliers (alpha): \n%s', self.lagmult)
            logger.debug('lagmult - lagmult.T (alpha): \n%s',
                         self.lagmult - self.lagmult.T)
            if not self.restricted:
                _nrlm_update_lagmult(self.lagmult_b, L_update[self.n_lagmtpl:],
                                     self.lagmul_triangular, N_b)
                logger.debug('Lagrange multipliers (beta): \n%s', self.lagmult_b)
                logger.debug('lagmult_b - lagmult_b.T (beta): \n%s',
                             self.lagmult_b - self.lagmult_b.T)
            self.orb[0][:, :N_a] += C_update[:, :N_a]
            self.orb.energies = np.diag(self.lagmult)
            if not self.restricted:
                self.orb[1][:, :N_b] += C_update[:, N_a:]
                self.orb.energies_b = np.diag(self.lagmult_b)

    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")

    def density_matrix_scf(self, i_SCF):
        raise NotImplementedError("Density matrix based SCF")


def _euclidean_gradient(fock, C, restricted):
    grad = (4 if restricted else 2) * fock @ C
    logger.debug('Gradient of energy:\n%r', grad)
    return grad


def _nrlm_grad(C, fock, lagmult,
               Id_N, S, invS, SC,
               restricted, lagmul_triangular, 
               grad_E, grad_R):
    n, N = C.shape  # This N is not N_a + N_b! It is N_a or N_b,
                    # depending on what has been passed
    proj = np.eye(n) - C @ C.T @ S
    Euc_grad = _euclidean_gradient(fock, C, restricted)
    Riem_grad = proj @ invS @ Euc_grad
    grad_E[:] = np.reshape(Euc_grad,
                           (n*N,), 'F')
    R = Id_N - C.T @ SC
    if lagmul_triangular:
        grad_E[:] -= np.reshape(SC @ lagmult, (len(grad_E),), 'F')
        grad_R[:] = R[np.tril_indices(N)]
        Jac = None
    else:
        B = [np.kron(SC, Id_N[:, i]) for i in range(N)]
        Jac = np.kron(Id_N, SC) + np.vstack(B)
        grad_E[:] -= Jac @ np.reshape(lagmult, (N**2,), 'F')
        grad_R[:] = np.reshape(R, (N**2,), 'F')
    return Jac, Riem_grad


def _nrlm_J_hess(SC, hess_J):
    def _start_block(k):
        return array_indices.triangular(k)

    n, N = SC.shape  # This N is not N_a + N_b! It is N_a or N_b,
                     # depending on what has been passed
    for k in range(N):
        p_slice = slice(n*k, n*k + n)
        hess_J[p_slice, _start_block(k): _start_block(k+1)] = -SC[:, :k+1]
        for j in range(k, N):
            hess_J[p_slice, _start_block(j) + k] += -SC[:, j]


def _nrlm_update_lagmult(ener, eta, is_triangular, N):
    """Add eta to ener, taking care of the version"""
    if is_triangular:
        _nrlm_update_triang_lagmult(ener, eta)
    else:
        ener[:, :] += np.reshape(eta[:N**2], (N, N), 'F')


def _nrlm_update_triang_lagmult(ener, eta):
    """Add eta to ener, with triangular version for multipliers
    
    This function does, in essence
    
    ener += eta.
    
    However, whereas ener is square (and the code assume to be symmetric):
    
    ener_11  ener_12  ...  ener_1N
    ener_12  ener_22  ...  ener_2N
      ...      ...           ...
    ener_1N  ener_N2  ...  ener_NN
    
    eta is a 1D array with the upper triangular part (includding diagonal):
    
    eta_11  eta_12  ...  eta_1N  eta_22  ...  eta_2N  ...  eta _NN
    
    Parameters:
    -----------
    ener (NxN array of floats) - the current multipliers
    eta (1D array of floats) - update for the multipliers
    
    Return:
    -------
    Nothing. array ener is updated
    
    """
    idx = np.tril_indices(ener.shape[0])
    for i, j, e in zip(idx[0], idx[1], eta):
        ener[i, j] += e
        if i != j:
            ener[j, i] += e


def _solve_nrlm_equation(H, g, d, N):
    """Solves the linear equation for the step
    
    Parameters:
    -----------
    H (2D array) - The Hessian
    g (1D array) - The gradient
    d (int) - number of orbitals
    N (int) - number of occupied orbitals
    
    Returns:
    --------
    A tuple with the update in C and in the Lagrange multipliers.
    In C is returned as a 2D array with shape (d, N)
    and in the multipliers they come as a 1D array
    
    """
    eta = np.linalg.lstsq(H, -g, rcond=None)
    if eta[1].size < 1:
        logger.warning('Hessian does not have full rank')
    elif eta[1] > 1e-10:
        logger.warning('Large conditioning number: %.5e', eta[1])
        self.large_cond_number.append(i_SCF)
    C = np.reshape(eta[0][0:d*N], (d, N), 'F')
    L = eta[0][d*N:]
    logger.debug('Update for orbital coefficients: \n%s', eta)
    logger.debug('Update for multipliers: \n%s', eta)
    return C, L


def _solve_rnr_equation(H, g, d, check_in_tgspace=False, C=None, S=None):
    """Solves the linear equation for the step
    
    Parameters:
    -----------
    H (2D array) - The Hessian
    g (1D array) - The gradient
    d (int) - number of orbitals
    check_in_tgspace (bool) - if True, check if the result is in the tangent space
                              In this case, S (the overlap matrix) and C (the orbital
                              coefficients) must be given
    S (2D array) - the overlap matrix
    C (2D array) - the orbitals coefficients. If unrestricted, alpha and beta should
                   be passed in one large matrix of size d x N, where
                   N = N_a + N_b.
    
    Returns:
    --------
    The tangent vector eta, a 2D array of shape (d,N)

    """
    eta = np.linalg.lstsq(H, -g, rcond=None)
    if eta[1].size < 1:
        logger.warning('Hessian does not have full rank')
    elif eta[1] > 1e-10:
        logger.warning('Large conditioning number: %.5e', eta[1])
        self.large_cond_number.append(i_SCF)
    N = eta[0].shape[0] // d
    eta = np.reshape(eta[0], (d, N), 'F')
    logger.debug('Tangent vector eta: \n%s', eta)
    if check_in_tgspace:
        if C is None or S is None:
            raise ValueError('When check_in_tgspace is True, C and S must be passed')
        cTse = C.T @ S @ eta[:,:N]
        logger.debug('C.T @ S @ eta[:,:N]:\n%s', cTse)
        logger.info('||C.T @ S @ eta|| = %s', linalg.norm(cTse))

    return eta
