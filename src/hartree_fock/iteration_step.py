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
        self.energy_prev = None
        self.diff_energy = 1.0
        self.P_a = None
        self.P_b = None
        self.grad_type = None
        self.grad_occvirtF = None # Do we really need both? Unify with grad from other methods
        self.grad = None
        self.grad_b = None
        self.Fock_a = None
        self.Fock_b = None
        self.RiemG_a = None
        self.diis_info = None
        self.diis_a = None
        self.diis_b = None
        self.one_el_energy = None
        self.two_el_energy = None
        self.grad_norm = 100.0
        self.g = None
        self.blocks_a = None
        self.blocks_b = None
        self.restriction_norm = 100.0
        self.large_cond_number = []
        self.energies = None
        self.energies_b = None
        self.energy_prev = float('inf')
        self.conjugacy = None
        self.step_size = 1.0

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

        elif step_type == 'RNR':
            self._init_P()
            self.Id_N_a = np.eye(self.N_a)
            self.Id_N_b = np.eye(self.N_b)
            self.Id_n = np.eye(self.n)
        
        elif step_type == 'orb_rot-Newton':
            pass
        
        elif step_type == 'RGD':
            self._init_P()
            self.Id_n = np.eye(self.n)
            self.Id_N_a = np.eye(self.N_a)
            self.Id_N_b = np.eye(self.N_b)
            self.energy_prev = float('inf')

        elif step_type == 'NRLM':
            self._init_P()
            self.energies = np.diag(self.orb.energies[:self.N_a])
            self.Id_N_a = np.eye(self.N_a)
            self.Id_N_b = np.eye(self.N_b)
            if not self.restricted:
                self.energies_b = np.diag(self.orb.energies_b[:self.N_b])

        elif step_type == 'GDLM':
            self._init_P()
            self.L_prev = float('inf')
            self.Id_N_a = np.eye(self.N_a)
            self.Id_N_b = np.eye(self.N_b)
            self.Id_n = np.eye(self.n)
            self.energies = np.diag(self.orb.energies[:self.N_a])
            if not self.restricted:
                self.energies_b = np.diag(self.orb.energies_b[:self.N_b])

        elif step_type == 'RCG':
            self._init_P()
            self.gamma = 0
            self.eta_a = 0
            self.eta_b = 0
            self.energy_prev = float('inf')
            self.Id_N_a = np.eye(self.N_a)
            self.Id_N_b = np.eye(self.N_b)
            self.Id_n = np.eye(self.n)

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
                                          for i in range(self.n)])
            if not self.restricted:
                F_MO = self.orb[1].T @ self.Fock_b @ self.orb[1]
                self.grad_b = F_MO[:self.N_b, self.N_b:]
                self.orb.energies_b = np.array([F_MO[i, i]
                                                   for i in range(self.n)])
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

    def _energy(self, C_a, C_b):
        P_a, P_b = C_a @ C_a.T, C_b @ C_b.T
        h, g = self.integrals.h, self.integrals.g._integrals
        Fock_a, Fock_b = np.copy(h), np.copy(h)

        tmp = np.einsum('ij,Lij->L', P_a, g)
        Fock_a += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lij->L', P_b, g)
        Fock_a += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lkj->Lik', P_a, g)
        Fock_a -= np.einsum('Lik,Lil->kl', tmp, g)

        tmp = np.einsum('ij,Lij->L', P_b, g)
        Fock_b += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lij->L', P_a, g)
        Fock_b += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lkj->Lik', P_b, g)
        Fock_b -= np.einsum('Lik,Lil->kl', tmp, g)

        return 0.5*((P_a + P_b)*h + P_a*Fock_a + P_b*Fock_b).sum()

    def _lagrange(self, C_a, C_b, eps_a, eps_b):
        P_a, P_b = C_a @ C_a.T, C_b @ C_b.T
        h, g = self.integrals.h, self.integrals.g._integrals
        Fock_a, Fock_b = np.copy(h), np.copy(h)
        S = self.integrals.S

        tmp = np.einsum('ij,Lij->L', P_a, g)
        Fock_a += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lij->L', P_b, g)
        Fock_a += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lkj->Lik', P_a, g)
        Fock_a -= np.einsum('Lik,Lil->kl', tmp, g)

        tmp = np.einsum('ij,Lij->L', P_b, g)
        Fock_b += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lij->L', P_a, g)
        Fock_b += np.einsum('L,Lkl->kl', tmp, g)
        tmp = np.einsum('ij,Lkj->Lik', P_b, g)
        Fock_b -= np.einsum('Lik,Lil->kl', tmp, g)

        return (0.5*((P_a + P_b)*h + P_a*Fock_a + P_b*Fock_b).sum()
                - np.trace(eps_a.T @ (C_a.T @ S @ C_a - self.Id_N_a))
                - np.trace(eps_b.T @ (C_b.T @ S @ C_b - self.Id_N_b)))

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

    def RGD(self, i_SCF):
        """Hartree--Fock using the Riemannian Gradient Descent Method.

        Update method: Backtracking Line Search.
        References: Boumal's book and Pymanopt.
        """
        C_a, C_b = self.orb[0][:, :self.N_a], self.orb[1][:, :self.N_b]
        S, invS = self.integrals.S, self.integrals.invS
        X, invX = self.integrals.X, self.integrals.invX
        g = self.integrals.g._integrals

        self.calc_density_matrix()
        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
        with logtime('computing the energy'):
            self.calc_energy()
        with logtime('computing the gradient'):
            proj_a = self.Id_n - self.P_a @ S
            grad_a = 2*proj_a @ invS @ self.Fock_a @ C_a
            logger.debug('gradient spin alpha:\n%r', grad_a)
            proj_b = self.Id_n - self.P_b @ S
            grad_b = 2*proj_b @ invS @ self.Fock_b @ C_b
            logger.debug('gradient spin beta:\n%r', grad_b)
            self.grad_norm = util.riem_norm(grad_a, grad_b, S)

        with logtime('updating the point'):
            self.orb[0][:, :self.N_a] = util.geodesic(C_a, -grad_a, X, invX,
                                                      t=self.step_size)
            self.orb[1][:, :self.N_b] = util.geodesic(C_b, -grad_b, X, invX,
                                                      t=self.step_size)
            # C_a, C_b, _ = util.RBLS(C_a, C_b, -grad_a, -grad_b,
            #                         self.grad_norm**2, self.energy,
            #                         self.energy_prev, S, X, invX, self)
            # self.orb[0][:, :self.N_a], self.orb[1][:, :self.N_b] = C_a, C_b
            logger.info('Is C.T @ S @ C close to Id (updated): %s (alpha); %s (beta)',
                        np.allclose(self.orb[0][:, :self.N_a].T @ S @ self.orb[0][:, :self.N_a],
                                    self.Id_N_a),
                        np.allclose(self.orb[1][:, :self.N_b].T @ S @ self.orb[1][:, :self.N_b],
                                    self.Id_N_b))

        self.diff_energy = self.energy_prev - self.energy
        self.energy_prev = self.energy

    def RCG(self, i_SCF):
        """Hartree--Fock using Riemannian Conjugate Gradient."""
        N_a, N_b, N, n = self.N_a, self.N_b, self.N_a + self.N_b, self.n
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, invS = self.integrals.S, self.integrals.invS
        X, invX = self.integrals.X, self.integrals.invX
        g = self.integrals.g._integrals

        # todo: this doesn't work if we start with another method because i_SCF
        # will be greater than 0.
        if i_SCF == 0:
            self.calc_density_matrix()
            with logtime('computing common blocks'):
                self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
                self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
            with logtime('computing Fock matrix'):
                self.calc_fock_matrix()
            with logtime('computing the energy'):
                self.calc_energy()
            with logtime('computing the gradient'):
                proj_a = self.Id_n - self.P_a @ S
                self.RiemG_a = 2*proj_a @ invS @ self.Fock_a @ C_a
                logger.debug('gradient spin alpha:\n%r', self.RiemG_a)
                proj_b = self.Id_n - self.P_b @ S
                self.RiemG_b = 2*proj_b @ invS @ self.Fock_b @ C_b
                logger.debug('gradient spin beta:\n%r', self.RiemG_b)
            self.eta_a = -self.RiemG_a
            self.eta_b = -self.RiemG_b
        elif i_SCF % (N_a*(n-N_a) + N_b*(n-N_b)) == 0:
            self.eta_a = -self.RiemG_a
            self.eta_b = -self.RiemG_b
        else:
            self.eta_a = -self.RiemG_a + self.gamma*self.eta_a
            self.eta_b = -self.RiemG_b + self.gamma*self.eta_b

        self.eta_norm = util.riem_norm(self.eta_a, self.eta_b, S)
        aux = util.rip(self.RiemG_a, self.RiemG_b, self.RiemG_a, self.RiemG_b, S)

        with logtime('backtracking line search'):
            newC_a, newC_b, t = util.RBLS(C_a, C_b, self.eta_a, self.eta_b,
                                          self.eta_norm**2, self.energy,
                                          self.energy_prev, S, X, invX, self)

        with logtime('parallel transport the vectors'):
            if self.conjugacy == 'PR':
                tGk_a = util.pt(self.RiemG_a, self.eta_a, C_a, self.Id_n, X, invX,
                                self.step_size)
                tGk_b = util.pt(self.RiemG_b, self.eta_b, C_b, self.Id_n, X, invX,
                                self.step_size)
            self.eta_a = util.pt(self.eta_a, self.eta_a, C_a, self.Id_n, X, invX,
                                 self.step_size)
            self.eta_b = util.pt(self.eta_b, self.eta_b, C_b, self.Id_n, X, invX,
                                 self.step_size)

        with logtime('updating the point'):
            self.orb[0][:, :N_a] = util.geodesic(C_a, self.eta_a, X, invX,
                                                 self.step_size)
            self.orb[1][:, :N_b] = util.geodesic(C_b, self.eta_b, X, invX,
                                                 self.step_size)
            # self.orb[0][:, :N_a], self.orb[1][:, :N_b] = newC_a, newC_b
            logger.info('Is C.T @ S @ C close to Id: %s (alpha); %s (beta)',
                        np.allclose(C_a.T @ S @ C_a, self.Id_N_a),
                        np.allclose(C_b.T @ S @ C_b, self.Id_N_b))

        self.diff_energy = self.energy_prev - self.energy
        self.energy_prev = self.energy

        self.calc_density_matrix()
        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
        with logtime('computing the energy'):
            self.calc_energy()
        with logtime('computing the gradient'):
            proj_a = self.Id_n - self.P_a @ S
            self.RiemG_a = 2*proj_a @ invS @ self.Fock_a @ C_a
            logger.debug('gradient spin alpha:\n%r', self.RiemG_a)
            proj_b = self.Id_n - self.P_b @ S
            self.RiemG_b = 2*proj_b @ invS @ self.Fock_b @ C_b
            logger.debug('gradient spin beta:\n%r', self.RiemG_b)
            self.grad_norm = util.riem_norm(self.RiemG_a, self.RiemG_b, S)

        if self.conjugacy == 'FR':
            self.gamma = (util.rip(self.RiemG_a, self.RiemG_b,
                                   self.RiemG_a, self.RiemG_b, S) / aux)
        elif self.conjugacy == 'PR':
            self.gamma = (util.rip(self.RiemG_a - tGk_a, self.RiemG_b - tGk_b,
                                   self.RiemG_a, self.RiemG_b, S) / aux)
        else:
            raise NotImplementedError("Conjugacy method not implemented.")


    def RNR(self, i_SCF):
        """Hartree--Fock using Riemannian Newton--Raphson Method.

        References: Edelman and Absil's papers.
        """
        N_a, N_b, N, n = self.N_a, self.N_b, self.N_a + self.N_b, self.n
        C_a, C_b = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, invS = self.integrals.S, self.integrals.invS
        X, invX = self.integrals.X, self.integrals.invX
        g = self.integrals.g._integrals
        aug_hess = np.zeros((n*N + N_a**2 + N_b**2, n*N))
        aug_grad = np.zeros((n*N + N_a**2 + N_b**2,))
        aux_hess = np.zeros((n*N, n*N))

        self.calc_density_matrix()
        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)
        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()
        with logtime('computing the energy'):
            self.calc_energy()
        with logtime('computing the gradient'):
            proj_a = self.Id_n - self.P_a @ S
            grad_a = 2*self.Fock_a @ C_a
            RiemG_a = proj_a @ invS @ grad_a
            logger.debug('gradient spin alpha:\n%r', grad_a)
            proj_b = self.Id_n - self.P_b @ S
            grad_b = 2*self.Fock_b @ C_b
            RiemG_b = proj_b @ invS @ grad_b
            logger.debug('gradient spin beta:\n%r', grad_b)
            aug_grad[:n*N_a] = np.reshape(RiemG_a, (n*N_a,), 'F')
            aug_grad[n*N_a : n*N] = np.reshape(RiemG_b, (n*N_b,), 'F')
            self.grad_norm = util.riem_norm(RiemG_a, RiemG_b, S)

        with logtime('computing the hessian'):
            hess = absil.hessian(C_a, C_b, self.Fock_a, self.Fock_b,
                                 self.blocks_a[1], self.blocks_b[1],
                                 self.blocks_a[2], self.blocks_b[2], g)
            logger.debug('euclidean hessian:\n%r', hess)
            aux_hess[:n*N_a, :n*N_a] = np.kron(self.Id_N_a, proj_a @ invS)
            aux_hess[n*N_a:, n*N_a:] = np.kron(self.Id_N_b, proj_b @ invS)
            hess = aux_hess @ hess
            aux_hess[:n*N_a, :n*N_a] = np.kron(grad_a.T @ C_a, self.Id_n)
            aux_hess[n*N_a:, n*N_a:] = np.kron(grad_b.T @ C_b, self.Id_n)
            hess -= aux_hess
            aug_hess[:n*N, :] = hess
            aug_hess[n*N:(n*N + N_a**2), :n*N_a] = np.kron(self.Id_N_a, C_a.T @ S)
            aug_hess[(n*N + N_a**2):, n*N_a:] = np.kron(self.Id_N_b, C_b.T @ S)

        with logtime('solving the main equation'):
            eta = np.linalg.lstsq(aug_hess, -aug_grad, rcond=None)

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
            self.orb[0][:, :N_a] = util.geodesic(C_a, eta[:, :N_a], X, invX)
            self.orb[1][:, :N_b] = util.geodesic(C_b, eta[:, N_a:], X, invX)
            logger.info('Is C.T @ S @ C close to Id: %s (alpha); %s (beta)',
                        np.allclose(C_a.T @ S @ C_a, self.Id_N_a),
                        np.allclose(C_b.T @ S @ C_b, self.Id_N_b))

    def NRLM(self, i_SCF):
        """Hartree--Fock using Newton--Raphson with Lagrange multipliers."""
        N_a, N_b, N, n = self.N_a, self.N_b, self.N_a + self.N_b, self.n
        S, g = self.integrals.S, self.integrals.g._integrals
        hess_Lag = np.zeros((n*N + N_a**2 + N_b**2, n*N + N_a**2 + N_b**2))
        grad_Lag = np.empty((n*N + N_a**2 + N_b**2,))

        with logtime('Orthogonalizing coefficients and computing the energy'):
            temp_a = np.copy(self.orb[0][:, :N_a])
            temp_b = np.copy(self.orb[1][:, :N_b])
            self.orb[0][:, :N_a] = absil.gram_schmidt(self.orb[0][:, :N_a], S)
            self.orb[1][:, :N_b] = absil.gram_schmidt(self.orb[1][:, :N_b], S)
            self.calc_density_matrix()
            self.blocks_a = absil.common_blocks(self.orb[0][:, :N_a], self.P_a, g)
            self.blocks_b = absil.common_blocks(self.orb[1][:, :N_b], self.P_b, g)
            with logtime('computing Fock matrix'):
                self.calc_fock_matrix()
            with logtime('computing the energy'):
                self.calc_energy()

        C_a = self.orb[0][:, :N_a] = temp_a
        C_b = self.orb[1][:, :N_b] = temp_b
        eps_a, eps_b = self.energies, self.energies_b
        aux_a, aux_b = C_a.T @ S, C_b.T @ S

        self.calc_density_matrix()

        with logtime('computing common blocks'):
            self.blocks_a = absil.common_blocks(C_a, self.P_a, g)
            self.blocks_b = absil.common_blocks(C_b, self.P_b, g)

        with logtime('computing Fock matrix'):
            self.calc_fock_matrix()

        with logtime('computing the gradient'):
            R_a = aux_a @ C_a - self.Id_N_a
            B = [np.kron(aux_a.T, self.Id_N_a[:, i]) for i in range(N_a)]
            Jac_a = np.kron(self.Id_N_a, aux_a) + np.vstack(B).T
            grad_a = 2*self.Fock_a @ C_a
            grad_Lag[:n*N_a] = (np.reshape(grad_a, (n*N_a,), 'F')
                                - Jac_a.T @ np.reshape(eps_a, (N_a**2,), 'F'))
            grad_Lag[n*N:(n*N + N_a**2)] = np.reshape(-R_a, (N_a**2,), 'F')
            logger.debug('gradient spin alpha:\n%r', grad_a)

            R_b = aux_b @ C_b - self.Id_N_b
            B = [np.kron(aux_b.T, self.Id_N_b[:, i]) for i in range(N_b)]
            Jac_b = np.kron(self.Id_N_b, aux_b) + np.vstack(B).T
            grad_b = 2*self.Fock_b @ C_b
            grad_Lag[n*N_a:n*N] = (np.reshape(grad_b, (n*N_b,), 'F')
                                   - Jac_b.T @ np.reshape(eps_b, (N_b**2,), 'F'))
            grad_Lag[n*N + N_a**2:] = np.reshape(-R_b, (N_b**2,), 'F')
            logger.debug('gradient spin beta:\n%r', grad_b)

        with logtime('computing the hessian'):
            hess_Lag[:n*N, :n*N] = absil.hessian(C_a, C_b,
                                                 self.Fock_a, self.Fock_b,
                                                 self.blocks_a[1],
                                                 self.blocks_b[1],
                                                 self.blocks_a[2],
                                                 self.blocks_b[2],
                                                 g)
            hess_Lag[:n*N_a, :n*N_a] -= np.kron(eps_a + eps_a.T, S)
            hess_Lag[n*N:(n*N + N_a**2), :n*N_a] = -Jac_a
            hess_Lag[:n*N_a, n*N:(n*N + N_a**2)] = -Jac_a.T
            hess_Lag[n*N_a:n*N, n*N_a:n*N] -= np.kron(eps_b + eps_b.T, S)
            hess_Lag[(n*N + N_a**2):, n*N_a:n*N] = -Jac_b
            hess_Lag[n*N_a:n*N, (n*N + N_a**2):] = -Jac_b.T
            logger.debug('hessian:\n%r', hess_Lag)

        self.grad_norm = linalg.norm(grad_Lag)
        self.restriction_norm = linalg.norm(R_a) + linalg.norm(R_b)

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
