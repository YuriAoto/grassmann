"""Class to run an iteration step in Hartree Fock optimisation


"""
import logging
import time
import copy

import numpy as np
from scipy import linalg

from util.variables import sqrt2
from input_output.log import logtime
from . import util
from . import absil
from . import absilnp

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


def getindex(i, j, k, l):
    """Convert the indexes of the two-electron integrals."""

    ij = j + i*(i + 1) // 2 if i >= j else i + j*(j + 1) // 2
    kl = l + k*(k + 1) // 2 if k >= l else k + l*(l + 1) // 2
    ijkl = (kl + ij*(ij + 1) // 2 if ij >= kl else ij + kl*(kl + 1) // 2)

    return ijkl

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
        self.g = None

    def initialise(self, step_type, three_indices):
        if step_type == 'RH-SCF':
            self.i_DIIS = -1
            self.grad = np.zeros((self.n_occ,
                                  len(self.orb) - self.n_occ,
                                  max(self.n_DIIS, 1)))
            self.Dmat = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
            if not three_indices:
                self.integrals.g.transform_to_ijkl()
        elif step_type == 'densMat-SCF':
            pass
        elif step_type == 'Absil':
            self.grad = np.zeros((len(self.orb), self.n_occ))
        elif step_type == 'orb_rot-Newton':
            pass
        elif step_type == 'gradient':
            self.grad = np.zeros((len(self.orb), self.n_occ))
        else:
            raise ValueError("Unknown type of Hartree-Fock step: "
                             + step_type)
        
    def roothan_hall(self, i_SCF, three_indices):
        """Roothan-Hall procedure as described, e.g, in Szabo
        
        """
        if i_SCF < self.n_DIIS:
            cur_n_DIIS = i_SCF
        else:
            cur_n_DIIS = self.n_DIIS
        if self.n_DIIS > 0:
            logger.info('current n DIIS: %d', cur_n_DIIS)

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
            if three_indices:
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
            else:
                n, N = len(self.orb), self.n_occ / 2
                g = self.integrals.g._integrals
                Fock = np.array(self.integrals.h)
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            for l in range(n):
                                Fock[i, j] += (self.Dmat[k, l, self.i_DIIS] *
                                               (g[getindex(i, j, l, k)]
                                                - g[getindex(i, k, l, j)] / 2))
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
            logger.info('current self.i_DIIS: %d', self.i_DIIS)
            
        with logtime('Fock matrix in the MO basis'):
            F_MO = self.orb[0].T @ Fock @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.n_occ, self.n_occ:]
            self.gradNorm = linalg.norm(self.grad[:, :, self.i_DIIS]) * sqrt2
            logger.info('Gradient norm = %f', self.gradNorm)
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies)

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
        """Hartree-Fock using Newton Method as it's in Absil."""
        N_a, N_b = self.n_occ_alpha, self.n_occ_beta
        N, n = N_a + N_b, len(self.orb)
        X, Y = self.orb[0][:, :N_a], self.orb[1][:, :N_b]
        S, Sqrt = self.integrals.S, self.integrals.X
        invS, invSqrt = self.integrals.invS, self.integrals.invX
        augD = np.zeros((n*N + N_a**2 + N_b**2, n*N))
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

        # with logtime("computing the gradient using three indices."):
        #     gradX = absil.gradient_three(X, Y, xxt, yyt, h, self.g)
        #     gradY = absil.gradient_three(Y, X, yyt, xxt, h, self.g)

        # with logtime("Gradient using Fock matrix."):
        #     gradX = absil.grad_fock(X, fock_a, xxt, self.g)
        #     gradY = absil.grad_fock(Y, fock_b, yyt, self.g)

        # with logtime("Compute the gradient with four indices."):
        #     gradX = absil.gradient(X, Y, xxt, yyt, h, g)
        #     gradY = absil.gradient(Y, X, yyt, xxt, h, g)
        # print(gradXd)

        # with logtime("computing the old hessian"):
        #     D = absil.directional_derivative(X, Y, xxt, yyt, projX, projY,
        #                                      gradX, gradY, invS, h, g)

        with logtime("computing the hessian in blocks"):
            if N_a != 0:
                PX = np.kron(np.eye(N_a), projX)
                QY = np.kron(np.eye(N_a), projY)
            if N_b != 0:
                QX = np.kron(np.eye(N_b), projX)
                PY = np.kron(np.eye(N_b), projY)
            inv = np.kron(np.eye(N), invS)
            D = inv @ absil.hessian_f(X, Y, g, fock_a, fock_b)
            dir_proj_a = absil.dir_proj(X.T, gradX)
            dir_proj_b = absil.dir_proj(Y.T, gradY)
            D[: n*N_a, : n*N_a] -= dir_proj_a
            D[n*N_a :, n*N_a :] -= dir_proj_b
            if N_a != 0:
                D[: n*N_a, : n*N_a] = PX @ D[: n*N_a, : n*N_a]
                D[: n*N_a, n*N_a :] = QY @ D[: n*N_a, n*N_a :]
            if N_b != 0:
                D[n*N_a :, n*N_a :] = PY @ D[n*N_a :, n*N_a :]
                D[n*N_a :, : n*N_a] = QX @ D[n*N_a :, : n*N_a]

        augD[: n*N, :] = D
        gradX, gradY = invS @ gradX, invS @ gradY
        R[:, :N_a], R[:, N_a:] = -projX @ gradX, -projY @ gradY
        R = np.reshape(R, (n*N,), 'F')
        augR[: n*N] = R

        if N_a != 0:
            augD[n*N : (n*N + N_a**2), : n*N_a] = np.kron(np.eye(N_a),
                                                          X.T @ S)
        if N_b != 0:
            augD[(n*N + N_a**2) :, n*N_a :] = np.kron(np.eye(N_b),
                                                      Y.T @ S)

        with logtime("Solving the main equation"):
            eta = np.linalg.lstsq(augD, augR, rcond=None)

        if eta[1].size < 1:
            logger.warning("Hessian does not have full rank.")
        elif eta[1] > 1e-10:
            logger.warning("Large conditioning number: %.5e", eta[1])

        eta = np.reshape(eta[0], (n, N), 'F')
        logger.debug("eta: \n%s", eta)

        if N_a != 0:
            u, s, v = linalg.svd(Sqrt @ eta[:, :N_a], full_matrices=False)
            sin, cos = np.diag(np.sin(s)), np.diag(np.cos(s))
            X = X @ v.T @ cos + invSqrt @ u @ sin
            self.orb[0][:, :N_a] = absil.gram_schmidt(X, S)

        if N_b != 0:
            u, s, v = linalg.svd(Sqrt @ eta[:, N_a:], full_matrices=False)
            sin, cos = np.diag(np.sin(s)), np.diag(np.cos(s))
            Y = Y @ v.T @ cos + invSqrt @ u @ sin
            self.orb[1][:, :N_b] = absil.gram_schmidt(Y, S)

        self.gradNorm = linalg.norm(R)
        
    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")

    def gradient_descent(self, i_SCF):
        """Hartree-Fock using Gradient Descent Method."""
        N_a, N_b = self.n_occ_alpha, self.n_occ_beta
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
            
        self.gradNorm = linalg.norm(R)
