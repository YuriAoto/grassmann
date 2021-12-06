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
        self.restricted = None
        self.n_DIIS = 0
        self.n_occ = 0
        self.n_occ_alpha = 0
        self.n_occ_beta = 0
        self.integrals = None
        self.orb = None
        self.energy = None
        self.grad = None
        self.P_a = None
        self.P_b = None
        self.Fock = None
        self.gradNorm = None
        self.g = None

    def initialise(self, step_type, three_indices=True):
        if step_type == 'RH-SCF':
            self.i_DIIS = -1
            self.grad = np.zeros((self.n_occ_alpha,
                                  len(self.orb) - self.n_occ_alpha,
                                  max(self.n_DIIS, 1)))
            self.P_a = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
            if not self.restricted:
                self.grad_beta = np.zeros((self.n_occ_beta,
                                           len(self.orb) - self.n_occ_beta,
                                           max(self.n_DIIS, 1)))
                self.P_b = np.zeros((len(self.orb),
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

    def calc_density_matrix(self):
        """Calculate the density matrix (or matrices)
        
        P = C @ C.T
        """
        self.P_a[:, :, self.i_DIIS] = np.einsum(
            'pi,qi->pq',
            self.orb[0][:, :self.n_occ_alpha],
            self.orb[0][:, :self.n_occ_alpha])
        if self.restricted:
            self.P_a[:, :, self.i_DIIS] = 2 * self.P_a[:, :, self.i_DIIS]
            logger.debug('Density matrix:\n%r', self.P_a[:, :, self.i_DIIS])
        else:
            self.P_b[:, :, self.i_DIIS] = np.einsum(
                'pi,qi->pq',
                self.orb[1][:, :self.n_occ_beta],
                self.orb[1][:, :self.n_occ_beta])
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
            Fock = np.array(self.integrals.h)
            tmp = np.einsum('rs,Frs->F',
                            self.P_a[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            Fock += np.einsum('F,Fmn->mn',
                              tmp,
                              self.integrals.g._integrals)
            tmp = np.einsum('rs,Fms->Frm',
                            self.P_a[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            Fock -= np.einsum('Frm,Frn->mn',
                                   tmp,
                                   self.integrals.g._integrals) / 2
            return Fock
        else:
            return (absil.fock(self.P_a[:, :, self.i_DIIS],
                               self.P_b[:, :, self.i_DIIS],
                               self.integrals.h, self.integrals.g._integrals),
                    absil.fock(self.P_b[:, :, self.i_DIIS],
                               self.P_a[:, :, self.i_DIIS],
                               self.integrals.h, self.integrals.g._integrals))

    def calc_energy(self, Fock):
        """Calculate the energy
        
        TODO: integrate restricted and unrestricted codes
        """
        if self.restricted:
            self.energy = np.tensordot(self.P_a[:, :, self.i_DIIS],
                                       self.integrals.h + Fock) / 2
            self.one_el_energy = np.tensordot(self.P_a[:, :, self.i_DIIS],
                                              self.integrals.h)
            self.two_el_energy = np.tensordot(self.P_a[:, :, self.i_DIIS],
                                              Fock - self.integrals.h) / 2
        else:
            self.one_el_energy = ((self.P_a[:, :, self.i_DIIS]
                                   + self.P_b[:, :, self.i_DIIS])* self.integrals.h).sum()
            self.energy = 0.5*((self.P_a[:, :, self.i_DIIS]*Fock[0]
                                + self.P_b[:, :, self.i_DIIS]*Fock[1]).sum()
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

    def calc_mo_fock(self, Fock):
        if self.restricted:
            F_MO = self.orb[0].T @ Fock @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.n_occ, self.n_occ:]
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            self.gradNorm = linalg.norm(self.grad[:, :, self.i_DIIS]) * sqrt2
            logger.info('Gradient norm = %f', self.gradNorm)
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies)
        else:
            F_MO = self.orb[0].T @ Fock[0] @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.n_occ_alpha, self.n_occ_alpha:]
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            F_MO = self.orb[1].T @ Fock[1] @ self.orb[1]
            self.grad_beta[:, :, self.i_DIIS] = F_MO[:self.n_occ_beta, self.n_occ_beta:]
            self.orb.energies_beta = np.array([F_MO[i, i]
                                               for i in range(len(self.orb))])
            self.gradNorm = sqrt(linalg.norm(self.grad[:, :, self.i_DIIS])**2
                                 + linalg.norm(self.grad_beta[:, :, self.i_DIIS])**2)
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r',
                            self.orb.energies)
                logger.info('Current beta orbital energies:\n%r',
                            self.orb.energies_beta)

    def diag_fock(self, Fock):
        if self.restricted:
            Fock = self.integrals.X.T @ Fock @ self.integrals.X
            e, C = linalg.eigh(Fock)
            #        e, C = linalg.eig(self.Fock)
            #        e_sort_index = np.argsort(e)
            #        e = e[e_sort_index]
            #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
        else:
            fock_a = self.integrals.X.T @ Fock[0] @ self.integrals.X
            fock_b = self.integrals.X.T @ Fock[1] @ self.integrals.X
            e, C = linalg.eigh(fock_a)
            eb, Cb = linalg.eigh(fock_b)
            #        e, C = linalg.eig(Fock)
            #        e_sort_index = np.argsort(e)
            #        e = e[e_sort_index]
            #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C
            self.orb[1][:, :] = self.integrals.X @ Cb


    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo
        
        """
        with logtime('Form the density matrix'):
            self.calc_density_matrix()
        with logtime('DIIS step'):
            self.calc_diis(i_SCF)
        with logtime('Form Fock matrix'):
            Fock = self.calc_fock_matrix()
        with logtime('Calculate Energy'):
            self.calc_energy(Fock)
        with logtime('Fock matrix in the MO basis'):
            self.calc_mo_fock(Fock)
        with logtime('Fock matrix in the orthogonal basis'):
            self.diag_fock(Fock)

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
            fock_a = absil.fock(xxt, yyt, h, g)
            fock_b = absil.fock(yyt, xxt, h, g)

        with logtime("computing the energy."):
            self.one_el_energy = ((xxt + yyt)*h).sum()
            self.energy = 0.5*((xxt*fock_a + yyt*fock_b).sum()
                               + self.one_el_energy)
            self.two_el_energy = self.energy - self.one_el_energy

        with logtime("computing the gradient."):
            gradX = fock_a @ X
            gradY = fock_b @ Y

        with logtime("computing the hessian in blocks"):
            if N_a != 0:
                PX = np.kron(np.eye(N_a), projX)
                QY = np.kron(np.eye(N_a), projY)
            if N_b != 0:
                QX = np.kron(np.eye(N_b), projX)
                PY = np.kron(np.eye(N_b), projY)
            inv = np.kron(np.eye(N), invS)
            D = inv @ absil.hessian(X, Y, g, fock_a, fock_b)
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
