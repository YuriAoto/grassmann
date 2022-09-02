"""Checks numerically if the Hessian of NMLM is correct.

Since Hess(g) = J(grad(g)), being J the jacobian, we use regular finite
difference in the gradient to check if the analytic Hessian is close to the
numeric Hessian. So, we basically use the formula

\partial{f}^2 / \partial{x_ix}(x) = (grad(f)(x + te_i) - grad(f)(x - te_i)) / 2t

varying i and for a small t to build the numeric Hessian and then checks if the
result is close to the analytic Hessian. Notice that the right side is a vector
containing all partial derivatives and when we increment in the e_i direction,
we are basically obtaining the vector with all second partial derivatives, ie,
\partial{f}^2 / \partial{x_ix_j} for all j.
"""


import unittest, tests
import numpy as np
from orbitals import orbitals
from hartree_fock import optimiser, absil
from molecular_geometry.molecular_geometry import MolecularGeometry


def build_fock(Ps, Pt, h, g):
    """Compute Fock matrix.
    
    F_ij = H_ij + [(Ps)_lk + (Pt)_lk] (ij|L)(L|lk) - (Ps)_lk (ik|L)(L|lj)
    """
    fock = np.array(h)

    tmp = np.einsum('ij,Lij->L', Ps, g)
    fock += np.einsum('L,Lkl->kl', tmp, g)
    tmp = np.einsum('ij,Lij->L', Pt, g)
    fock += np.einsum('L,Lkl->kl', tmp, g)
    tmp = np.einsum('ij,Lkj->Lik', Ps, g)
    fock -= np.einsum('Lik,Lil->kl', tmp, g)

    return fock

def build_gradient(C_a, C_b, fock_a, fock_b, energies_a, energies_b, S):
    _, N_a = C_a.shape; n, N_b = C_b.shape; N = N_a + N_b
    grad = np.zeros((n*N + N_a**2 + N_b**2,))
    restr_a = C_a.T @ S @ C_a - np.eye(N_a)
    restr_b = C_b.T @ S @ C_b - np.eye(N_b)
    aux_a = C_a.T @ S; aux_b = C_b.T @ S
        
    if N_a:
        jacob_restr_a = np.empty((N_a**2, N_a*n))
        e_j = np.zeros((N_a, 1))
        for j in range(N_a):
            e_j[j] = 1.0
            jacob_restr_a[:, n*j : n*(j+1)] = np.kron(aux_a, e_j)
            jacob_restr_a[N_a*j : N_a*(j+1), n*j : n*(j+1)] += aux_a
            e_j[j] = 0.0
        jacob_restr_a = jacob_restr_a.T@np.reshape(energies_a, (N_a**2,) , 'F')
        grad_energy_a = np.reshape(2*fock_a @ C_a, (n*N_a,), 'F')
        # jacob_restr_a = np.reshape(2*S @ C_a @ energies_a, (n*N_a,), 'F')
        grad[: n*N_a] = grad_energy_a - jacob_restr_a
        grad[n*N : n*N + N_a**2] = np.reshape(-restr_a, (N_a**2,), 'F')

    if N_b:
        jacob_restr_b = np.empty((N_b**2, N_b*n))
        e_j = np.zeros((N_b, 1))
        for j in range(N_b):
            e_j[j] = 1.0
            jacob_restr_b[:, n*j : n*(j+1)] = np.kron(aux_b, e_j)
            jacob_restr_b[N_b*j : N_b*(j+1), n*j : n*(j+1)] += aux_b
            e_j[j] = 0.0
        jacob_restr_b = jacob_restr_b.T@np.reshape(energies_b, (N_b**2,) , 'F')
        grad_energy_b = np.reshape(2*fock_b @ C_b, (n*N_b,), 'F')
        # jacob_restr_b = np.reshape(2*S @ C_b @ energies_b, (n*N_b,), 'F')
        grad[n*N_a : n*N] = grad_energy_b - jacob_restr_b
        grad[n*N + N_a**2:] = np.reshape(-restr_b, (N_b**2,) , 'F')

    return grad

def build_numeric_hessian(C_a, C_b, S, h, g, t=0.001):
    _, N_a = C_a.shape; n, N_b = C_b.shape; N = N_a + N_b; counter = 0
    numeric_hess = np.zeros((n*N + N_a**2 + N_b**2, n*N + N_a**2 + N_b**2))
    P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
    fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)
    energies_a, _ = np.linalg.eigh(fock_a)
    energies_a = np.diag(energies_a[:N_a])
    energies_b, _ = np.linalg.eigh(fock_b)
    energies_b = np.diag(energies_b[:N_b])

    for j in range(N_a):
        for i in range(n):
            C_a[i, j] += t; P_a = C_a @ C_a.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            grad_p = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            C_a[i, j] -= 2*t; P_a = C_a @ C_a.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            grad_m = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            numeric_hess[counter, :] = (grad_p - grad_m) / (2*t)
            C_a[i, j] += t; counter += 1

    P_a = C_a @ C_a.T
    for j in range(N_b):
        for i in range(n):
            C_b[i, j] += t; P_b = C_b @ C_b.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            grad_p = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            C_b[i, j] -= 2*t; P_b = C_b @ C_b.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            grad_m = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            numeric_hess[counter, :] = (grad_p - grad_m) / (2*t)
            C_b[i, j] += t; counter += 1

    P_b = C_b @ C_b.T
    fock_a = build_fock(P_a, P_b, h, g)
    fock_b = build_fock(P_b, P_a, h, g)
    for j in range(N_a):
        for i in range(N_a):
            energies_a[i, j] += t
            grad_p = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            energies_a[i, j] -= 2*t
            grad_m = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            numeric_hess[counter, :] = (grad_p - grad_m) / (2*t)
            energies_a[i, j] += t; counter += 1

    for j in range(N_b):
        for i in range(N_b):
            energies_b[i, j] += t
            grad_p = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            energies_b[i, j] -= 2*t
            grad_m = build_gradient(C_a, C_b, fock_a, fock_b,
                                    energies_a, energies_b, S)
            numeric_hess[counter, :] = (grad_p - grad_m) / (2*t)
            energies_b[i, j] += t; counter += 1

    return numeric_hess

def build_analytic_hessian(C_a, C_b, S, h, g):
    _, N_a = C_a.shape; n, N_b = C_b.shape; N = N_a + N_b
    aux_a = C_a.T @ S; aux_b = C_b.T @ S
    P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
    blocks = absil.common_blocks(C_a, C_b, P_a, P_b, g)
    fock_a = build_fock(P_a, P_b, h, g)
    fock_b = build_fock(P_b, P_a, h, g)
    energies_a, _ = np.linalg.eigh(fock_a)
    energies_a = np.diag(energies_a[:N_a])
    energies_b, _ = np.linalg.eigh(fock_b)
    energies_b = np.diag(energies_b[:N_b])
    analytic_hess = np.zeros((n*N + N_a**2 + N_b**2, n*N + N_a**2 + N_b**2))

    if N_a:
        jacob_restr_a = np.empty((N_a**2, N_a*n))
        e_j = np.zeros((N_a, 1))
        for j in range(N_a):
            e_j[j] = 1.0
            jacob_restr_a[:, n*j : n*(j+1)] = np.kron(aux_a, e_j)
            jacob_restr_a[N_a*j : N_a*(j+1), n*j : n*(j+1)] += aux_a
            e_j[j] = 0.0
        analytic_hess[:n*N_a, :n*N_a] -= 2*np.kron(energies_a.T, S)
        analytic_hess[n*N : n*N + N_a**2, : n*N_a] = -jacob_restr_a
        analytic_hess[: n*N_a, n*N : n*N + N_a**2] = -jacob_restr_a.T

    if N_b:
        jacob_restr_b = np.empty((N_b**2, N_b*n))
        e_j = np.zeros((N_b, 1))
        for j in range(N_b):
            e_j[j] = 1.0
            jacob_restr_b[:, n*j : n*(j+1)] = np.kron(aux_b, e_j)
            jacob_restr_b[N_b*j : N_b*(j+1), n*j : n*(j+1)] += aux_b
            e_j[j] = 0.0
        analytic_hess[n*N_a:n*N, n*N_a:n*N] -= 2*np.kron(energies_b.T, S)
        analytic_hess[n*N + N_a**2 :, n*N_a : n*N] = -jacob_restr_b
        analytic_hess[n*N_a : n*N, n*N + N_a**2 :] = -jacob_restr_b.T

    analytic_hess[:n*N, :n*N] += 2*absil.hessian(C_a, C_b, fock_a, fock_b,
                                                 blocks[2], blocks[3],
                                                 blocks[4], blocks[5], g)

    return analytic_hess


class HessianTestLagrange(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_H2_ccpvdz(self):
        C_a = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('H2', '5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(np.zeros(analytic_hess.shape),
                         analytic_hess - numeric_hess)

    def test_H2O_631g(self):
        C_a = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('H2O', 'Req'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(np.zeros(analytic_hess.shape),
                         analytic_hess - numeric_hess)

    def test_Be_631g(self):
        C_a = np.load(tests.get_references('Orb__Be__at__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_Be_631g_ms22(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_Be_631g_ms2m2(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_Be_631g_ms24(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_Be_631g_ms2m4(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_He2_631g(self):
        C_a = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_He2_ccpvdz(self):
        C_a = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_He8_631g(self):
        C_a = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He8_cage', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_Li2_631g(self):
        C_a = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Li2', '5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)

    def test_N2_631g(self):
        C_a = np.load(tests.get_references('Orb__N2__3__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('N2', '3'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals

        analytic_hess = build_analytic_hessian(C_a, C_b, S, h, g)
        numeric_hess = build_numeric_hessian(C_a, C_b, S, h, g)

        self.assertEqual(analytic_hess, numeric_hess)
