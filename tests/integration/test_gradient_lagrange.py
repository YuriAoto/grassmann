import unittest, tests
import numpy as np
from orbitals import orbitals
from hartree_fock import optimiser
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

def build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S, t=0.01):
    n, N_a = C_a.shape
    _, N_b = C_b.shape
    N = N_a + N_b
    grad_numeric = np.zeros((n*N + N_a**2 + N_b**2))
    energies_a, _ = np.linalg.eigh(fock_a)
    energies_a = np.diag(energies_a[:N_a])
    energies_b, _ = np.linalg.eigh(fock_b)
    energies_b = np.diag(energies_b[:N_b])
    P_a = C_a @ C_a.T; tr_a = np.trace(energies_a.T @ C_a.T @ S @ C_a - energies_a.T)
    P_b = C_b @ C_b.T; tr_b = np.trace(energies_b.T @ C_b.T @ S @ C_b - energies_b.T)
    fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)
    energy = 0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
    

    for j in range(N_a):
        for i in range(n):
            C_a[i, j] += t
            P_a = C_a @ C_a.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energyp = (0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
                       - np.trace(energies_a.T @ C_a.T @ S @ C_a - energies_a.T))
            C_a[i, j] -= 2 * t
            P_a = C_a @ C_a.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energym = (0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
                       - np.trace(energies_a.T @ C_a.T @ S @ C_a - energies_a.T))
            grad_numeric[i + j*n] = (energyp - energym) / (2 * t)
            C_a[i, j] += t

    P_a = C_a @ C_a.T
    for j in range(N_b):
        for i in range(n):
            C_b[i, j] += t
            P_b = C_b @ C_b.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energyp = (0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
                       - np.trace(energies_b.T @ C_b.T @ S @ C_b - energies_b.T))
            C_b[i, j] -= 2 * t
            P_b = C_b @ C_b.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energym = (0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
                       - np.trace(energies_b.T @ C_b.T @ S @ C_b - energies_b.T))
            grad_numeric[n*N_a + i + n*j] = (energyp - energym) / (2 * t)
            C_b[i, j] += t

    g_a = C_a.T @ S @ C_a
    for i in range(N_a):
        for j in range(N_a):
            energies_a[i, j] += t
            tr_a_p = -np.trace(energies_a.T @ g_a - energies_a.T)
            energies_a[i, j] -= 2*t
            tr_a_m = -np.trace(energies_a.T @ g_a - energies_a.T)
            grad_numeric[n*N + i + j*N_a] = (tr_a_p - tr_a_m) / (2*t)
            energies_a[i, j] += t

    g_b = C_b.T @ S @ C_b
    for i in range(N_b):
        for j in range(N_b):
            energies_b[i, j] += t
            tr_b_p = -np.trace(energies_b.T @ g_b - energies_b.T)
            energies_b[i, j] -= 2*t
            tr_b_m = -np.trace(energies_b.T @ g_b - energies_b.T)
            grad_numeric[n*N + N_a**2 + i + j*N_b] = (tr_b_p - tr_b_m) / (2*t)
            energies_b[i, j] += t

    return grad_numeric

def build_analytic_gradient(C_a, C_b, fock_a, fock_b, S):
    _, N_a = C_a.shape; n, N_b = C_b.shape; N = N_a + N_b
    analytic_grad = np.zeros((n*N + N_a**2 + N_b**2,))
    restr_a = C_a.T @ S @ C_a - np.eye(N_a)
    restr_b = C_b.T @ S @ C_b - np.eye(N_b)
    energies_a, _ = np.linalg.eigh(fock_a)
    energies_a = np.diag(energies_a[:N_a])
    energies_b, _ = np.linalg.eigh(fock_b)
    energies_b = np.diag(energies_b[:N_b])
    invS = np.linalg.inv(S)

    if N_a:
        # grad_energy_a = np.reshape(2*fock_a @ C_a, (n*N_a,), 'F')
        # jacob_restr_a = np.reshape(2*S @ C_a @ energies_a, (n*N_a,), 'F')
        grad_energy_a = np.reshape(2*invS @ fock_a @ C_a, (n*N_a,), 'F')
        jacob_restr_a = np.reshape(2*C_a @ energies_a, (n*N_a,), 'F')
        analytic_grad[: n*N_a] = grad_energy_a - jacob_restr_a
        analytic_grad[n*N : n*N + N_a**2] = np.reshape(restr_a, (N_a**2,), 'F')

    if N_b:
        # grad_energy_b = np.reshape(2*fock_b @ C_b, (n*N_b,), 'F')
        # jacob_restr_b = np.reshape(2*S @ C_b @ energies_b, (n*N_b,), 'F')
        grad_energy_b = np.reshape(2*invS @ fock_b @ C_b, (n*N_b,), 'F')
        jacob_restr_b = np.reshape(2*C_b @ energies_b, (n*N_b,), 'F')
        analytic_grad[n*N_a : n*N] = grad_energy_b - jacob_restr_b
        analytic_grad[n*N + N_a**2:] = np.reshape(restr_b, (N_b**2,) , 'F')

    return analytic_grad

class GradientTestLagrange(unittest.TestCase):

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
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_H2O_631g(self):
        C_a = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('H2O', 'Req'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_Be_631g(self):
        C_a = np.load(tests.get_references('Orb__Be__at__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_Be_631g_ms22(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_Be_631g_ms2m2(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_Be_631g_ms24(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_Be_631g_ms2m4(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_He2_631g(self):
        C_a = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_He2_ccpvdz(self):
        C_a = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_He8_631g(self):
        C_a = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He8_cage', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_Li2_631g(self):
        C_a = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Li2', '5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)

    def test_N2_631g(self):
        C_a = np.load(tests.get_references('Orb__N2__3__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('N2', '3'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S
        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        P_a = C_a @ C_a.T; P_b = C_b @ C_b.T
        fock_a = build_fock(P_a, P_b, h, g); fock_b = build_fock(P_b, P_a, h, g)

        analytic_grad = build_analytic_gradient(C_a, C_b, fock_a, fock_b, S)
        numeric_grad = build_numeric_gradient(C_a, C_b, fock_a, fock_b, h, g, S)

        self.assertEqual(analytic_grad, numeric_grad)
