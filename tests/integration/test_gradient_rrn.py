import unittest, tests
import numpy as np
from orbitals import orbitals
from hartree_fock import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry


def build_numeric_gradient(C_a, C_b, h, g, t):
    n, N_a = C_a.shape
    _, N_b = C_b.shape
    numeric_grad = np.zeros((n, N_a + N_b))

    P_b = C_b @ C_b.T
    for i in range(n):
        for j in range(N_a):
            C_a[i, j] += t
            P_a = C_a @ C_a.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energyp = 0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
            C_a[i, j] -= 2 * t
            P_a = C_a @ C_a.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energym = 0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
            numeric_grad[i, j] = (energyp - energym) / (2 * t)
            C_a[i, j] += t

    P_a = C_a @ C_a.T
    for i in range(n):
        for j in range(N_b):
            C_b[i, j] += t
            P_b = C_b @ C_b.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energyp = 0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
            C_b[i, j] -= 2 * t
            P_b = C_b @ C_b.T
            fock_a = build_fock(P_a, P_b, h, g)
            fock_b = build_fock(P_b, P_a, h, g)
            energym = 0.5 * ((P_a + P_b)*h + P_a*fock_a + P_b*fock_b).sum()
            numeric_grad[i, N_a+j] = (energyp - energym) / (2 * t)
            C_b[i, j] += t

    return numeric_grad

def build_fock(Ps, Pt, h, g):
    """Compute Fock matrix.
    
    F_ij = H_ij + [(Ps)_lk + (Pt)_lk] (ij|L)(L|lk) - (Ps)_lk (ik|L)(L|lj)
    """
    Fock = np.array(h)

    tmp = np.einsum('ij,Lij->L', Ps, g)
    Fock += np.einsum('L,Lkl->kl', tmp, g)
    tmp = np.einsum('ij,Lij->L', Pt, g)
    Fock += np.einsum('L,Lkl->kl', tmp, g)
    tmp = np.einsum('ij,Lkj->Lik', Ps, g)
    Fock -= np.einsum('Lik,Lil->kl', tmp, g)

    return Fock

def grad_fock(C, fock, Ps, g):
    grad = 2 * (fock @ C)
    return grad


class GradientTestFock(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_H2_ccpvdz(self):
        C = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('H2', '5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_H2O_631g(self):
        C = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('H2O', 'Req'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_Be_631g(self):
        C = np.load(tests.get_references('Orb__Be__at__631g.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_Be_631g_ms22(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
        n, N_a = C_a.shape
        n, N_b = C_b.shape
        P_a = C_a @ C_a.T
        P_b = C_b @ C_b.T
        grad = np.empty((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)
        grad[:, :N_a] = grad_fock(C_a, fock_a, P_a, g)
        grad[:, N_a:] = grad_fock(C_b, fock_b, P_b, g)
        numeric_grad = build_numeric_gradient(C_a, C_b, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_Be_631g_ms2m2(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
        n, N_a = C_a.shape
        n, N_b = C_b.shape
        P_a = C_a @ C_a.T
        P_b = C_b @ C_b.T
        grad = np.empty((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)
        grad[:, :N_a] = grad_fock(C_a, fock_a, P_a, g)
        grad[:, N_a:] = grad_fock(C_b, fock_b, P_b, g)
        numeric_grad = build_numeric_gradient(C_a, C_b, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_Be_631g_ms24(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
        n, N_a = C_a.shape
        n, N_b = C_b.shape
        P_a = C_a @ C_a.T
        P_b = C_b @ C_b.T
        grad = np.empty((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)
        grad[:, :N_a] = grad_fock(C_a, fock_a, P_a, g)
        grad[:, N_a:] = grad_fock(C_b, fock_b, P_b, g)
        numeric_grad = build_numeric_gradient(C_a, C_b, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_Be_631g_ms2m4(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
        n, N_a = C_a.shape
        n, N_b = C_b.shape
        P_a = C_a @ C_a.T
        P_b = C_b @ C_b.T
        grad = np.empty((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(P_a, P_b, h, g)
        fock_b = build_fock(P_b, P_a, h, g)
        grad[:, :N_a] = grad_fock(C_a, fock_a, P_a, g)
        grad[:, N_a:] = grad_fock(C_b, fock_b, P_b, g)
        numeric_grad = build_numeric_gradient(C_a, C_b, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_He2_631g(self):
        C = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_He2_ccpvdz(self):
        C = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_He8_631g(self):
        C = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('He8_cage', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_Li2_631g(self):
        C = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Li2', '5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)

    def test_N2_631g(self):
        C = np.load(tests.get_references('Orb__N2__3__631g.npy'))
        n, N = C.shape
        P = C @ C.T
        grad = np.empty((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('N2', '3'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(P, P, h, g)
        grad[:, :N] = grad_fock(C, fock, P, g)
        grad[:, N:] = grad_fock(C, fock, P, g)
        numeric_grad = build_numeric_gradient(C, C, h, g, 0.001)

        self.assertEqual(grad, numeric_grad)
