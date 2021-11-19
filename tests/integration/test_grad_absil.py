import unittest, tests
import numpy as np
from orbitals import orbitals
from hartree_fock import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
# from hartree_fock.absil import gradient_three
from hartree_fock.absil import gradient
# from hartree_fock.absil import grad_fock
from hartree_fock.absil import fock_three
from hartree_fock.absil import fock_four


def _check_gradient(X, Y, h, g, t):
    n, N_a = X.shape
    _, N_b = Y.shape
    M = np.zeros((n, N_a + N_b))

    yyt = Y @ Y.T
    for i in range(n):
        for j in range(N_a):
            X[i, j] += t
            xxt = X @ X.T
            fock_a = fock_three(xxt, yyt, h, g)
            fock_b = fock_three(yyt, xxt, h, g)
            energyp = 0.5 * ((xxt + yyt)*h + xxt*fock_a + yyt*fock_b).sum()
            X[i, j] -= 2 * t
            xxt = X @ X.T
            fock_a = fock_three(xxt, yyt, h, g)
            fock_b = fock_three(yyt, xxt, h, g)
            energym = 0.5 * ((xxt + yyt)*h + xxt*fock_a + yyt*fock_b).sum()
            M[i, j] = (energyp - energym) / (2 * t)
            X[i, j] += t

    xxt = X @ X.T
    for i in range(n):
        for j in range(N_b):
            Y[i, j] += t
            yyt = Y @ Y.T
            fock_a = fock_three(xxt, yyt, h, g)
            fock_b = fock_three(yyt, xxt, h, g)
            energyp = 0.5 * ((xxt + yyt)*h + xxt*fock_a + yyt*fock_b).sum()
            Y[i, j] -= 2 * t
            yyt = Y @ Y.T
            fock_a = fock_three(xxt, yyt, h, g)
            fock_b = fock_three(yyt, xxt, h, g)
            energym = 0.5 * ((xxt + yyt)*h + xxt*fock_a + yyt*fock_b).sum()
            M[i, N_a+j] = (energyp - energym) / (2 * t)
            Y[i, j] += t

    return M

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

def grad_fock(W, F, Ps, g):
    grad = 2 * (F @ W)
    return grad


class GradientTestFock(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_H2_ccpvdz(self):
        X = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('H2', '5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_H2O_631g(self):
        X = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('H2O', 'Req'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_Be_631g(self):
        X = np.load(tests.get_references('Orb__Be__at__631g.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_Be_631g_ms22(self):
        X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
        Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
        n, N_a = X.shape
        n, N_b = Y.shape
        xxt = X @ X.T
        yyt = Y @ Y.T
        grad = np.zeros((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(xxt, yyt, h, g)
        fock_b = build_fock(yyt, xxt, h, g)
        grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
        grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
        numeric_grad = _check_gradient(X, Y, h, g, 0.001)
        zeros = np.zeros((n, N_a + N_b))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_Be_631g_ms2m2(self):
        X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
        Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
        n, N_a = X.shape
        n, N_b = Y.shape
        xxt = X @ X.T
        yyt = Y @ Y.T
        grad = np.zeros((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(xxt, yyt, h, g)
        fock_b = build_fock(yyt, xxt, h, g)
        grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
        grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
        numeric_grad = _check_gradient(X, Y, h, g, 0.001)
        zeros = np.zeros((n, N_a + N_b))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_Be_631g_ms24(self):
        X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
        Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
        n, N_a = X.shape
        n, N_b = Y.shape
        xxt = X @ X.T
        yyt = Y @ Y.T
        grad = np.zeros((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(xxt, yyt, h, g)
        fock_b = build_fock(yyt, xxt, h, g)
        grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
        grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
        numeric_grad = _check_gradient(X, Y, h, g, 0.001)
        zeros = np.zeros((n, N_a + N_b))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_Be_631g_ms2m4(self):
        X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
        Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
        n, N_a = X.shape
        n, N_b = Y.shape
        xxt = X @ X.T
        yyt = Y @ Y.T
        grad = np.zeros((n, N_a + N_b))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock_a = build_fock(xxt, yyt, h, g)
        fock_b = build_fock(yyt, xxt, h, g)
        grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
        grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
        numeric_grad = _check_gradient(X, Y, h, g, 0.001)
        zeros = np.zeros((n, N_a + N_b))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_He2_631g(self):
        X = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_He2_ccpvdz(self):
        X = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_He8_631g(self):
        X = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('He8_cage', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_Li2_631g(self):
        X = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('Li2', '5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)

    def test_N2_631g(self):
        X = np.load(tests.get_references('Orb__N2__3__631g.npy'))
        n, N = X.shape
        xxt = X @ X.T
        grad = np.zeros((n, 2*N))

        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('N2', '3'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        h = molecular_system.integrals.h
        g = molecular_system.integrals.g._integrals
        fock = build_fock(xxt, xxt, h, g)
        grad[:, :N] = grad_fock(X, fock, xxt, g)
        grad[:, N:] = grad_fock(X, fock, xxt, g)
        numeric_grad = _check_gradient(X, X, h, g, 0.001)
        zeros = np.zeros((n, 2*N))

        self.assertEqual(zeros, grad - numeric_grad)


# @unittest.skip('Not now')
# class GradientTestFock(unittest.TestCase):

#     def setUp(self):
#         self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

#     def test_H2_ccpvdz(self):
#         X = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('H2', '5'))
#         molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_H2O_631g(self):
#         X = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('H2O', 'Req'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g(self):
#         X = np.load(tests.get_references('Orb__Be__at__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms22(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock_a = fock_three(xxt, yyt, h, g)
#         fock_b = fock_three(yyt, xxt, h, g)
#         grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
#         grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms2m2(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock_a = fock_three(xxt, yyt, h, g)
#         fock_b = fock_three(yyt, xxt, h, g)
#         grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
#         grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms24(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock_a = fock_three(xxt, yyt, h, g)
#         fock_b = fock_three(yyt, xxt, h, g)
#         grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
#         grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms2m4(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock_a = fock_three(xxt, yyt, h, g)
#         fock_b = fock_three(yyt, xxt, h, g)
#         grad[:, :N_a] = grad_fock(X, fock_a, xxt, g)
#         grad[:, N_a:] = grad_fock(Y, fock_b, yyt, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He2_631g(self):
#         X = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He2', '1.5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He2_ccpvdz(self):
#         X = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He2', '1.5'))
#         molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He8_631g(self):
#         X = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He8_cage', '1.5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Li2_631g(self):
#         X = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Li2', '5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_N2_631g(self):
#         X = np.load(tests.get_references('Orb__N2__3__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('N2', '3'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         fock = fock_three(xxt, xxt, h, g)
#         grad[:, :N] = grad_fock(X, fock, xxt, g)
#         grad[:, N:] = grad_fock(X, fock, xxt, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)


# @unittest.skip('Not now.')
# class GradientTestThree(unittest.TestCase):

#     def setUp(self):
#         self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

#     def test_H2_ccpvdz(self):
#         X = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('H2', '5'))
#         molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_H2O_631g(self):
#         X = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('H2O', 'Req'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g(self):
#         X = np.load(tests.get_references('Orb__Be__at__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms22(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient_three(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient_three(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms2m2(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient_three(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient_three(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms24(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient_three(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient_three(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms2m4(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient_three(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient_three(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He2_631g(self):
#         X = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He2', '1.5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He2_ccpvdz(self):
#         X = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He2', '1.5'))
#         molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He8_631g(self):
#         X = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He8_cage', '1.5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Li2_631g(self):
#         X = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Li2', '5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_N2_631g(self):
#         X = np.load(tests.get_references('Orb__N2__3__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('N2', '3'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient_three(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient_three(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)


# @unittest.skip('Not now.')
# class GradientTestFour(unittest.TestCase):

#     def setUp(self):
#         self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

#     def test_H2_ccpvdz(self):
#         X = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('H2', '5'))
#         molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_H2O_631g(self):
#         X = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('H2O', 'Req'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g(self):
#         X = np.load(tests.get_references('Orb__Be__at__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms22(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms2m2(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms24(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Be_631g_ms2m4(self):
#         X = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
#         Y = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
#         n, N_a = X.shape
#         n, N_b = Y.shape
#         xxt = X @ X.T
#         yyt = Y @ Y.T
#         grad = np.zeros((n, N_a + N_b))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Be', 'at'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N_a] = gradient(X, Y, xxt, yyt, h, g)
#         grad[:, N_a:] = gradient(Y, X, yyt, xxt, h, g)
#         numeric_grad = _check_gradient(X, Y, h, g, 0.001)
#         zeros = np.zeros((n, N_a + N_b))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He2_631g(self):
#         X = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He2', '1.5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He2_ccpvdz(self):
#         X = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He2', '1.5'))
#         molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_He8_631g(self):
#         X = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('He8_cage', '1.5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_Li2_631g(self):
#         X = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('Li2', '5'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)

#     def test_N2_631g(self):
#         X = np.load(tests.get_references('Orb__N2__3__631g.npy'))
#         n, N = X.shape
#         xxt = X @ X.T
#         grad = np.zeros((n, 2*N))

#         molecular_system = MolecularGeometry.from_xyz_file(
#         tests.geom_file('N2', '3'))
#         molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

#         h = molecular_system.integrals.h
#         molecular_system.integrals.g.transform_to_ijkl()
#         g = molecular_system.integrals.g._integrals
#         grad[:, :N] = gradient(X, X, xxt, xxt, h, g)
#         grad[:, N:] = gradient(X, X, xxt, xxt, h, g)
#         numeric_grad = _check_gradient(X, X, h, g, 0.001)
#         zeros = np.zeros((n, 2*N))

#         self.assertEqual(zeros, grad - numeric_grad)


# # investigar: rodar gradiente com 3 índices, 4 índices e com matriz de fock.
