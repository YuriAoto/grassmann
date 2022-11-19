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


def build_numeric_jacobian(C_a, C_b, S, t=0.001):
    _, N_a = C_a.shape; n, N_b = C_b.shape; N = N_a + N_b; counter = 0
    numeric_jacob = np.zeros((N_a**2 + N_b**2, n*N))

    for j in range(N_a):
        for i in range(n):
            C_a[i, j] += t
            restr_p = C_a.T @ S @ C_a
            C_a[i, j] -= 2*t
            restr_m = C_a.T @ S @ C_a
            numeric_jacob[:N_a**2, counter] = np.reshape(((restr_p - restr_m)
                                                          / (2*t)),
                                                         (N_a**2) , 'F')
            C_a[i, j] += t; counter += 1

    for j in range(N_b):
        for i in range(n):
            C_b[i, j] += t
            restr_p = C_b.T @ S @ C_b
            C_b[i, j] -= 2*t
            restr_m = C_b.T @ S @ C_b
            numeric_jacob[N_a**2:, counter] = np.reshape(((restr_p - restr_m)
                                                          / (2*t)),
                                                         (N_b**2) , 'F')
            C_b[i, j] += t; counter += 1

    return numeric_jacob

def build_analytic_jacobian(C_a, C_b, S):
    _, N_a = C_a.shape; n, N_b = C_b.shape; N = N_a + N_b
    aux_a = C_a.T @ S; aux_b = C_b.T @ S
    analytic_jacob = np.zeros((N_a**2 + N_b**2, n*N))

    if N_a:
        jacob_restr_a = np.empty((N_a**2, N_a*n))
        e_j = np.zeros((N_a, 1))
        for j in range(N_a):
            e_j[j] = 1.0
            jacob_restr_a[:, n*j : n*(j+1)] = np.kron(aux_a, e_j)
            jacob_restr_a[N_a*j : N_a*(j+1), n*j : n*(j+1)] += aux_a
            e_j[j] = 0.0
        analytic_jacob[:N_a**2, :n*N_a] = jacob_restr_a

    if N_b:
        jacob_restr_b = np.empty((N_b**2, N_b*n))
        e_j = np.zeros((N_b, 1))
        for j in range(N_b):
            e_j[j] = 1.0
            jacob_restr_b[:, n*j : n*(j+1)] = np.kron(aux_b, e_j)
            jacob_restr_b[N_b*j : N_b*(j+1), n*j : n*(j+1)] += aux_b
            e_j[j] = 0.0
        analytic_jacob[N_a**2:, n*N_a:] = jacob_restr_b

    return analytic_jacob


class JacobTestLagrange(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_H2_ccpvdz(self):
        C_a = np.load(tests.get_references('Orb__H2__5__ccpvdz.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('H2', '5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_H2O_631g(self):
        C_a = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('H2O', 'Req'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_Be_631g(self):
        C_a = np.load(tests.get_references('Orb__Be__at__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_Be_631g_ms22(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms2.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_Be_631g_ms2m2(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm2.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm2.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_Be_631g_ms24(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2ms4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2ms4.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_Be_631g_ms2m4(self):
        C_a = np.load(tests.get_references('Orb_alpha__Be__at__631g__2msm4.npy'))
        C_b = np.load(tests.get_references('Orb_beta__Be__at__631g__2msm4.npy'))
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Be', 'at'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_He2_631g(self):
        C_a = np.load(tests.get_references('Orb__He2__1.5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_He2_ccpvdz(self):
        C_a = np.load(tests.get_references('Orb__He2__1.5__ccpvdz.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He2', '1.5'))
        molecular_system.calculate_integrals('cc-pVDZ', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_He8_631g(self):
        C_a = np.load(tests.get_references('Orb__He8_cage__1.5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('He8_cage', '1.5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_Li2_631g(self):
        C_a = np.load(tests.get_references('Orb__Li2__5__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('Li2', '5'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)

    def test_N2_631g(self):
        C_a = np.load(tests.get_references('Orb__N2__3__631g.npy'))
        C_b = C_a.copy()
        molecular_system = MolecularGeometry.from_xyz_file(
            tests.geom_file('N2', '3'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')

        S = molecular_system.integrals.S

        analytic_jacob = build_analytic_jacobian(C_a, C_b, S)
        numeric_jacob = build_numeric_jacobian(C_a, C_b, S)

        self.assertEqual(analytic_jacob, numeric_jacob)
