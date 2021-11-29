import unittest, tests
import numpy as np
from orbitals import orbitals
from hartree_fock import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
from hartree_fock.absil import hessian


class HessianTest(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_H2O_631g(self):
        X = np.load(tests.get_references('Orb__H2O__Req__631g.npy'))
        n, N_a = X.shape
        my_hess = np.zeros((2*n*N_a, 2*n*N_a))
        molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file('H2O', 'Req'))
        molecular_system.calculate_integrals('6-31g', int_meth='ir-wmme')
        my_hess[:, :n*N_a] = hessian(X, X, X @ X.T, X @ X.T,
                                     molecular_system.integrals.h,
                                     molecular_system.integrals.g._integrals)
        my_hess[:, n*N_a:] = hessian(X, X, X @ X.T, X @ X.T,
                                     molecular_system.integrals.h,
                                     molecular_system.integrals.g._integrals)
        ref_hess = np.load(tests.get_references('Hess__H2O__Req__631g.npy'))
        for i in range(2*n*N_a):
            for j in range(2*n*N_a):
                print(f'{my_hess[i, j]:.5f} {ref_hess[i, j]:.5f} {i} {j}')
        # self.assertEqual(my_hess, ref_hess)

