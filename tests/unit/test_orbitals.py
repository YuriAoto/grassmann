"""Tests for orbitals

"""
import unittest
import cProfile

from scipy import linalg
import numpy as np

import tests
from orbitals import orbitals
from orbitals.orbital_space import OrbitalSpace


@tests.category('PROFILE')
class ProfileTestCase(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.prng = np.random.RandomState(tests.init_random_state)

    def test_constr_1(self):
        orb_dim = OrbitalSpace(dim=[2], orb_type='R')
        U = [np.zeros((2, 1))]
        U[0][0, 0] = 1.0
        cProfile.runctx('orbitals.complete_orb_space(U, orb_dim)', globals(), locals())


@tests.category('SHORT', 'ESSENTIAL')
class ContructExtSpaceTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.prng = np.random.RandomState(tests.init_random_state)

    def test_constr_1(self):
        orb_dim = OrbitalSpace(dim=[2], orb_type='R')
        U = [np.zeros((2, 1))]
        U[0][0, 0] = 1.0
        full_U = orbitals.complete_orb_space(U, orb_dim)
        self.assertEqual(full_U[0][:, 0], U[0][:, 0])
        self.assertEqual(full_U[0][:, 1], np.array([0.0, 1.0]))
    
    def test_orbdim_exception(self):
        orb_dim = OrbitalSpace(dim=[4], orb_type='R')
        U = [np.zeros((2, 1))]
        with self.assertRaises(ValueError):
            orbitals.complete_orb_space(U, orb_dim)

    def test_constr_2(self):
        orb_dim = OrbitalSpace(dim=[4], orb_type='R')
        U = [np.zeros((4, 1))]
        U[0][0, 0] = 1.0
        full_U = orbitals.complete_orb_space(U, orb_dim)
        self.assertEqual(full_U[0][:, 0], U[0][:, 0])
        self.assertEqual(np.matmul(full_U[0], full_U[0].T),
                         np.identity(orb_dim[0]))

    def test_constr_3(self):
        orb_dim = OrbitalSpace(dim=[4], orb_type='R')
        newU = self.prng.random_sample(size=(4, 2))
        newU = linalg.orth(newU)
        U = [newU]
        full_U = orbitals.complete_orb_space(U, orb_dim)
        self.assertEqual(U[0][:, 0], U[0][:, 0])
        self.assertEqual(U[0][:, 1], U[0][:, 1])
        self.assertEqual(np.matmul(full_U[0], full_U[0].T),
                         np.identity(orb_dim[0]))

    def test_constr_4(self):
        orb_dim = OrbitalSpace(dim=[10], orb_type='R')
        newU = self.prng.random_sample(size=(10, 3))
        newU = linalg.orth(newU)
        U = [newU]
        full_U = orbitals.complete_orb_space(U, orb_dim)
        self.assertEqual(full_U[0][:, 0], U[0][:, 0])
        self.assertEqual(full_U[0][:, 1], U[0][:, 1])
        self.assertEqual(full_U[0][:, 2], U[0][:, 2])
        self.assertEqual(np.matmul(full_U[0], full_U[0].T),
                         np.identity(orb_dim[0]))

    def test_constr_5(self):
        orb_dim = OrbitalSpace(dim=[10, 5, 5, 0], orb_type='R')
        U = []
        newU = self.prng.random_sample(size=(10, 4))
        newU = linalg.orth(newU)
        U.append(newU)
        newU = self.prng.random_sample(size=(5, 2))
        newU = linalg.orth(newU)
        U.append(newU)
        newU = np.zeros((5, 0))
        U.append(newU)
        newU = np.zeros((0, 0))
        U.append(newU)
        full_U = orbitals.complete_orb_space(U, orb_dim)
        self.assertEqual(full_U[0][:, 0], U[0][:, 0])
        self.assertEqual(full_U[0][:, 1], U[0][:, 1])
        self.assertEqual(full_U[0][:, 2], U[0][:, 2])
        self.assertEqual(full_U[0][:, 3], U[0][:, 3])
        self.assertEqual(np.matmul(full_U[0], full_U[0].T),
                         np.identity(orb_dim[0]))
        self.assertEqual(full_U[1][:, 0], U[1][:, 0])
        self.assertEqual(full_U[1][:, 1], U[1][:, 1])
        self.assertEqual(np.matmul(full_U[1], full_U[1].T),
                         np.identity(orb_dim[1]))
        self.assertEqual(np.matmul(full_U[2], full_U[2].T),
                         np.identity(orb_dim[2]))

    def test_constr_6(self):
        orb_dim = OrbitalSpace(dim=[2], orb_type='R')
        U = [np.zeros((2, 1))]
        U[0][1, 0] = 1.0
        full_U = orbitals.complete_orb_space(U, orb_dim)
        self.assertEqual(full_U[0][:, 0], U[0][:, 0])
        self.assertEqual(full_U[0][:, 1], np.array([1.0, 0.0]))

    def test_constr_7(self):
        orb_dim = OrbitalSpace(dim=[6], orb_type='R')
        newU = np.zeros((6, 2))
        newU[3, 0] = 1.0
        newU[5, 1] = 1.0
        U = [newU]
        tests.logger.info('U, before completing it:\n%s', U)
        full_U = orbitals.complete_orb_space(U, orb_dim)
        tests.logger.info('U, after completing it:\n%s', full_U)
        self.assertEqual(U[0][:, 0], U[0][:, 0])
        self.assertEqual(U[0][:, 1], U[0][:, 1])
        self.assertEqual(np.matmul(full_U[0], full_U[0].T),
                         np.identity(orb_dim[0]))
