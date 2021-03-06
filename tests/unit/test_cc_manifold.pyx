"""Tests for CC manifold

"""
import unittest

import numpy as np

import tests
from util.variables import int_dtype
from util.other import int_array
from orbitals.occ_orbitals cimport OccOrbital
from orbitals.occ_orbitals import OccOrbital
from orbitals.orbital_space cimport FullOrbitalSpace, OrbitalSpace
from orbitals.orbital_space import FullOrbitalSpace, OrbitalSpace
from coupled_cluster.manifold_util cimport SingleExc, DoubleExc
from coupled_cluster.manifold_term1 cimport (term1_a, term1_b,
                                             term1_aa, term1_bb,
                                             term1_ab)
from coupled_cluster.manifold_term2 cimport (term2_diag_a, term2_diag_b,
                                             term2_diag_aa,
                                             term2_diag_ab)



@tests.category('SHORT')
class Terms2el6orbTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.nel = 1
        self.norb = 3
        self.occ_a = np.empty(self.nel, dtype=int_dtype)
        self.exc_occ_a = np.empty(self.nel, dtype=int_dtype)
        self.occ_b = np.empty(self.nel, dtype=int_dtype)
        self.exc_occ_b = np.empty(self.nel, dtype=int_dtype)
        self.wf = np.array([[1.0,  0.7, -0.5],
                            [0.2, -0.4,  0.8],
                            [0.7,  0.6, -0.3]])
        self.wf_cc = np.array([[ 1.0, -0.2, 0.5],
                               [ 0.7,  0.3, 0.9],
                               [-0.2,  0.1, 0.1]])
        self.str_gr = int_array([[0],
                                 [1],
                                 [2]])

    def test_term1_singles(self):
        cdef SingleExc single_exc
        single_exc.i = 0
        single_exc.a = 1
        self.assertAlmostEqual(term1_a(single_exc,
                                       self.wf, self.wf_cc,
                                       self.str_gr,
                                       self.occ_a, self.exc_occ_a),
                               0.41)
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term1_a(single_exc,
                                       self.wf, self.wf_cc,
                                       self.str_gr,
                                       self.occ_a, self.exc_occ_a),
                               -0.6)
        single_exc.i = 0
        single_exc.a = 1
        self.assertAlmostEqual(term1_b(single_exc,
                                       self.wf, self.wf_cc,
                                       self.str_gr,
                                       self.occ_b, self.exc_occ_b),
                               -0.31)
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term1_b(single_exc,
                                       self.wf, self.wf_cc,
                                       self.str_gr,
                                       self.occ_b, self.exc_occ_b),
                               0.99)
    
    def test_term1_doubles(self):
        cdef DoubleExc double_exc
        double_exc.i = 0
        double_exc.a = 1
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.str_gr, self.str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               0.7)
        double_exc.i = 0
        double_exc.a = 1
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.str_gr, self.str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               0.1)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.str_gr, self.str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               -0.5)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.str_gr, self.str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               0.4)
    
    def test_term2_diag_singles(self):
        cdef SingleExc single_exc
        single_exc.i = 0
        single_exc.a = 1
        self.assertAlmostEqual(term2_diag_a(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -1.29)
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term2_diag_a(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -1.29)
        single_exc.i = 0
        single_exc.a = 1
        self.assertAlmostEqual(term2_diag_b(single_exc,
                                            self.wf_cc,
                                            self.occ_b),
                               -1.53)
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term2_diag_b(single_exc,
                                            self.wf_cc,
                                            self.occ_b),
                               -1.53)
    
    def test_term2_diag_doubles(self):
        cdef DoubleExc double_exc
        double_exc.i = 0
        double_exc.a = 1
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.0)
        double_exc.i = 0
        double_exc.a = 1
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.0)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.0)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.0)


@tests.category('SHORT')
class Terms3el7orbTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.beta_nel = 1
        self.beta_norb = 3
        self.alpha_nel = 2
        self.alpha_norb = 4
        self.occ_a = np.empty(self.alpha_nel, dtype=int_dtype)
        self.exc_occ_a = np.empty(self.alpha_nel, dtype=int_dtype)
        self.exc_occ_a2 = np.empty(self.alpha_nel, dtype=int_dtype)
        self.occ_b = np.empty(self.beta_nel, dtype=int_dtype)
        self.exc_occ_b = np.empty(self.beta_nel, dtype=int_dtype)
        self.wf = np.array([[ 1.0,  0.1, -0.7],
                            [-0.3,  0.2,  0.8],
                            [ 0.2, -0.4, -0.9],
                            [-0.7,  0.2,  0.1],
                            [ 0.8, -0.3,  0.5],
                            [ 0.1,  0.7,  0.6]])
        self.wf_cc = np.array([[ 1.0, -0.7, -0.5],
                               [ 0.2,  0.8,  0.7],
                               [ 0.7, -0.3,  0.6],
                               [-0.2,  0.1, -0.8],
                               [-0.1, -0.4,  0.9],
                               [ 0.6,  0.5,  0.3]])
        self.alpha_str_gr = int_array([[0, 0],
                                       [1, 1],
                                       [2, 3]])
        self.beta_str_gr = int_array([[0],
                                      [1],
                                      [2]])
      
    def test_term1_singles(self):
        cdef SingleExc single_exc
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term1_a(single_exc,
                                       self.wf, self.wf_cc,
                                       self.alpha_str_gr,
                                       self.occ_a, self.exc_occ_a),
                               0.44)
        single_exc.i = 0
        single_exc.a = 3
        self.assertAlmostEqual(term1_a(single_exc,
                                       self.wf, self.wf_cc,
                                       self.alpha_str_gr,
                                       self.occ_a, self.exc_occ_a),
                               1.3)
        single_exc.i = 1
        single_exc.a = 2
        self.assertAlmostEqual(term1_a(single_exc,
                                       self.wf, self.wf_cc,
                                       self.alpha_str_gr,
                                       self.occ_a, self.exc_occ_a),
                               -0.11)
        single_exc.i = 1
        single_exc.a = 3
        self.assertAlmostEqual(term1_a(single_exc,
                                       self.wf, self.wf_cc,
                                       self.alpha_str_gr,
                                       self.occ_a, self.exc_occ_a),
                               0.79)
        single_exc.i = 0
        single_exc.a = 1
        self.assertAlmostEqual(term1_b(single_exc,
                                       self.wf, self.wf_cc,
                                       self.beta_str_gr,
                                       self.occ_b, self.exc_occ_b),
                               -0.7)
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term1_b(single_exc,
                                       self.wf, self.wf_cc,
                                       self.beta_str_gr,
                                       self.occ_b, self.exc_occ_b),
                               1.19)
    
    def test_term1_doubles(self):
        cdef DoubleExc double_exc
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 1
        double_exc.b = 3
        self.assertAlmostEqual(term1_aa(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr,
                                        self.occ_a, self.exc_occ_a),
                               0.79)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               -0.06)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               -1.44)
        double_exc.i = 0
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               0.14)
        double_exc.i = 0
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               -0.34)
        double_exc.i = 1
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               0.62)
        double_exc.i = 1
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               -0.07)
        double_exc.i = 1
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               0.04)
        double_exc.i = 1
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term1_ab(double_exc,
                                        self.wf, self.wf_cc,
                                        self.alpha_str_gr, self.beta_str_gr,
                                        self.occ_a, self.exc_occ_a,
                                        self.occ_b, self.exc_occ_b),
                               -0.69)
        
    def test_term2_diag_singles(self):
        cdef SingleExc single_exc
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term2_diag_a(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -2.43)
        single_exc.i = 0
        single_exc.a = 3
        self.assertAlmostEqual(term2_diag_a(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -2.91)
        single_exc.i = 1
        single_exc.a = 2
        self.assertAlmostEqual(term2_diag_a(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -2.72)
        single_exc.i = 1
        single_exc.a = 3
        self.assertAlmostEqual(term2_diag_a(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -2.68)
        single_exc.i = 0
        single_exc.a = 1
        self.assertAlmostEqual(term2_diag_b(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -1.94)
        single_exc.i = 0
        single_exc.a = 2
        self.assertAlmostEqual(term2_diag_b(single_exc,
                                            self.wf_cc,
                                            self.occ_a),
                               -1.94)
    
    def test_term2_diag_doubles(self):
        cdef DoubleExc double_exc
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 1
        double_exc.b = 3
        self.assertAlmostEqual(term2_diag_aa(double_exc,
                                             self.wf_cc,
                                             self.occ_a),
                               -1.74)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term2_diag_aa(double_exc,
                                             self.wf_cc,
                                             self.occ_a),
                               -1.04)
        double_exc.i = 0
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.04)
        double_exc.i = 0
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.04)
        double_exc.i = 0
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.04)
        double_exc.i = 1
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.01)
        double_exc.i = 1
        double_exc.a = 2
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.01)
        double_exc.i = 1
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 1
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.49)
        double_exc.i = 1
        double_exc.a = 3
        double_exc.j = 0
        double_exc.b = 2
        self.assertAlmostEqual(term2_diag_ab(double_exc,
                                             self.wf_cc,
                                             self.occ_a,
                                             self.occ_b),
                               -1.49)


@tests.category('SHORT', 'ESSENTIAL')
class OccOrbitalTestCase(unittest.TestCase):

    def setUp(self):
        self.orbspace = FullOrbitalSpace(n_irrep=4)
        self.orbspace.set_full(OrbitalSpace(dim=[10, 5, 5, 2], orb_type='R'), update=False)
        self.orbspace.set_ref(OrbitalSpace(dim=[5, 2, 2, 0,
                                                4, 2, 2, 0], orb_type='F'))
##        self.orbspace.corr = [5, 2, 2, 0, 4, 2, 2, 0]
##        self.orbspace.n_orb_before = [0, 10, 15, 20, 22]

    def test_alpha(self):
        cdef OccOrbital i
        i = OccOrbital(self.orbspace, True)
        self.assertEqual(i.pos_in_occ, 0)
        self.assertEqual(i.orb, 0)
        self.assertEqual(i.spirrep, 0)
        self.assertTrue(i.alive)
        i.next_()
        self.assertEqual(i.pos_in_occ, 1)
        self.assertEqual(i.orb, 1)
        self.assertEqual(i.spirrep, 0)
        self.assertTrue(i.alive)
        i.next_()
        i.next_()
        i.next_()
        self.assertEqual(i.pos_in_occ, 4)
        self.assertEqual(i.orb, 4)
        self.assertEqual(i.spirrep, 0)
        self.assertTrue(i.alive)
        i.next_()
        self.assertEqual(i.pos_in_occ, 5)
        self.assertEqual(i.orb, 10)
        self.assertEqual(i.spirrep, 1)
        self.assertTrue(i.alive)
        i.next_()
        i.next_()
        i.next_()
        self.assertEqual(i.pos_in_occ, 8)
        self.assertEqual(i.orb, 16)
        self.assertEqual(i.spirrep, 2)
        self.assertTrue(i.alive)
        i.next_()
        self.assertEqual(i.pos_in_occ, 9)
        self.assertFalse(i.alive)
        i.rewind()
        self.assertEqual(i.pos_in_occ, 0)
        self.assertEqual(i.orb, 0)
        self.assertEqual(i.spirrep, 0)
        self.assertTrue(i.alive)

    def test_beta(self):
        cdef OccOrbital i
        i = OccOrbital(self.orbspace, False)
        self.assertEqual(i.pos_in_occ, 0)
        self.assertEqual(i.orb, 0)
        self.assertEqual(i.spirrep, 4)
        self.assertTrue(i.alive)
        i.next_()
        self.assertEqual(i.pos_in_occ, 1)
        self.assertEqual(i.orb, 1)
        self.assertEqual(i.spirrep, 4)
        self.assertTrue(i.alive)
        i.next_()
        i.next_()
        i.next_()
        self.assertEqual(i.pos_in_occ, 4)
        self.assertEqual(i.orb, 10)
        self.assertEqual(i.spirrep, 5)
        self.assertTrue(i.alive)
        i.next_()
        self.assertEqual(i.pos_in_occ, 5)
        self.assertEqual(i.orb, 11)
        self.assertEqual(i.spirrep, 5)
        self.assertTrue(i.alive)
        i.next_()
        i.next_()
        i.next_()
        self.assertEqual(i.pos_in_occ, 8)
        self.assertFalse(i.alive)
        i.rewind()
        self.assertEqual(i.pos_in_occ, 0)
        self.assertEqual(i.orb, 0)
        self.assertEqual(i.spirrep, 4)
        self.assertTrue(i.alive)
