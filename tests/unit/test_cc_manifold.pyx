"""Tests for CC manifold

"""
import unittest

import numpy as np

import tests
from coupled_cluster.manifold cimport (
    EXC_TYPE_A, EXC_TYPE_B, EXC_TYPE_AA, EXC_TYPE_AB, EXC_TYPE_BB,
    _term1, _term2_diag, _exc_on_string)
from wave_functions.fci import make_occ
from util.variables import int_dtype


class ExcOnStringTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test1(self):
        self.assertEqual(make_occ([1, 2, 3, 4, 5, 6, -1]),
                         make_occ(_exc_on_string(
                             0, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 2, 3, 4, 5, 6, 1]),
                         make_occ(_exc_on_string(
                             1, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 3, 4, 5, 6, -1]),
                         make_occ(_exc_on_string(
                             2, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 5, 6, 1]),
                         make_occ(_exc_on_string(
                             3, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 5, 6, -1]),
                         make_occ(_exc_on_string(
                             4, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 4, 6, 1]),
                         make_occ(_exc_on_string(
                             5, 6, make_occ([0, 1, 2, 3, 4, 5]))))

    def test2(self):
        self.assertEqual(make_occ([1, 2, 3, 4, 5, 8, -1]),
                         make_occ(_exc_on_string(
                             0, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 2, 3, 4, 5, 8, 1]),
                         make_occ(_exc_on_string(
                             1, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 3, 4, 5, 8, -1]),
                         make_occ(_exc_on_string(
                             2, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 5, 8, 1]),
                         make_occ(_exc_on_string(
                             3, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 5, 8, -1]),
                         make_occ(_exc_on_string(
                             4, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 4, 8, 1]),
                         make_occ(_exc_on_string(
                             5, 8, make_occ([0, 1, 2, 3, 4, 5]))))

    def test3(self):
        self.assertEqual(make_occ([1, 2, 4, 6, 7, 8, 1]),
                         make_occ(_exc_on_string(
                             0, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 2, 4, 6, 7, 8, -1]),
                         make_occ(_exc_on_string(
                             1, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 4, 6, 7, 8, 1]),
                         make_occ(_exc_on_string(
                             2, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 7, 8, 1]),
                         make_occ(_exc_on_string(
                             6, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 6, 8, -1]),
                         make_occ(_exc_on_string(
                             7, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 6, 7, 1]),
                         make_occ(_exc_on_string(
                             8, 4, make_occ([0, 1, 2, 6, 7, 8]))))

    def test4(self):
        self.assertEqual(make_occ([1, 2, 6, 7, 8, 9, -1]),
                         make_occ(_exc_on_string(
                             0, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 2, 6, 7, 8, 9, 1]),
                         make_occ(_exc_on_string(
                             1, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 6, 7, 8, 9, -1]),
                         make_occ(_exc_on_string(
                             2, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 7, 8, 9, 1]),
                         make_occ(_exc_on_string(
                             6, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 6, 8, 9, -1]),
                         make_occ(_exc_on_string(
                             7, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 6, 7, 9, 1]),
                         make_occ(_exc_on_string(
                             8, 9, make_occ([0, 1, 2, 6, 7, 8]))))

    def test5(self):
        self.assertEqual(make_occ([0, 4, 5, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             3, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 5, 10, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             4, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             5, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 5, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             10, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 5, 10, 12, 1]),
                         make_occ(_exc_on_string(
                             11, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 5, 10, 11, -1]),
                         make_occ(_exc_on_string(
                             12, 0, make_occ([3, 4, 5, 10, 11, 12]))))

    def test6(self):
        self.assertEqual(make_occ([2, 4, 5, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             3, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 5, 10, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             4, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             5, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 5, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             10, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 5, 10, 12, 1]),
                         make_occ(_exc_on_string(
                             11, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 5, 10, 11, -1]),
                         make_occ(_exc_on_string(
                             12, 2, make_occ([3, 4, 5, 10, 11, 12]))))

    def test7(self):
        self.assertEqual(make_occ([4, 5, 6, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             3, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 5, 6, 10, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             4, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 6, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             5, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 6, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             10, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 6, 10, 12, -1]),
                         make_occ(_exc_on_string(
                             11, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 6, 10, 11, 1]),
                         make_occ(_exc_on_string(
                             12, 6, make_occ([3, 4, 5, 10, 11, 12]))))

    def test8(self):
        self.assertEqual(make_occ([4, 5, 10, 11, 12, 15, -1]),
                         make_occ(_exc_on_string(
                             3, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 5, 10, 11, 12, 15, 1]),
                         make_occ(_exc_on_string(
                             4, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 10, 11, 12, 15, -1]),
                         make_occ(_exc_on_string(
                             5, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 11, 12, 15, 1]),
                         make_occ(_exc_on_string(
                             10, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 10, 12, 15, -1]),
                         make_occ(_exc_on_string(
                             11, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 10, 11, 15, 1]),
                         make_occ(_exc_on_string(
                             12, 15, make_occ([3, 4, 5, 10, 11, 12]))))


class Terms2el6orbTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.nel = 1
        self.norb = 3
        self.wf = np.array([[1.0,  0.7, -0.5],
                            [0.2, -0.4,  0.8],
                            [0.7,  0.6, -0.3]])
        self.wf_cc = np.array([[ 1.0, -0.2, 0.5],
                               [ 0.7,  0.3, 0.9],
                               [-0.2,  0.1, 0.1]])
        self.str_gr = np.array([[0],
                                [1],
                                [2]], dtype=int_dtype)

    def test_term1_singles(self):
        self.assertAlmostEqual(_term1(make_occ([0, 1]),
                                      EXC_TYPE_A,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               -0.41)
        self.assertAlmostEqual(_term1(make_occ([0, 2]),
                                      EXC_TYPE_A,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               0.6)
        self.assertAlmostEqual(_term1(make_occ([0, 1]),
                                      EXC_TYPE_B,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               0.31)
        self.assertAlmostEqual(_term1(make_occ([0, 2]),
                                      EXC_TYPE_B,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               -0.99)
    
    def test_term1_doubles(self):
        self.assertAlmostEqual(_term1(make_occ([0, 1, 0, 1]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               -0.7)
        self.assertAlmostEqual(_term1(make_occ([0, 1, 0, 2]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               -0.1)
        self.assertAlmostEqual(_term1(make_occ([0, 2, 0, 1]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               0.5)
        self.assertAlmostEqual(_term1(make_occ([0, 2, 0, 2]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.str_gr,
                                      self.str_gr),
                               -0.4)
    
    def test_term2_diag_singles(self):
        self.assertAlmostEqual(_term2_diag(make_occ([0, 1]),
                                           EXC_TYPE_A,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.29)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2]),
                                           EXC_TYPE_A,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.29)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 1]),
                                           EXC_TYPE_B,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.53)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2]),
                                           EXC_TYPE_B,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.53)
    
    def test_term2_diag_doubles(self):
        self.assertAlmostEqual(_term2_diag(make_occ([0, 1, 0, 1]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.0)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 1, 0, 2]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.0)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2, 0, 1]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.0)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2, 0, 2]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.nel,
                                           self.nel),
                               1.0)


class Terms3el7orbTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.beta_nel = 1
        self.beta_norb = 3
        self.alpha_nel = 2
        self.alpha_norb = 4
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
        self.alpha_str_gr = np.array([[0, 0],
                                      [1, 1],
                                      [2, 3]], dtype=int_dtype)
        self.beta_str_gr = np.array([[0],
                                     [1],
                                     [2]], dtype=int_dtype)
        
    def test_term1_singles(self):
        self.assertAlmostEqual(_term1(make_occ([0, 2]),
                                      EXC_TYPE_A,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -0.44)
        self.assertAlmostEqual(_term1(make_occ([0, 3]),
                                      EXC_TYPE_A,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -1.3)
        self.assertAlmostEqual(_term1(make_occ([1, 2]),
                                      EXC_TYPE_A,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               0.11)
        self.assertAlmostEqual(_term1(make_occ([1, 3]),
                                      EXC_TYPE_A,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -0.79)
        self.assertAlmostEqual(_term1(make_occ([0, 1]),
                                      EXC_TYPE_B,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               0.7)
        self.assertAlmostEqual(_term1(make_occ([0, 2]),
                                      EXC_TYPE_B,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -1.19)
    
    def test_term1_doubles(self):
        self.assertAlmostEqual(_term1(make_occ([0, 2, 1, 3]),
                                      EXC_TYPE_AA,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -0.79)
        self.assertAlmostEqual(_term1(make_occ([0, 2, 0, 1]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               0.06)
        self.assertAlmostEqual(_term1(make_occ([0, 2, 0, 2]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               1.44)
        self.assertAlmostEqual(_term1(make_occ([0, 3, 0, 1]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -0.14)
        self.assertAlmostEqual(_term1(make_occ([0, 3, 0, 2]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               0.34)
        self.assertAlmostEqual(_term1(make_occ([1, 2, 0, 1]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -0.62)
        self.assertAlmostEqual(_term1(make_occ([1, 2, 0, 2]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               0.07)
        self.assertAlmostEqual(_term1(make_occ([1, 3, 0, 1]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               -0.04)
        self.assertAlmostEqual(_term1(make_occ([1, 3, 0, 2]),
                                      EXC_TYPE_AB,
                                      self.wf, self.wf_cc,
                                      self.alpha_str_gr,
                                      self.beta_str_gr),
                               0.69)
        
    def test_term2_diag_singles(self):
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2]),
                                           EXC_TYPE_A,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               2.43)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 3]),
                                           EXC_TYPE_A,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               2.91)
        self.assertAlmostEqual(_term2_diag(make_occ([1, 2]),
                                           EXC_TYPE_A,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               2.72)
        self.assertAlmostEqual(_term2_diag(make_occ([1, 3]),
                                           EXC_TYPE_A,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               2.68)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 1]),
                                           EXC_TYPE_B,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.94)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2]),
                                           EXC_TYPE_B,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.94)
    
    def test_term2_diag_doubles(self):
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2, 1, 3]),
                                           EXC_TYPE_AA,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.74)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2, 0, 1]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.04)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 2, 0, 2]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.04)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 3, 0, 1]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.04)
        self.assertAlmostEqual(_term2_diag(make_occ([0, 3, 0, 2]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.04)
        self.assertAlmostEqual(_term2_diag(make_occ([1, 2, 0, 1]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.01)
        self.assertAlmostEqual(_term2_diag(make_occ([1, 2, 0, 2]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.01)
        self.assertAlmostEqual(_term2_diag(make_occ([1, 3, 0, 1]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.49)
        self.assertAlmostEqual(_term2_diag(make_occ([1, 3, 0, 2]),
                                           EXC_TYPE_AB,
                                           self.wf_cc,
                                           self.alpha_nel,
                                           self.beta_nel),
                               1.49)
