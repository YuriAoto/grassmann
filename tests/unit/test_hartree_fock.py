"""Tests for input parser

"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

import tests
from hartree_fock.main import _define_hfstep_func
from hartree_fock.iteration_step import _nrlm_update_triang_eps, _nrlm_J_hess


@tests.category('SHORT')
class InternalsTestCase(unittest.TestCase):

    def test_SCF(self):
        f = _define_hfstep_func('SCF')
        self.assertEqual(f(0, 0.0), 'RH-SCF')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'RH-SCF')

    def test_Absil(self):
        f = _define_hfstep_func('Absil')
        self.assertEqual(f(0, 0.0), 'Absil')
        self.assertEqual(f(3, 4.0), 'Absil')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_SCF_Absil_n5(self):
        f = _define_hfstep_func('SCF-Absil_n5')
        self.assertEqual(f(0, 0.0), 'RH-SCF')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_SCF_Absil_n2(self):
        f = _define_hfstep_func('SCF-Absil_n2')
        self.assertEqual(f(0, 0.0), 'RH-SCF')
        self.assertEqual(f(3, 4.0), 'Absil')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_SCF_Absil_g1_0(self):
        f = _define_hfstep_func('SCF-Absil_grad1.0')
        self.assertEqual(f(0, 0.0), 'Absil')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'RH-SCF')

    def test_SCF_Absil_g3_0(self):
        f = _define_hfstep_func('SCF-Absil_grad3.0')
        self.assertEqual(f(0, 0.0), 'Absil')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_raise(self):
        with self.assertRaises(ValueError):
            f = _define_hfstep_func('SCF-Absil_3.0')
     

@tests.category('SHORT')
class NRLMTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_update_eta(self):
        A = np.array([[1, 2, 4, 7],
                      [2, 3, 5, 8],
                      [4, 5, 6, 9],
                      [7, 8, 9,10.0]])
        eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        A_eta = np.array([[1.1, 2.2, 4.4, 7.7],
                          [2.2, 3.3, 5.5, 8.8],
                          [4.4, 5.5, 6.6, 9.9],
                          [7.7, 8.8, 9.9, 11.0]])
        _nrlm_update_triang_eps(A, eta)
        self.assertEqual(A_eta, A)

    def test_Jhess(self):
        SC = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype=float)
        hessJ = np.zeros((6, 6))
        #                  ij     00   01   11   02  12  22       p  k
        hessJcheck = -np.array([[  2,   2,   0,   3,  0,  0],   # 0  0
                                [  8,   5,   0,   6,  0,  0],   # 1  0
                                [  0,   1,   4,   0,  3,  0],   # 0  1
                                [  0,   4,  10,   0,  6,  0],   # 1  1
                                [  0,   0,   0,   1,  2,  6],   # 0  2
                                [  0,   0,   0,   4,  5, 12]], dtype=float)  # 1  2
        _nrlm_J_hess(SC, hessJ)
        self.assertEqual(hessJ, hessJcheck)
