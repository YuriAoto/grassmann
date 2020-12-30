"""Tests for fci

"""
import unittest

import numpy as np

from wave_functions import fci, general
from wave_functions.fci import make_occ
import test


class SlaterDetTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        
    def test_simple_construction_and_namedtuple(self):
        det = fci.SlaterDet(c=0.123,
                            alpha_occ=np.array([0, 1, 2], dtype=np.int8),
                            beta_occ=np.array([1], dtype=np.int8))
        self.assertEqual(det.c, 0.123)
        self.assertEqual(det[0], 0.123)
        self.assertEqual(det.alpha_occ[1], 1)
        self.assertEqual(det[1][1], 1)
        self.assertEqual(det.beta_occ[0], 1)
        self.assertEqual(det[2][0], 1)
        with self.assertRaises(IndexError):
            det.beta_occ[1]

    def test_get_from_FCI_line_1(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        n_core = general.Orbitals_Sets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.162676901257, places=10)
        self.assertEqual(det.alpha_occ, make_occ([0, 1, 6]))
        self.assertEqual(det.beta_occ, make_occ([0, 1, 6]))

    def test_get_from_FCI_line_2(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.Orbitals_Sets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.049624632911, places=10)
        self.assertEqual(det.alpha_occ, make_occ([0, 1, 3]))
        self.assertEqual(det.beta_occ, make_occ([0, 1, 5]))

    def test_get_from_FCI_line_3(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.Orbitals_Sets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core, zero_coefficient=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)

    def test_get_from_FCI_line_4(self):
        line = '0.000000000000  1  2  9  1  2 10'
        n_core = general.Orbitals_Sets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, 0.0, places=10)
        self.assertEqual(det.alpha_occ, make_occ([0, 1, 8]))
        self.assertEqual(det.beta_occ, make_occ([0, 1, 9]))

    def test_get_from_FCI_line_5(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        n_core = general.Orbitals_Sets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.162676901257, places=10)
        self.assertEqual(det.alpha_occ, make_occ([4]))
        self.assertEqual(det.beta_occ, make_occ([4]))

    def test_get_from_FCI_line_6(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.Orbitals_Sets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.049624632911, places=10)
        self.assertEqual(det.alpha_occ, make_occ([1]))
        self.assertEqual(det.beta_occ, make_occ([3]))

    def test_get_from_FCI_line_7(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.Orbitals_Sets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core, zero_coefficient=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)

    def test_get_from_FCI_line_8(self):
        line = '0.000000000000  1  2  9  1  2 10'
        n_core = general.Orbitals_Sets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, 0.0, places=10)
        self.assertEqual(det.alpha_occ, make_occ([6]))
        self.assertEqual(det.beta_occ, make_occ([7]))

