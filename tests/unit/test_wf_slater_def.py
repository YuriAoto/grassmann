"""Tests for fci

"""
import unittest

import numpy as np

import tests
from util.other import int_array
from orbitals.orbital_space import OrbitalSpace
from wave_functions.slater_det import SlaterDet

@tests.category('SHORT', 'ESSENTIAL')
class SlaterDetTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        
    def test_simple_construction_and_namedtuple(self):
        det = SlaterDet(c=0.123,
                        alpha_occ=int_array(0, 1, 2),
                        beta_occ=int_array(1))
        self.assertEqual(det.c, 0.123)
        self.assertEqual(det.alpha_occ[1], 1)
        self.assertEqual(det.beta_occ[0], 1)
        with self.assertRaises(IndexError):
            det.beta_occ[1]

    def test_get_from_FCI_line_1(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        n_core = OrbitalSpace(dim=[0, 0, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core)
        self.assertEqual(det, SlaterDet(c=-0.162676901257,
                                        alpha_occ=[0, 1, 6],
                                        beta_occ=[0, 1, 6]))

    def test_get_from_FCI_line_2(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = OrbitalSpace(dim=[0, 0, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core)
        self.assertEqual(det, SlaterDet(c=-0.049624632911,
                                        alpha_occ=[0, 1, 3],
                                        beta_occ=[0, 1, 5]))

    def test_get_from_FCI_line_3(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = OrbitalSpace(dim=[0, 0, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core, zero_coefficient=True)
        self.assertEqual(det, SlaterDet(c=0.0,
                                        alpha_occ=[0, 1, 3],
                                        beta_occ=[0, 1, 5]))

    def test_get_from_FCI_line_4(self):
        line = '0.000000000000  1  2  9  1  2 10'
        n_core = OrbitalSpace(dim=[0, 0, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core)
        self.assertEqual(det, SlaterDet(c=0.0,
                                        alpha_occ=[0, 1, 8],
                                        beta_occ=[0, 1, 9]))

    def test_get_from_FCI_line_5(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        n_core = OrbitalSpace(dim=[1, 1, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core)
        self.assertEqual(det, SlaterDet(c=-0.162676901257,
                                        alpha_occ=[4],
                                        beta_occ=[4]))

    def test_get_from_FCI_line_6(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = OrbitalSpace(dim=[1, 1, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core)
        self.assertEqual(det, SlaterDet(c=-0.049624632911,
                                        alpha_occ=[1],
                                        beta_occ=[3]))

    def test_get_from_FCI_line_7(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = OrbitalSpace(dim=[1, 1, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core, zero_coefficient=True)
        self.assertEqual(det, SlaterDet(c=0.0,
                                        alpha_occ=[1],
                                        beta_occ=[3]))

    def test_get_from_FCI_line_8(self):
        line = '0.000000000000  1  2  9  1  2 10'
        n_core = OrbitalSpace(dim=[1, 1, 0, 0], orb_type='R')
        Ms = 0.0
        det = SlaterDet.from_molpro_line(line, Ms, n_core)
        self.assertEqual(det, SlaterDet(c=0.0,
                                        alpha_occ=[6],
                                        beta_occ=[7]))
