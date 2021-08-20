"""Tests for occupied orbitals

"""
import unittest

import numpy as np

from util.other import int_array
import tests
from orbitals.occ_orbitals cimport OrbitalSpace
from orbitals.occ_orbitals import OrbitalSpace

@tests.category('SHORT', 'ESSENTIAL')
class OrbitalSpaceTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test1(self):
        orbspace = OrbitalSpace()
        orbspace.initialize(4)
        orbspace.dim[0] = 5
        orbspace.dim[1] = 3
        orbspace.dim[2] = 3
        orbspace.dim[3] = 1
        orbspace.ref[0] = 2
        orbspace.ref[1] = 0
        orbspace.ref[2] = 0
        orbspace.ref[3] = 0
        orbspace.ref[4] = 2
        orbspace.ref[5] = 0
        orbspace.ref[6] = 0
        orbspace.ref[7] = 0
        orbspace.act[0] = 0
        orbspace.act[1] = 0
        orbspace.act[2] = 0
        orbspace.act[3] = 0
        orbspace.froz[0] = 1
        orbspace.froz[1] = 0
        orbspace.froz[2] = 0
        orbspace.froz[3] = 0
        self.assertEqual(orbspace.corr().as_array(),
                         int_array(1, 0, 0, 0, 1, 0, 0, 0))
        self.assertEqual(orbspace.virt().as_array(),
                         int_array(3, 3, 3, 1, 3, 3, 3, 1))
        orbspace.calc_orbs_before()
        self.assertEqual(np.array(orbspace.orbs_before),
                         int_array(0, 4, 7, 10, 11, 0, 0, 0, 0))
        self.assertEqual(np.array(orbspace.corr_orbs_before),
                         int_array(0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
