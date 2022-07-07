"""Tests for occupied orbitals

"""
import unittest

import numpy as np

from util.other import int_array
import tests
from orbitals.orbital_space cimport FullOrbitalSpace, OrbitalSpace
from orbitals.orbital_space import FullOrbitalSpace, OrbitalSpace

@tests.category('SHORT', 'ESSENTIAL')
class OrbitalSpaceTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test1(self):
        orbspace = FullOrbitalSpace(n_irrep=4)
        orbspace.set_full(OrbitalSpace(dim=[5, 3, 3, 1], orb_type='R'), update=False)
        orbspace.set_ref(OrbitalSpace(dim=[2, 0, 0, 0,
                                       2, 0, 0, 0], orb_type='F'), update=False)
        orbspace.set_act(OrbitalSpace(dim=[0, 0, 0, 0], orb_type='A'), update=False)
        orbspace.set_froz(OrbitalSpace(dim=[1, 0, 0, 0], orb_type='R'), update=True)
        self.assertEqual(np.array(orbspace.corr),
                         int_array(1, 0, 0, 0, 1, 0, 0, 0))
        self.assertEqual(np.array(orbspace.virt),
                         int_array(3, 3, 3, 1, 3, 3, 3, 1))
        self.assertEqual(int_array(orbspace.orbs_before),
                         int_array(0, 4, 7, 10, 11, 0, 0, 0, 0))
        self.assertEqual(int_array(orbspace.corr_orbs_before),
                         int_array(0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
