
"""Tests for input parser

"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

import tests
from hartree_fock import stiefel
from hartree_fock.starting_orbitals import initial_orbitals



def get_stiefel_element_from_molecule(molecule, geometry, basis):
    molsys = tests.get_test_system(molecule=molecule,
                                   geometry=geometry,
                                   basis=basis)
    S = molsys.integrals.S
    N = molsys.n_elec // 2
    C = initial_orbitals('Hcore', molsys, restricted=True,
                         conjugacy=None, step_size=None)[0][:, :N]
    return C, S




@tests.category('SHORT')
class TgSpaceTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_prj_tg_space(self):
        C, S = get_stiefel_element_from_molecule(molecule='HCl',
                                                 geometry='1.5',
                                                 basis='cc-pVDZ')
        np.random.seed(tests.init_random_state)
        for i in range(100):
            Z = np.random.rand(*C.shape)
            projZ = stiefel.proj_tgspace(C, Z, S=S)
            self.assertTrue(stiefel.check_in_tgspace(C, projZ, S=S))


    def test_gen_tg_space(self):
        C, S = get_stiefel_element_from_molecule(molecule='HCl',
                                                 geometry='1.5',
                                                 basis='cc-pVDZ')

        for name_v, v in stiefel.tg_vecs(C,
                                         mode='2.5',
                                         S=S,
                                         in_Gr_tgspace=True,
                                         in_Gr_tgspace_orth=True):
            self.assertTrue(stiefel.check_in_tgspace(C, v, S=S))

        for name_v, v in stiefel.tg_vecs(C,
                                         mode='2.6',
                                         S=S,
                                         in_Gr_tgspace=True,
                                         in_Gr_tgspace_orth=True):
            self.assertTrue(stiefel.check_in_tgspace(C, v, S=S))


    def test_gen_ordered_tg_space(self):
        C, S = get_stiefel_element_from_molecule(molecule='HCl',
                                                 geometry='1.5',
                                                 basis='cc-pVDZ')

        i = 0
        for name_v, v in stiefel.tg_vecs(C,
                                         mode='2.5',
                                         S=S,
                                         in_Gr_tgspace=True,
                                         in_Gr_tgspace_orth=True):
            self.assertEqual(v, stiefel.ith_tg_vecs(C, i, mode='2.5', S=S))
            i += 1

        i = 0
        for name_v, v in stiefel.tg_vecs(C,
                                         mode='2.6',
                                         S=S,
                                         in_Gr_tgspace=True,
                                         in_Gr_tgspace_orth=True):
            self.assertEqual(v, stiefel.ith_tg_vecs(C, i, mode='2.6', S=S))
            i += 1

