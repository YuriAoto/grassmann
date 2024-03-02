
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
    st = stiefel.StiefelManifold(C.shape, S=S)
    return C, st


mymolecule = 'H2O'
mygeometry = '1.5'
mybasis = 'cc-pVTZ'



@tests.category('SHORT')
class TgSpaceTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_prj_tg_space(self):
        C, st = get_stiefel_element_from_molecule(molecule=mymolecule,
                                                  geometry=mygeometry,
                                                  basis=mybasis)
        np.random.seed(tests.init_random_state)
        for i in range(100):
            Z = np.random.rand(*st.shape)
            projZ = st.proj_tgspace(C, Z)
            self.assertTrue(st.check_in_tgspace(C, projZ))


    def test_gen_tg_space(self):
        C, st = get_stiefel_element_from_molecule(molecule=mymolecule,
                                                  geometry=mygeometry,
                                                  basis=mybasis)

        for name_v, v in st.tg_vecs(C,
                                    mode='2.5',
                                    in_Gr_tgspace=True,
                                    in_Gr_tgspace_orth=True):
            self.assertTrue(st.check_in_tgspace(C, v))

        for name_v, v in st.tg_vecs(C,
                                    mode='2.6',
                                    in_Gr_tgspace=True,
                                    in_Gr_tgspace_orth=True):
            self.assertTrue(st.check_in_tgspace(C, v))


    def test_gen_ordered_tg_space(self):
        C, st = get_stiefel_element_from_molecule(molecule=mymolecule,
                                                  geometry=mygeometry,
                                                  basis=mybasis)

        i = 0
        for name_v, v in st.tg_vecs(C,
                                    mode='2.5',
                                    in_Gr_tgspace=True,
                                    in_Gr_tgspace_orth=True):
            self.assertEqual(v, st.ith_tg_vecs(C, i, mode='2.5'))
            i += 1

        i = 0
        for name_v, v in st.tg_vecs(C,
                                    mode='2.6',
                                    in_Gr_tgspace=True,
                                    in_Gr_tgspace_orth=True):
            self.assertEqual(v, st.ith_tg_vecs(C, i, mode='2.6'))
            i += 1


    def test_orthonormal_basis(self):
        C, st = get_stiefel_element_from_molecule(molecule=mymolecule,
                                                  geometry=mygeometry,
                                                  basis=mybasis)
        basis = st.basis_for_tg_space(C)
        self.assertEqual(len(basis), st.dim)
        
        gram = np.empty((st.dim, st.dim))
        for i in range(st.dim):
            for j in range(st.dim):
                gram[i,j] = st.rip(basis[i], basis[j])
        
        self.assertEqual(np.linalg.matrix_rank(gram), st.dim)
        self.assertEqual(gram, np.eye(st.dim))
