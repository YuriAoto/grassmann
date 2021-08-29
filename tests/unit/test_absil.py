"""Tests for absil

"""
import unittest

import numpy as np
from scipy import linalg

import tests
from wave_functions import cisd, interm_norm
from orbitals import orbitals
from dist_grassmann import absil
from util.other import int_array


np.set_printoptions(formatter={'float': lambda x: '{0:>9.6f}'.format(x)})


@tests.category('SHORT')
class YieldExcitations523TestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.n_el = 5
        self.n_corr = 2
        self.n_virt = 3

    def test_all_singles(self):
        for i_Ind, ia_Ind in enumerate(
                absil._all_singles(self.n_el, self.n_corr, self.n_virt)):
            i, a, Ind = ia_Ind
            if i_Ind == 0:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 0)
                    self.assertEqual(a, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 5))
            elif i_Ind == 1:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 0)
                    self.assertEqual(a, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 6))
            elif i_Ind == 2:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 7))
            elif i_Ind == 3:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(a, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 5))
            elif i_Ind == 4:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(a, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 6))
            elif i_Ind == 5:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(a, 2)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 7))
            else:
                self.fail('Too many indices')

    def test_all_doubles(self):
        for i_Ind, ijab_Ind in enumerate(
                absil._all_doubles(self.n_el, self.n_corr, self.n_virt)):
            i, j, a, b, Ind = ijab_Ind
            if i_Ind == 0:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 1)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 5, 6))
            elif i_Ind == 1:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 5, 7))
            elif i_Ind == 2:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 6, 7))
            else:
                self.fail('Too many indices')


@tests.category('SHORT')
class YieldExcitations633TestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.n_el = 6
        self.n_corr = 3
        self.n_virt = 3

    def test_all_singles(self):
        for i_Ind, ia_Ind in enumerate(
                absil._all_singles(self.n_el, self.n_corr, self.n_virt)):
            i, a, Ind = ia_Ind
            if i_Ind == 0:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 0)
                    self.assertEqual(a, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 5, 6))
            elif i_Ind == 1:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 0)
                    self.assertEqual(a, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 5, 7))
            elif i_Ind == 2:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 5, 8))
            elif i_Ind == 3:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(a, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 5, 6))
            elif i_Ind == 4:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(a, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 5, 7))
            elif i_Ind == 5:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(a, 2)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 5, 8))
            elif i_Ind == 6:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(a, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 4, 6))
            elif i_Ind == 7:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(a, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 4, 7))
            elif i_Ind == 8:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(a, 2)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 4, 8))
            else:
                self.fail('Too many indices')

    def test_all_doubles(self):
        for i_Ind, ijab_Ind in enumerate(
                absil._all_doubles(self.n_el, self.n_corr, self.n_virt)):
            i, j, a, b, Ind = ijab_Ind
            if i_Ind == 0:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 1)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 5, 6, 7))
            elif i_Ind == 1:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 5, 6, 8))
            elif i_Ind == 2:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 1)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 5, 7, 8))
            elif i_Ind == 3:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 1)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 6, 7))
            elif i_Ind == 4:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 6, 8))
            elif i_Ind == 5:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(j, 0)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 4, 7, 8))
            elif i_Ind == 6:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(j, 1)
                    self.assertEqual(a, 1)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 6, 7))
            elif i_Ind == 7:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(j, 1)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 0)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 6, 8))
            elif i_Ind == 8:
                with self.subTest(i_Ind=i_Ind):
                    self.assertEqual(i, 2)
                    self.assertEqual(j, 1)
                    self.assertEqual(a, 2)
                    self.assertEqual(b, 1)
                    self.assertEqual(Ind, int_array(0, 1, 2, 3, 7, 8))
            else:
                self.fail('Too many indices')


@tests.category('SHORT', 'ESSENTIAL')
class CalcOverlap(unittest.TestCase):

    def setUp(self):
        prng = np.random.RandomState(tests.init_random_state)
        # H2:
        self.intN_wf_H2 = interm_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file('H2__5__sto3g__D2h'))
        self.wf_H2 = cisd.CISDWaveFunction.from_interm_norm(self.intN_wf_H2)
        self.Uid_H2 = orbitals.construct_Id_orbitals(self.wf_H2.ref_orb,
                                                     self.wf_H2.orb_dim,
                                                     self.wf_H2.n_irrep)
        # Li2:
        self.intN_wf_Li2 = interm_norm.IntermNormWaveFunction.from_Molpro(
            tests.CISD_file('Li2__5__631g__C2v'))
        self.wf_Li2 = cisd.CISDWaveFunction.from_interm_norm(self.intN_wf_Li2)
        self.U_Li2 = tests.construct_random_orbitals(
            self.wf_Li2.ref_orb,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep,
            prng)
        self.U_Li2_non_orth = tests.construct_random_orbitals(
            self.wf_Li2.ref_orb,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep,
            prng,
            orthogonalise=False)
        self.Uid_Li2 = orbitals.construct_Id_orbitals(
            self.wf_Li2.ref_orb,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep)

    def test_ref_overlap(self):
        self.assertAlmostEqual(absil.overlap_to_det(self.wf_H2,
                                                    self.Uid_H2),
                               self.wf_H2.C0)
        self.assertAlmostEqual(absil.overlap_to_det(self.wf_Li2,
                                                    self.Uid_Li2),
                               self.wf_Li2.C0)
        orbitals.extend_to_unrestricted(self.Uid_H2)
        orbitals.extend_to_unrestricted(self.Uid_Li2)
        self.assertAlmostEqual(absil.overlap_to_det(self.intN_wf_H2,
                                                    self.Uid_H2),
                               self.intN_wf_H2.C0)
        self.assertAlmostEqual(absil.overlap_to_det(self.intN_wf_Li2,
                                                    self.Uid_Li2),
                               self.intN_wf_Li2.C0)

    def test_compare_overlap_and_gen_sys(self):
        f, X, C = absil.generate_lin_system(self.wf_Li2,
                                            self.U_Li2,
                                            absil.make_slice_XC(self.U_Li2))
        self.assertAlmostEqual(absil.overlap_to_det(self.wf_Li2,
                                                    self.U_Li2),
                               f)
        orbitals.extend_to_unrestricted(self.U_Li2)
        f, X, C = absil.generate_lin_system(self.intN_wf_Li2,
                                            self.U_Li2,
                                            absil.make_slice_XC(self.U_Li2))
        self.assertAlmostEqual(absil.overlap_to_det(self.intN_wf_Li2,
                                                    self.U_Li2),
                               f)

    def test_non_orth(self):
        f_non_orth = absil.overlap_to_det(self.wf_Li2,
                                          self.U_Li2_non_orth)
        tests.logger.info('U before orth:\n%s',
                          self.U_Li2_non_orth)
        for irrep, Ui in enumerate(self.U_Li2_non_orth):
            if Ui.shape[0] * Ui.shape[1] != 0:
                self.U_Li2_non_orth[irrep] = linalg.orth(Ui)
        tests.logger.info('U after orth:\n%s',
                          self.U_Li2_non_orth)
        f_orth = absil.overlap_to_det(self.wf_Li2,
                                      self.U_Li2_non_orth)
        f_non_orth = absil.overlap_to_det(self.wf_Li2,
                                          self.U_Li2_non_orth,
                                          assume_orth=False)
        self.assertAlmostEqual(f_orth, f_non_orth)
        orbitals.extend_to_unrestricted(self.U_Li2_non_orth)
        f_non_orth = absil.overlap_to_det(self.intN_wf_Li2,
                                          self.U_Li2_non_orth)
        tests.logger.info('U before orth:\n%s',
                         self.U_Li2_non_orth)
        for irrep, Ui in enumerate(self.U_Li2_non_orth):
            if Ui.shape[0] * Ui.shape[1] != 0:
                self.U_Li2_non_orth[irrep] = linalg.orth(Ui)
        tests.logger.info('U after orth:\n%s',
                          self.U_Li2_non_orth)
        f_orth = absil.overlap_to_det(self.intN_wf_Li2,
                                      self.U_Li2_non_orth)
        f_non_orth = absil.overlap_to_det(self.intN_wf_Li2,
                                          self.U_Li2_non_orth,
                                          assume_orth=False)
        self.assertAlmostEqual(f_orth, f_non_orth)


@tests.category('SHORT', 'ESSENTIAL')
class CalcXCmatrices(unittest.TestCase):

    def setUp(self):
        prng = np.random.RandomState(tests.init_random_state)
        # H2:
        self.intN_wf_H2 = interm_norm.IntermNormWaveFunction.from_Molpro(
            tests.CISD_file('H2__5__sto3g__D2h'))
        self.wf_H2 = cisd.CISDWaveFunction.from_interm_norm(self.intN_wf_H2)
        self.Uid_H2 = orbitals.construct_Id_orbitals(
            self.wf_H2.ref_orb,
            self.wf_H2.orb_dim,
            self.wf_H2.n_irrep)
        # Li2:
        self.intN_wf_Li2 = interm_norm.IntermNormWaveFunction.from_Molpro(
            tests.CISD_file('Li2__5__631g__C2v'))
        self.wf_Li2 = cisd.CISDWaveFunction.from_interm_norm(self.intN_wf_Li2)
        self.U_Li2 = tests.construct_random_orbitals(
            self.wf_Li2.ref_orb,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep,
            prng)
        self.Uid_Li2 = orbitals.construct_Id_orbitals(
            self.wf_Li2.ref_orb,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep)

    def test_check_Absil_eq(self):
        slice_XC = absil.make_slice_XC(self.Uid_H2)
        f, X, C = absil.generate_lin_system(self.wf_H2,
                                            self.Uid_H2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_H2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape))
        self.assertTrue(absil.check_Newton_eq(self.wf_H2,
                                              self.Uid_H2,
                                              eta,
                                              True))
        slice_XC = absil.make_slice_XC(self.Uid_Li2)
        f, X, C = absil.generate_lin_system(self.wf_Li2,
                                            self.Uid_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape))
        self.assertTrue(absil.check_Newton_eq(self.wf_Li2,
                                              self.Uid_Li2,
                                              eta,
                                              True,
                                              eps=0.001))
        slice_XC = absil.make_slice_XC(self.U_Li2)
        f, X, C = absil.generate_lin_system(self.wf_Li2,
                                            self.U_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.U_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape))
        tests.logger.info('2C:\n%r', 2*C)
        tests.logger.info('2X @ eta:\n%r', np.matmul(2*X, lin_sys_solution[0]))
        self.assertTrue(absil.check_Newton_eq(self.wf_Li2,
                                              self.U_Li2,
                                              eta,
                                              True,
                                              eps=0.0001))

    def test_check_Absil_eq_gen_WF(self):
        orbitals.extend_to_unrestricted(self.Uid_H2)
        slice_XC = absil.make_slice_XC(self.Uid_H2)
        f, X, C = absil.generate_lin_system(self.intN_wf_H2,
                                            self.Uid_H2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_H2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape,
                                  order='F'))
        self.assertTrue(absil.check_Newton_eq(self.intN_wf_H2,
                                              self.Uid_H2,
                                              eta,
                                              False))
        orbitals.extend_to_unrestricted(self.Uid_Li2)
        slice_XC = absil.make_slice_XC(self.Uid_Li2)
        f, X, C = absil.generate_lin_system(self.intN_wf_Li2,
                                            self.Uid_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape,
                                  order='F'))
        self.assertTrue(absil.check_Newton_eq(self.intN_wf_Li2,
                                              self.Uid_Li2,
                                              eta,
                                              False,
                                              eps=0.001))
        orbitals.extend_to_unrestricted(self.U_Li2)
        slice_XC = absil.make_slice_XC(self.U_Li2)
        f, X, C = absil.generate_lin_system(self.intN_wf_Li2,
                                            self.U_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.U_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape,
                                  order='F'))
        tests.logger.info('C:\n%r', C)
        tests.logger.info('X @ eta:\n%r', np.matmul(X, lin_sys_solution[0]))
        self.assertTrue(absil.check_Newton_eq(self.intN_wf_Li2,
                                              self.U_Li2,
                                              eta,
                                              False,
                                              eps=0.0001))
