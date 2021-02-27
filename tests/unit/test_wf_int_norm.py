"""Tests for wave_functions.int_norm

"""
import unittest

import numpy as np

import tests
from util.other import int_array
from wave_functions import int_norm
from orbitals.symmetry import OrbitalsSets
from string_indices.string_indices import SD_StringIndex


@tests.category('SHORT')
class ConstructorTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_from_zero_1(self):
        point_group = 'C1'
        orb_dim = OrbitalsSets([3],
                               occ_type='R')
        ref_occ = OrbitalsSets([1],
                               occ_type='R')
        core_orb = OrbitalsSets([0],
                                occ_type='R')
        wf = int_norm.IntermNormWaveFunction.from_zero_amplitudes(
            point_group, ref_occ, orb_dim, core_orb, level='SD')
        self.assertEqual(wf.n_elec, 2)
        self.assertEqual(wf.n_alpha, 1)
        self.assertEqual(wf.n_beta, 1)
        self.assertEqual(len(wf.singles), 1)
        self.assertEqual(len(wf.doubles), 1)
        self.assertEqual(wf.singles[0], np.array([[0.0, 0.0]]))
        self.assertEqual(wf.doubles[0][0], np.array([[0.0, 0.0],
                                                     [0.0, 0.0]]))


    def test_from_zero_2(self):
        point_group = 'C1'
        orb_dim = OrbitalsSets([5],
                               occ_type='R')
        ref_occ = OrbitalsSets([2],
                               occ_type='R')
        core_orb = OrbitalsSets([0],
                                occ_type='R')
        wf = int_norm.IntermNormWaveFunction.from_zero_amplitudes(
            point_group, ref_occ, orb_dim, core_orb, level='SD')
        self.assertEqual(wf.n_elec, 4)
        self.assertEqual(wf.n_alpha, 2)
        self.assertEqual(wf.n_beta, 2)
        self.assertEqual(len(wf.singles), 1)
        self.assertEqual(len(wf.doubles), 3)
        self.assertEqual(wf.singles[0], np.array([[0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0]]))
        self.assertEqual(wf.doubles[0][0], np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0]]))
        self.assertEqual(wf.doubles[1][0], np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0]]))
        self.assertEqual(wf.doubles[2][0], np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0]]))


@tests.category('SHORT')
class He2StringIndicesTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.He2_wf = int_norm.IntermNormWaveFunction.from_Molpro(
            tests.CISD_file('He2__1.5__631g__D2h'))
        self.He2_CCwf = int_norm.IntermNormWaveFunction.from_Molpro(
            tests.CCSD_file('He2__1.5__631g__D2h'))

    def test_string_indices_singles_cisd_He2(self):
        irrep = 0
        for i, Index in enumerate(self.He2_wf._string_indices_sing_exc(irrep)):
            if i == 0 or i == 1:
                with self.subTest('He2, C of singles',
                                  i=i, irrep=irrep):
                    self.assertAlmostEqual(Index.C,
                                           0.00339734 / self.He2_wf.norm)
                if i == 0:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[0].occ_orb,
                                         int_array(1))
                        self.assertEqual(Index[8].occ_orb,
                                         int_array(0))
                elif i == 1:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[0].occ_orb,
                                         int_array(0))
                        self.assertEqual(Index[8].occ_orb,
                                         int_array(1))
            else:
                self.fail(msg='Too many indices!')
        irrep = 4
        for i, Index in enumerate(self.He2_wf._string_indices_sing_exc(irrep)):
            if i == 0 or i == 1:
                with self.subTest('He2, C of singles',
                                  i=i, irrep=irrep):
                    self.assertAlmostEqual(Index.C,
                                           0.00235879 / self.He2_wf.norm)
                if i == 0:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[4].occ_orb,
                                         int_array(1))
                        self.assertEqual(Index[12].occ_orb,
                                         int_array(0))
                elif i == 1:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[4].occ_orb,
                                         int_array(0))
                        self.assertEqual(Index[12].occ_orb,
                                         int_array(1))
            else:
                self.fail(msg='Too many indices!')

    def test_string_indices_singles_ccsd_He2(self):
        irrep = 0
        for i, Index in enumerate(self.He2_CCwf._string_indices_sing_exc(irrep)):
            if i == 0 or i == 1:
                with self.subTest('He2, C of singles',
                                  i=i, irrep=irrep):
                    self.assertAlmostEqual(Index.C,
                                           0.00344764 / self.He2_CCwf.norm)
                if i == 0:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[0].occ_orb,
                                         int_array(1))
                        self.assertEqual(Index[8].occ_orb,
                                         int_array(0))
                elif i == 1:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[0].occ_orb,
                                         int_array(0))
                        self.assertEqual(Index[8].occ_orb,
                                         int_array(1))
            else:
                self.fail(msg='Too many indices!')
        irrep = 4
        for i, Index in enumerate(self.He2_CCwf._string_indices_sing_exc(irrep)):
            if i == 0 or i == 1:
                with self.subTest('He2, C of singles',
                                  i=i, irrep=irrep):
                    self.assertAlmostEqual(Index.C,
                                           0.00240016 / self.He2_CCwf.norm)
                if i == 0:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[4].occ_orb,
                                         int_array(1))
                        self.assertEqual(Index[12].occ_orb,
                                         int_array(0))
                elif i == 1:
                    with self.subTest('He2, occ of singles',
                                      i=i, irrep=irrep):
                        self.assertEqual(Index[4].occ_orb,
                                         int_array(0))
                        self.assertEqual(Index[12].occ_orb,
                                         int_array(1))
            else:
                self.fail(msg='Too many indices!')

    def test_string_indices_Dii_cisd_He2(self):
        i = 0
        i_irrep = 0
        a_irrep = 0
        D = self.He2_wf.doubles[self.He2_wf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_wf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i == 0 or i == 1:
                self.assertAlmostEqual(Index.C, -0.02677499 / self.He2_wf.norm)
                if i == 0:
                    self.assertEqual(Index[0].occ_orb,
                                     int_array(1))
                    self.assertEqual(Index[8].occ_orb,
                                     int_array(1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        # ----------
        i = 0
        i_irrep = 0
        a_irrep = 4
        D = self.He2_wf.doubles[self.He2_wf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_wf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i == 0 or i == 1:
                self.assertAlmostEqual(Index.C, -0.02720985 / self.He2_wf.norm)
                if i == 0:
                    self.assertEqual(Index[4].occ_orb,
                                     int_array(0, 1))
                    self.assertEqual(Index[12].occ_orb,
                                     int_array(0, 1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        # ----------
        i = 0
        i_irrep = 4
        a_irrep = 0
        D = self.He2_wf.doubles[self.He2_wf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_wf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i_Ind == 0 or i_Ind == 1:
                self.assertAlmostEqual(Index.C, -0.03191829 / self.He2_wf.norm)
                if i_Ind == 0:
                    self.assertEqual(Index[0].occ_orb,
                                     int_array(0, 1))
                    self.assertEqual(Index[8].occ_orb,
                                     int_array(0, 1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        # ----------
        i = 0
        i_irrep = 4
        a_irrep = 4
        D = self.He2_wf.doubles[self.He2_wf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_wf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i_Ind == 0 or i_Ind == 1:
                self.assertAlmostEqual(Index.C, -0.05013654 / self.He2_wf.norm)
                if i_Ind == 0:
                    self.assertEqual(Index[4].occ_orb,
                                     int_array(1))
                    self.assertEqual(Index[12].occ_orb,
                                     int_array(1))
            else:
                self.assertTrue(False, msg='Too many indices!')

    def test_string_indices_Dii_ccsd_He2(self):
        i = 0
        i_irrep = 0
        a_irrep = 0
        D = self.He2_CCwf.doubles[self.He2_CCwf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_CCwf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i == 0 or i == 1:
                self.assertAlmostEqual(Index.C,
                                       (-0.02692981 + 0.00344764**2)
                                       / self.He2_CCwf.norm)
                if i == 0:
                    self.assertEqual(Index[0].occ_orb,
                                     int_array(1))
                    self.assertEqual(Index[8].occ_orb,
                                     int_array(1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        # ----------
        i = 0
        i_irrep = 0
        a_irrep = 4
        D = self.He2_CCwf.doubles[self.He2_CCwf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_CCwf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i == 0 or i == 1:
                self.assertAlmostEqual(Index.C,
                                       -0.02730136 / self.He2_CCwf.norm)
                if i == 0:
                    self.assertEqual(Index[4].occ_orb,
                                     int_array(0, 1))
                    self.assertEqual(Index[12].occ_orb,
                                     int_array(0, 1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        # ----------
        i = 0
        i_irrep = 4
        a_irrep = 0
        D = self.He2_CCwf.doubles[self.He2_CCwf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_CCwf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i_Ind == 0 or i_Ind == 1:
                self.assertAlmostEqual(Index.C,
                                       -0.03210796 / self.He2_CCwf.norm)
                if i_Ind == 0:
                    self.assertEqual(Index[0].occ_orb,
                                     int_array(0, 1))
                    self.assertEqual(Index[8].occ_orb,
                                     int_array(0, 1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        # ----------
        i = 0
        i_irrep = 4
        a_irrep = 4
        D = self.He2_CCwf.doubles[self.He2_CCwf.N_from_ij(
            i, i, i_irrep, i_irrep, 'aa')][a_irrep]
        for i_Ind, Index in enumerate(
                self.He2_CCwf._string_indices_D_ii(i, i_irrep, a_irrep, D)):
            if i_Ind == 0 or i_Ind == 1:
                self.assertAlmostEqual(Index.C,
                                       (-0.05029521 + 0.00240016**2)
                                       / self.He2_CCwf.norm)
                if i_Ind == 0:
                    self.assertEqual(Index[4].occ_orb,
                                     int_array(1))
                    self.assertEqual(Index[12].occ_orb,
                                     int_array(1))
            else:
                self.assertTrue(False, msg='Too many indices!')

    def test_string_indices_Dij_cisd_He2(self):
        i = j = 0
        i_irrep = 4
        j_irrep = 0
        a_irrep = 4
        b_irrep = 0
        D = self.He2_wf.doubles[self.He2_wf.N_from_ij(
            i, j, i_irrep, j_irrep, 'aa')][a_irrep]
        D_other = self.He2_wf.doubles[self.He2_wf.N_from_ij(
            i, j, i_irrep, j_irrep, 'aa')][b_irrep]
        for i_Ind, Index in enumerate(
                self.He2_wf._string_indices_D_ij(
                    i, j, i_irrep, j_irrep, a_irrep, b_irrep, D, D_other)):
            if i_Ind == 0:
                # baba
                self.assertAlmostEqual(Index.C,
                                       -0.03492970
                                       / self.He2_wf.norm)
            elif i_Ind == 1:
                # abab
                self.assertAlmostEqual(Index.C,
                                       -0.03492970
                                       / self.He2_wf.norm)
            elif i_Ind == 2:
                # abba
                self.assertAlmostEqual(Index.C,
                                       -(-0.02890619)
                                       / self.He2_wf.norm)
            elif i_Ind == 3:
                # baab
                self.assertAlmostEqual(Index.C,
                                       -(-0.02890619)
                                       / self.He2_wf.norm)
            elif i_Ind == 4:
                # aaaa
                self.assertAlmostEqual(Index.C,
                                       (-0.03492970 - -0.02890619)
                                       / self.He2_wf.norm)
            elif i_Ind == 0:
                # bbbb
                self.assertAlmostEqual(Index.C,
                                       (-0.03492970 - -0.02890619)
                                       / self.He2_wf.norm)

    def test_string_indices_Dij_ccsd_He2(self):
        i = j = 0
        i_irrep = 4
        j_irrep = 0
        a_irrep = 4
        b_irrep = 0
        D = self.He2_CCwf.doubles[self.He2_CCwf.N_from_ij(
            i, j, i_irrep, j_irrep, 'aa')][a_irrep]
        D_other = self.He2_CCwf.doubles[self.He2_CCwf.N_from_ij(
            i, j, i_irrep, j_irrep, 'aa')][b_irrep]
        for i_Ind, Index in enumerate(
                self.He2_CCwf._string_indices_D_ij(
                    i, j, i_irrep, j_irrep, a_irrep, b_irrep, D, D_other)):
            if i_Ind == 0:
                # baba
                self.assertAlmostEqual(Index.C,
                                       (-0.03508008 + 0.00344764 * 0.00240016)
                                       / self.He2_CCwf.norm)
            elif i_Ind == 1:
                # abab
                self.assertAlmostEqual(Index.C,
                                       (-0.03508008 + 0.00344764 * 0.00240016)
                                       / self.He2_CCwf.norm)
            elif i_Ind == 2:
                # abba
                self.assertAlmostEqual(Index.C,
                                       -(-0.02903510)
                                       / self.He2_CCwf.norm)
            elif i_Ind == 3:
                # baab
                self.assertAlmostEqual(Index.C,
                                       -(-0.02903510)
                                       / self.He2_CCwf.norm)
            elif i_Ind == 4:
                # aaaa
                self.assertAlmostEqual(Index.C,
                                       (-0.03508008 + 0.00344764 * 0.00240016
                                        - -0.02903510)
                                       / self.He2_CCwf.norm)
            elif i_Ind == 0:
                # bbbb
                self.assertAlmostEqual(Index.C,
                                       (-0.03508008 + 0.00344764 * 0.00240016
                                        - -0.02903510)
                                       / self.He2_CCwf.norm)

    def test_make_occ_ind_doub_He2(self):
        zero_arr_1 = int_array(0)
        one_arr_1 = int_array(1)
        i = j = 0
        irrep_i = irrep_j = 0
        irrep_a = irrep_b = 0
        Index = self.He2_wf._make_occ_indices_for_doubles(
            i, j, irrep_i, irrep_j, irrep_a, irrep_b)
        self.assertTrue(isinstance(Index, SD_StringIndex))
        tests.logger.info('(i, irrep_i) = (%d, %d); (j, irrep_j) = (%d, %d);'
                          + ' irrep_a = %d; irrep_b = %d:\n%s',
                          i, irrep_i, j, irrep_j, irrep_a, irrep_b, Index)
        self.assertEqual(len(Index[0]), 1)
        self.assertEqual(Index[4], zero_arr_1)
        self.assertEqual(len(Index[8]), 1)
        self.assertEqual(Index[12], zero_arr_1)
        # ----------
        i = j = 0
        irrep_i = irrep_j = 0
        irrep_a = irrep_b = 4
        Index = self.He2_wf._make_occ_indices_for_doubles(
            i, j, irrep_i, irrep_j, irrep_a, irrep_b)
        self.assertTrue(isinstance(Index, SD_StringIndex))
        tests.logger.info('(i, irrep_i) = (%d, %d); (j, irrep_j) = (%d, %d);'
                          + ' irrep_a = %d; irrep_b = %d:\n%s',
                          i, irrep_i, j, irrep_j, irrep_a, irrep_b, Index)
        self.assertEqual(len(Index[0]), 0)
        self.assertEqual(Index[4][:1], zero_arr_1)
        self.assertEqual(len(Index[8]), 0)
        self.assertEqual(Index[12][:1], zero_arr_1)
        # ----------
        i = j = 0
        irrep_i = irrep_j = 4
        irrep_a = irrep_b = 0
        Index = self.He2_wf._make_occ_indices_for_doubles(
            i, j, irrep_i, irrep_j, irrep_a, irrep_b)
        self.assertTrue(isinstance(Index, SD_StringIndex))
        tests.logger.info('(i, irrep_i) = (%d, %d); (j, irrep_j) = (%d, %d);'
                          + ' irrep_a = %d; irrep_b = %d:\n%s',
                          i, irrep_i, j, irrep_j, irrep_a, irrep_b, Index)
        self.assertEqual(Index[0][:1], zero_arr_1)
        self.assertEqual(len(Index[4]), 0)
        self.assertEqual(Index[8][:1], zero_arr_1)
        self.assertEqual(len(Index[12]), 0)
        # ----------
        i = j = 0
        irrep_i = irrep_j = 4
        irrep_a = irrep_b = 4
        Index = self.He2_wf._make_occ_indices_for_doubles(
            i, j, irrep_i, irrep_j, irrep_a, irrep_b)
        self.assertTrue(isinstance(Index, SD_StringIndex))
        tests.logger.info('(i, irrep_i) = (%d, %d); (j, irrep_j) = (%d, %d);'
                          + ' irrep_a = %d; irrep_b = %d:\n%s',
                          i, irrep_i, j, irrep_j, irrep_a, irrep_b, Index)
        self.assertEqual(Index[0], zero_arr_1)
        self.assertEqual(len(Index[4]), 1)
        self.assertEqual(Index[8], zero_arr_1)
        self.assertEqual(len(Index[12]), 1)
        # ----------
        i = j = 0
        irrep_i = 4
        irrep_j = 0
        irrep_a = 4
        irrep_b = 0
        n_irrep = self.He2_wf.n_irrep
        Index = self.He2_wf._make_occ_indices_for_doubles(
            i, j, irrep_i, irrep_j, irrep_a, irrep_b)
        self.assertTrue(isinstance(Index, int_norm.DoublesTypes))
        tests.logger.info('(i, irrep_i) = (%d, %d); (j, irrep_j) = (%d, %d);'
                          + ' irrep_a = %d; irrep_b = %d:\n'
                          + 'baba: %s\nabab: %s\n'
                          + 'abba: %s\nbaab: %s\n'
                          + 'aaaa: %s\nbbbb: %s\n',
                          i, irrep_i, j, irrep_j, irrep_a, irrep_b,
                          Index.baba, Index.abab,
                          Index.abba, Index.baab,
                          Index.aaaa, Index.bbbb)
        # ----
        self.assertEqual(Index.baba[irrep_b + n_irrep], one_arr_1)
        self.assertEqual(Index.baba[irrep_a], one_arr_1)
        # ----
        self.assertEqual(Index.abab[irrep_b], one_arr_1)
        self.assertEqual(Index.abab[irrep_a + n_irrep], one_arr_1)
        # ----
        self.assertEqual(len(Index.abba[irrep_j]), 0)
        self.assertEqual(len(Index.abba[irrep_i + n_irrep]), 0)
        self.assertEqual(Index.abba[irrep_b + n_irrep][:1], zero_arr_1)
        self.assertEqual(Index.abba[irrep_a][:1], zero_arr_1)
        # ----
        self.assertEqual(len(Index.baab[irrep_j + n_irrep]), 0)
        self.assertEqual(len(Index.baab[irrep_i]), 0)
        self.assertEqual(Index.baab[irrep_b][:1], zero_arr_1)
        self.assertEqual(Index.baab[irrep_a + n_irrep][:1], zero_arr_1)
        # ----
        self.assertEqual(Index.aaaa[irrep_b], one_arr_1)
        self.assertEqual(Index.aaaa[irrep_a], one_arr_1)
        # ----
        self.assertEqual(Index.bbbb[irrep_b + n_irrep], one_arr_1)
        self.assertEqual(Index.bbbb[irrep_a + n_irrep], one_arr_1)
        # ----------
        i = j = 0
        irrep_i = 4
        irrep_j = 0
        irrep_a = 0
        irrep_b = 4
        n_irrep = self.He2_wf.n_irrep
        Index = self.He2_wf._make_occ_indices_for_doubles(
            i, j, irrep_i, irrep_j, irrep_a, irrep_b)
        self.assertTrue(isinstance(Index, int_norm.DoublesTypes))
        tests.logger.info('(i, irrep_i) = (%d, %d); (j, irrep_j) = (%d, %d);'
                          + ' irrep_a = %d; irrep_b = %d:\n'
                          + 'baba: %s\nabab: %s\n'
                          + 'abba: %s\nbaab: %s\n'
                          + 'aaaa: %s\nbbbb: %s\n',
                          i, irrep_i, j, irrep_j, irrep_a, irrep_b,
                          Index.baba, Index.abab,
                          Index.abba, Index.baab,
                          Index.aaaa, Index.bbbb)
        # ----
        self.assertEqual(len(Index.baba[irrep_j + n_irrep]), 0)
        self.assertEqual(len(Index.baba[irrep_i]), 0)
        self.assertEqual(Index.baba[irrep_b + n_irrep][:1], zero_arr_1)
        self.assertEqual(Index.baba[irrep_a][:1], zero_arr_1)
        # ----
        self.assertEqual(len(Index.abab[irrep_j]), 0)
        self.assertEqual(len(Index.abab[irrep_i + n_irrep]), 0)
        self.assertEqual(Index.abab[irrep_b][:1], zero_arr_1)
        self.assertEqual(Index.abab[irrep_a + n_irrep][:1], zero_arr_1)
        # ----
        self.assertEqual(Index.abba[irrep_b + n_irrep], one_arr_1)
        self.assertEqual(Index.abba[irrep_a], one_arr_1)
        # ----
        self.assertEqual(Index.baab[irrep_b], one_arr_1)
        self.assertEqual(Index.baab[irrep_a + n_irrep], one_arr_1)
        # ----
        self.assertEqual(Index.aaaa[irrep_b], one_arr_1)
        self.assertEqual(Index.aaaa[irrep_a], one_arr_1)
        # ----
        self.assertEqual(Index.bbbb[irrep_b + n_irrep], one_arr_1)
        self.assertEqual(Index.bbbb[irrep_a + n_irrep], one_arr_1)
        # ----------

    def test_string_indices_spirrep_He2(self):
        for i_Ind, Index in enumerate(self.He2_wf._string_indices_spirrep(0)):
            if i_Ind == 0:
                self.assertEqual(Index.occ_orb,
                                 int_array(0))
            elif i_Ind == 1:
                self.assertEqual(Index.occ_orb,
                                 int_array(1))
            else:
                self.assertTrue(False, msg='Too many indices!')
        for i_Ind, Index in enumerate(self.He2_wf._string_indices_spirrep(
                0, only_this_occ=OrbitalsSets([2, 0, 0, 0, 0, 0, 0, 0],
                                              occ_type='R'))):
            if i_Ind == 0:
                self.assertEqual(Index.occ_orb,
                                 int_array(0, 1))
            else:
                self.assertTrue(False, msg='Too many indices!')

