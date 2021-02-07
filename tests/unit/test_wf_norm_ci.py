"""Tests for norm_ci

"""
import unittest

from scipy import linalg
import numpy as np

import test
from util.variables import int_dtype
from wave_functions import general, norm_ci, int_norm
from wave_functions.fci import make_occ
from string_indices.string_indices import SpirrepStringIndex, SD_StringIndex
from orbitals.symmetry import OrbitalsSets


class WFConstructorsTestCase(unittest.TestCase):
    
    def setUp(self):
        self.int_N_WF = int_norm.IntermNormWaveFunction.from_Molpro(
            test.CISD_file('H2__5__sto3g__D2h'))
        test.logger.info('end of WFConstructorsTestCase.SetUp')
    
    def test_int_norm_contructor(self):
        wf = norm_ci.NormCI_WaveFunction.from_Molpro_FCI(
            test.FCI_file('H2__5__sto3g__D2h'),
            zero_coefficients=True)
        wf.get_coeff_from_int_norm_WF(self.int_N_WF,
                                      change_structure=False,
                                      use_structure=True)
        self.assertAlmostEqual(wf.C0, 0.7568532707525314, places=10)
        self.assertAlmostEqual(wf[0].c, 0.7568532707525314, places=10)
        self.assertEqual(wf[0].occupation[0], make_occ([0]))
        self.assertEqual(wf[0].occupation[8], make_occ([0]))
        self.assertAlmostEqual(wf[1].c, -0.6535848273569357, places=10)
        self.assertEqual(wf[1].occupation[4], make_occ([0]))
        self.assertEqual(wf[1].occupation[12], make_occ([0]))
        test.logger.info('end of WFConstructorsTestCase.test_int_norm_contructor')


class SlaterDetTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        self.n_irrep = 4
        self.string_index = SD_StringIndex()
        self.string_index.C = 0.12345
        self.string_index.append(
            SpirrepStringIndex.make_hole(3, 1))
        self.string_index[-1][2] = 4
        self.string_index.append(SpirrepStringIndex(1))
        self.string_index.append(SpirrepStringIndex(0))
        self.string_index.append(SpirrepStringIndex(1))
        self.string_index.append(SpirrepStringIndex(3))
        self.string_index.append(
            SpirrepStringIndex.make_hole(2, 0))
        self.string_index[-1][1] = 3
        self.string_index.append(SpirrepStringIndex(0))
        self.string_index.append(SpirrepStringIndex(1))
        # ------
        self.string_index_2 = SD_StringIndex()
        self.string_index_2.C = 0.6789
        self.string_index_2.append(
            SpirrepStringIndex.make_hole(3, 1))
        self.string_index_2[-1][2] = 4
        self.string_index_2.append(SpirrepStringIndex(1))
        self.string_index_2.append(SpirrepStringIndex(0))
        self.string_index_2.append(SpirrepStringIndex(1))
        self.string_index_2.append(SpirrepStringIndex(3))
        self.string_index_2.append(
            SpirrepStringIndex.make_hole(2, 0))
        self.string_index_2[-1][1] = 3
        self.string_index_2.append(SpirrepStringIndex(0))
        self.string_index_2.append(SpirrepStringIndex(1))
    
    def test_simple_construction(self):
        det = norm_ci.SlaterDet(c=0.123,
                                occupation=(make_occ([0, 1, 2]),
                                            make_occ([1])))
        self.assertEqual(det.c, 0.123)
        self.assertEqual(det[0], 0.123)
        self.assertEqual(det.occupation[0][1], 1)
        self.assertEqual(det.occupation[1][0], 1)
        with self.assertRaises(IndexError):
            det.occupation[1][1]

    def test_get_from_FCI_line_1(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([0, 0, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms)
        self.assertAlmostEqual(det.c, -0.162676901257, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 1]))
        self.assertEqual(det.occupation[1], make_occ([0]))
        self.assertEqual(det.occupation[2], make_occ([]))
        self.assertEqual(det.occupation[3], make_occ([]))
        self.assertEqual(det.occupation[4], make_occ([0, 1]))
        self.assertEqual(det.occupation[5], make_occ([0]))
        self.assertEqual(det.occupation[6], make_occ([]))
        self.assertEqual(det.occupation[7], make_occ([]))

    def test_get_from_FCI_line_2(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([0, 0, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms)
        self.assertAlmostEqual(det.c, -0.049624632911, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 1, 3]))
        self.assertEqual(det.occupation[1], make_occ([]))
        self.assertEqual(det.occupation[2], make_occ([]))
        self.assertEqual(det.occupation[3], make_occ([]))
        self.assertEqual(det.occupation[4], make_occ([0, 1, 5]))
        self.assertEqual(det.occupation[5], make_occ([]))
        self.assertEqual(det.occupation[6], make_occ([]))
        self.assertEqual(det.occupation[7], make_occ([]))

    def test_get_from_FCI_line_3(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([0, 0, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms, zero_coefficients=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)

    def test_get_from_FCI_line_4(self):
        line = '0.000000000000  1  2  9  1  2 10'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([0, 0, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms)
        self.assertAlmostEqual(det.c, -0.000000000000, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 1]))
        self.assertEqual(det.occupation[1], make_occ([]))
        self.assertEqual(det.occupation[2], make_occ([0]))
        self.assertEqual(det.occupation[3], make_occ([]))
        self.assertEqual(det.occupation[4], make_occ([0, 1]))
        self.assertEqual(det.occupation[5], make_occ([]))
        self.assertEqual(det.occupation[6], make_occ([1]))
        self.assertEqual(det.occupation[7], make_occ([]))

    def test_get_from_FCI_line_5(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([1, 1, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms)
        self.assertAlmostEqual(det.c, -0.162676901257, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 5]))
        self.assertEqual(det.occupation[1], make_occ([0]))
        self.assertEqual(det.occupation[2], make_occ([]))
        self.assertEqual(det.occupation[3], make_occ([]))
        self.assertEqual(det.occupation[4], make_occ([0, 5]))
        self.assertEqual(det.occupation[5], make_occ([0]))
        self.assertEqual(det.occupation[6], make_occ([]))
        self.assertEqual(det.occupation[7], make_occ([]))

    def test_get_from_FCI_line_6(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([1, 1, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms)
        self.assertAlmostEqual(det.c, -0.049624632911, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 2]))
        self.assertEqual(det.occupation[1], make_occ([0]))
        self.assertEqual(det.occupation[2], make_occ([]))
        self.assertEqual(det.occupation[3], make_occ([]))
        self.assertEqual(det.occupation[4], make_occ([0, 4]))
        self.assertEqual(det.occupation[5], make_occ([0]))
        self.assertEqual(det.occupation[6], make_occ([]))
        self.assertEqual(det.occupation[7], make_occ([]))

    def test_get_from_FCI_line_7(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([1, 1, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms, zero_coefficients=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)

    def test_get_from_FCI_line_8(self):
        line = '0.000000000000  1  2  9  1  2 10'
        orb_dim = OrbitalsSets([6, 2, 2, 0], occ_type='R')
        n_core = OrbitalsSets([1, 1, 0, 0], occ_type='R')
        n_irrep = 4
        Ms = 0.0
        det = norm_ci._get_Slater_Det_from_FCI_line(
            line, orb_dim, n_core, n_irrep, Ms)
        self.assertAlmostEqual(det.c, -0.000000000000, places=10)
        self.assertEqual(det.occupation[0], make_occ([0]))
        self.assertEqual(det.occupation[1], make_occ([0]))
        self.assertEqual(det.occupation[2], make_occ([0]))
        self.assertEqual(det.occupation[3], make_occ([]))
        self.assertEqual(det.occupation[4], make_occ([0]))
        self.assertEqual(det.occupation[5], make_occ([0]))
        self.assertEqual(det.occupation[6], make_occ([1]))
        self.assertEqual(det.occupation[7], make_occ([]))
    
    def test_get_from_String_Index_1(self):
        det = norm_ci._get_Slater_Det_from_String_Index(self.string_index,
                                                    self.n_irrep)
        self.assertAlmostEqual(det.c, 0.12345, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 2, 4]))
        self.assertEqual(det.occupation[1], make_occ([0]))
        self.assertEqual(det.occupation[2], make_occ([]))
        self.assertEqual(det.occupation[3], make_occ([0]))
        self.assertEqual(det.occupation[4], make_occ([0, 1, 2]))
        self.assertEqual(det.occupation[5], make_occ([1, 3]))
        self.assertEqual(det.occupation[6], make_occ([]))
        self.assertEqual(det.occupation[7], make_occ([0]))
    
    def test_get_from_String_Index_2(self):
        det = norm_ci._get_Slater_Det_from_String_Index(
            self.string_index, self.n_irrep, zero_coefficients=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)
    
    def test_get_from_String_Index_3(self):
        det = norm_ci._get_Slater_Det_from_String_Index(self.string_index_2,
                                                    self.n_irrep)
        self.assertAlmostEqual(det.c, 0.6789, places=10)
        self.assertEqual(det.occupation[0], make_occ([0, 2, 4]))
        self.assertEqual(det.occupation[1], make_occ([0]))
        self.assertEqual(det.occupation[2], make_occ([]))
        self.assertEqual(det.occupation[3], make_occ([0]))
        self.assertEqual(det.occupation[4], make_occ([0, 1, 2]))
        self.assertEqual(det.occupation[5], make_occ([1, 3]))
        self.assertEqual(det.occupation[6], make_occ([]))
        self.assertEqual(det.occupation[7], make_occ([0]))


class JacHess_H2_TestCase(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        self.WF = norm_ci.NormCI_WaveFunction.from_Molpro_FCI(
            test.FCI_file('H2__5__sto3g__D2h'))
        Ures = []
        Uunres = []
        prng = np.random.RandomState(test.init_random_state)
        for spirrep in self.WF.spirrep_blocks(restricted=True):
            K = self.WF.orb_dim[spirrep]
            newU = prng.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            Ures.append(newU)
        for spirrep in self.WF.spirrep_blocks(restricted=False):
            K = self.WF.orb_dim[spirrep]
            newU = prng.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            Uunres.append(newU)
        self.WFres = self.WF.change_orb_basis(Ures)
        self.WFunres = self.WF.change_orb_basis(Uunres)
        self.WFunres.restricted = False
    
    def test_Num_Anal_original(self):
        Janal, Hanal = self.WFres.make_Jac_Hess_overlap(analytic=True,
                                                        restricted=True)
        Jnum, Hnum = self.WFres.make_Jac_Hess_overlap(analytic=False,
                                                      restricted=True)
        test.logger.info('Wave function:\n%r', self.WF)
        test.logger.info('Analytic Jacobian:\n%r', Janal)
        test.logger.info('Numeric Jacobian:\n%r', Jnum)
        test.logger.info('Analytic Hessian:\n%r', Hanal)
        test.logger.info('Numeric Hessian:\n%r', Hnum)
        self.assertEqual(Janal, Jnum)
        self.assertEqual(Hanal, Hnum)
    
    def test_Num_Anal_res(self):
        Janal, Hanal = self.WFres.make_Jac_Hess_overlap(analytic=True,
                                                        restricted=True)
        Jnum, Hnum = self.WFres.make_Jac_Hess_overlap(analytic=False,
                                                      restricted=True)
        test.logger.info('Wave function:\n%r', self.WFres)
        test.logger.info('Analytic Jacobian:\n%r', Janal)
        test.logger.info('Numeric Jacobian:\n%r', Jnum)
        test.logger.info('Analytic Hessian:\n%r', Hanal)
        test.logger.info('Numeric Hessian:\n%r', Hnum)
        self.assertEqual(Janal, Jnum)
        self.assertEqual(Hanal, Hnum)
    
    def test_Num_Anal_unres(self):
        Janal, Hanal = self.WFunres.make_Jac_Hess_overlap(analytic=True,
                                                          restricted=False)
        Jnum, Hnum = self.WFunres.make_Jac_Hess_overlap(analytic=False,
                                                        restricted=False)
        test.logger.info('Wave function:\n%r', self.WFunres)
        test.logger.info('Analytic Jacobian:\n%r', Janal)
        test.logger.info('Numeric Jacobian:\n%r', Jnum)
        test.logger.info('Analytic Hessian:\n%r', Hanal)
        test.logger.info('Numeric Hessian:\n%r', Hnum)
        self.assertEqual(Janal, Jnum)
        self.assertEqual(Hanal, Hnum)


class WForbChangeTestCase(unittest.TestCase):
    
    def setUp(self):
        self.WF = norm_ci.NormCI_WaveFunction.from_Molpro_FCI(
            test.FCI_file('H2__5__sto3g__D2h'))
        self.U1 = []
        self.U2 = []
        self.U3 = []
        self.U = []
        self.UT = []
        for spirrep in self.WF.spirrep_blocks(restricted=True):
            K = self.WF.orb_dim[spirrep]
            newU = np.random.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            self.U1.append(newU)
            newU = np.random.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            self.U2.append(newU)
            newU = np.random.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            self.U3.append(newU)
            self.U.append(np.matmul(np.matmul(self.U1[-1],
                                              self.U2[-1]),
                                    self.U3[-1]))
            self.UT.append(self.U[-1].T)
    
    def test_go_and_back(self):
        backWF = self.WF.change_orb_basis(self.U).change_orb_basis(self.UT)
        self.assertEqual(self.WF, backWF, msg='Wave functions are not equal.')
    
    def test_three_vs_one_steps(self):
        newWF1 = self.WF.change_orb_basis(self.U1)
        newWF2 = newWF1.change_orb_basis(self.U2)
        newWF3 = newWF2.change_orb_basis(self.U3)
        newWF3_direct = self.WF.change_orb_basis(self.U)
        self.assertEqual(newWF3, newWF3_direct)


if __name__ == '__main__':
    unittest.main()
