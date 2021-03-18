"""Tests for fci

"""
import unittest
from math import sqrt

from scipy.linalg import norm
import numpy as np

import tests
from wave_functions.fci import FCIWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from util.other import int_array
from input_output.log import logtime
from wave_functions.slater_det import SlaterDet
from wave_functions.strings_rev_lexical_order import get_index

@tests.category('SHORT')
class FromMolproTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        self.assertEqual(wf._coefficients,
                         np.array([[0.756853272220, 0.0],
                                   [0.0, -0.653584825658]]))
        self.assertEqual(wf._alpha_string_graph, int_array([[0],
                                                            [1]]))
        self.assertEqual(wf._beta_string_graph, int_array([[0],
                                                           [1]]))
        self.assertEqual(wf.n_irrep, 8)
        self.assertTrue(wf.restricted)

    def test_h2_631g_c2v(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__C2v'))
        my_coeff = np.array(
            [[ 0.781771118719, 0.000000000000, 0.061389372090, 0.000000000000],
             [ 0.000000000000,-0.612568504984, 0.000000000000,-0.054814463843],
             [ 0.061389372090, 0.000000000000, 0.002689301121, 0.000000000000],
             [ 0.000000000000,-0.054814463843, 0.000000000000,-0.006320711531]])
        self.assertEqual(wf._coefficients, my_coeff)
        self.assertEqual(wf._alpha_string_graph, int_array([[0],
                                                            [1],
                                                            [2],
                                                            [3]]))
        self.assertEqual(wf._beta_string_graph, int_array([[0],
                                                           [1],
                                                           [2],
                                                           [3]]))
        self.assertEqual(wf.n_irrep, 4)
        self.assertTrue(wf.restricted)

    def test_h2_ccpvdz_c2v(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__ccpVDZ__C2v'))
        my_coeff = np.zeros((10, 10))
        my_coeff[0, :6] = [0.781815404715,  0.000000000000, -0.084281730429,
                           0.000000000000, -0.004432884704,  0.000000000000]
        my_coeff[1, :6] = [0.000000000000, -0.601382497214,  0.000000000000,
                           0.079354517044,  0.000000000000, -0.004408335035]
        my_coeff[2, :6] = [-0.084281730429, 0.000000000000, 0.006565211778, 
                           0.000000000000, 0.000649650003, 0.000000000000] 
        my_coeff[3, :6] = [0.000000000000,  0.079354517044, 0.000000000000, 
                           -0.011874773981, 0.000000000000, 0.000538284860] 
        my_coeff[4, :6] = [-0.004432884704, 0.000000000000, 0.000649650003, 
                           0.000000000000, -0.004272971755, 0.000000000000] 
        my_coeff[5, :6] = [0.000000000000, -0.004408335035, 0.000000000000, 
                           0.000538284860, 0.000000000000, -0.000939891958] 
        my_coeff[6, 6:8] = [-0.002927344434, 0.000000000000]
        my_coeff[7, 6:8] = [0.000000000000, -0.001158032360]
        my_coeff[8, 8:] = [-0.002927344434, 0.000000000000]
        my_coeff[9, 8:] = [0.000000000000, -0.001158032360]
        self.assertEqual(wf._coefficients, my_coeff)
        self.assertEqual(wf._alpha_string_graph, int_array([[0],
                                                            [1],
                                                            [2],
                                                            [3],
                                                            [4],
                                                            [5],
                                                            [6],
                                                            [7],
                                                            [8],
                                                            [9]]))
        self.assertEqual(wf._beta_string_graph, int_array([[0],
                                                           [1],
                                                           [2],
                                                           [3],
                                                           [4],
                                                           [5],
                                                           [6],
                                                           [7],
                                                           [8],
                                                           [9]]))
        self.assertEqual(wf.n_irrep, 4)
        self.assertTrue(wf.restricted)

    def test_hcl_plus_631g_c2v(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('HCl_plus__1.5__631g__C2v'))
        self.assertEqual(wf._coefficients[0, 0], 0.000015255089)
        self.assertEqual(
            wf._coefficients[
                get_index(int_array(0, 1, 2, 3), wf._alpha_string_graph),
                get_index(int_array(0, 1, 2), wf._beta_string_graph)],
            0.000015255089)
        self.assertEqual(
            wf._coefficients[
                get_index(int_array(2, 3, 6, 8), wf._alpha_string_graph),
                get_index(int_array(5, 6, 9), wf._beta_string_graph)],
            -0.000086583490)
        self.assertEqual(
            wf._coefficients[
                get_index(int_array(1, 6, 7, 8), wf._alpha_string_graph),
                get_index(int_array(1, 4, 8), wf._beta_string_graph)],
            0.000010336692)
        self.assertEqual(wf._alpha_string_graph, int_array([[0,  0,  0,  0],
                                                            [1,  1,  1,  1],
                                                            [2,  3,  4,  5],
                                                            [3,  6, 10, 15],
                                                            [4, 10, 20, 35],
                                                            [5, 15, 35, 70],
                                                            [6, 21, 56, 126]]))
        self.assertEqual(wf._beta_string_graph, int_array([[0,  0,  0],
                                                           [1,  1,  1],
                                                           [2,  3,  4],
                                                           [3,  6, 10],
                                                           [4, 10, 20],
                                                           [5, 15, 35],
                                                           [6, 21, 56],
                                                           [7, 28, 84]]))
        self.assertEqual(wf.n_irrep, 4)
        self.assertFalse(wf.restricted)


class FromIntermNormCCDTestCase(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__sto3g__D2h')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)
    
    @tests.category('SHORT')
    def test_h2_631g_c2v(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('H2__5__631g__C2v')))
        my_coeff = np.array(
            [[ 1.00000000, 0.00000000, 0.00000000, 0.00000000],
             [ 0.00000000,-0.75250839, 0.00000000,-0.05419745],
             [ 0.00000000, 0.00000000, 0.00270811, 0.00000000],
             [ 0.00000000,-0.05419745, 0.00000000,-0.00238158]])
        self.assertEqual(wf._coefficients, my_coeff)
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.unrestrict(
                IntermNormWaveFunction.from_Molpro(
                    tests.CCD_file('H2__5__631g__C2v'))))
        self.assertEqual(wf._coefficients, my_coeff)
    
    @tests.category('SHORT')
    def test_h2_631g_d2h(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('H2__5__631g__D2h')))
        my_coeff = np.array(
            [[ 1.00000000, 0.00000000, 0.00000000, 0.00000000],
             [ 0.00000000, 0.00270811, 0.00000000, 0.00000000],
             [ 0.00000000, 0.00000000,-0.75250839,-0.05419745],
             [ 0.00000000, 0.00000000,-0.05419745,-0.00238158]])
        self.assertEqual(wf._coefficients, my_coeff)
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.unrestrict(
                IntermNormWaveFunction.from_Molpro(
                    tests.CCD_file('H2__5__631g__D2h'))))
        self.assertEqual(wf._coefficients, my_coeff)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('H2__5__ccpVDZ__D2h')))
        my_coeff = np.zeros((10, 10))
        my_coeff[:3, :3] = [[1.00000000, 0.00000000,  0.00000000],
                            [0.00000000, 0.00414731,  0.00116415],
                            [0.00000000, 0.00116415, -0.00468952]]
        my_coeff[3, 3] = -0.00292749
        my_coeff[4, 4] = -0.00292749
        my_coeff[5:8,5:8] = [[-0.72461125,  0.07727448, -0.00845294],
                             [ 0.07727448, -0.00662715,  0.00157013],
                             [-0.00845294,  0.00157013, -0.00053259]]
        my_coeff[8, 8] = -0.00075374
        my_coeff[9, 9] = -0.00075374
        self.assertEqual(wf._coefficients, my_coeff)
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.unrestrict(
                IntermNormWaveFunction.from_Molpro(
                    tests.CCD_file('H2__5__ccpVDZ__D2h'))))
        self.assertEqual(wf._coefficients, my_coeff)
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h')))
        my_coeff = np.zeros((8, 8))
        my_coeff[:2, :2] = [[1.00000000, 0.00000000],
                            [0.00000000,-0.09591407]]
        my_coeff[2, 2] = -0.17367068
        my_coeff[3, 3] = -0.17367068
        my_coeff[4:6,4:6] = [[-0.13010175, -0.05296460],
                             [-0.05296460, -0.02815273]]
        my_coeff[6, 6] = -0.00001730
        my_coeff[7, 7] = -0.00001730
        self.assertEqual(wf._coefficients, my_coeff)
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.unrestrict(
                IntermNormWaveFunction.from_Molpro(
                    tests.CCD_file('Li2__5__sto3g__D2h'))))
        self.assertEqual(wf._coefficients, my_coeff)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_alle(self):
        wfcc = IntermNormWaveFunction.from_Molpro(
            tests.CCD_file('Li2__5__sto3g__D2h', allE=True))
        wf = FCIWaveFunction.from_int_norm(wfcc)
        # Ref det: SlaterDet(c=0.0,
        #                    alpha_occ=int_array(0,1,5),
        #                    beta_occ=int_array(0,1,5)))
        self.assertAlmostEqual(wf.C0, 1.0)
        self.assertEqual(wf.ref_det.alpha_occ, int_array(0,1,5))
        self.assertEqual(wf.ref_det.beta_occ, int_array(0,1,5))
        # --------- some singles
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,2),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,3),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,4),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,6),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,7),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,8),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,9),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(0,1,2)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(0,1,8)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(1,3,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(1,5,9)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,5,8),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,2,5),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        # --------- some doubles, alpha,beta
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,2,5),
            beta_occ=int_array(1,2,5)))],
                               -0.00091750)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,3,5),
            beta_occ=int_array(1,3,5)))],
                               -0.00042147)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,5,6),
            beta_occ=int_array(1,5,6)))],
                               -0.00202693)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,5,6),
            beta_occ=int_array(1,5,7)))],
                               -0.00092670)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,5,7),
            beta_occ=int_array(1,5,7)))],
                               -0.00260880)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,2,5),
            beta_occ=int_array(0,2,5)))],
                               -(-0.00328470))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,3,5),
            beta_occ=int_array(1,3,5)))],
                               -(-0.00284484))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,5,6),
            beta_occ=int_array(0,5,6)))],
                               -(0.00117550))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(1,5,6),
            beta_occ=int_array(0,5,7)))],
                               -(0.00227618))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,5,6),
            beta_occ=int_array(1,5,7)))],
                               -(0.00252029))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,5,7),
            beta_occ=int_array(1,5,6)))],
                               -(0.00227618))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,2,5),
            beta_occ=int_array(0,2,5)))],
                               -0.09496718)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,3,5),
            beta_occ=int_array(0,3,5)))],
                               -0.17272984)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,5,6),
            beta_occ=int_array(0,5,6)))],
                               -0.12989194)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,5,6),
            beta_occ=int_array(0,5,7)))],
                               -0.05293789)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,2),
            beta_occ=int_array(0,1,2)))],
                               -0.00079094)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,6),
            beta_occ=int_array(0,1,7)))],
                               -0.00085545)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,2),
            beta_occ=int_array(1,5,6)))],
                               0.00085432)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,2),
            beta_occ=int_array(1,5,7)))],
                               -0.00086412)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,7),
            beta_occ=int_array(1,2,5)))],
                               -(-0.00097751))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,7),
            beta_occ=int_array(0,2,5)))],
                               -0.00347713)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,2,5),
            beta_occ=int_array(0,1,7)))],
                               -0.00347713)
        # --------- some doubles alpha,alpha or beta,beta
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(2,3,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(4,5,6),
            beta_occ=int_array(0,1,5)))],
                               0.0)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(5,6,7),
            beta_occ=int_array(0,1,5)))],
                               0.00227618-0.00252029)
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(1,2,6)))],
                               -(0.00091881-0.00085432))
        self.assertAlmostEqual(wf[wf.index(SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,5),
            beta_occ=int_array(1,3,8)))],
                               -(-0.00102550-(-0.00096368)))
        # --------- some quadruples
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(2,3,5),  #  0a 1a 0b 1b -> 2a 3a 2b 3b
                beta_occ=int_array(2,3,5)))],
            (-0.00091750 * -0.17272984)   # 0a 0b -> 2a 2b ; 1a 1b -> 3a 3b
            +(-0.00042147 * -0.09496718)  # 0a 0b -> 3a 3b ; 1a 1b -> 2a 2b
            -(-0.00328470 * -0.00284484)  # 0a 1b -> 2a 2b ; 1a 0b -> 3a 3b
            -(-0.00284484 * -0.00328470)) # 0a 1b -> 3a 3b ; 1a 0b -> 2a 2b
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(5,6,7),  #  0a 1a 0b 1b -> 6a 7a 6b 7b
                beta_occ=int_array(5,6,7)))],
             (-0.00202693 * -0.02804540)  # 0a 0b -> 6a 6b ; 1a 1b -> 7a 7b
            +(-0.00260880 * -0.12989194)  # 0a 0b -> 7a 7b ; 1a 1b -> 6a 6b
            -(-0.00092670 * -0.05293789)  # 0a 0b -> 6a 7b ; 1a 1b -> 7a 6b
            -(-0.00092670 * -0.05293789)  # 0a 0b -> 7a 6b ; 1a 1b -> 6a 7b
            -( 0.00117550 * -0.00113985)  # 0a 1b -> 6a 6b ; 1a 0b -> 7a 7b
            -(-0.00113985 *  0.00117550)  # 0a 1b -> 7a 7b ; 1a 0b -> 6a 6b
            +( 0.00252029 *  0.00252029)  # 0a 1b -> 6a 7b ; 1a 0b -> 7a 6b
            +( 0.00227618 *  0.00227618)  # 0a 1b -> 7a 6b ; 1a 0b -> 6a 7b
            +( (0.00227618-0.00252029)
               * (0.00227618-0.00252029)))  # 0a 1a -> 6a 7a ; 0b 1b -> 6b 7b
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(5,6,7), 
                beta_occ=int_array(0,5,6)))],
            0.0)
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(0,2,6),  # 1a 5a 1b 5b -> 2a 6a 2b 6b
                beta_occ=int_array(0,2,6)))],
             (-0.09496718 * -0.00205421)  # 1a 1b -> 2a 2b ; 5a 5b -> 6a 6b
            +(-0.00079094 * -0.12989194)  # 1a 1b -> 6a 6b ; 5a 5b -> 2a 2b
            +( 0.00125126 *  0.00125126)  # 1a 5b -> 2a 6b ; 5a 1b -> 6a 2b
            +( 0.00089563 *  0.00089563)  # 1a 5b -> 6a 2b ; 5a 1b -> 2a 6b
            +((0.00125126 - 0.00089563)
              * (0.00125126 - 0.00089563))  # 1a 5a -> 2a 6a ; 1b 5b -> 2b 6b
            )
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(0,3,6),  # 1a 5a 1b 5b -> 3a 6a 3b 6b
                beta_occ=int_array(0,3,6)))],
             (-0.17272984 * -0.00205421)  # 1a 1b -> 3a 3b ; 5a 5b -> 6a 6b
            +(-0.12989194 * -0.00036760)  # 1a 1b -> 6a 6b ; 5a 5b -> 3a 3b
            )
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(4,8,9),
                beta_occ=int_array(0,1,2)))],
            0.0)
        self.assertAlmostEqual(
            wf[wf.index(SlaterDet(
                c=0.0,
                alpha_occ=int_array(2,6,7),  # 0a 1a 5a 5b -> 2a 6a 7a 2b
                beta_occ=int_array(0,1,2)))],
            ((0.00227618 - 0.00252029) * -0.00079094)  # 0a 1a -> 6a 7a ; 5a 5b -> 2a 2b
            -((0.00091881 - 0.00085432) * -0.00190054)  # 0a 5a -> 2a 6a ; 1a 5b -> 7a 2b
            +((-0.00097751 - -0.00086412) * 0.00089563)  # 0a 5a -> 2a 7a ; 1a 5b -> 6a 2b
            -(0.00085432 * (-0.00347713 - -0.00190054))  # 0a 5b -> 6a 2b ; 1a 5a -> 2a 7a
            +(-0.00086412 * (0.00125126 - 0.00089563)))  # 0a 5b -> 7a 2b ; 1a 5a -> 2a 6a
        # --------- Some sextuple:
        # self.assertAlmostEqual(
        #     wf[wf.index(SlaterDet(
        #         c=0.0,
        #         alpha_occ=int_array(2,6,7),  # 0a 1a 5a 0b 1b 5b -> 2a 6a 7a 2b 6a 7a
        #         beta_occ=int_array(2,6,7)))],
        #     () # 0a 0b -> 2a 2b; 1a 1b -> 6a 6b; 5a 5b -> 7a 7b
        #     () # 0a 0b -> 2a 2b; 1a 1b -> 6a 7b; 5a 5b -> 7a 6b
        #     () # 0a 0b -> 2a 2b; 1a 1b -> 7a 6b; 5a 5b -> 6a 7b
        #     () # 0a 0b -> 2a 2b; 1a 1b -> 7a 7b; 5a 5b -> 6a 6b
        #     () # 0a 0b -> 6a 6b; 1a 1b -> 2a 2b; 5a 5b -> 7a 7b
        #     () # 0a 0b -> 6a 7b; 1a 1b -> 2a 2b; 5a 5b -> 7a 6b
        #     () # 0a 0b -> 7a 6b; 1a 1b -> 2a 2b; 5a 5b -> 6a 7b
        #     () # 0a 0b -> 6a 6b; 1a 1b -> 7a 7b; 5a 5b -> 2a 2b
        #     () # 0a 0b -> 7a 7b; 1a 1b -> 6a 6b; 5a 5b -> 2a 2b
        #     () # 0a 0b -> 7a 6b; 1a 1b -> 6a 7b; 5a 5b -> 2a 2b
        #     () # 0a 0b -> 6a 7b; 1a 1b -> 7a 6b; 5a 5b -> 2a 2b
        #     () # 0a 0b -> 7a 7b; 1a 1b -> 2a 2b; 5a 5b -> 7a 7b

        #     () # 0a 1b -> 2a 2b; 1a 0b -> 6a 6b; 5a 5b -> 7a 7b
        #     () # 0a 1b -> 2a 2b; 1a 0b -> 6a 7b; 5a 5b -> 7a 6b
        #     () # 0a 1b -> 2a 2b; 1a 0b -> 7a 6b; 5a 5b -> 6a 7b
        #     () # 0a 1b -> 2a 2b; 1a 0b -> 7a 7b; 5a 5b -> 6a 6b
        #     () # 0a 1b -> 6a 6b; 1a 0b -> 2a 2b; 5a 5b -> 7a 7b
        #     () # 0a 1b -> 6a 7b; 1a 0b -> 2a 2b; 5a 5b -> 7a 6b
        #     () # 0a 1b -> 7a 6b; 1a 0b -> 2a 2b; 5a 5b -> 6a 7b
        #     () # 0a 1b -> 6a 6b; 1a 0b -> 7a 7b; 5a 5b -> 2a 2b
        #     () # 0a 1b -> 7a 7b; 1a 0b -> 6a 6b; 5a 5b -> 2a 2b
        #     () # 0a 1b -> 7a 6b; 1a 0b -> 6a 7b; 5a 5b -> 2a 2b
        #     () # 0a 1b -> 6a 7b; 1a 0b -> 7a 6b; 5a 5b -> 2a 2b
        #     () # 0a 1b -> 7a 7b; 1a 0b -> 2a 2b; 5a 5b -> 7a 7b
            
        #     () # 0a 5b -> .....


@tests.category('SHORT')
class FromIntermNormCCSDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_h2_631g_c2v(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__631g__C2v')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__C2v'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)

    def test_h2_631g_d2h(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__631g__D2h')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)

    def test_h2_ccpvdz_c2v(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__631g__C2v')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__C2v'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)

    def test_li2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__sto3g__D2h'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)

    def test_li2_to2s_c2v(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__to2s__C2v')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__to2s__C2v'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)

    def test_li2_to3s_c2v(self):
        wf = FCIWaveFunction.from_int_norm(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__to3s__C2v')))
        wf_from_molpro_fci = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__to3s__C2v'))
        wf_from_molpro_fci.normalise(mode='intermediate')
        self.assertEqual(wf._coefficients,
                         wf_from_molpro_fci._coefficients)
    
    def test_he2_sto3g_d2h(self):
        wfcc = IntermNormWaveFunction.from_Molpro(
            tests.CCSD_file('He2__1.5__631g__C2v'))
        wf = FCIWaveFunction.from_int_norm(wfcc)
        self.assertEqual(wf[0, 2], -0.00344764)


@tests.category('SHORT', 'ESSENTIAL')
class ExcInfoTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.wf = FCIWaveFunction.from_Molpro_FCI(
            'tests/inputs_outputs/h2o__Req__sto3g__C2v/FCI_allE.out')

    def test_get_exc_1(self):
        rank, alpha_hp, beta_hp = self.wf.get_exc_info(
            SlaterDet(c=0.0,
                      alpha_occ=int_array(0,1,2,4,8),
                      beta_occ=int_array(0,1,2,4,5)))
        self.assertEqual(rank, 1)
        self.assertEqual(alpha_hp[0], int_array(5))
        self.assertEqual(alpha_hp[1], int_array(8))
        self.assertEqual(beta_hp[0], int_array())
        self.assertEqual(beta_hp[1], int_array())

    def test_get_exc_2(self):
        rank, alpha_hp, beta_hp = self.wf.get_exc_info(
            SlaterDet(c=0.0,
                      alpha_occ=int_array(0,1,2,4,8),
                      beta_occ=int_array(0,1,4,5,9)))
        self.assertEqual(rank, 2)
        self.assertEqual(alpha_hp[0], int_array(5))
        self.assertEqual(alpha_hp[1], int_array(8))
        self.assertEqual(beta_hp[0], int_array(2))
        self.assertEqual(beta_hp[1], int_array(9))

    def test_get_exc_3(self):
        rank, alpha_hp, beta_hp = self.wf.get_exc_info(
            SlaterDet(c=0.0,
                      alpha_occ=int_array(0,4,7,8,11),
                      beta_occ=int_array(0,1,4,5,9)))
        self.assertEqual(rank, 4)
        self.assertEqual(alpha_hp[0], int_array(1,2,5))
        self.assertEqual(alpha_hp[1], int_array(7,8,11))
        self.assertEqual(beta_hp[0], int_array(2))
        self.assertEqual(beta_hp[1], int_array(9))


@tests.category('SHORT')
class NormTestCase(unittest.TestCase):

    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise()
        self.assertAlmostEqual(norm(wf._coefficients), 1.0)
        wf.normalise(mode='intermediate')
        self.assertAlmostEqual(wf.C0, 1.0)
        wf.normalise()
        S = 0.0
        for det in wf:
            S += det.c**2
        S = sqrt(S)
        self.assertAlmostEqual(S, 1.0)
 
    def test_h2_631g_c2v(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__C2v'))
        wf.normalise()
        self.assertAlmostEqual(norm(wf._coefficients), 1.0)
        wf.normalise(mode='intermediate')
        self.assertAlmostEqual(wf.C0, 1.0)
        wf.normalise()
        S = 0.0
        for det in wf:
            S += det.c**2
        S = sqrt(S)
        self.assertAlmostEqual(S, 1.0)
 
    def test_h2_ccpvdz_c2v(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__ccpVDZ__C2v'))
        wf.normalise()
        self.assertAlmostEqual(norm(wf._coefficients), 1.0)
        wf.normalise(mode='intermediate')
        self.assertAlmostEqual(wf.C0, 1.0)
        wf.normalise()
        S = 0.0
        for det in wf:
            S += det.c**2
        S = sqrt(S)
        self.assertAlmostEqual(S, 1.0)

    def test_hcl_plus_631g_c2v(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('HCl_plus__1.5__631g__C2v'))
        wf.normalise()
        self.assertAlmostEqual(norm(wf._coefficients), 1.0)
        wf.normalise(mode='intermediate')
        self.assertAlmostEqual(wf.C0, 1.0)
        wf.normalise()
        S = 0.0
        for det in wf:
            S += det.c**2
        S = sqrt(S)
        self.assertAlmostEqual(S, 1.0)
