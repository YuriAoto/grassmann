"""Tests for fci

"""
import unittest

import numpy as np

import tests
from wave_functions.fci import FCIWaveFunction
from util.other import int_array
from wave_functions.slater_det import SlaterDet
from wave_functions.strings_rev_lexical_order import get_index


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

    def test_hcl_631g_c2v(self):
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



class FromIntermNormTestCase(unittest.TestCase):

    def test_1(self):
        pass


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

