"""Tests for fci

"""
import unittest

import numpy as np

import tests
from wave_functions import fci
from util.other import int_array
from wave_functions.slater_det import SlaterDet


class ExcInfoTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.wf = fci.FCIWaveFunction.from_Molpro_FCI(
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

