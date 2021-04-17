"""

"""
import unittest

import numpy as np

import tests
from util.other import int_array
from wave_functions.fci import FCIWaveFunction
from wave_functions.slater_det import get_slater_det_from_excitation

@tests.category('SHORT', 'ESSENTIAL')
class ExcInfoTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.wf = FCIWaveFunction.from_Molpro_FCI(
            'tests/inputs_outputs/h2o__Req__sto3g__C2v/FCI_allE.out')

    def test_1(self):
        alpha_hp = (int_array(5), int_array(8))
        beta_hp = (int_array(), int_array())
        det = get_slater_det_from_excitation(self.wf.ref_det, 0.0, alpha_hp, beta_hp)
        rank, new_alpha_hp, new_beta_hp = self.wf.get_exc_info(det)
        self.assertEqual(alpha_hp[0], new_alpha_hp[0])
        self.assertEqual(alpha_hp[1], new_alpha_hp[1])
        self.assertEqual(beta_hp[0], new_beta_hp[0])
        self.assertEqual(beta_hp[1], new_beta_hp[1])

    def test_2(self):
        alpha_hp = (int_array(5), int_array(8))
        beta_hp = (int_array(2), int_array(9))
        det = get_slater_det_from_excitation(self.wf.ref_det, 0.0, alpha_hp, beta_hp)
        rank, new_alpha_hp, new_beta_hp = self.wf.get_exc_info(det)
        self.assertEqual(alpha_hp[0], new_alpha_hp[0])
        self.assertEqual(alpha_hp[1], new_alpha_hp[1])
        self.assertEqual(beta_hp[0], new_beta_hp[0])
        self.assertEqual(beta_hp[1], new_beta_hp[1])

    def test_3(self):
        alpha_hp = (int_array(1,2,5), int_array(7,8,11))
        beta_hp = (int_array(2), int_array(9))
        det = get_slater_det_from_excitation(self.wf.ref_det, 0.0, alpha_hp, beta_hp)
        rank, new_alpha_hp, new_beta_hp = self.wf.get_exc_info(det)
        self.assertEqual(alpha_hp[0], new_alpha_hp[0])
        self.assertEqual(alpha_hp[1], new_alpha_hp[1])
        self.assertEqual(beta_hp[0], new_beta_hp[0])
        self.assertEqual(beta_hp[1], new_beta_hp[1])

