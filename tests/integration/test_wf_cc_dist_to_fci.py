"""Integration tests for wave_functions.fci

"""
import sys
import unittest

import numpy as np

from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.fci import FCIWaveFunction
from coupled_cluster.dist_to_fci import (vertical_proj_to_cc_manifold,
                                         calc_dist_to_cc_manifold)

import tests


class VertDistTwoElecCCDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCD_file('H2__5__sto3g__D2h'))
        res = vertical_proj_to_cc_manifold(wf, level="D", restore_wf=False)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_h2_631g_d2h_noS(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        wf._coefficients[0, 1:] = 0.0
        wf._coefficients[1:, 0] = 0.0
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, level="D", restore_wf=False)        
        self.assertEqual(res.distance, 0.0)

class VertDistTwoElecCCSDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCSD_file('H2__5__sto3g__D2h'))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_h2_631g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__631g__D2h')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_h2_ccpvdz_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__ccpVDZ__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__ccpVDZ__D2h')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        np.testing.assert_almost_equal(res.wave_function.amplitudes,
                                       cc_wf.amplitudes, 6)
#        print(res.wave_function.amplitudes- cc_wf.amplitudes)
#        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_li2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_li2_to2s_c2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__to2s__C2v'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__to2s__C2v')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_li2_to3s_c2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__to3s__C2v'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__to3s__C2v')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)


class MinDistTwoElecCCDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCD_file('H2__5__sto3g__D2h'))
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=sys.stdout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    def test_h2_631g_d2h_noS(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        wf._coefficients[0, 1] = 0.0
        wf._coefficients[1, 0] = 0.0
        wf.normalise(mode='intermediate')
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=sys.stdout)
        self.assertEqual(res.distance, 0.0)
