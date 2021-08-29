"""Integration tests for wave_functions.fci

"""
import sys
import unittest

import numpy as np

from util.other import int_array
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.slater_det import SlaterDet
from wave_functions.fci import FCIWaveFunction
from coupled_cluster.dist_to_fci import (vertical_proj_to_cc_manifold,
                                         calc_dist_to_cc_manifold)

import tests


fout = None

@tests.category('SHORT')
class VertDistTwoElecCCDTestCase(unittest.TestCase):

    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCD_file('H2__5__sto3g__D2h'))
        res = vertical_proj_to_cc_manifold(wf, level="D", restore_wf=False)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    def test_h2_631g_d2h_noS(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        for i in range(1, wf.shape[0]):
            wf[i, 0] = 0.0
        for i in range(1, wf.shape[1]):
            wf[0, i] = 0.0
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, level="D", restore_wf=False)        
        self.assertEqual(res.distance, 0.0)

class VertDistTwoElecCCSDTestCase(unittest.TestCase):

    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCSD_file('H2__5__sto3g__D2h'))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_h2_631g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__631g__D2h')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__ccpVDZ__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('H2__5__ccpVDZ__D2h')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_li2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('VERY LONG')
    def test_li2_to2s_c2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__to2s__C2v'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__to2s__C2v')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('VERY LONG')
    def test_li2_to3s_c2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('Li2__5__to3s__C2v'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__to3s__C2v')))
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False)        
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class VertDistCCDwfCCDTestCase(unittest.TestCase):

    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=True, level='D')
        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=True, level='D')
        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('VERY LONG')
    def test_li2_sto3g_c2v_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__C2v', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=True, level='D')
        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('VERY LONG')
    def test_li2_sto3g_c1_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__C1', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=True, level='D')
        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Be__at__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=True, level='D')
        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_be_ccpvdz_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Be__at__ccpVDZ__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=True, level='D')
        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)


class VertDistCCSDwfCCSDTestCase(unittest.TestCase):
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False, level='SD')
##        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
##        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False, level='SD')
##        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
##        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_be_ccpvdz_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Be__at__ccpVDZ__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = vertical_proj_to_cc_manifold(wf, restore_wf=False, level='SD')
##        explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
#        print(res.right_dir)
#        print(res.wave_function)
#        print(cc_wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
##        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistTwoElecCCDTestCase(unittest.TestCase):

    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCD_file('H2__5__sto3g__D2h'))
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_h2_631g_d2h_noS(self):
        wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__631g__D2h'))
        wf[0, 1] = 0.0
        wf[1, 0] = 0.0
        wf.normalise(mode='intermediate')
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)


class MinDistCCDwfCCDTestCase(unittest.TestCase):

    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = calc_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_li2_sto3g_d2h_ini_0(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        ini_cc = IntermNormWaveFunction.similar_to(wf, 'CCD', False)
        res = calc_dist_to_cc_manifold(wf, level="D",
                                       ini_wf=ini_cc, f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__C2v')))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = calc_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


    @unittest.skip('This is not working corretly...')
    @tests.category('LONG')
    def test_li2_sto3g_d2h_ini_half(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        ini_cc = IntermNormWaveFunction.unrestrict(cc_wf)
        ini_cc.amplitudes *= 0.9
        res = calc_dist_to_cc_manifold(wf, level="D",
                                       ini_wf=ini_cc, f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__C2v', allE=True)))
        wf = FCIWaveFunction.from_interm_norm(cc_wf)
        wf.normalise(mode='intermediate')
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
