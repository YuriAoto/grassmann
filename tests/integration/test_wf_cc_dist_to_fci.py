"""Integration tests for wave_functions.fci

"""
import sys
import unittest

import numpy as np

from util.other import int_array
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.slater_det import SlaterDet
from wave_functions.fci import FCIWaveFunction
from coupled_cluster.dist_to_fci import (vertical_dist_to_cc_manifold,
                                         calc_dist_to_cc_manifold)

from tests import FCI_file
import tests


fout = sys.stdout

@tests.category('SHORT')
class VertDistTwoElecCCDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_h2_sto3g_d2h(self):
        mol_system = 'H2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)

    def test_h2_631g_d2h_noS(self):
        mol_system = 'H2__5__631g__D2h'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        wf._coefficients[0, 1:] = 0.0
        wf._coefficients[1:, 0] = 0.0
        res = vertical_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)

class VertDistTwoElecCCSDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    @tests.category('VERY SHORT')
    def test_h2_sto3g_d2h(self):
        mol_system = 'H2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)

    @tests.category('VERY SHORT')
    def test_h2_631g_d2h(self):
        mol_system = 'H2__5__631g__D2h'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)

    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        mol_system = 'H2__5__ccpVDZ__D2h'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)

    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        mol_system = 'Li2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)

    @tests.category('SHORT')
    def test_li2_to2s_c2h(self):
        mol_system = 'Li2__5__to2s__C2v'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)

    @tests.category('SHORT')
    def test_li2_to3s_c2h(self):
        mol_system = 'Li2__5__to3s__C2v'
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)


class VertDistCCDwfCCDTestCase(unittest.TestCase):

    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        mol_system = 'Li2__5__sto3g__D2h'
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file(mol_system)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='D')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)

    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        mol_system = 'Be__at__sto3g__D2h'
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file(mol_system, allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='D')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        mol_system = 'Li2__5__sto3g__D2h'
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file(mol_system, allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='D')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('LONG')
    def test_be_ccpvdz_d2h_allel(self):
        mol_system = 'Be__at__ccpVDZ__D2h'
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file(mol_system, allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='D')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        
    @tests.category('VERY LONG')
    def test_li2_sto3g_c2v_allel(self):
        mol_system = 'Li2__5__sto3g__C2v'
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file(mol_system, allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='D')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)

    @tests.category('VERY LONG')
    def test_li2_sto3g_c1_allel(self):
        mol_system = 'Li2__5__sto3g__C1'
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file(mol_system, allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='D')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)


class VertDistCCSDwfCCSDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='SD')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='SD')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('LONG')
    def test_be_ccpvdz_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Be__at__ccpVDZ__D2h', allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = vertical_dist_to_cc_manifold(wf, level='SD')
        explicitly_dist = FCIWaveFunction.from_int_norm(res.wave_function).dist_to(wf)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)


class MinDistTwoElecCCDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file('H2__5__sto3g__D2h'))
        cc_wf = IntermNormWaveFunction.from_Molpro(
            tests.CCD_file('H2__5__sto3g__D2h'))
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('SHORT')
    def test_h2_631g_d2h_noS(self):
        wf = FCIWaveFunction.from_Molpro_FCI(FCI_file('H2__5__631g__D2h'))
        wf._coefficients[0, 1] = 0.0
        wf._coefficients[1, 0] = 0.0
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)


class MinDistCCDwfCCDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = calc_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('SHORT')
    def test_li2_sto3g_d2h_ini_0(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h')))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        ini_cc = IntermNormWaveFunction.similar_to(wf, 'CCD', False)
        res = calc_dist_to_cc_manifold(wf, level="D",
                                       ini_wf=ini_cc, f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__C2v')))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = calc_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)


##    @unittest.skip('This is not working corretly...')
    @tests.category('LONG')
    def test_li2_sto3g_d2h_ini_half(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        ini_cc = IntermNormWaveFunction.unrestrict(cc_wf)
        ini_cc.amplitudes *= 0.9
        res = calc_dist_to_cc_manifold(wf, level="D",
                                       ini_wf=ini_cc, f_out=fout,
                                       diag_hess=False)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)

    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCD_file('Li2__5__sto3g__C2v', allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        res = calc_dist_to_cc_manifold(wf, level="D", f_out=fout)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function.amplitudes, cc_wf.amplitudes)
