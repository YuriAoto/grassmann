"""Integration tests for coupled_cluster.dist_to_fci

"""
import sys
import unittest
import copy

import numpy as np

from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.fci import FCIWaveFunction
from wave_functions.general import WaveFunction
from coupled_cluster.dist_to_fci import (vertical_dist_to_cc_manifold,
                                         calc_dist_to_cc_manifold)

from tests import FCI_file, CCSD_file
import tests


fout = None


@tests.category('SHORT')
class VertDistTwoElecCCDTestCase(unittest.TestCase):
    """Vertical distance from FCI to the CCD manifold for a two electron system
    
    These FCI have no contribution of singles, thus CCD is potentially exact,
    and the distance is zero
    """
    
    def test_h2_sto3g_d2h(self):
        mol_system = 'H2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro(FCI_file(mol_system))
        res = vertical_dist_to_cc_manifold(wf, level="D")
        self.assertEqual(res.distance, 0.0)

    def test_h2_631g_d2h_noS(self):
        wf = FCIWaveFunction.from_Molpro(FCI_file('H2__5__631g__D2h'))
        for i in range(1, wf.shape[0]):
            wf[i, 0] = 0.0
        for i in range(1, wf.shape[1]):
            wf[0, i] = 0.0
        wf.normalise(mode='intermediate')
        res = vertical_dist_to_cc_manifold(wf, level="D")        
        self.assertEqual(res.distance, 0.0)


class VertDistTwoElecCCSDTestCase(unittest.TestCase):
    """Vertical distance from FCI to the CCSD manifold for a two electron system
    
    The CCSD is potentially exact, and the distance is zero
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
    
    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        mol_system = 'H2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro(FCI_file(mol_system))
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(CCSD_file(mol_system)))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_h2_631g_d2h(self):
        mol_system = 'H2__5__631g__D2h'
        wf = FCIWaveFunction.from_Molpro(FCI_file(mol_system))
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(CCSD_file(mol_system)))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        mol_system = 'H2__5__ccpVDZ__D2h'
        wf = FCIWaveFunction.from_Molpro(tests.FCI_file(mol_system))
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(mol_system)))
        res = vertical_dist_to_cc_manifold(wf)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('LONG')
    def test_li2_sto3g_d2h(self):
        mol_system = 'Li2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro(FCI_file(mol_system))
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(mol_system)))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('VERY LONG')
    def test_li2_to2s_c2h(self):
        mol_system = 'Li2__5__to2s__C2v'
        wf = FCIWaveFunction.from_Molpro(FCI_file(mol_system))
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(mol_system)))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('VERY LONG')
    def test_li2_to3s_c2h(self):
        mol_system = 'Li2__5__to3s__C2v'
        wf = FCIWaveFunction.from_Molpro(FCI_file(mol_system))
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(mol_system)))
        res = vertical_dist_to_cc_manifold(wf)
        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


def _calc_vertdist(molecular_system, level, allE=False):
    """Helper for tests of VertDist"""
    system_file = (tests.CCD_file(molecular_system, allE=allE)
                   if level == 'D' else
                   tests.CCSD_file(molecular_system, allE=allE))
    cc_wf = IntermNormWaveFunction.unrestrict(
        IntermNormWaveFunction.from_Molpro(system_file))
    wf = FCIWaveFunction.from_interm_norm(cc_wf)
    res = vertical_dist_to_cc_manifold(wf, level=level)
    explicitly_dist = FCIWaveFunction.from_interm_norm(res.wave_function).dist_to(wf)
    return res, explicitly_dist, cc_wf

class VertDistCCDwfCCDTestCase(unittest.TestCase):
    """The vertical distance from a CCD wave function to the CCD manifold
    
    Since the wave function belongs to the manifold, all coefficients
    should be in the right directions and the distance should be zero
    """
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__D2h', level='D')
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Be__at__sto3g__D2h', level='D', allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__D2h', level='D', allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('LONG')
    def test_be_ccpvdz_d2h_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Be__at__ccpVDZ__D2h', level='D', allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        
    @tests.category('VERY LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__C2v', level='D', allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('VERY LONG')
    def test_li2_sto3g_c1_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__C1', level='D', allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        self.assertEqual(res.wave_function, cc_wf)


class VertDistCCSDwfCCSDTestCase(unittest.TestCase):
    """The vertical distance from a CCSD wave function to the CCSD manifold
    
    Since the wave function belongs to the manifold, all coefficients
    should be in the right directions and the distance should be zero
    """
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__D2h',
                                                     level='SD')
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Be__at__sto3g__D2h',
                                                     level='SD',
                                                     allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)

    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__D2h',
                                                     level='SD',
                                                     allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('LONG')
    def test_be_ccpvdz_d2h_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Be__at__ccpVDZ__D2h',
                                                     level='SD',
                                                     allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
        
    @tests.category('VERY LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__C2v',
                                                     level='SD',
                                                     allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('VERY LONG')
    def test_li2_sto3g_c1_allel(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('Li2__5__sto3g__C1',
                                                     level='SD',
                                                     allE=True)
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)
    
    @tests.category('VERY LONG')
    def test_hclplus_631g_c2v(self):
        res, explicitly_dist, cc_wf = _calc_vertdist('HCl_plus__1.5__631g__C2v',
                                                     level='SD')
        for k in res.right_dir:
            with self.subTest(rank=k):
                self.assertEqual(res.right_dir[k][0], res.right_dir[k][1])
        self.assertAlmostEqual(res.distance, 0.0, places=9)
        self.assertAlmostEqual(explicitly_dist, 0.0, places=9)


def _calc_mindist_twoel(molsys, level, diag_hess=False, factor=None):
    """Helper to MinDistTwoElec*TestCase"""
    wf = FCIWaveFunction.from_Molpro(FCI_file(molsys))
    if '631g' in molsys and level == 'D':
        wf[0, 1] = 0.0
        wf[1, 0] = 0.0
        wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.from_projected_fci(wf, wf_type='CCD')
    elif level == 'D':
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(tests.CCD_file(molsys)))
    else:
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(tests.CCSD_file(molsys)))
    if factor is None:
        ini_wf = None
    else:
        ini_wf = IntermNormWaveFunction.unrestrict(cc_wf)
        ini_wf *= factor
    res = calc_dist_to_cc_manifold(wf,
                                   level=level,
                                   f_out=fout,
                                   ini_wf=ini_wf,
                                   diag_hess=diag_hess,
                                   max_iter=20)
    return res, cc_wf


class MinDistTwoElecCCDTestCase(unittest.TestCase):
    """Calculates the minDist(CCD) to FCI for two electron systems
    
    These FCI have no contribution of singles, thus CCD is potentially exact,
    and we should obtain the CCD wave function and zero distance.
    
    This starts from the CCD wave function, thus already there and we should
    have one iteration only
    
    """
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)

    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__D2h', level='D')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)

    @tests.category('SHORT')
    def test_h2_631g_d2h_noS(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__D2h', level='D')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistTwoElecCCDOptDiagHessTestCase(unittest.TestCase):
    """Calculates the minDist(CCD) to FCI for two electron systems
    
    These FCI have no contribution of singles, thus CCD is potentially exact,
    and we should obtain the CCD wave function and zero distance.
    
    We start from a multiple of the exact amplitudes.
    
    The diagonal approximation for the Hessian is used.
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 2.0
    
    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__D2h',
                                         level='D',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_d2h_noS(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__D2h',
                                         level='D',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistTwoElecCCDOptTestCase(unittest.TestCase):
    """Calculates the minDist(CCD) to FCI for two electron systems
    
    These FCI have no contribution of singles, thus CCD is potentially exact,
    and we should obtain the CCD wave function and zero distance.
    
    We start from a multiple of the exact amplitudes.
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 4.0
    
    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__D2h',
                                         level='D',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_d2h_noS(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__D2h',
                                         level='D',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistTwoElecCCSDTestCase(unittest.TestCase):
    """Calculates the minDist(CCSD) to FCI for two electron systems
    
    Since CCSD is potentially exact, we should obtain the CCSD wave
    function and zero distance.
    
    This starts from the CCSD wave function, thus already there and we should
    have one iteration only
    
    """
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        
    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__D2h', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_sto3g_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__C1', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__D2h', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_c2v(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__C2v', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__C1', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__D2h', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0E-6, rtol=1.0E-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_c2v(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__C2v', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0E-6, rtol=1.0E-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_cs(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__Cs', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0E-6, rtol=1.0E-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__C1', level='SD')
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0E-6, rtol=1.0E-5)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistTwoElecCCSDOptDiagHessTestCase(unittest.TestCase):
    """Calculates the minDist(CCSD) to FCI for two electron systems
    
    Since CCSD is potentially exact, we should obtain the CCSD wave
    function and zero distance.
    
    We start from a multiple of the exact amplitudes.
    
    The diagonal approximation for the Hessian is used.
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 4.0
    
    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__D2h',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_sto3g_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__C1',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__D2h',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_c2v(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__C2v',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__C1',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__D2h',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_c2v(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__C2v',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_cs(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__Cs',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__C1',
                                         level='SD',
                                         diag_hess=True,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistTwoElecCCSDOptTestCase(unittest.TestCase):
    """Calculates the minDist(CCSD) to FCI for two electron systems
    
    Since CCSD id potentially exact, we should obtain the CCSD wave
    function and zero distance
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 4.0
    
    @tests.category('SHORT')
    def test_h2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__D2h',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_sto3g_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__sto3g__C1',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__D2h',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_c2v(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__C2v',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_631g_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__631g__C1',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_d2h(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__D2h',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_c2v(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__C2v',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_cs(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__Cs',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2_ccpvdz_c1(self):
        res, cc_wf = _calc_mindist_twoel('H2__5__ccpVDZ__C1',
                                         level='SD',
                                         diag_hess=False,
                                         factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        res.wave_function.set_eq_tol(atol=1.0e-6, rtol=1.0e-5)
        self.assertEqual(res.wave_function, cc_wf)


def _calc_mindist(molecular_system, level, diag_hess, factor=None, allE=False):
    """Helper for tests of MinDist"""
    system_file = (tests.CCD_file(molecular_system, allE=allE)
                   if level == 'D' else
                   tests.CCSD_file(molecular_system, allE=allE))
    cc_wf = IntermNormWaveFunction.unrestrict(
        IntermNormWaveFunction.from_Molpro(system_file))
    wf = FCIWaveFunction.from_interm_norm(cc_wf)
    if factor is None:
        res = calc_dist_to_cc_manifold(wf,
                                       level=level,
                                       f_out=fout,
                                       diag_hess=diag_hess,
                                       max_iter=20)
    else:
        ini_wf = IntermNormWaveFunction.unrestrict(cc_wf)
        ini_wf *= factor
        res = calc_dist_to_cc_manifold(wf,
                                       level=level,
                                       f_out=fout,
                                       diag_hess=diag_hess,
                                       ini_wf=ini_wf,
                                       max_iter=20)
    return res, cc_wf


class MinDistCCDwfCCDTestCase(unittest.TestCase):
    """The minimun distance from a CCD wave function to the CCD manifold
    
    Since the wave function belongs to the manifold, the distance should
    be zero and we recover the CCD wave function.
    
    This starts from the CCD wave function, thus already there and we should
    have one iteration only
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='D', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='D', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='D', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='D', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='D', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_be_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='D', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_ccpvdz_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__ccpVDZ__D2h',
                                   level='D', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='D', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='D', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v(self):
        res, cc_wf = _calc_mindist('He2__1.5__631g__C2v',
                                   level='D', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistCCDwfCCDOptTestCase(unittest.TestCase):
    """The minimun distance from a CCD wave function to the CCD manifold
    
    Since the wave function belongs to the manifold, the distance should
    be zero and we recover the CCD wave function.
    
    We start from a multiple of the exact amplitudes.
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 3.0
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='D', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='D', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='D', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='D', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='D', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='D', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_ccpvdz_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__ccpVDZ__D2h',
                                   level='D', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='D', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='D', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v(self):
        res, cc_wf = _calc_mindist('He2__1.5__631g__C2v',
                                   level='D', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    

class MinDistCCDwfCCDOptDiagHessTestCase(unittest.TestCase):
    """The minimun distance from a CCD wave function to the CCD manifold
    
    Since the wave function belongs to the manifold, the distance should
    be zero and we recover the CCD wave function.
    
    We start from a multiple of the exact amplitudes.
        
    The diagonal approximation for the Hessian is used
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 3.0
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='D', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='D', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='D', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='D', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='D', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='D', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_ccpvdz_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__ccpVDZ__D2h',
                                   level='D', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='D', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='D', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v(self):
        res, cc_wf = _calc_mindist('He2__1.5__631g__C2v',
                                   level='D', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistCCSDwfCCSDTestCase(unittest.TestCase):
    """The minimun distance from a CCSD wave function to the CCSD manifold
    
    Since the wave function belongs to the manifold, the distance should
    be zero and we recover the CCSD wave function.
    
    This starts from the CCSD wave function, thus already there and we should
    have one iteration only
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='SD', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='SD', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='SD', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='SD', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='SD', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_be_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='SD', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_ccpvdz_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__ccpVDZ__D2h',
                                   level='SD', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='SD', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='SD', diag_hess=False,
                                   allE=True)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v(self):
        res, cc_wf = _calc_mindist('He2__1.5__631g__C2v',
                                   level='SD', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('VERY LONG')
    def test_hclplus_631g_c2v(self):
        res, cc_wf = _calc_mindist('HCl_plus__1.5__631g__C2v',
                                   level='SD', diag_hess=False)
        self.assertEqual(res.n_iter, 1)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistCCSDwfCCSDOptTestCase(unittest.TestCase):
    """The minimun distance from a CCSD wave function to the CCSD manifold
    
    Since the wave function belongs to the manifold, the distance should
    be zero and we recover the CCSD wave function.
    
    We start from a multiple of the exact amplitudes.
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 3.0
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='SD', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='SD', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='SD', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='SD', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='SD', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='SD', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_ccpvdz_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__ccpVDZ__D2h',
                                   level='SD', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='SD', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='SD', diag_hess=False,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v(self):
        res, cc_wf = _calc_mindist('He2__1.5__631g__C2v',
                                   level='SD', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('VERY LONG')
    def test_hclplus_631g_c2v(self):
        res, cc_wf = _calc_mindist('HCl_plus__1.5__631g__C2v',
                                   level='SD', diag_hess=False,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)


class MinDistCCSDwfCCSDOptDiagHessTestCase(unittest.TestCase):
    """The minimun distance from a CCSD wave function to the CCSD manifold
    
    Since the wave function belongs to the manifold, the distance should
    be zero and we recover the CCSD wave function.
    
    We start from a multiple of the exact amplitudes.
    
    The diagonal approximation for the Hessian is used.
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(IntermNormWaveFunction, tests.assert_wave_functions)
        self.factor = 3.0
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='SD', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_li2_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='SD', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__D2h',
                                   level='SD', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('LONG')
    def test_li2_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('Li2__5__sto3g__C2v',
                                   level='SD', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='SD', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__sto3g__D2h',
                                   level='SD', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_be_ccpvdz_d2h_allel(self):
        res, cc_wf = _calc_mindist('Be__at__ccpVDZ__D2h',
                                   level='SD', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='SD', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_h2o_15_sto3g_c2v_allel(self):
        res, cc_wf = _calc_mindist('h2o__1.5__sto3g__C2v',
                                   level='SD', diag_hess=True,
                                   factor=self.factor,
                                   allE=True)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v(self):
        res, cc_wf = _calc_mindist('He2__1.5__631g__C2v',
                                   level='SD', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
    
    @tests.category('VERY LONG')
    def test_hclplus_631g_c2v(self):
        res, cc_wf = _calc_mindist('HCl_plus__1.5__631g__C2v',
                                   level='SD', diag_hess=True,
                                   factor=self.factor)
        self.assertAlmostEqual(res.distance, 0.0)
        self.assertEqual(res.wave_function, cc_wf)
