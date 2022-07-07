"""Checks for CC manifold

"""
import unittest


import numpy as np

import tests
from wave_functions.fci import FCIWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from coupled_cluster.manifold import min_dist_jac_hess, min_dist_jac_hess_num
from coupled_cluster.dist_to_fci import (calc_all_distances,
                                         calc_dist_to_cc_manifold,
                                         vertical_dist_to_cc_manifold)


def _print_hess(H, Hnum):
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            print(f'{i:3d} {j:3d} : {H[i, j]:20.10f}  {HNum[i, j]:20.10f}'
                  + f'  {H[i, j]-HNum[i, j]:20.10f}')


def _print_coef(wf):
    print('---------------')
    print('wf coef:')
    for i in range(wf._coefficients.shape[0]):
        for j in range(wf._coefficients.shape[1]):
            print(f' {cc_wf_as_fci._coefficients[i, j]:12.7f} ', end='')
        print()
    print('---------------')


def _calc_anal_num_jac_hess(mol_system, allE, wf_type, factor=1.0):
    """Helper to calculate compare analitical and numerical Jacobian and Hessian
    
    Parameters:
    -----------
    mol_system (str)
        A molecular system (basically a directory of tests/inputs_outputs)
    
    allE (bool)
        True for all electron case
    
    wf_type (str)
        'CCD' or 'CCSD'
    
    factor (float, optional, default=1.0)
        A factor to multiply all amplitudes of the CCD or CCSD wave function
        (just to test in several cases)
    
    
    """
    wf = FCIWaveFunction.from_Molpro(tests.FCI_file(mol_system, allE=allE))
    wf.normalise(mode='intermediate')
    tests.logger.info('wf from Molpro, interm norm:\n%s', wf)
    wf.set_max_coincidence_orbitals()
    tests.logger.info('wf with max coincidence orbitals:\n%s', wf)
    cc_wf = IntermNormWaveFunction.from_projected_fci(wf, wf_type=wf_type)
    cc_wf *= factor
    cc_wf_as_fci = FCIWaveFunction.from_interm_norm(cc_wf, ordered_orbitals=True)
    tests.logger.info('cc wf:\n%s\nas FCI\n%s', cc_wf, cc_wf_as_fci)
    wf.set_ordered_orbitals()
    Jac = np.empty(cc_wf.n_indep_ampl)
    Hess = np.empty((cc_wf.n_indep_ampl, cc_wf.n_indep_ampl))
    min_dist_jac_hess(
        wf,
        cc_wf_as_fci,
        Jac,
        Hess,
        diag_hess=False,
        level=wf_type[2:])
    JacNum, HessNum = min_dist_jac_hess_num(
        wf,
        cc_wf,
        np.array(Jac)*2,
        np.array(Hess)*2,
        eps = 0.0002)
    return np.array(Jac), np.array(JacNum)/2, np.array(Hess), np.array(HessNum)/2


class MinDVertExplicitDistTestCase(unittest.TestCase):

    @tests.category('SHORT')
    def test_li2_sto3g_d2h_ccd(self):
        level = 'D'
        mol_system = 'Li2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro(tests.FCI_file(mol_system, allE=True))
        res_vert = vertical_dist_to_cc_manifold(wf, level=level)
        cc_wf = res_vert.wave_function
        res_min_d = calc_dist_to_cc_manifold(wf, level=level, ini_wf=cc_wf)
        res_all_dists = calc_all_distances(wf, res_vert, res_min_d,
                                           cc_wf=None, ci_wf=None, level=level,
                                           explicit_calcs=True)
        self.assertAlmostEqual(res_all_dists.FCI__minD,
                               res_all_dists.FCI__minD_expl)
        self.assertAlmostEqual(res_all_dists.FCI__vertCC,
                               res_all_dists.FCI__vertCC_expl)

    @tests.category('SHORT')
    def test_li2_sto3g_d2h_ccsd(self):
        level = 'SD'
        mol_system = 'Li2__5__sto3g__D2h'
        wf = FCIWaveFunction.from_Molpro(tests.FCI_file(mol_system, allE=True))
        res_vert = vertical_dist_to_cc_manifold(wf, level=level)
        cc_wf = res_vert.wave_function
        res_min_d = calc_dist_to_cc_manifold(wf, level=level, ini_wf=cc_wf)
        res_all_dists = calc_all_distances(wf, res_vert, res_min_d,
                                           cc_wf=None, ci_wf=None, level=level,
                                           explicit_calcs=True)
        self.assertAlmostEqual(res_all_dists.FCI__minD,
                               res_all_dists.FCI__minD_expl)
        self.assertAlmostEqual(res_all_dists.FCI__vertCC,
                               res_all_dists.FCI__vertCC_expl)

    @tests.category('LONG')
    def test_he2_631g_d2h_ccd(self):
        level = 'D'
        mol_system = 'He2__1.5__631g__D2h'
        wf = FCIWaveFunction.from_Molpro(tests.FCI_file(mol_system))
        res_vert = vertical_dist_to_cc_manifold(wf, level=level)
        cc_wf = res_vert.wave_function
        res_min_d = calc_dist_to_cc_manifold(wf, level=level,
                                             ini_wf=IntermNormWaveFunction.unrestrict(cc_wf))
        res_all_dists = calc_all_distances(wf, res_vert, res_min_d,
                                           cc_wf=None, ci_wf=None, level=level,
                                           explicit_calcs=True)
        self.assertAlmostEqual(res_all_dists.FCI__minD,
                               res_all_dists.FCI__minD_expl)
        self.assertAlmostEqual(res_all_dists.FCI__vertCC,
                               res_all_dists.FCI__vertCC_expl)

    @tests.category('SHORT')
    def test_he2_631g_d2h_ccsd(self):
        level = 'SD'
        mol_system = 'He2__1.5__631g__D2h'
        wf = FCIWaveFunction.from_Molpro(tests.FCI_file(mol_system))
        res_vert = vertical_dist_to_cc_manifold(wf, level=level)
        cc_wf = res_vert.wave_function
        res_min_d = calc_dist_to_cc_manifold(wf, level=level, ini_wf=cc_wf)
        res_all_dists = calc_all_distances(wf, res_vert, res_min_d,
                                           cc_wf=None, ci_wf=None, level=level,
                                           explicit_calcs=True)
        self.assertAlmostEqual(res_all_dists.FCI__minD,
                               res_all_dists.FCI__minD_expl)
        self.assertAlmostEqual(res_all_dists.FCI__vertCC,
                               res_all_dists.FCI__vertCC_expl)



class CheckNumAnalJacHessTestCase(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        
    @tests.category('SHORT')
    def test_li2_sto3g_d2h_ccsd(self):
        mol_system = 'Li2__5__sto3g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('SHORT')
    def test_li2_sto3g_d2h_ccd(self):
        mol_system = 'Li2__5__sto3g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_ccd(self):
        mol_system = 'Be__at__sto3g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('SHORT')
    def test_be_sto3g_d2h_ccd_allel(self):
        mol_system = 'Be__at__sto3g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        
    @tests.category('VERY LONG')
    def test_li2_sto3g_c2v_ccd_allel(self):
        mol_system = 'Li2__5__sto3g__C2v'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=3.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('VERY LONG')
    def test_li2_sto3g_d2h_ccd_allel(self):
        mol_system = 'Li2__5__sto3g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=3.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=True,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('SHORT')
    def test_he2_631g_d2h_ccsd(self):
        mol_system = 'He2__1.5__631g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('SHORT')
    def test_he2_631g_d2h_ccd(self):
        mol_system = 'He2__1.5__631g__D2h'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('SHORT')
    def test_he2_631g_c2v_ccd(self):
        mol_system = 'He2__1.5__631g__C2v'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('LONG')
    def test_h2o_sto3g_c2v_ccsd(self):
        mol_system = 'h2o__1.5__sto3g__C2v'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCSD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
    
    @tests.category('LONG')
    def test_h2o_sto3g_c2v_ccd(self):
        mol_system = 'h2o__1.5__sto3g__C2v'
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=1.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.2)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=0.6)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
        J, JNum, H, HNum = _calc_anal_num_jac_hess(mol_system,
                                                   allE=False,
                                                   wf_type='CCD',
                                                   factor=5.0)
        self.assertEqual(J, JNum)
        self.assertEqual(H, HNum)
