"""Checks for CC manifold

"""
import unittest


import numpy as np

import tests
from wave_functions.fci import FCIWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from coupled_cluster.manifold import min_dist_jac_hess, min_dist_jac_hess_num


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
    wf = FCIWaveFunction.from_Molpro_FCI(tests.FCI_file(mol_system, allE=allE))
    wf.normalise(mode='intermediate')
    # print('========================')
    # print('wf from Molpro, interm norm ')
    # print(wf)
    # for i in range(wf._coefficients.shape[0]):
    #     for j in range(wf._coefficients.shape[1]):
    #         print(f' {wf._coefficients[i, j]:12.7f} ', end='')
    #     print()
    # print('========================')
    wf.set_max_coincidence_orbitals()
    # print('========================')
    # print('wf with max coincidence orbs')
    # print(wf)
    # for i in range(wf._coefficients.shape[0]):
    #     for j in range(wf._coefficients.shape[1]):
    #         print(f' {wf._coefficients[i, j]:12.7f} ', end='')
    #     print()
    # print('========================')
    cc_wf = IntermNormWaveFunction.from_projected_fci(wf, wf_type=wf_type)
    cc_wf.amplitudes *= factor
    cc_wf_as_fci = FCIWaveFunction.from_int_norm(cc_wf, ordered_orbitals=True)
    # print(cc_wf)
    # print('========================')
    # print('cc wf')
    # print(cc_wf_as_fci)
    # for i in range(wf._coefficients.shape[0]):
    #     for j in range(wf._coefficients.shape[1]):
    #         print(f' {cc_wf_as_fci._coefficients[i, j]:12.7f} ', end='')
    #     print()
    # print('========================')
    wf.set_ordered_orbitals()
    Jac, Hess = min_dist_jac_hess(
        wf._coefficients,
        cc_wf_as_fci._coefficients,
        len(cc_wf),
        wf.orbs_before,
        wf.corr_orb.as_array(),
        wf.virt_orb.as_array(),
        wf._alpha_string_graph,
        wf._beta_string_graph,
        diag_hess=False,
        level=wf_type[2:])
    JacNum, HessNum = min_dist_jac_hess_num(
        wf,
        cc_wf,
        cc_wf.ini_blocks_D[0, 0],
        len(cc_wf),
        wf.orbs_before,
        wf.corr_orb.as_array(),
        wf.virt_orb.as_array(),
        eps = 0.0002)
    return np.array(Jac), np.array(JacNum)/2, np.array(Hess), np.array(HessNum)/2


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
        mol_system = 'Li2__5__sto3g__D2h'
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
        # for i in range(H.shape[0]):
        #     for j in range(H.shape[1]):
        #         print(f'{i:3d} {j:3d} : {H[i, j]:20.10f}  {HNum[i, j]:20.10f}  {H[i, j]-HNum[i, j]:20.10f}')
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
        # for i in range(H.shape[0]):
        #     for j in range(H.shape[1]):
        #         print(f'{i:3d} {j:3d} : {H[i, j]:20.10f}  {HNum[i, j]:20.10f}  {H[i, j]-HNum[i, j]:20.10f}')
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
