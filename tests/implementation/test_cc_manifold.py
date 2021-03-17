"""Checks for CC manifold

"""
import unittest


import numpy as np

import tests
from wave_functions.fci import FCIWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from coupled_cluster.manifold import min_dist_jac_hess, min_dist_jac_hess_num

#@tests.category('SHORT')
class CheckNumAnalJacTestCase(unittest.TestCase):

    def setUp(self):
        self.system = 'He2__1.5__631g__D2h'
#        self.system = 'Li2__5__sto3g__C2v'
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.wf = FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file(self.system, allE=False))
        print('fci from molpro: OK')
        self.wf.normalise(mode='intermediate')
        self.cc_wf = IntermNormWaveFunction.similar_to(
            self.wf, wf_type='CCSD', restricted=False)
        self.cc_wf.amplitudes *= 0.2
        print('similar to: OK')
        self.cc_wf_as_fci = FCIWaveFunction.from_int_norm(self.cc_wf)
        print('cc as fci: OK')
##        print(self.cc_wf)
        print(repr(self.wf))
        print(self.wf)
        print(self.wf._coefficients)
        print()

    def test1(self):
        Jac, Hess = min_dist_jac_hess(
            self.wf._coefficients,
            self.cc_wf_as_fci._coefficients,
            len(self.cc_wf),
            self.wf.orbs_before,
            self.wf.corr_orb.as_array(),
            self.wf.virt_orb.as_array(),
            self.wf._alpha_string_graph,
            self.wf._beta_string_graph,
            diag_hess=False,
            level='SD')
        Jac = np.array(Jac)
        Hess = np.array(Hess)
        JacNum, HessNum = min_dist_jac_hess_num(
            self.wf,
            self.cc_wf,
            self.cc_wf.ini_blocks_D[0, 0],
            len(self.cc_wf),
            self.wf.orbs_before,
            self.wf.corr_orb.as_array(),
            self.wf.virt_orb.as_array())
        JacNum = np.array(JacNum)/2
        HessNum = np.array(JacNum)/2
        self.cc_wf.amplitudes = Jac
        print(self.cc_wf)
        self.assertEqual(np.array(Jac), JacNum)
#        self.assertEqual(np.array(Hess), HessNum)
