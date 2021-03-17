"""Integration tests for wave_functions.fci

"""
import unittest

import numpy as np

from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.fci import FCIWaveFunction
from coupled_cluster.manifold import min_dist_jac_hess

import tests


@tests.category('SHORT')
class MinDistJacHessTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test1(self):
        cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file('Li2__5__sto3g__D2h', allE=True)))
        wf = FCIWaveFunction.from_int_norm(cc_wf)
        cc_wf.amplitudes *= 0.8
        cc_wf_as_fci = FCIWaveFunction.from_int_norm(cc_wf)
        print(cc_wf)
        J, H = min_dist_jac_hess(
            wf._coefficients,
            cc_wf_as_fci._coefficients,
            cc_wf.ini_blocks_S[cc_wf.n_irrep],
            cc_wf.ini_blocks_D[0, 0],
            cc_wf.ini_blocks_D[cc_wf.first_bb_pair, 0],
            cc_wf.ini_blocks_D[cc_wf.first_ab_pair, 0],
            len(cc_wf),
            wf.orbs_before,
            wf.corr_orb.as_array(),
            wf.virt_orb.as_array(),
            wf._alpha_string_graph,
            wf._beta_string_graph,
            diag_hess=False,
            level='SD')
        print(np.array(J))
        print('=========')
        print(np.array(H)[:5,:5])
