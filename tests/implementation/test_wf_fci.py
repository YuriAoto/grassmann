"""Tests for fci

"""
import unittest

import numpy as np
from scipy.special import comb
from scipy import linalg

from wave_functions import fci
import wave_functions.strings_rev_lexical_order as str_order
import tests

@tests.category('SHORT', 'ESSENTIAL')
class StringGraphsTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.norb1 = 6
        self.nel1 = 4
        self.norb2 = 5
        self.nel2 = 3
        self.string_gr_1 = str_order.generate_graph(self.nel1, self.norb1)
        self.string_gr_2 = str_order.generate_graph(self.nel2, self.norb2)

    def test_occ_fromto_strind(self):
        for i in range(comb(self.norb1, self.nel1, exact=True)):
            self.assertEqual(
                str_order.get_index(str_order.occ_from_pos(
                    i, self.string_gr_1), self.string_gr_1),
                i)

    def test_occ_fromto_strind2(self):
        for i in range(comb(self.norb2, self.nel2, exact=True)):
            self.assertEqual(
                str_order.get_index(str_order.occ_from_pos(
                    i, self.string_gr_2), self.string_gr_2),
                i)



@tests.category('SHORT')
class JacHess_H2_TestCase(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.WF = fci.FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        Ures = []
        Uunres = []
        prng = np.random.RandomState(tests.init_random_state)
        for spirrep in self.WF.spirrep_blocks(restricted=True):
            K = self.WF.orbspace.full[spirrep]
            newU = prng.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            Ures.append(newU)
        for spirrep in self.WF.spirrep_blocks(restricted=False):
            K = self.WF.orbspace.full[spirrep]
            newU = prng.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            Uunres.append(newU)
        self.WFres = self.WF.change_orb_basis(Ures)
        self.WFunres = self.WF.change_orb_basis(Uunres)
        self.WFunres.restricted = False
    
    def test_Num_Anal_original(self):
        Janal, Hanal = self.WFres.make_Jac_Hess_overlap(analytic=True,
                                                        restricted=True)
        Jnum, Hnum = self.WFres.make_Jac_Hess_overlap(analytic=False,
                                                      restricted=True)
        tests.logger.info('Wave function:\n%r', self.WF)
        tests.logger.info('Analytic Jacobian:\n%r', Janal)
        tests.logger.info('Numeric Jacobian:\n%r', Jnum)
        tests.logger.info('Analytic Hessian:\n%r', Hanal)
        tests.logger.info('Numeric Hessian:\n%r', Hnum)
        self.assertEqual(Janal, Jnum)
        self.assertEqual(Hanal, Hnum)
    
    def test_Num_Anal_res(self):
        Janal, Hanal = self.WFres.make_Jac_Hess_overlap(analytic=True,
                                                        restricted=True)
        Jnum, Hnum = self.WFres.make_Jac_Hess_overlap(analytic=False,
                                                      restricted=True)
        tests.logger.info('Wave function:\n%r', self.WFres)
        tests.logger.info('Analytic Jacobian:\n%r', Janal)
        tests.logger.info('Numeric Jacobian:\n%r', Jnum)
        tests.logger.info('Analytic Hessian:\n%r', Hanal)
        tests.logger.info('Numeric Hessian:\n%r', Hnum)
        self.assertEqual(Janal, Jnum)
        self.assertEqual(Hanal, Hnum)
    
    def test_Num_Anal_unres(self):
        Janal, Hanal = self.WFunres.make_Jac_Hess_overlap(analytic=True,
                                                          restricted=False)
        Jnum, Hnum = self.WFunres.make_Jac_Hess_overlap(analytic=False,
                                                        restricted=False)
        tests.logger.info('Wave function:\n%r', self.WFunres)
        tests.logger.info('Analytic Jacobian:\n%r', Janal)
        tests.logger.info('Numeric Jacobian:\n%r', Jnum)
        tests.logger.info('Analytic Hessian:\n%r', Hanal)
        tests.logger.info('Numeric Hessian:\n%r', Hnum)
        self.assertEqual(Janal, Jnum)
        self.assertEqual(Hanal, Hnum)


@tests.category('SHORT')
class WForbChangeTestCase(unittest.TestCase):
    
    def setUp(self):
        self.WF = fci.FCIWaveFunction.from_Molpro_FCI(
            tests.FCI_file('H2__5__sto3g__D2h'))
        self.U1 = []
        self.U2 = []
        self.U3 = []
        self.U = []
        self.UT = []
        for spirrep in self.WF.spirrep_blocks(restricted=True):
            K = self.WF.orbspace.full[spirrep]
            newU = np.random.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            self.U1.append(newU)
            newU = np.random.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            self.U2.append(newU)
            newU = np.random.random_sample(size=(K, K))
            if K > 0:
                newU = linalg.orth(newU)
            self.U3.append(newU)
            self.U.append(np.matmul(np.matmul(self.U1[-1],
                                              self.U2[-1]),
                                    self.U3[-1]))
            self.UT.append(self.U[-1].T)
    
    def test_go_and_back(self):
        backWF = self.WF.change_orb_basis(self.U).change_orb_basis(self.UT)
        self.assertEqual(self.WF, backWF, msg='Wave functions are not equal.')
    
    def test_three_vs_one_steps(self):
        newWF1 = self.WF.change_orb_basis(self.U1)
        newWF2 = newWF1.change_orb_basis(self.U2)
        newWF3 = newWF2.change_orb_basis(self.U3)
        newWF3_direct = self.WF.change_orb_basis(self.U)
        self.assertEqual(newWF3, newWF3_direct)

