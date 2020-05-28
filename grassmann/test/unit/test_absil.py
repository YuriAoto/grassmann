"""Tests for absil

"""
import unittest

import numpy as np
from scipy import linalg

from wave_functions import cisd, int_norm
import orbitals
import absil
import test


np.set_printoptions(formatter={'all': lambda x: '{0:>9.6f}'.format(x)})


class CalcOverlap(unittest.TestCase):

    def setUp(self):
        prng = np.random.RandomState(test.init_random_state)
        # H2:
        self.intN_wf_H2 = int_norm.Wave_Function_Int_Norm.from_Molpro(
                test.CISD_file('H2__R_5__sto3g__D2h'))
        self.intN_wf_H2.calc_norm()
        self.wf_H2 = cisd.Wave_Function_CISD.from_intNorm(self.intN_wf_H2)
        self.Uid_H2 = orbitals.construct_Id_orbitals(self.wf_H2.ref_occ,
                                                     self.wf_H2.orb_dim,
                                                     self.wf_H2.n_irrep)
        # Li2:
        self.intN_wf_Li2 = int_norm.Wave_Function_Int_Norm.from_Molpro(
            test.CISD_file('Li2__R_5__631g__C2v'))
        self.intN_wf_Li2.calc_norm()
        self.wf_Li2 = cisd.Wave_Function_CISD.from_intNorm(self.intN_wf_Li2)
        self.U_Li2 = test.construct_random_orbitals(
            self.wf_Li2.ref_occ,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep,
            prng)
        self.U_Li2_non_orth = test.construct_random_orbitals(
            self.wf_Li2.ref_occ,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep,
            prng,
            orthogonalise=False)
        self.Uid_Li2 = orbitals.construct_Id_orbitals(
            self.wf_Li2.ref_occ,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep)

    def test_ref_overlap(self):
        self.assertAlmostEqual(absil.overlap_to_det(self.wf_H2,
                                                    self.Uid_H2),
                               self.wf_H2.C0)
        self.assertAlmostEqual(absil.overlap_to_det(self.wf_Li2,
                                                    self.Uid_Li2),
                               self.wf_Li2.C0)
        orbitals.extend_to_unrestricted(self.Uid_H2)
        orbitals.extend_to_unrestricted(self.Uid_Li2)
        self.assertAlmostEqual(absil.overlap_to_det(self.intN_wf_H2,
                                                    self.Uid_H2),
                               self.intN_wf_H2.C0)
        self.assertAlmostEqual(absil.overlap_to_det(self.intN_wf_Li2,
                                                    self.Uid_Li2),
                               self.intN_wf_Li2.C0)

    def test_compare_overlap_and_gen_sys(self):
        f, X, C = absil.generate_lin_system(self.wf_Li2,
                                            self.U_Li2,
                                            absil.make_slice_XC(self.U_Li2))
        self.assertAlmostEqual(absil.overlap_to_det(self.wf_Li2,
                                                    self.U_Li2),
                               f)
        orbitals.extend_to_unrestricted(self.U_Li2)
        f, X, C = absil.generate_lin_system(self.intN_wf_Li2,
                                            self.U_Li2,
                                            absil.make_slice_XC(self.U_Li2))
        self.assertAlmostEqual(absil.overlap_to_det(self.intN_wf_Li2,
                                                    self.U_Li2),
                               f)

    def test_non_orth(self):
        f_non_orth = absil.overlap_to_det(self.wf_Li2,
                                          self.U_Li2_non_orth)
        test.logger.info('U before orth:\n%s',
                         self.U_Li2_non_orth)
        for irrep, Ui in enumerate(self.U_Li2_non_orth):
            if Ui.shape[0] * Ui.shape[1] != 0:
                self.U_Li2_non_orth[irrep] = linalg.orth(Ui)
        test.logger.info('U after orth:\n%s',
                         self.U_Li2_non_orth)
        f_orth = absil.overlap_to_det(self.wf_Li2,
                                      self.U_Li2_non_orth)
        f_non_orth = absil.overlap_to_det(self.wf_Li2,
                                          self.U_Li2_non_orth,
                                          assume_orth=False)
        self.assertAlmostEqual(f_orth, f_non_orth)
        orbitals.extend_to_unrestricted(self.U_Li2_non_orth)
        f_non_orth = absil.overlap_to_det(self.intN_wf_Li2,
                                          self.U_Li2_non_orth)
        test.logger.info('U before orth:\n%s',
                         self.U_Li2_non_orth)
        for irrep, Ui in enumerate(self.U_Li2_non_orth):
            if Ui.shape[0] * Ui.shape[1] != 0:
                self.U_Li2_non_orth[irrep] = linalg.orth(Ui)
        test.logger.info('U after orth:\n%s',
                         self.U_Li2_non_orth)
        f_orth = absil.overlap_to_det(self.intN_wf_Li2,
                                      self.U_Li2_non_orth)
        f_non_orth = absil.overlap_to_det(self.intN_wf_Li2,
                                          self.U_Li2_non_orth,
                                          assume_orth=False)
        self.assertAlmostEqual(f_orth, f_non_orth)


class CalcXCmatrices(unittest.TestCase):

    def setUp(self):
        prng = np.random.RandomState(test.init_random_state)
        # H2:
        self.intN_wf_H2 = int_norm.Wave_Function_Int_Norm.from_Molpro(
            test.CISD_file('H2__R_5__sto3g__D2h'))
        self.intN_wf_H2.calc_norm()
        self.wf_H2 = cisd.Wave_Function_CISD.from_intNorm(self.intN_wf_H2)
        self.Uid_H2 = orbitals.construct_Id_orbitals(
            self.wf_H2.ref_occ,
            self.wf_H2.orb_dim,
            self.wf_H2.n_irrep)
        # Li2:
        self.intN_wf_Li2 = int_norm.Wave_Function_Int_Norm.from_Molpro(
            test.CISD_file('Li2__R_5__631g__C2v'))
        self.intN_wf_Li2.calc_norm()
        self.wf_Li2 = cisd.Wave_Function_CISD.from_intNorm(self.intN_wf_Li2)
        self.U_Li2 = test.construct_random_orbitals(
            self.wf_Li2.ref_occ,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep,
            prng)
        self.Uid_Li2 = orbitals.construct_Id_orbitals(
            self.wf_Li2.ref_occ,
            self.wf_Li2.orb_dim,
            self.wf_Li2.n_irrep)

    def test_check_Absil_eq(self):
        slice_XC = absil.make_slice_XC(self.Uid_H2)
        f, X, C = absil.generate_lin_system(self.wf_H2,
                                            self.Uid_H2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_H2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape))
        self.assertTrue(absil.check_Newton_eq(self.wf_H2,
                                              self.Uid_H2,
                                              eta,
                                              True))
        slice_XC = absil.make_slice_XC(self.Uid_Li2)
        f, X, C = absil.generate_lin_system(self.wf_Li2,
                                            self.Uid_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape))
        self.assertTrue(absil.check_Newton_eq(self.wf_Li2,
                                              self.Uid_Li2,
                                              eta,
                                              True,
                                              eps=0.001))
        slice_XC = absil.make_slice_XC(self.U_Li2)
        f, X, C = absil.generate_lin_system(self.wf_Li2,
                                            self.U_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.U_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape))
        test.logger.info('2C:\n%r', 2*C)
        test.logger.info('2X @ eta:\n%r', np.matmul(2*X, lin_sys_solution[0]))
        self.assertTrue(absil.check_Newton_eq(self.wf_Li2,
                                              self.U_Li2,
                                              eta,
                                              True,
                                              eps=0.0001))

    def test_check_Absil_eq_gen_WF(self):
        orbitals.extend_to_unrestricted(self.Uid_H2)
        slice_XC = absil.make_slice_XC(self.Uid_H2)
        f, X, C = absil.generate_lin_system(self.intN_wf_H2,
                                            self.Uid_H2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_H2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape,
                                  order='F'))
        self.assertTrue(absil.check_Newton_eq(self.intN_wf_H2,
                                              self.Uid_H2,
                                              eta,
                                              False))
        orbitals.extend_to_unrestricted(self.Uid_Li2)
        slice_XC = absil.make_slice_XC(self.Uid_Li2)
        f, X, C = absil.generate_lin_system(self.intN_wf_Li2,
                                            self.Uid_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.Uid_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape,
                                  order='F'))
        self.assertTrue(absil.check_Newton_eq(self.intN_wf_Li2,
                                              self.Uid_Li2,
                                              eta,
                                              False,
                                              eps=0.001))
        orbitals.extend_to_unrestricted(self.U_Li2)
        slice_XC = absil.make_slice_XC(self.U_Li2)
        f, X, C = absil.generate_lin_system(self.intN_wf_Li2,
                                            self.U_Li2,
                                            slice_XC)
        lin_sys_solution = linalg.lstsq(X, C, cond=None)
        eta = []
        for i, Ui in enumerate(self.U_Li2):
            eta.append(np.reshape(lin_sys_solution[0][slice_XC[i]],
                                  Ui.shape,
                                  order='F'))
        test.logger.info('C:\n%r', C)
        test.logger.info('X @ eta:\n%r', np.matmul(X, lin_sys_solution[0]))
        self.assertTrue(absil.check_Newton_eq(self.intN_wf_Li2,
                                              self.U_Li2,
                                              eta,
                                              False,
                                              eps=0.0001))
