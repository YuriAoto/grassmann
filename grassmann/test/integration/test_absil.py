"""Integration tests for absil

"""
import unittest

import numpy as np

import absil
import orbitals
from wave_functions import int_norm, cisd
import test


class GenCisdAlgorithmsTestCase(unittest.TestCase):
    """Compares Absil algorithm for general and CISD wave functions
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        prng = np.random.RandomState(test.init_random_state)
        # ----------
        self.He2_wf_intN = int_norm.Wave_Function_Int_Norm.from_Molpro(
            test.CISD_file('He2__R_1.5__631g__D2h'))
        self.He2_wf_intN.calc_norm()
        self.He2_wf_CISD = cisd.Wave_Function_CISD.from_intNorm(
            self.He2_wf_intN)
        self.U_He2 = test.construct_random_orbitals(
            self.He2_wf_CISD.ref_occ,
            self.He2_wf_CISD.orb_dim,
            self.He2_wf_CISD.n_irrep,
            prng)
        test.logger.debug("Int norm WF (%s):\n%s",
                          'He2__R_1.5__631g__D2h', self.He2_wf_intN)
        test.logger.debug("CISD WF (%s):\n%s",
                          'He2__R_1.5__631g__D2h', self.He2_wf_CISD)
        # ----------
        self.Li2_wf_intN = int_norm.Wave_Function_Int_Norm.from_Molpro(
            test.CISD_file('Li2__R_5__631g__C2v'))
        self.Li2_wf_intN.calc_norm()
        self.Li2_wf_CISD = cisd.Wave_Function_CISD.from_intNorm(
            self.Li2_wf_intN)
        self.U_Li2 = test.construct_random_orbitals(
            self.Li2_wf_CISD.ref_occ,
            self.Li2_wf_CISD.orb_dim,
            self.Li2_wf_CISD.n_irrep,
            prng)
        test.logger.debug("Int norm WF (%s):\n%s",
                          'Li2__R_5__631g__C2v', self.Li2_wf_intN)
        test.logger.debug("CISD WF (%s):\n%s",
                          'Li2__R_5__631g__C2v', self.Li2_wf_CISD)

    def test_overlap(self):
        # ----------
        f_CI = absil.overlap_to_det(
            self.He2_wf_CISD, self.U_He2)
        orbitals.extend_to_unrestricted(self.U_He2)
        fgen = absil.overlap_to_det(
            self.He2_wf_intN, self.U_He2)
        self.assertAlmostEqual(f_CI, fgen)
        # ----------
        f_CI = absil.overlap_to_det(
            self.Li2_wf_CISD, self.U_Li2)
        orbitals.extend_to_unrestricted(self.U_Li2)
        fgen = absil.overlap_to_det(
            self.Li2_wf_intN, self.U_Li2)
        self.assertAlmostEqual(f_CI, fgen)

    def test_create_XC_matrices(self):
        # ----------
        f_CI, X_CI, C_CI = absil.generate_lin_system(
            self.He2_wf_CISD,
            self.U_He2,
            absil.make_slice_XC(self.U_He2))
        # Add more to U_He2, because general algorithm is for unrestricted WF
        for i in self.He2_wf_CISD.spirrep_blocks(restricted=True):
            self.U_He2.append(np.array(self.U_He2[i]))
        slice_XC = absil.make_slice_XC(self.U_He2)
        fgen, Xgen, Cgen = absil.generate_lin_system(
            self.He2_wf_intN,
            self.U_He2,
            slice_XC)
        test.logger.info('Original Xgen aa:\n%s',
                         Xgen[slice_XC[0],
                              slice_XC[0]].reshape(self.U_He2[0].shape
                                                   + self.U_He2[0].shape,
                                                   order='F'))
        test.logger.info('Original Xgen ab:\n%s',
                         Xgen[slice_XC[0],
                              slice_XC[self.He2_wf_intN.n_irrep]].reshape(
                                  self.U_He2[0].shape
                                  + self.U_He2[0].shape,
                                  order='F'))
        sep = '\n' + '=' * 30 + '\n'
        sep2 = '\n' + '-' * 30 + '\n'
        test.logger.info(sep + 'f:\n'
                         + 'General algorithm:  %.12f\n'
                         + 'CISD-opt algorithm: %.12f', fgen, f_CI)
        self.assertAlmostEqual(fgen, f_CI)
        for i in self.He2_wf_CISD.spirrep_blocks(restricted=True):
            if self.U_He2[i].shape[0] * self.U_He2[i].shape[1] == 0:
                continue
            Mgen = Cgen[slice_XC[i]].reshape(self.U_He2[i].shape, order='F')
            M_CI = C_CI[slice_XC[i]].reshape(self.U_He2[i].shape, order='C')
            self.assertEqual(Mgen, M_CI)
            test.logger.debug(sep
                              + 'C[irrep = %d]:\n'
                              + 'General algorithm:\n%r\n' + sep2
                              + 'CISD-opt algorithm:\n%r' + sep2
                              + 'Cgen - C_CI:\n%r' + sep2,
                              i, Mgen, M_CI, Mgen - M_CI)
            for j in self.He2_wf_CISD.spirrep_blocks(restricted=True):
                if self.U_He2[j].shape[0] * self.U_He2[j].shape[1] == 0:
                    continue
                Mgen = (Xgen[slice_XC[i],
                             slice_XC[j]]
                        + Xgen[slice_XC[i],
                               slice_XC[j + self.He2_wf_CISD.n_irrep]]
                        ).reshape(self.U_He2[i].shape
                                  + self.U_He2[j].shape,
                                  order='F')
                M_CI = X_CI[slice_XC[i], slice_XC[j]].reshape(
                    self.U_He2[i].shape + self.U_He2[j].shape, order='C')
                self.assertEqual(Mgen, M_CI)
                test.logger.debug(sep
                                  + 'X[irrep = %d, irrep = %d]:\n'
                                  + 'General algorithm:\n%r\n' + sep2
                                  + 'CISD-opt algorithm:\n%r' + sep2
                                  + 'Xgen - X_CI:\n%r' + sep2,
                                  i, j, Mgen, M_CI, Mgen - M_CI)
        # ----------
        f_CI, X_CI, C_CI = absil.generate_lin_system(
            self.Li2_wf_CISD,
            self.U_Li2,
            absil.make_slice_XC(self.U_Li2))
        # Add more to U_Li2, because general algorithm is for unrestricted WF
        for i in self.Li2_wf_CISD.spirrep_blocks(restricted=True):
            self.U_Li2.append(np.array(self.U_Li2[i]))
        slice_XC = absil.make_slice_XC(self.U_Li2)
        fgen, Xgen, Cgen = absil.generate_lin_system(
            self.Li2_wf_intN,
            self.U_Li2,
            slice_XC)
        test.logger.info('Original Xgen aa:\n%s',
                         Xgen[slice_XC[0],
                              slice_XC[0]].reshape(self.U_Li2[0].shape
                                                   + self.U_Li2[0].shape,
                                                   order='F'))
        test.logger.info('Original Xgen ab:\n%s',
                         Xgen[slice_XC[0],
                              slice_XC[self.Li2_wf_intN.n_irrep]].reshape(
                                  self.U_Li2[0].shape
                                  + self.U_Li2[0].shape,
                                  order='F'))
        sep = '\n' + '=' * 30 + '\n'
        sep2 = '\n' + '-' * 30 + '\n'
        test.logger.info(sep + 'f:\n'
                         + 'General algorithm:  %.12f\n'
                         + 'CISD-opt algorithm: %.12f', fgen, f_CI)
        self.assertAlmostEqual(fgen, f_CI)
        for i in self.Li2_wf_CISD.spirrep_blocks(restricted=True):
            if self.U_Li2[i].shape[0] * self.U_Li2[i].shape[1] == 0:
                continue
            Mgen = Cgen[slice_XC[i]].reshape(self.U_Li2[i].shape, order='F')
            M_CI = C_CI[slice_XC[i]].reshape(self.U_Li2[i].shape, order='C')
            self.assertEqual(Mgen, M_CI)
            test.logger.debug(sep
                              + 'C[irrep = %d]:\n'
                              + 'General algorithm:\n%r\n' + sep2
                              + 'CISD-opt algorithm:\n%r' + sep2
                              + 'Cgen - C_CI:\n%r' + sep2,
                              i, Mgen, M_CI, Mgen - M_CI)
            for j in self.Li2_wf_CISD.spirrep_blocks(restricted=True):
                if self.U_Li2[j].shape[0] * self.U_Li2[j].shape[1] == 0:
                    continue
                Mgen = (Xgen[slice_XC[i],
                             slice_XC[j]]
                        + Xgen[slice_XC[i],
                               slice_XC[j + self.Li2_wf_CISD.n_irrep]]
                        ).reshape(self.U_Li2[i].shape
                                  + self.U_Li2[j].shape,
                                  order='F')
                M_CI = X_CI[slice_XC[i], slice_XC[j]].reshape(
                    self.U_Li2[i].shape + self.U_Li2[j].shape, order='C')
                self.assertEqual(Mgen, M_CI)
                test.logger.debug(sep
                                  + 'X[irrep = %d, irrep = %d]:\n'
                                  + 'General algorithm:\n%r\n' + sep2
                                  + 'CISD-opt algorithm:\n%r' + sep2
                                  + 'Xgen - X_CI:\n%r' + sep2,
                                  i, j, Mgen, M_CI, Mgen - M_CI)
