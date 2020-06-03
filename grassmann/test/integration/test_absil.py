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
        self.prng = np.random.RandomState(test.init_random_state)

    def test_overlap(self):
        for cisd_sys in test.test_systems(has_method='CISD',
                                          molecule=None):
            wf_intN = int_norm.Wave_Function_Int_Norm.from_Molpro(
                test.CISD_file(cisd_sys))
            wf_intN.calc_norm()
            wf_CISD = cisd.Wave_Function_CISD.from_intNorm(wf_intN)
            U = test.construct_random_orbitals(wf_CISD.ref_occ,
                                               wf_CISD.orb_dim,
                                               wf_CISD.n_irrep,
                                               self.prng)
            test.logger.debug("Int norm WF (%s):\n%s",
                              cisd_sys, wf_intN)
            test.logger.debug("CISD WF (%s):\n%s",
                              cisd_sys, wf_CISD)
            f_CI = absil.overlap_to_det(wf_CISD, U)
            orbitals.extend_to_unrestricted(U)
            fgen = absil.overlap_to_det(wf_intN, U)
            with self.subTest(system=cisd_sys):
                self.assertAlmostEqual(f_CI, fgen)

    def test_create_XC_matrices(self):
        for cisd_sys in test.test_systems(has_method='CISD',
                                          molecule=None):
            wf_intN = int_norm.Wave_Function_Int_Norm.from_Molpro(
                test.CISD_file(cisd_sys))
            wf_intN.calc_norm()
            wf_CISD = cisd.Wave_Function_CISD.from_intNorm(wf_intN)
            U = test.construct_random_orbitals(wf_CISD.ref_occ,
                                               wf_CISD.orb_dim,
                                               wf_CISD.n_irrep,
                                               self.prng)
            f_CI, X_CI, C_CI = absil.generate_lin_system(
                wf_CISD,
                U,
                absil.make_slice_XC(U))
            orbitals.extend_to_unrestricted(U)
            slice_XC = absil.make_slice_XC(U)
            fgen, Xgen, Cgen = absil.generate_lin_system(
                wf_intN,
                U,
                slice_XC)
            test.logger.info('Original Xgen aa:\n%s',
                             Xgen[slice_XC[0],
                                  slice_XC[0]].reshape(U[0].shape
                                                       + U[0].shape,
                                                       order='F'))
            test.logger.info('Original Xgen ab:\n%s',
                             Xgen[slice_XC[0],
                                  slice_XC[wf_intN.n_irrep]].reshape(
                                      U[0].shape + U[0].shape,
                                      order='F'))
            sep = '\n' + '=' * 30 + '\n'
            sep2 = '\n' + '-' * 30 + '\n'
            test.logger.info(sep + 'f:\n'
                             + 'General algorithm:  %.12f\n'
                             + 'CISD-opt algorithm: %.12f', fgen, f_CI)
            with self.subTest(system=cisd_sys):
                self.assertAlmostEqual(fgen, f_CI)
            for i in wf_CISD.spirrep_blocks(restricted=True):
                if U[i].shape[0] * U[i].shape[1] == 0:
                    continue
                Mgen = Cgen[slice_XC[i]].reshape(U[i].shape, order='F')
                M_CI = C_CI[slice_XC[i]].reshape(U[i].shape, order='C')
                with self.subTest(system=cisd_sys, irrep=i):
                    self.assertEqual(Mgen, M_CI)
                test.logger.info(sep
                                 + 'C[irrep = %d]:\n'
                                 + 'General algorithm:\n%r\n' + sep2
                                 + 'CISD-opt algorithm:\n%r' + sep2
                                 + 'Cgen - C_CI:\n%r' + sep2,
                                 i, Mgen, M_CI, Mgen - M_CI)
                for j in wf_CISD.spirrep_blocks(restricted=True):
                    if U[j].shape[0] * U[j].shape[1] == 0:
                        continue
                    Mgen = (Xgen[slice_XC[i],
                                 slice_XC[j]]
                            + Xgen[slice_XC[i],
                                   slice_XC[j + wf_CISD.n_irrep]]
                            ).reshape(U[i].shape + U[j].shape,
                                      order='F')
                    M_CI = X_CI[slice_XC[i], slice_XC[j]].reshape(
                        U[i].shape + U[j].shape, order='C')
                    with self.subTest(system=cisd_sys, irrep=i, irrep2=j):
                        self.assertEqual(Mgen, M_CI)
                    test.logger.info(sep
                                     + 'X[irrep = %d, irrep = %d]:\n'
                                     + 'General algorithm:\n%r\n' + sep2
                                     + 'CISD-opt algorithm:\n%r' + sep2
                                     + 'Xgen - X_CI:\n%r' + sep2,
                                     i, j, Mgen, M_CI, Mgen - M_CI)
