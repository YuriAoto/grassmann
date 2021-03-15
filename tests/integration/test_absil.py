"""Integration tests for absil

"""
import unittest

import numpy as np

from dist_grassmann import absil
from orbitals import orbitals
from wave_functions import int_norm, cisd
import tests

molecules = ('H2', 'Li2')

@tests.category('SHORT')
class GenCisdAlgorithmsTestCase(unittest.TestCase):
    """Compares Absil algorithm for general and CISD wave functions
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.prng = np.random.RandomState(tests.init_random_state)

    @tests.category('SHORT')
    def test_overlap(self):
        for cisd_sys in tests.test_systems(has_method='CISD',
                                          molecule=molecules):
            wf_intN = int_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(cisd_sys))
            wf_CISD = cisd.CISD_WaveFunction.from_int_norm(wf_intN)
            U = tests.construct_random_orbitals(wf_CISD.ref_orb,
                                                wf_CISD.orb_dim,
                                                wf_CISD.n_irrep,
                                                self.prng)
            tests.logger.debug("Int norm WF (%s):\n%s",
                               cisd_sys, wf_intN)
            tests.logger.debug("CISD WF (%s):\n%s",
                               cisd_sys, wf_CISD)
            f_CI = absil.overlap_to_det(wf_CISD, U)
            orbitals.extend_to_unrestricted(U)
            fgen = absil.overlap_to_det(wf_intN, U)
            with self.subTest(system=cisd_sys):
                self.assertAlmostEqual(f_CI, fgen)

    @tests.category('LONG')
    def test_create_XC_matrices(self):
        for cisd_sys in tests.test_systems(has_method='CISD',
                                          molecule=molecules):
            wf_intN = int_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(cisd_sys))
            wf_CISD = cisd.CISD_WaveFunction.from_int_norm(wf_intN)
            U = tests.construct_random_orbitals(wf_CISD.ref_orb,
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
            tests.logger.info('Original Xgen aa:\n%s',
                              Xgen[slice_XC[0],
                                   slice_XC[0]].reshape(U[0].shape
                                                        + U[0].shape,
                                                        order='F'))
            tests.logger.info('Original Xgen ab:\n%s',
                              Xgen[slice_XC[0],
                                   slice_XC[wf_intN.n_irrep]].reshape(
                                       U[0].shape + U[0].shape,
                                       order='F'))
            sep = '\n' + '=' * 30 + '\n'
            sep2 = '\n' + '-' * 30 + '\n'
            tests.logger.info(sep + 'f:\n'
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
                tests.logger.info(sep
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
                    tests.logger.info(sep
                                      + 'X[irrep = %d, irrep = %d]:\n'
                                      + 'General algorithm:\n%r\n' + sep2
                                      + 'CISD-opt algorithm:\n%r' + sep2
                                      + 'Xgen - X_CI:\n%r' + sep2,
                                      i, j, Mgen, M_CI, Mgen - M_CI)
