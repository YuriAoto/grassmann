"""Integration tests for wave_functions.int_norm

"""
import unittest

import numpy as np

from wave_functions import interm_norm, cisd
import tests


@tests.category('SHORT')
class CISDvsCCSDTestCase(unittest.TestCase):
    """Compares CISD and CCSD wave functions for H2
    
    Being a two-electron system, the CCSD and CISD wave functions
    must be the same.
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.prng = np.random.RandomState(tests.init_random_state)

    def test_check_CCvsCI_cisd_for_H2(self):
        for H2_sys in tests.test_systems(has_method=('CISD', 'CCSD'),
                                         molecule='H2'):
            wf_intN = interm_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(H2_sys))
            wf_CISD = cisd.CISDWaveFunction.from_interm_norm(wf_intN)
            wf_intN = interm_norm.IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(H2_sys))
            wf_intN.use_CISD_norm = True
            wf_CCSD = cisd.CISDWaveFunction.from_interm_norm(wf_intN)
            with self.subTest(system=H2_sys, coef='C0'):
                self.assertAlmostEqual(wf_CCSD.C0, wf_CISD.C0, places=5)
            for irp in wf_CCSD.spirrep_blocks(restricted=True):
                with self.subTest(system=H2_sys, irrep=irp, coef='Cs'):
                    self.assertEqual(wf_CCSD.Cs[irp], wf_CISD.Cs[irp])
                with self.subTest(system=H2_sys, irrep=irp, coef='Cd'):
                    self.assertEqual(wf_CCSD.Cd[irp], wf_CISD.Cd[irp])
                for irp2 in range(irp + 1):
                    with self.subTest(system=H2_sys, irrep=irp,
                                      irrep2=irp2, coef='Cs'):
                        self.assertEqual(wf_CCSD.Csd[irp][irp2],
                                         wf_CISD.Csd[irp][irp2])
