"""Integration tests for wave_functions.int_norm

"""
import unittest

import numpy as np

from wave_functions.fci import FCIWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions import int_norm, cisd
import tests



@tests.category('SHORT')
class CISDgoFCIbackCISDTestCase(unittest.TestCase):
    """Check CISD -> FCI -> CISD
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_he2_631g_c2v(self):
        mol_system = 'He2__1.5__631g__C2v'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CISD_file(mol_system))
        goback_cc_wf = IntermNormWaveFunction.restrict(
            IntermNormWaveFunction.from_projected_fci(
                FCIWaveFunction.from_int_norm(cc_wf), 'CISD'))
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    def test_he2_631g_d2h(self):
        mol_system = 'He2__1.5__631g__D2h'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CISD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CISD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    def test_li2_ccpvdz_d2h(self):
        mol_system = 'Li2__5__ccpVDZ__D2h'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CISD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CISD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)


@tests.category('SHORT')
class CCDgoFCIbackCCDTestCase(unittest.TestCase):
    """Check CCD -> FCI -> CCD
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_he2_631g_c2v(self):
        mol_system = 'He2__1.5__631g__C2v'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCD_file(mol_system))
        goback_cc_wf = IntermNormWaveFunction.restrict(
            IntermNormWaveFunction.from_projected_fci(
                FCIWaveFunction.from_int_norm(cc_wf), 'CCD'))
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    def test_he2_631g_d2h(self):
        mol_system = 'He2__1.5__631g__D2h'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CCD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    def test_li2_ccpvdz_d2h(self):
        mol_system = 'Li2__5__ccpVDZ__D2h'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CCD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    @unittest.skip('This is too long and requires too much memory...')
    def test_h2o_631g_c2v(self):
        mol_system = 'h2o__Req__631g__C2v'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CCD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)


@tests.category('SHORT')
class CCSDgoFCIbackCCSDTestCase(unittest.TestCase):
    """Check CCSD -> FCI -> CCSD
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_he2_631g_c2v(self):
        mol_system = 'He2__1.5__631g__C2v'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCSD_file(mol_system))
        goback_cc_wf = IntermNormWaveFunction.restrict(
            IntermNormWaveFunction.from_projected_fci(
                FCIWaveFunction.from_int_norm(cc_wf), 'CCSD'))
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    def test_he2_631g_d2h(self):
        mol_system = 'He2__1.5__631g__D2h'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCSD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CCSD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)

    def test_li2_ccpvdz_d2h(self):
        mol_system = 'Li2__5__ccpVDZ__D2h'
        cc_wf = IntermNormWaveFunction.from_Molpro(tests.CCSD_file(mol_system))
        goback_cc_wf = FCIWaveFunction.from_int_norm(cc_wf)
        goback_cc_wf = IntermNormWaveFunction.from_projected_fci(goback_cc_wf, 'CCSD')
        goback_cc_wf = IntermNormWaveFunction.restrict(goback_cc_wf)
        self.assertEqual(cc_wf.amplitudes, goback_cc_wf.amplitudes)


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
            wf_intN = int_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(H2_sys))
            wf_CISD = cisd.CISD_WaveFunction.from_int_norm(wf_intN)
            wf_intN = int_norm.IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(H2_sys))
            wf_intN.use_CISD_norm = True
            wf_CCSD = cisd.CISD_WaveFunction.from_int_norm(wf_intN)
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

    def test_check_CCvsCI_string_indices_for_H2(self):
        for H2_sys in tests.test_systems(has_method=('CISD', 'CCSD'),
                                         molecule='H2'):
            wf_CI = int_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(H2_sys))
            wf_CC = int_norm.IntermNormWaveFunction.from_Molpro(
                tests.CCSD_file(H2_sys))
            wf_CC.use_CISD_norm = True
            str_ind_CC = wf_CC.string_indices()
            for Ind_CI in wf_CI.string_indices():
                Ind_CC = next(str_ind_CC)
                self.assertAlmostEqual(Ind_CC.C, Ind_CI.C, places=5)
                for irp in wf_CI.spirrep_blocks(restricted=False):
                    self.assertEqual(Ind_CC[irp].occ_orb, Ind_CI[irp].occ_orb)
