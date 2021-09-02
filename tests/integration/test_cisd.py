"""Integration tests for wave_functions.cisd

"""
import unittest

import numpy as np

from wave_functions import interm_norm, cisd, fci
import tests

molecule = ('H2', 'Li2')
basis = None
symmetry = None


@tests.category('SHORT')
class CisdFciJacHessTestCase(unittest.TestCase):
    """Compares CISD and FCI Jacobian and Hessian
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_check_Jac_Hess_use_FCItempl(self):
        for test_sys in tests.test_systems(has_method=('CISD', 'FCI'),
                                          molecule=molecule,
                                          basis=basis,
                                          symmetry=symmetry):
            wf_intN = interm_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(test_sys))
            wf_CISD = cisd.CISDWaveFunction.from_interm_norm(wf_intN)
            wf_FCI = fci.FCIWaveFunction.from_Molpro_FCI(
                tests.FCI_file(test_sys), zero_coefficients=False)
            tests.logger.debug("FCI before:\n%r", wf_FCI)
            wf_FCI.get_coefficients_from_interm_norm_wf(wf_intN)
            tests.logger.debug("CISD:\n%r", wf_CISD)
            tests.logger.debug("FCI:\n%r", wf_FCI)
            Jac_fci, Hess_fci = wf_FCI.make_Jac_Hess_overlap()
            Jac_cisd, Hess_cisd = wf_CISD.make_Jac_Hess_overlap()
            slices_cisd = []
            slices_fci = []
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                nK = wf_CISD.corr_orb[irp] * wf_CISD.virt_orb[irp]
                slice_start = 0 if irp == 0 else slices_cisd[-1].stop
                slices_cisd.append(slice(slice_start, slice_start + nK))
                nK = wf_FCI.ref_orb[irp] * wf_FCI.virt_orb[irp]
                slice_start = 0 if irp == 0 else slices_fci[-1].stop
                slices_fci.append(slice(slice_start, slice_start + nK))
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                with self.subTest(system=test_sys, irrep=irp, coef='Jac'):
                    self.assertEqual(
                        np.reshape(Jac_cisd[slices_cisd[irp]],
                                   (wf_CISD.corr_orb[irp],
                                    wf_CISD.virt_orb[irp])),
                        np.reshape(Jac_fci[slices_fci[irp]],
                                   (wf_FCI.ref_orb[irp],
                                    wf_FCI.virt_orb[irp]))[
                                        wf_FCI.froz_orb[irp]:, :])
                for irp2 in range(irp + 1):
                    with self.subTest(system=test_sys, irrep=irp,
                                      irrep2=irp2, coef='Hess'):
                        self.assertEqual(
                            np.reshape(Hess_cisd[slices_cisd[irp],
                                                 slices_cisd[irp2]],
                                       (wf_CISD.corr_orb[irp],
                                        wf_CISD.virt_orb[irp],
                                        wf_CISD.corr_orb[irp2],
                                        wf_CISD.virt_orb[irp2])),
                            np.reshape(Hess_fci[slices_fci[irp],
                                                slices_fci[irp2]],
                                       (wf_FCI.ref_orb[irp],
                                        wf_FCI.virt_orb[irp],
                                        wf_FCI.ref_orb[irp2],
                                        wf_FCI.virt_orb[irp2]))[
                                            wf_FCI.froz_orb[irp]:, :,
                                            wf_FCI.froz_orb[irp2]:, :])

    def test_check_Jac_Hess_FCIdirectly(self):
        for test_sys in tests.test_systems(has_method=('CISD', 'FCI'),
                                          molecule=molecule,
                                          basis=basis,
                                          symmetry=symmetry):
            wf_intN = interm_norm.IntermNormWaveFunction.from_Molpro(
                tests.CISD_file(test_sys))
            wf_CISD = cisd.CISDWaveFunction.from_interm_norm(wf_intN)
            wf_FCI = fci.FCIWaveFunction.from_interm_norm(wf_intN)
            tests.logger.debug("CISD:\n%r", wf_CISD)
            tests.logger.debug("FCI:\n%r", wf_FCI)
            Jac_fci, Hess_fci = wf_FCI.make_Jac_Hess_overlap()
            Jac_cisd, Hess_cisd = wf_CISD.make_Jac_Hess_overlap()
            slices_cisd = []
            slices_fci = []
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                nK = wf_CISD.corr_orb[irp] * wf_CISD.virt_orb[irp]
                slice_start = 0 if irp == 0 else slices_cisd[-1].stop
                slices_cisd.append(slice(slice_start, slice_start + nK))
                nK = wf_FCI.ref_orb[irp] * wf_FCI.virt_orb[irp]
                slice_start = 0 if irp == 0 else slices_fci[-1].stop
                slices_fci.append(slice(slice_start, slice_start + nK))
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                with self.subTest(system=test_sys, irrep=irp, coef='Jac'):
                    self.assertEqual(
                        np.reshape(Jac_cisd[slices_cisd[irp]],
                                   (wf_CISD.corr_orb[irp],
                                    wf_CISD.virt_orb[irp])),
                        np.reshape(Jac_fci[slices_fci[irp]],
                                   (wf_FCI.ref_orb[irp],
                                    wf_FCI.virt_orb[irp]))[
                                        wf_FCI.froz_orb[irp]:, :])
                for irp2 in range(irp + 1):
                    with self.subTest(system=test_sys, irrep=irp,
                                      irrep2=irp2, coef='Hess'):
                        self.assertEqual(
                            np.reshape(Hess_cisd[slices_cisd[irp],
                                                 slices_cisd[irp2]],
                                       (wf_CISD.corr_orb[irp],
                                        wf_CISD.virt_orb[irp],
                                        wf_CISD.corr_orb[irp2],
                                        wf_CISD.virt_orb[irp2])),
                            np.reshape(Hess_fci[slices_fci[irp],
                                                slices_fci[irp2]],
                                       (wf_FCI.ref_orb[irp],
                                        wf_FCI.virt_orb[irp],
                                        wf_FCI.ref_orb[irp2],
                                        wf_FCI.virt_orb[irp2]))[
                                            wf_FCI.froz_orb[irp]:, :,
                                            wf_FCI.froz_orb[irp2]:, :])
