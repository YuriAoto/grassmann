"""Integration tests for wave_functions.cisd

"""
import unittest

import numpy as np

from wave_functions import int_norm, cisd, fci
import test

systems = [
     'H2__5__631g__C1',
     'H2__5__631g__C2v',
     'H2__5__631g__D2h',
     'H2__5__ccpVDZ__C1',
     'H2__5__ccpVDZ__C2v',
     'H2__5__ccpVDZ__Cs',
     'H2__5__ccpVDZ__D2h',
     'H2__5__sto3g__C1',
     'H2__5__sto3g__D2h',
     'H2__5__631g__C1',
     'He2__1.5__631g__C2v',
     'He2__1.5__631g__D2h',
     'h2o__1.5__sto3g__C2v',
     'h2o__Req__631g__C2v',
     'h2o__Req__sto3g__C2v',
     'Li2__5__sto3g__C1',
     'Li2__5__sto3g__C2v',
     'Li2__5__sto3g__D2h',
     'Li2__5__to2s__C2v',
     'Li2__5__to3s__C2v',
     'Li2__5__631g__C1',
     'Li2__5__631g__C2v',
     'Li2__5__631g__D2h',
     'Li2__5__ccpVDZ__C1',
     'Li2__5__ccpVDZ__C2v',
     'Li2__5__ccpVDZ__Cs',
     'Li2__5__ccpVDZ__D2h',
     'Li2__5__ccpVTZ__D2h',
     'Li2__5__ccpVQZ__D2h',
#     'N2__3__sto3g__D2h',
     'N2__3__631g__D2h',
     'N2__3__631g__D2h_occ_21101110',
     'N2__3__cc-pVDZ__D2h',
     'H8_cage__1.5__631g__D2h',
     'He8_cage__1.5__631g__D2h',
     'He8_cage__1.5__ccpVDZ__D2h',
     'Li8_cage__1.5__631g__D2h'
]


class CisdFciJacHessTestCase(unittest.TestCase):
    """Compares CISD and FCI Jacobian and Hessian
    
    """
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)

    def test_check_Jac_Hess_use_FCItempl(self):
        for cisd_sys in systems:
            wf_intN = int_norm.Wave_Function_Int_Norm.from_Molpro(
                test.CISD_file(cisd_sys))
            wf_intN.calc_norm()
            wf_CISD = cisd.Wave_Function_CISD.from_intNorm(wf_intN)
            wf_FCI = fci.Wave_Function_Norm_CI.from_Molpro_FCI(
                test.FCI_file(cisd_sys), zero_coefficients=True)
            wf_FCI.get_coeff_from_Int_Norm_WF(wf_intN,
                                              change_structure=False,
                                              use_structure=True)
            Jac_fci, Hess_fci = wf_FCI.make_Jac_Hess_overlap()
            Jac_cisd, Hess_cisd = wf_CISD.make_Jac_Hess_overlap()
            slices_cisd = []
            slices_fci = []
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                nK = wf_CISD.n_corr_orb[irp] * wf_CISD.n_ext[irp]
                slice_start = 0 if irp == 0 else slices_cisd[-1].stop
                slices_cisd.append(slice(slice_start, slice_start + nK))
                nK = wf_FCI.ref_occ[irp] * wf_FCI.n_ext[irp]
                slice_start = 0 if irp == 0 else slices_fci[-1].stop
                slices_fci.append(slice(slice_start, slice_start + nK))
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                with self.subTest(system=cisd_sys, irrep=irp, coef='Jac'):
                    self.assertEqual(
                        np.reshape(Jac_cisd[slices_cisd[irp]],
                                   (wf_CISD.n_corr_orb[irp],
                                    wf_CISD.n_ext[irp])),
                        np.reshape(Jac_fci[slices_fci[irp]],
                                   (wf_FCI.ref_occ[irp],
                                    wf_FCI.n_ext[irp]))[
                                        wf_FCI.n_core[irp]:, :])
                for irp2 in range(irp + 1):
                    with self.subTest(system=cisd_sys, irrep=irp,
                                      irrep2=irp2, coef='Hess'):
                        self.assertEqual(
                            np.reshape(Hess_cisd[slices_cisd[irp],
                                                 slices_cisd[irp2]],
                                       (wf_CISD.n_corr_orb[irp],
                                        wf_CISD.n_ext[irp],
                                        wf_CISD.n_corr_orb[irp2],
                                        wf_CISD.n_ext[irp2])),
                            np.reshape(Hess_fci[slices_fci[irp],
                                                slices_fci[irp2]],
                                       (wf_FCI.ref_occ[irp],
                                        wf_FCI.n_ext[irp],
                                        wf_FCI.ref_occ[irp2],
                                        wf_FCI.n_ext[irp2]))[
                                            wf_FCI.n_core[irp]:, :,
                                            wf_FCI.n_core[irp2]:, :])

    def test_check_Jac_Hess_FCIdirectly(self):
        for cisd_sys in systems:
            wf_intN = int_norm.Wave_Function_Int_Norm.from_Molpro(
                test.CISD_file(cisd_sys))
            wf_intN.calc_norm()
            wf_CISD = cisd.Wave_Function_CISD.from_intNorm(wf_intN)
            wf_FCI = fci.Wave_Function_Norm_CI.from_Int_Norm(wf_intN)
            test.logger.debug("CISD:\n%r", wf_CISD)
            test.logger.debug("FCI:\n%r", wf_FCI)
            Jac_fci, Hess_fci = wf_FCI.make_Jac_Hess_overlap()
            Jac_cisd, Hess_cisd = wf_CISD.make_Jac_Hess_overlap()
            slices_cisd = []
            slices_fci = []
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                nK = wf_CISD.n_corr_orb[irp] * wf_CISD.n_ext[irp]
                slice_start = 0 if irp == 0 else slices_cisd[-1].stop
                slices_cisd.append(slice(slice_start, slice_start + nK))
                nK = wf_FCI.ref_occ[irp] * wf_FCI.n_ext[irp]
                slice_start = 0 if irp == 0 else slices_fci[-1].stop
                slices_fci.append(slice(slice_start, slice_start + nK))
            for irp in wf_CISD.spirrep_blocks(restricted=True):
                with self.subTest(system=cisd_sys, irrep=irp, coef='Jac'):
                    self.assertEqual(
                        np.reshape(Jac_cisd[slices_cisd[irp]],
                                   (wf_CISD.n_corr_orb[irp],
                                    wf_CISD.n_ext[irp])),
                        np.reshape(Jac_fci[slices_fci[irp]],
                                   (wf_FCI.ref_occ[irp],
                                    wf_FCI.n_ext[irp]))[
                                        wf_FCI.n_core[irp]:, :])
                for irp2 in range(irp + 1):
                    with self.subTest(system=cisd_sys, irrep=irp,
                                      irrep2=irp2, coef='Hess'):
                        self.assertEqual(
                            np.reshape(Hess_cisd[slices_cisd[irp],
                                                 slices_cisd[irp2]],
                                       (wf_CISD.n_corr_orb[irp],
                                        wf_CISD.n_ext[irp],
                                        wf_CISD.n_corr_orb[irp2],
                                        wf_CISD.n_ext[irp2])),
                            np.reshape(Hess_fci[slices_fci[irp],
                                                slices_fci[irp2]],
                                       (wf_FCI.ref_occ[irp],
                                        wf_FCI.n_ext[irp],
                                        wf_FCI.ref_occ[irp2],
                                        wf_FCI.n_ext[irp2]))[
                                            wf_FCI.n_core[irp]:, :,
                                            wf_FCI.n_core[irp2]:, :])
