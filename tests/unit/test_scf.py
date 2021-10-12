"""Tests for Hartree-Fock - SCF


"""
import unittest


from hartree_fock import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
import tests


def _run_SCF(molecule, geometry, basis, max_iter=20):
    molecular_system = MolecularGeometry.from_xyz_file(tests.geom_file(molecule,
                                                                       geometry))
    molecular_system.calculate_integrals(basis, int_meth='ir-wmme')
    return optimiser.Restricted_Closed_Shell_SCF(molecular_system.integrals,
						 molecular_system.nucl_rep,
						 molecular_system.n_elec,
						 f_out=None,
						 n_DIIS=0,
                                                 max_iter=max_iter)


class RestrictedSCFTestCase(unittest.TestCase):
    """TestCase for the """

    @tests.category('VERY SHORT', 'ESSENTIAL')
    def test_h2_ccpvdz(self):
        resHF = _run_SCF(molecule='H2',
                         geometry='5',
                         basis='cc-pVDZ')
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -0.852458440109)
        self.assertEqual(resHF.n_iter, 6)

    @tests.category('SHORT')
    def test_h2o_631g_no_convergence(self):
        resHF = _run_SCF(molecule='h2o',
                         geometry='Rref_Helgaker',
                         basis='6-31g')
        self.assertFalse(resHF.success)
        self.assertAlmostEqual(resHF.energy, -75.984068514944)
        self.assertEqual(resHF.n_iter, 19)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_h2o_631g(self):
        resHF = _run_SCF(molecule='h2o',
                         geometry='Rref_Helgaker',
                         basis='6-31g',
                         max_iter=40)
        # print(resHF)
        # print(resHF.n_iter)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -75.9840686313831)
        self.assertEqual(resHF.n_iter, 27)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Li8_631g(self):
        resHF = _run_SCF(molecule='Li8_cage',
                         geometry='1.5',
                         basis='6-31g',
                         max_iter=40)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -59.51324535794139)
        self.assertEqual(resHF.n_iter, 11)

    @tests.category('LONG')
    def test_Li8_ccpvdz(self):
        resHF = _run_SCF(molecule='Li8_cage',
                         geometry='1.5',
                         basis='cc-pVDZ',
                         max_iter=40)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -59.52534568767985)
        self.assertEqual(resHF.n_iter, 15)
