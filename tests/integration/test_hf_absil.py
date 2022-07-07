"""Tests for Hartree-Fock - Absil


"""
import unittest


from orbitals import orbitals
from hartree_fock import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
import tests


def _run_Absil(molecule, geometry, basis, max_iter=20, ms2=0, kwargsSCF=None):
    molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file(molecule, geometry))
    molecular_system.calculate_integrals(basis, int_meth='ir-wmme')
    orb = None
    if kwargsSCF is not None:
        HF = optimiser.Restricted_Closed_Shell_HF(molecular_system.integrals,
                                                  molecular_system.nucl_rep,
						  molecular_system.n_elec,
                                                  f_out=None,
                                                  **kwargsSCF)
        orb = orbitals.MolecularOrbitals.unrestrict(HF.orbitals)
    return optimiser.Unrestricted_HF(molecular_system.integrals,
				     molecular_system.nucl_rep,
				     molecular_system.n_elec,
                                     ms2=ms2,
                                     max_iter=max_iter,
                                     ini_orb=orb,
                                     f_out=None,
                                     grad_thresh=1.0E-5)
            

class HFAbsilTestCase(unittest.TestCase):
    """TestCase for the HF-Absil implementation."""

    @tests.category('VERY SHORT', 'ESSENTIAL')
    def test_H2_ccpvdz(self):
        resHF = _run_Absil(molecule='H2',
                           geometry='5',
                           basis='cc-pVDZ')
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -0.852458440109)
        self.assertEqual(resHF.n_iter, 3)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_H2O_631g_nc(self):
        """Test for H2O with basis 6-31g that didn't converge.

        In previous tests this converged to other points because of numerical
        errors due to computing the gradient and the hessian using other
        formulas, such as four indices integrals.
        """
        resHF = _run_Absil(molecule='H2O',
                           geometry='Rref_Helgaker',
                           basis='6-31g')
        self.assertFalse(resHF.success)
        self.assertAlmostEqual(resHF.energy, -68.58118956706662)
        self.assertEqual(resHF.n_iter, 19)
    
    @tests.category('SHORT', 'ESSENTIAL')
    def test_H2O_631g(self):
        kwargsSCF = {'max_iter': 3, 'n_DIIS': 2}
        resHF = _run_Absil(molecule='H2O',
                           geometry='Rref_Helgaker',
                           basis='6-31g',
                           kwargsSCF=kwargsSCF)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -75.98406863141355)
        self.assertEqual(resHF.n_iter, 3)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_H2O_631g_2(self):
        kwargsSCF = {'max_iter': 4, 'n_DIIS': 2}
        resHF = _run_Absil(molecule='H2O',
                           geometry='Rref_Helgaker',
                           basis='6-31g',
                           kwargsSCF=kwargsSCF)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -75.98406863141355)
        self.assertEqual(resHF.n_iter, 2)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_H2O_ccpvdz_nc(self):
        resHF = _run_Absil(molecule='H2O',
                           geometry='Rref_Helgaker',
                           basis='cc-pVDZ')
        self.assertFalse(resHF.success)
        self.assertAlmostEqual(resHF.energy, -62.154788593887375)
        self.assertEqual(resHF.n_iter, 19)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Be_631g(self):
        resHF = _run_Absil(molecule='Be',
                           geometry='at',
                           basis='6-31g')
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -13.580831335633)
        self.assertEqual(resHF.n_iter, 6)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Be_631g_ms22(self):
        resHF = _run_Absil(molecule='Be',
                           geometry='at',
                           basis='6-31g',
                           ms2=2)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -13.965506285000233)
        self.assertEqual(resHF.n_iter, 6)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Be_631g_ms2m2(self):
        resHF = _run_Absil(molecule='Be',
                           geometry='at',
                           basis='6-31g',
                           ms2=-2)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -13.965506285000)
        self.assertEqual(resHF.n_iter, 6)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Be_631g_ms24(self):
        resHF = _run_Absil(molecule='Be',
                           geometry='at',
                           basis='6-31g',
                           ms2=4)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -10.310318082828122)
        self.assertEqual(resHF.n_iter, 5)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Be_631g_ms2m4(self):
        resHF = _run_Absil(molecule='Be',
                           geometry='at',
                           basis='6-31g',
                           ms2=-4)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -10.310318082828122)
        self.assertEqual(resHF.n_iter, 5)
        
    @tests.category('SHORT', 'ESSENTIAL')
    def test_He2_631g_nc(self):
        """Test for He2 with basis 6-31g that didn't converge.

        In previous tests this converged to other points because of numerical
        errors due to computing the gradient and the hessian using other
        formulas, such as four indices integrals.
        """
        resHF = _run_Absil(molecule='He2',
                           geometry='1.5',
                           basis='6-31g')
        self.assertFalse(resHF.success)
        self.assertAlmostEqual(resHF.energy, -2.0343634735980247)
        self.assertEqual(resHF.n_iter, 19)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_He2_ccpvdz(self):
        kwargsSCF = {'max_iter': 1, 'n_DIIS': 0}
        resHF = _run_Absil(molecule='He2',
                           geometry='1.5',
                           basis='cc-pVDZ',
                           kwargsSCF=kwargsSCF)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -5.361487027302651)
        self.assertEqual(resHF.n_iter, 2)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_He8_631g(self):
        kwargsSCF = {'max_iter': 1, 'n_DIIS': 0}
        resHF = _run_Absil(molecule='He8_cage',
                           geometry='1.5',
                           basis='6-31g',
                           kwargsSCF=kwargsSCF)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -22.841474436008095)
        self.assertEqual(resHF.n_iter, 2)

    @tests.category('SHORT', 'ESSENTIAL')
    def test_Li2_631g(self):
        kwargsSCF = {'max_iter': 1, 'n_DIIS': 0}
        resHF = _run_Absil(molecule='Li2',
                           geometry='5',
                           basis='6-31g',
                           kwargsSCF=kwargsSCF)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -14.86545209295543)
        self.assertEqual(resHF.n_iter, 4)

    @tests.category('LONG', 'ESSENTIAL')
    def test_N2_631g(self):
        kwargsSCF = {'max_iter': 1, 'n_DIIS': 0}
        resHF = _run_Absil(molecule='N2',
                           geometry='3',
                           basis='6-31g',
                           kwargsSCF=kwargsSCF)
        self.assertTrue(resHF.success)
        self.assertAlmostEqual(resHF.energy, -108.33006303532665)
        self.assertEqual(resHF.n_iter, 3)
