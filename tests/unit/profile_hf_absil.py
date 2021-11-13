"""Profiling Hartree-Fock as in Absil.

"""
import unittest
import cProfile
import pstats
from pstats import SortKey
import tracemalloc

from orbitals import orbitals
from hartree_fock import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
import tests


def _run_Absil(molecule, geometry, basis, max_iter=5, ms2=0, kwargsSCF=None):
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


class HFAbsilProfile(unittest.TestCase):

    @tests.category('SHORT', 'ESSENTIAL')
    def test_h2o_631g(self):
        cProfile.runctx('_run_Absil(molecule="h2o",\
        geometry="Rref_Helgaker",\
        basis="6-31g")', globals(), locals(), 'run_stats')
        p = pstats.Stats('run_stats')
        p.sort_stats(SortKey.TIME)
        p.print_stats()
