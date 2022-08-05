"""Profiling Hartree-Fock as in Absil.

"""
import unittest
import cProfile
import pstats
from pstats import SortKey
import tracemalloc

from orbitals import orbitals
from hartree_fock import optimiser, starting_orbitals
from molecular_geometry.molecular_geometry import MolecularGeometry
import tests


def _run_Lagrange(molecule, geometry, basis, ini_orb, max_iter=10, ms2=0, kwargsSCF=None):
    molecular_system = MolecularGeometry.from_xyz_file(
        tests.geom_file(molecule, geometry))
    molecular_system.calculate_integrals(basis, int_meth='ir-wmme')
    orb = starting_orbitals.initial_orbitals(ini_orb, molecular_system, restricted=False)
    # if kwargsSCF is not None:
    #     HF = optimiser.hartree_fock(molecular_system.integrals,
    #                                 molecular_system.nucl_rep,
    #     			    molecular_system.n_elec,
    #                                 f_out=None,
    #                                 **kwargsSCF)
    #     orb = orbitals.MolecularOrbitals(HF.orbitals)
    return optimiser.hartree_fock(molecular_system.integrals,
				  molecular_system.nucl_rep,
				  molecular_system.n_elec,
                                  ms2=ms2,
                                  max_iter=max_iter,
                                  ini_orb=orb,
                                  restricted=False,
                                  f_out=None,
                                  HF_step_type=lambda **x: "lagrange",
                                  n_DIIS=0,
                                  grad_thresh=1.0E-5)


class HFLagrangeProfile(unittest.TestCase):

    @tests.category('SHORT', 'ESSENTIAL')
    def test_H2O_ccPVQZ(self):
        cProfile.runctx('_run_Lagrange(molecule="H2O",\
        geometry="Rref_Helgaker",\
        basis="cc-pVTZ",\
        ini_orb="SAD")', globals(), locals(), 'run_stats')
        p = pstats.Stats('run_stats')
        p.sort_stats(SortKey.TIME)
        p.print_stats()
