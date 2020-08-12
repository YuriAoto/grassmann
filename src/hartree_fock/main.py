"""Main caller for Hartree-Fock


TODO: Loading molecular geometry elsewhere

Yuri Aoto, 2020
"""
from . import naive
from . import optimiser
from molecular_geometry import MolecularGeometry
from util import logtime


def main(args, f_out):
    """Main function for Hartree-Fock programs"""
    
    molecular_system = MolecularGeometry.from_xyz_file(args.input_file)
    with logtime('Calculate integrals'):
        molecular_system.calculate_integrals(args.basis, int_meth='ir-wmme')
    RHF = optimiser.Restricted_Closed_Shell_SCF(
        molecular_system, f_out=f_out, n_DIIS=0)
    f_out.write(RHF)
