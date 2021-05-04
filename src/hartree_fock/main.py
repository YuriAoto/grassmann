"""Main caller for Hartree-Fock


TODO: Loading molecular geometry elsewhere

Yuri Aoto, 2020
"""

from . import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
from input_output.log import logtime


def main(args, f_out):
    """Main function for Hartree-Fock programs"""
    
    molecular_system = MolecularGeometry.from_xyz_file(args.geometry)
    with logtime('Calculate integrals'):
#        args.integrals = integrals.mol_geo(args.basis, int_meth='ir-wmme') ##TODO:
        molecular_system.calculate_integrals(args.basis, int_meth='ir-wmme')
    RHF = optimiser.Restricted_Closed_Shell_SCF(molecular_system.integrals,
												molecular_system.nucl_rep,
												molecular_system.n_elec,
												f_out=f_out,
												n_DIIS=0)
    f_out.write(str(RHF))
    return RHF
