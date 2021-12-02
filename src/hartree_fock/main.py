"""Main caller for Hartree-Fock


TODO: Loading molecular geometry elsewhere

Yuri Aoto, 2020
"""

import numpy as np

from . import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
from input_output.log import logtime
from orbitals import orbitals


def main(args, f_out):
    """Main function for Hartree-Fock programs"""

    molecular_system = MolecularGeometry.from_xyz_file(args.geometry)
    with logtime('Calculate integrals'):
        molecular_system.calculate_integrals(args.basis, int_meth='ir-wmme')
    if args.restricted:
        HF = optimiser.Restricted_Closed_Shell_HF(molecular_system.integrals,
						  molecular_system.nucl_rep,
						  molecular_system.n_elec,
                                                  max_iter=args.max_iter,
						  f_out=f_out,
						  n_DIIS=args.diis,
                                                  grad_thresh=1E-05)
    else:
        orb = None
        if args.max_iter_scf > 0:
            HF = optimiser.Restricted_Closed_Shell_HF(molecular_system.integrals,
                                                      molecular_system.nucl_rep,
						      molecular_system.n_elec,
                                                      max_iter=args.max_iter_scf,
						      f_out=f_out,
                                                      n_DIIS=args.diis)
            orb = orbitals.MolecularOrbitals.unrestrict(HF.orbitals)
            # def myf(i_scf, grad):
            #     if i_scf < args.max_iter_scf:
            #         return 'RH-scf'
            #     return 'Absil'
        HF = optimiser.Unrestricted_HF(molecular_system.integrals,
				       molecular_system.nucl_rep,
				       molecular_system.n_elec,
                                       args.ms2,
                                       max_iter=args.max_iter,
				       f_out=f_out,
                                       ini_orb=orb,
                                       n_DIIS=args.diis,
                                       grad_thresh=1.0E-5)
        
    f_out.write(str(HF))
    return HF
