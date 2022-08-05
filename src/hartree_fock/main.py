"""Main caller for Hartree-Fock


TODO: Loading molecular geometry elsewhere

Yuri Aoto, 2020
"""
import re

import numpy as np

from . import optimiser
from molecular_geometry.molecular_geometry import MolecularGeometry
from input_output.log import logtime
from orbitals import orbitals
from hartree_fock import starting_orbitals

def _define_hfstep_func(hf_step):
    """Create the function for Hartree-Fock steps from string hf_step"""
    if hf_step == 'SCF':
        return lambda i_SCF=None, grad_norm=None: 'RH-SCF'

    if hf_step == 'Absil':
        return lambda i_SCF=None, grad_norm=None: 'Absil'
    
    if hf_step == 'lagrange':
        return lambda i_SCF=None, grad_norm=None: 'lagrange'

    if hf_step == 'gradient':
        return lambda i_SCF=None, grad_norm=None: 'gradient'

    rematch = re.match('SCF-Absil_n(\d+)', hf_step)
    if rematch:
        n = int(rematch.group(1))
        return lambda i_SCF=0, grad_norm=None: 'RH-SCF' if i_SCF < n else 'Absil'

    rematch = re.match('SCF-Lagrange_n(\d+)', hf_step)
    if rematch:
        n = int(rematch.group(1))
        return lambda i_SCF=0, grad_norm=None: 'RH-SCF' if i_SCF < n else 'lagrange'

    rematch = re.match('SCF-Absil_grad(.+)', hf_step)
    if rematch:
        g = float(rematch.group(1))
        return lambda i_SCF=None, grad_norm=100.0: 'RH-SCF' if grad_norm > g else 'Absil'

    rematch = re.match('Gradient-Lagrange_n(\d+)', hf_step)
    if rematch:
        n = int(rematch.group(1))
        return lambda i_SCF=0, grad_norm=None: 'gradient' if i_SCF < n else 'lagrange'

    raise ValueError('Invalid HF step')


def main(args, f_out):
    """Main function for Hartree-Fock"""
    molecular_system = MolecularGeometry.from_xyz_file(args.geometry)
    with logtime('Calculate integrals'):
        molecular_system.calculate_integrals(args.basis, int_meth='ir-wmme')

    with logtime('Hartree-Fock optimisation') as T:
        HF = optimiser.hartree_fock(molecular_system.integrals,
                                    molecular_system.nucl_rep,
                                    molecular_system.n_elec,
                                    ms2=args.ms2,
                                    restricted=args.restricted,
                                    max_iter=args.max_iter,
                                    grad_thresh=1E-08,
                                    f_out=f_out,
                                    n_DIIS=args.diis,
                                    HF_step_type=_define_hfstep_func(args.step_type),
                                    ini_orb=starting_orbitals.initial_orbitals(args.ini_orb,
                                                                               molecular_system,
                                                                               args.restricted)
        )
    HF.totaltime = T.end_time - T.ini_time
    if f_out is not None: f_out.write(str(HF))
    return HF
