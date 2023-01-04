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

    if hf_step == 'RNR':
        return lambda i_SCF=None, grad_norm=None: 'RNR'
    
    if hf_step == 'NRLM':
        return lambda i_SCF=None, grad_norm=None: 'NRLM'

    if hf_step == 'RGD':
        return lambda i_SCF=None, grad_norm=None: 'RGD'

    if hf_step == 'GDLM':
        return lambda i_SCF=None, grad_norm=None: 'GDLM'

    if hf_step == 'RCG':
        return lambda i_SCF=None, grad_norm=None: 'RCG'

    rematch = re.match('SCF-RNR_n(\d+)', hf_step)
    if rematch:
        n = int(rematch.group(1))
        return lambda i_SCF=0, grad_norm=None: 'RH-SCF' if i_SCF < n else 'RNR'

    rematch = re.match('SCF-NRLM_n(\d+)', hf_step)
    if rematch:
        n = int(rematch.group(1))
        return lambda i_SCF=0, grad_norm=None: 'RH-SCF' if i_SCF < n else 'NRLM'

    rematch = re.match('SCF-RNR_grad(.+)', hf_step)
    if rematch:
        g = float(rematch.group(1))
        return lambda i_SCF=None, grad_norm=100.0: 'RH-SCF' if grad_norm > g else 'RNR'

    rematch = re.match('SCF-NRLM_grad(.+)', hf_step)
    if rematch:
        g = float(rematch.group(1))
        return lambda i_SCF=0, grad_norm=100.0: 'RH-SCF' if grad_norm > g else 'NRLM'

    rematch = re.match('RNR-RCG_n(\d+)', hf_step)
    if rematch:
        n = int(rematch.group(1))
        return lambda i_SCF=0, grad_norm=None: 'RNR' if i_SCF < n else 'RCG'

    raise ValueError('Invalid HF step')


def main(args, f_out):
    """Main function for Hartree-Fock"""
    molecular_system = MolecularGeometry.from_xyz_file(args.geometry)
    with logtime('Setting integrals'):
        molecular_system.calculate_integrals(args.basis, int_meth='ir-wmme')

    diis_info = optimiser.DiisInfo(n=args.diis,
                                   at_F=args.diis_at_F,
                                   at_P=args.diis_at_P)

    if args.grad_type is None:
        grad_type = 'F_occ_virt' if args.diis and args.diis_at_P else 'F_asym'
    else:
        grad_type = args.grad_type

    ini_orb = starting_orbitals.initial_orbitals(args.ini_orb,
                                                 molecular_system,
                                                 args.restricted,
                                                 args.conjugacy,
                                                 args.step_size)

    with logtime('Hartree-Fock optimisation') as T:
        HF = optimiser.hartree_fock(molecular_system.integrals,
                                    molecular_system.nucl_rep,
                                    molecular_system.n_elec - args.charge,
                                    ms2=args.ms2,
                                    restricted=args.restricted,
                                    max_iter=args.max_iter,
                                    grad_thresh=1E-8,
                                    grad_type=grad_type,
                                    f_out=f_out,
                                    diis_info=diis_info,
                                    conjugacy=args.conjugacy,
                                    step_size=args.step_size,
                                    HF_step_type=_define_hfstep_func(args.step_type),
                                    ini_orb=ini_orb)

    HF.totaltime = T.end_time - T.ini_time
    if f_out is not None: f_out.write(str(HF))
    return HF
