"""
"""
import copy

from util.results import OptResults
from wave_functions.interm_norm import IntermNormWaveFunction
from orbitals.orbital_space import OrbitalSpace
import coupled_cluster.coupled_cluster as ccsd

def cc_closed_shell(hf_energy,
                    mol_orb,
                    atom_int,
                    wf_ini=None,
                    preserve_wf_ini=False,
                    level='SD',
                    max_inter=20):
    """A Restricted Closed Shell Coupled-Cluster (CC) procedure
    
    Parameters:
    -----------
    
    hf_energy (float)
        The energy of the slater determinant reference

    mol_orb (MolecularOrbital)
        The molecular orbitals 

    atom_int (Integrals)
        Integrals from atomic basis set with the associated MolecularGeometry properly setted.

    wf_ini (IntermNormWaveFunction, optional, default=None)
        A previous wf can be used as initial guess, if None a new one is generated

    preserve_wf_ini (bool, optional, default=False)
        If True a wf_ini input will be duplicated and the copy modified, otherwise the wf_ini will be modified.

    level (str, optional, default='SD')
        Types of excitations included in the calculation:
        - SD: single and duble excitations, T1-transformation is used;
        -  D: only duble excitations.

    max_iter (int, optional, default=20)
        Maximum number of iterations

    """
    if wf_ini == None:
        point_group = 'C1'
        orb_dim = OrbitalSpace([len(mol_orb)],
                               occ_type='R')
        ref_occ = OrbitalSpace([atom_int.mol_geo.n_elec],
                               occ_type='R')
        core_orb = OrbitalSpace([0],
                                occ_type='R')
        wave_function = IntermNormWaveFunction.from_zero_amplitudes(
            point_group, ref_occ, orb_dim, core_orb, level=level)
    elif isinstance(wf_ini, IntermNormWaveFunction):
        if preserve_wf_ini:
            wave_function = copy.deepcopy(wf_ini)
        else:
            wave_function = wf_ini
    else:
        raise ValueError(
            'wf_ini must be an instance of IntermNormWaveFunction.')
    mol_orb.molecular_integrals_gen(atom_int)
    F = ccsd.make_F(mol_orb.molecular_integrals.h,mol_orb.molecular_integrals.g._integrals,len(wave_function.ref_orb)//2)
    omega = None

    n_inter = -1
    while True:
        n_inter+=1
        energy = ccsd.energy(wave_function,hf_energy,mol_orb.molecular_integrals.g._integrals)
        conv_status = ccsd.test_conv(omega) 
        if conv_status or n_inter == max_inter:
            break
 
        omega,update = ccsd.equation(wave_function,F,mol_orb.molecular_integrals.g._integrals)
        print(omega,wave_function.amplitudes)
        wave_function.update_amplitudes(update)

    results = OptResults('CC'+level)
    results.wave_function=wave_function
    results.energy=energy
    results.n_inter=n_inter
    results.success = conv_status
    return results 
