"""
"""
import copy

from wave_function.int_norm import IntermNormWaveFunction
from orbitals.symmetry import OrbitalsSets
import .ccsd

def cc_closed_shell(hf_energy,
                    mol_geom,
                    wf_ini=None,
                    preserve_wf_ini=False,
                    level='SD',
                    max_iter=20,
                    inter_matrix=True):
    """A Restricted Closed Shell Coupled-Cluster (CC) procedure
    
    Parameters:
    -----------
    
    hf_energy (float)
        The energy of the slater determinant reference

    mol_geom (MolecularGeometry)
        The molecular geometry, with all integrals already calculated

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

    inter_matrix (bool, optional, default=True)
        If True intermediate matrices are used. These are u, L and F matrices.
        If False all calculations uses only the one and two electrons (h and g) matrices.

    """
    if cc_ini == None:
        point_group = 'C1'
        orb_dim = OrbitalsSets([mol_geom.integrals.n_func],
                               occ_type='R')
        ref_occ = OrbitalsSets([mol_geom.n_elec],
                               occ_type='R')
        core_orb = OrbitalsSets([0],
                                occ_type='R')
        cc_wf = IntermNormWaveFunction.from_zero_amplitudes(
            point_group, ref_occ, orb_dim, core_orb, level=level)
    elif isinstance(wf_ini, IntermNormWaveFunction):
        if preserve_wf_ini:
            cc_wf = copy.deepcopy(wf_ini)
        else:
            cc_wf = wf_ini
    else:
        raise ValueError(
            'wf_ini must be an instance of IntermNormWaveFunction.')

    g = get_2e_MO(mol_geom.integrals.g,D_matrix)
    h = get_1e_MO(mol_geom.integrals.h,d_matrix)

    if cc_wf.inter_matrix:
        L = ccsd.make_L(g)
        F = ccsd.make_F(h,L,use_L=True)
    else:
        u = None
        L = None
        F = None
    
    omega1 = None

    i=0
    while True:
        i+=1
        CC_E = ccsd.energy(cc_wf,hf_energy)
        conv_status = ccsd.test_conv(omega1,omega2) 
        if conv_status or i == max_iter:
            break
 
        omega1,omega2=ccsd.equation(wf_cc,h,g,F,L)
        cc_wf.update_amplitudes(omega1,omega2)
