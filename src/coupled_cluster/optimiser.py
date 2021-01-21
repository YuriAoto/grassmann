"""
"""
import copy

from wave_function.int_norm import IntermNormWaveFunction
from wave_functions.general import OrbitalsSets
from coupled_cluster import ccsd

def cc_closed_shell(hf_energy,
                    mol_geom,
                    wf_ini = None,
                    preserve_wf_ini=False,
                    level='SD',
                    maxit=30,
                    inter_matrix=True):
    """
    
    Parameters:
    -----------

    

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

    if cc_wf.inter_matrix:
        F = ccsd.make_F(h,g)
        L = ccsd.make_L(g)
    else:
        u=None
        L=None
        F=None

    i=0
    while True:
        i+=1
        ccsd.energy(cc_wf)
 
        if ccsd.test_conv(cc_wf) or i == maxit:
            break
 
        if cc_wf.level == 'SD':
            ccsd.get_t1_transf()
            if cc_wf.inter_matrix:
                u = ccsd.make_u(cc_wf)
                F = ccsd.make_t1_F(h,g,F)
                L = ccsd.make_t1_L(g,L)   
        elif cc_wf.level == 'D':
            if cc_wf.inter_matrix:
                u = ccsd.matrix(cc_wf,('u'))
        else:
            raise ValueError(
                'CC version '+cc_wf.level+' unknown.')
 
        ccsd.equation(wf_cc,u,L,F)
        cc_wf.update_amplitudes
