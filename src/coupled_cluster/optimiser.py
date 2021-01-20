"""
"""

from wave_function import int_norm
from coupled_cluster import ccsd

def cc_closed_shell(hf_energy,
                    h,
                    g,
                    wf_ini = None,
                    level = 'SD'
                    maxit = 30,
                    inter_matrix = True):
    if cc_ini == None:
        cc_wf = int_norm.IntermNormWaveFunction()
        cc_wf.level = level
    elif isinstance(wf_ini, IntermNormWaveFunction):
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
