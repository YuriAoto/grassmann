'''
'''

import numpy as np

from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.interm_norm_full import IntermNormWaveFunctionFull
from tests.implementation import residual_cy

def cal_V2_A2(wf,g):
    residual = IntermNormWaveFunction.similar_to(
                 wf, 'residual of '+wf.wf_type, wf.restricted)
    no = wf.corr_orb[0]
    residual_cy.calc_A2(g,wf.amplitudes,residual.amplitudes,no)
    return residual

def cal_V3_A2(wf,g):
    residual = IntermNormWaveFunctionFull.similar_to(
                 wf, 'residual of '+wf.wf_type, wf.restricted)
    no = wf.corr_orb[0]
    residual.amplitudes[:,:,:,:] = np.einsum('aibj->ijab',g[no:,:no,no:,:no])
    residual.amplitudes[:,:,:,:]+= np.einsum('ijcd,acbd->ijab',wf.amplitudes,g[no:,no:,no:,no:])
    return residual
