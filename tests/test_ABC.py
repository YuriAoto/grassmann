import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

import dGr_util as util
import dGr_WF_int_norm as IntN
import dGr_FCI_Molpro as FCI
import dGr_general_WF as genWF

Ua, Ub = FCI.transf_orb_from_to('Li2_min_basis_R_5.0/RHF.out',
                                'Li2_min_basis_R_5.0/UHF.out')


print('Interm norm...')
wf = IntN.Wave_Function_Int_Norm.from_Molpro('Li2_min_basis_R_5.0/CISD_wf.out')
Ua = Ua[:,:wf.n_alpha]
Ub = Ub[:,:wf.n_beta]
wf.calc_norm()
A1, B1, C1 = wf.get_ABC_matrices((Ua, Ub))

print('Using FCI...')
wf = FCI.Molpro_FCI_Wave_Function('Li2_min_basis_R_5.0/CISD_wf.out',
                                  FCI_file_name='Li2_min_basis_R_5.0/FCI_templ.out')
Ua = Ua[:,:wf.n_alpha]
Ub = Ub[:,:wf.n_beta]
A2, B2, C2 = wf.get_ABC_matrices((Ua, Ub))

print('Are both matrices A similar:', np.allclose(A1, A2, atol=1e-15))
print('Are both matrices B similar:', np.allclose(B1, B2, atol=1e-15))
print('Are both matrices C similar:', np.allclose(C1, C2, atol=1e-15))

