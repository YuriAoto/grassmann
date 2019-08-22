import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

import dGr_WF_int_norm as IntN
import dGr_FCI_Molpro as FCI

Ua, Ub = FCI.transf_orb_from_to('Li2_min_basis_R_5.0/RHF.out',
                                'Li2_min_basis_R_5.0/UHF.out')

wf = IntN.Wave_Function_Int_Norm.from_Molpro('Li2_min_basis_R_5.0/CISD_wf.out')
Ua = Ua[:,:wf.n_alpha]
Ub = Ub[:,:wf.n_beta]

wf.calc_norm()
print(wf.distance_to_det((Ua, Ub)))

wf = FCI.Molpro_FCI_Wave_Function('Li2_min_basis_R_5.0/CISD_wf.out',
                                  FCI_file_name='Li2_min_basis_R_5.0/FCI_templ.out')

print(wf.distance_to_det((Ua, Ub)))


