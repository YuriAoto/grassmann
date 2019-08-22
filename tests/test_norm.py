import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + '/../')

import dGr_WF_int_norm as IntN
import dGr_FCI_Molpro as FCI

wf = IntN.Wave_Function_Int_Norm.from_Molpro('Li2_min_basis_R_5.0/CISD_wf.out')
wf.calc_norm()
print(wf.norm)

wf = FCI.Molpro_FCI_Wave_Function('Li2_min_basis_R_5.0/CISD_wf.out',
                                  FCI_file_name='Li2_min_basis_R_5.0/FCI_templ.out')


