import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + '/../')

import dGr_CISD_WF as CIwf
import dGr_WF_int_norm as IntN


print('checking list initialization in Wave_Function_CISD:')
wf = IntN.Wave_Function_Int_Norm.from_Molpro('Li2__R_5__631g__D2h/CISD.out')
print(wf)
print(wf.doubles[0][0])

wf = CIwf.Wave_Function_CISD.from_intNorm(wf)
print()
print(wf)
print()
print(wf.Cd)
