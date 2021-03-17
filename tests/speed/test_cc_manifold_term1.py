import sys

sys.path.append('../../')

import tests
from wave_functions.fci import FCIWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from input_output import log
from cc_manifold_term1 import testspeed, testspeed_2


system = 'Li2__5__sto3g__D2h'

wf = FCIWaveFunction.from_Molpro_FCI(
    tests.FCI_file(system, allE=True))
cc = IntermNormWaveFunction.from_Molpro(
    tests.CCSD_file(system, allE=True))
cc.amplitudes *= 0.8
wf_cc = FCIWaveFunction.from_int_norm(cc)

i=1
j=0
a=2
b=3

with log.logtime('V1') as T1:
    x1 = testspeed(i,j,a,b,
                   wf._coefficients,
                   wf_cc._coefficients,
                   wf._alpha_string_graph,
                   wf._beta_string_graph)

with log.logtime('V2') as T2:
    x2 = testspeed_2(i,j,a,b,
                     wf._coefficients,
                     wf_cc._coefficients,
                     wf._alpha_string_graph,
                     wf._beta_string_graph)

print(f'x1 = {x1}')
print(f'x2 = {x2}')

print(f'T1 = {T1.elapsed_time}')
print(f'T2 = {T2.elapsed_time}')
