'''
'''

import sys
import pathlib 

import numpy as np

#sys.path.append(str(pathlib.Path(__file__).cwd())+'/../../')
sys.path.append(str(pathlib.Path(__file__).cwd())+'/')
print(sys.path)
from input_output import log
from orbitals.symmetry import OrbitalsSets
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.interm_norm_full import IntermNormWaveFunctionFull
from test.speed import residual 
from util import memory


no = int(sys.argv[1])
nt = int(sys.argv[2])

memory.set_total_memory(1000,'MB')
memory.allocate(memory.mem_of_floats(nt**4),'g matrix')
print('Memory for g matrix:')
print(memory.show_status())
g=np.random.rand(nt,nt,nt,nt)

occ = OrbitalsSets([no], occ_type='R')
tot = OrbitalsSets([nt], occ_type='R')
frozen = OrbitalsSets([0], occ_type='R')

wf = IntermNormWaveFunction.from_zero_amplitudes('C1',occ,tot,frozen,level='D')
wf.amplitudes[:] = np.random.rand(len(wf))

print(f'Memory of wf V2 = {wf.mem} {memory.unit()}')

wf_f = IntermNormWaveFunctionFull.from_V2(wf)

print(f'Memory of wf V3 = {wf_f.mem} {memory.unit()}')

with log.logtime('V2') as t_v2:
    A2_V2 = residual.cal_V2_A2(wf,g)

with log.logtime('V3') as t_v3:
    A2_V3 = residual.cal_V3_A2(wf_f,g)

#print('Is equal: ',A2_V2 == A2_V3)

#print(A2_V3.amplitudes)

print('For V2')
print(t_v2.elapsed_time)

print('For V3')
print(t_v3.elapsed_time)

