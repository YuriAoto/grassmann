"""Tests for Hartree-Fock over the G2/97 dataset

Run from the main directory of grassmann

"""
import os
import sys

sys.path.append(os.getcwd())

import tests
from src.hartree_fock.main import main as run_HF
import dataset

basis = '6-31g'
max_iter = 30
n_diis = 0
step_type = 'SCF'

keep_xyz_files = False



res_RRN = {}
res_SCF = {}
res_DIIS_5 = {}

def fill_args(args):
    """Add needed things to args"""
    args.ms2 = None  # hartree_fock code should find it
    args.restricted = None  # hartree_fock code should find it
    args.basis = basis
    args.max_iter = max_iter
    args.diis = n_diis
    args.ini_orb = None
    args.step_type = step_type


f_out = open('iterations_from_SCF.gr', 'w')
for mol, args in dataset.iterate_over('g2_1'):
    f_out.write(f'\n\n -----\n\n{mol}\n\n -----\n')
    fill_args(args)
    res_SCF[mol] = run_HF(args, f_out)
    if not keep_xyz_files:
        os.remove(args.geometry)
    break
f_out.close()


n_conv = 0
n_iter_total = 0
total_time = 0

for mol, res in res_SCF.items():
    if res.success:
        total_time += res.totaltime
        n_conv += 1
        n_iter_total += res.n_iter

print(f'Converged: {n_conv} from a total of {len(res_SCF)} ({100*n_conv/len(res_SCF):.2f}%)')
print(f'Average n_iter: {n_iter_total/n_conv}')
print(f'Total time (for converged calculations): {total_time}')


