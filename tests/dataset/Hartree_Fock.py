"""Tests for Hartree-Fock over the G2/97 dataset

Run from the main directory of grassmann

Usage:

python Hartree_Fock.py <step_type> <n_diis> <ini_orb> <basis> <max_iter>


Example:

python Hartree_Fock.py SCF 0 SAD 6-31g 30

"""
import os
import sys

sys.path.append(os.getcwd())

import tests
from src.hartree_fock.main import main as run_HF
import dataset

step_type = sys.argv[1]
n_diis = int(sys.argv[2])
ini_orb = sys.argv[3]
basis = sys.argv[4]
max_iter = int(sys.argv[5])



out_name = f'iterations_{step_type}__{n_diis}__{ini_orb}__{basis}.gr'



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
    args.ini_orb = ini_orb
    args.step_type = step_type


f_out = open(out_name, 'w')
for mol, args in dataset.iterate_over('g2_1'):
    f_out.write(f'\n\n -----\n\n{mol}\n\n -----\n')
    fill_args(args)
    try:
        res_SCF[mol] = run_HF(args, f_out)
    except Exception as e:
        f_out.write('\nException:\n{e}\n\n')
    os.remove(args.geometry)

n_conv = 0
n_iter_total = 0
total_time = 0

for mol, res in res_SCF.items():
    if res.success:
        total_time += res.totaltime
        n_conv += 1
        n_iter_total += res.n_iter

f_out.write('\n\n----------\nStatistics:\n\n')
f_out.write(f'Converged: {n_conv} from a total of {len(res_SCF)} ({100*n_conv/len(res_SCF):.2f}%)\n')
f_out.write(f'Average n_iter: {n_iter_total/n_conv:.2f}\n')
f_out.write(f'Total time (for converged calculations): {total_time:.5f} seconds\n')


f_out.close()
