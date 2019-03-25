#!/usr/bin/python3
import sys
import re
import os
import numpy as np
from scipy import linalg
import dist_to_FCI_Molpro as dFCI
from opt_dist_FCI import optimize_distance_to_FCI
import datetime
import time
import argparse
import math


# ----- command line
parser = argparse.ArgumentParser(description='Optimize the Psi_minD.')
parser.add_argument('molpro_output')
parser.add_argument('-U', '--ini_orb')
parser.add_argument('--UHF_orb')
parser.add_argument('--state')
parser.add_argument('-v', '--verbose')
parser.add_argument('-f', '--FCIwf')

cmd_args = parser.parse_args()

file_name = cmd_args.molpro_output
if cmd_args.ini_orb is None:
    file_name_iniU = None
else:
    file_name_iniU = (cmd_args.ini_orb + '_Ua.npy',
                      cmd_args.ini_orb + '_Ub.npy')
if cmd_args.UHF_orb is None:
    uhf_orb = None
else:
    uhf_orb = cmd_args.UHF_orb
state = cmd_args.state if cmd_args.state is not None else ''
verbose = 0 if cmd_args.verbose is None else int(cmd_args.verbose)

FCI_file_name = cmd_args.FCIwf

if 'QPY_JOB_ID' in os.environ:
    QPY_ID = os.environ['QPY_JOB_ID']
else:
    QPY_ID = 'not in qpy'

# ----- output
file_name_out = re.sub('\.out$', '', file_name)
f_out = open(file_name_out + '.min_dist' + \
             state + \
             ('_U' if uhf_orb is not None else ''), 'w')

start_time = time.time()

if verbose > 10:
    print('Molpro output: ', file_name)
    if FCI_file_name is not None:
        print('Molpro FCI output: ', FCI_file_name)
    print('State: ', state)
    print('Verbose', verbose)


ext_wf = dFCI.Molpro_FCI_Wave_Function(file_name,
                                       state if state else '1.1',
                                       FCI_file_name = FCI_file_name)

##print('External wave function, initial coefficients:\n' + str(ext_wf))

# S = 0.0
# for det in ext_wf.determinants:
#     S += det[0]**2
# print('Norm of external WF: ',math.sqrt(S))

### exit()

if uhf_orb is not None:
    Ua, Ub = dFCI.get_transf_RHF_to_UHF(file_name, uhf_orb)
    f_out.write('Using external wave function in the basis of UHF orbitals: ' + uhf_orb + '\n')
    if verbose > 20:
        f_out.write('External Wave function, coefficients before changing to new basis:\n'
                    + str(ext_wf))
    ext_wf = dFCI.transform_wf(ext_wf, Ua, Ub)

if file_name_iniU is not None:
    Ua = np.load(file_name_iniU[0])
    Ub = np.load(file_name_iniU[1])
    if verbose > 5:
        f_out.write('Using initial guess for U: {0:s}, {1:s}\n'.\
                    format(file_name_iniU[0], file_name_iniU[1]))

else:
    Ua, Ub = dFCI.get_trans_max_coef(ext_wf)
    if verbose > 5:
        f_out.write('Using initial guess for U from det with largest coefficient.\n')
        f_out.write('Ua for alpha orbitals:\n' + str(Ua) + '\n')
        f_out.write('Ub for beta orbitals:\n' + str(Ub) + '\n')

if verbose > 5:
    f_out.write('Molpro output: {0:s}\n'.\
                format(file_name))
    f_out.write('qpy job ID: {:s}\n'.\
                format(QPY_ID))
    f_out.write('Lenght of FCI vector: {0:d}\n'.\
                format(len(ext_wf.determinants)))
    f_out.write('Dimension of orbital space: {0:d}\n'.\
                format(ext_wf.orb_dim))
    f_out.write('Number of alpha and beta electrons: {0:d} {1:d}\n'.\
                format(ext_wf.n_alpha, ext_wf.n_beta))
    f_out.write('First determinant: {0}\n'.\
                format(ext_wf.determinants[0]))
    if verbose > 20:
        f_out.write('External Wave function, initial coefficients:\n' + str(ext_wf))

res = optimize_distance_to_FCI(ext_wf,
                               dFCI.construct_Jac_Hess,
                               dFCI.print_Jac_Hess,
                               dFCI.calc_wf_from_z,
                               dFCI.transform_wf,
                               verbose = verbose,
                               f_out = f_out,
                               ini_U = (Ua, Ub))

Ua, Ub = res[0]
norm_Z, norm_J = res[1]
n_iter = res[2]
conv = res[3]


if conv:
    f_out.write('Iterations converged!\n')
else:
    f_out.write('WARNING: Iterations not converged!!!\n')
    f_out.write('Norm of final z vector: {0:8.4e}\n'.format(norm_Z))
    f_out.write('Norm of final Jacobian: {0:8.4e}\n'.format(norm_J))


np.save(file_name_out + '-min_dist_Ua', Ua)
np.save(file_name_out + '-min_dist_Ub', Ub)

f_out.write('\n' + '='*30 + '\n')
f_out.write('First determinant in initial basis:\n|min_E> = {0:s}\n'.\
            format(str(ext_wf.determinants[0])))

f_out.write('Transforming FCI initial wave function to the new basis.\n')
new_WF = dFCI.transform_wf(ext_wf, Ua, Ub)

f_out.write('Checking Hessian engenvalues...\n')
Jac, Hess = dFCI.construct_Jac_Hess(new_WF, f_out, verb = 0)
eig_val, eig_vec = linalg.eigh(Hess)
n_pos_eigVal = 0
for i in eig_val:
    if i > 0.0:
        n_pos_eigVal += 1
if n_pos_eigVal > 0:
    f_out.write('WARNING: Found {0:d} positive eigenvalue(s) for Hessian.\n'.\
                format(n_pos_eigVal))
else:
    f_out.write('All eigenvalues are negative: OK!\n')

max_coef = 0.0
i_max_coef = -1
for i,c in enumerate(new_WF.determinants):
    if abs(c[0]) > max_coef:
        max_coef = abs(c[0])
        i_max_coef = i
f_out.write('Determinant with largest coefficient: {0:d}\n|min_dist> = {1:s}\n'.\
            format(i_max_coef, str(new_WF.determinants[i_max_coef])))

f_out.write('Transforming determinant of FCI in initial basis to the new basis.\n')
# Note: This is destroing original FCI!
first_det = True
for det in ext_wf.determinants:
    if first_det:
        det[0] = 1.0
        first_det = False
    else:
        det[0] = 0.0

det_in_new_basis = dFCI.transform_wf(ext_wf, Ua, Ub)

if verbose > 20:
    f_out.write('Determinant in new basis:\n' + str(det_in_new_basis))

end_time = time.time()
elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))

if verbose > 5:
    f_out.write('Total time: {0:s}\n'.format(elapsed_time))

f_out.write('|<min_dist|min_E>| = {0:15.12f}'.\
            format(abs(det_in_new_basis.determinants[i_max_coef][0])))

f_out.close()

exit()
