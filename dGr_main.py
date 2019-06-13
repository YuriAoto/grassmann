"""Main function of dGr

"""
import os
import datetime
import time
import math
import logging

import numpy as np
from scipy import linalg

import dGr_FCI_Molpro as dFCI
from dGr_opt_FCI import optimize_distance_to_FCI

logger = logging.getLogger(__name__)

def dGr_main(args):
    f_out = open(args.basename + '.min_dist' + args.state, 'w')
    def toout(x):
        f_out.write(x + '\n')
    toout('dGr - optimize the distance in the Grassmannian')
    toout('Yuri Aoto - 2018, 2019')
    toout('')
    toout('Directory:\n' + args.wdir)
    toout('')
    toout('Command:\n' + args.command)
    toout('')
    toout('From command line and environment:')
    toout('External wave function, |extWF>: ' + args.molpro_output)
    if args.WF_templ is not None:
        toout('Template for |extWF>: ' + args.WF_templ)
    toout('Orbital basis of |extWF>: ' + args.WF_orb)
    toout('Hartree-Fock orbitals, |minE>: ' + args.HF_orb)
    if args.ini_orb is not None:
        f_out.write('Initial orbitals: ')
        if isinstance(args.ini_orb, tuple):
            toout('Numpy files ' + str(args.ini_orb[0]) +
                        ' and ' + str(args.ini_orb[1]))
        else:
            toout('Molpro output ' + args.ini_orb)
    toout('State: ' + args.state)
    toout('Log level: ' + logging.getLevelName(args.loglevel))
    toout('qpy job ID: {:s}'.\
                format(os.environ['QPY_JOB_ID']
                       if 'QPY_JOB_ID' in os.environ
                       else 'not in qpy'))
    toout('')
    start_time = time.time()
    toout('Starting at {}'.format(
        time.strftime("%d %b %Y - %H:%M",time.gmtime(start_time))))
    toout('')
    # ----- loading wave function
    ext_wf = dFCI.Molpro_FCI_Wave_Function(args.molpro_output,
                                           args.state if args.state else '1.1',
                                           FCI_file_name = args.WF_templ)

    logger.info('External wave function, initial coefficients:\n' + str(ext_wf))
    if logger.level <= logging.INFO:
        S = 0.0
        for det in ext_wf.determinants:
            S += det[0]**2
        logger.info('Norm of external WF: %f',
                    math.sqrt(S))

    if args.HF_orb is not None:
        Ua, Ub = dFCI.transf_orb_from_to(args.WF_orb, args.HF_orb)
        if logger.level <= logging.DEBUG:
            logger.debug('Wave function, before changing to new basis:\n%s',
                         str(ext_wf))
        ext_wf = dFCI.transform_wf(ext_wf, Ua, Ub)

    if args.ini_orb is not None:
        if isinstance(args.ini_orb, tuple):
            Ua = np.load(args.ini_orb[0])
            Ub = np.load(args.ini_orb[1])
        else:
            if args.HF_orb is not None:
                Ua, Ub = dFCI.transf_orb_from_to(args.HF_orb,
                                                 args.ini_orb)
            else:
                Ua, Ub = dFCI.transf_orb_from_to(args.WF_orb,
                                                 args.ini_orb)
    else:
        Ua, Ub = dFCI.get_trans_max_coef(ext_wf)
        toout('Using initial guess for U from det with largest coefficient.')
        logger.debug('Ua for alpha orbitals:\n' + str(Ua))
        logger.debug('Ub for beta orbitals:\n' + str(Ub))

    toout('Lenght of FCI vector: {0:d}'.\
                format(len(ext_wf.determinants)))
    toout('Dimension of orbital space: {0:d}'.\
                format(ext_wf.orb_dim))
    toout('Number of alpha and beta electrons: {0:d} {1:d}'.\
                format(ext_wf.n_alpha, ext_wf.n_beta))
    toout('First determinant: {0}'.\
                format(ext_wf.determinants[0]))

    if logger.level <= logging.DEBUG:
        logger.debug('Wave function, initial coefficients:\n%s',
                     str(ext_wf))

    toout('')
    toout('Starting optimisation')
    toout('-'*30)
    logger.info('Starting optimisation')
    res = optimize_distance_to_FCI(ext_wf,
                                   dFCI.construct_Jac_Hess,
                                   dFCI.str_Jac_Hess,
                                   dFCI.calc_wf_from_z,
                                   dFCI.transform_wf,
                                   f_out = f_out,
                                   ini_U = (Ua, Ub))
    toout('-'*30)
    logger.info('Optimisation completed')

    Ua, Ub = res[0]
    norm_Z, norm_J = res[1]
    n_iter = res[2]
    conv = res[3]

    if conv:
        toout('Iterations converged!')
    else:
        toout('WARNING: Iterations not converged!!!')
        toout('Norm of final z vector: {0:8.4e}'.format(norm_Z))
        toout('Norm of final Jacobian: {0:8.4e}'.format(norm_J))
    toout('')
        
    np.save(args.basename + '-min_dist_Ua', Ua)
    np.save(args.basename + '-min_dist_Ub', Ub)

    toout('First determinant in initial basis:\n|min_E> = {0:s}'.\
                format(str(ext_wf.determinants[0])))

    logger.info('Transforming FCI initial wave function to the new basis.')
    new_WF = dFCI.transform_wf(ext_wf, Ua, Ub)

    logger.info('Checking Hessian engenvalues')
    Jac, Hess = dFCI.construct_Jac_Hess(new_WF, f_out)
    eig_val, eig_vec = linalg.eigh(Hess)
    n_pos_eigVal = 0
    for i in eig_val:
        if i > 0.0:
            n_pos_eigVal += 1
    if n_pos_eigVal > 0:
        toout('WARNING: Found {0:d} positive eigenvalue(s) for Hessian.'.\
                    format(n_pos_eigVal))
        logger.warning('Found {0:d} positive eigenvalue(s) for Hessian.'.\
                    format(n_pos_eigVal))
    else:
        logger.info('All eigenvalues are negative: OK!')

    max_coef = 0.0
    i_max_coef = -1
    for i,c in enumerate(new_WF.determinants):
        if abs(c[0]) > max_coef:
            max_coef = abs(c[0])
            i_max_coef = i
    toout('Determinant with largest coefficient: {0:d}\n|min_dist> = {1:s}'.\
                format(i_max_coef, str(new_WF.determinants[i_max_coef])))

    logger.info('Transforming determinant of FCI in initial basis to the new basis.')
    # Note: This is destroing original FCI!
    first_det = True
    for det in ext_wf.determinants:
        if first_det:
            det[0] = 1.0
            first_det = False
        else:
            det[0] = 0.0

    det_in_new_basis = dFCI.transform_wf(ext_wf, Ua, Ub)
    logger.debug('Determinant in new basis:\n%s',
                 str(det_in_new_basis))

    toout('|<min_dist|min_E>| = {0:15.12f}'.\
                format(abs(det_in_new_basis.determinants[i_max_coef][0])))

    toout('')
    end_time = time.time()
    elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))
    toout('Ending at {}'.format(
        time.strftime("%d %b %Y - %H:%M",time.gmtime(end_time))))
    toout('Total time: {}'.format(elapsed_time))
    f_out.close()
