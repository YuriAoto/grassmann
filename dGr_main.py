"""Main function of dGr

Functions:
----------
dGr_main
"""
import os
import datetime
import time
import math
import logging

import git
import numpy as np
from scipy import linalg

from dGr_util import dist_from_ovlp, ovlp_Slater_dets
import dGr_orbitals as orb
import dGr_FCI_Molpro as FCI
import dGr_WF_int_norm as IntN
import dGr_optimiser

logger = logging.getLogger(__name__)

def dGr_main(args, f_out):
    """The main function to calculate the distance to the Grassmannian.
    
    Parameters:
    ---------
    args (argparse.Namespace)
        The parsed information. See dGrParseError.parse_cmd_line
    
    f_out (file)
        where the output goes
    """
    git_repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
    git_sha = git_repo.head.object.hexsha
    use_FCIopt = False
    def toout(x=''):
        f_out.write(x + '\n')
    def print_ovlp_D(pt1, pt2, ovlp):
        toout('|<{0:s}|{1:s}>| = {2:12.8f} ; D({0:s}, {1:s}) = {3:12.8f}'.\
              format(pt1, pt2,
                     abs(ovlp),
                     dist_from_ovlp(ovlp)))
    toout('dGr - optimise the distance in the Grassmannian')
    toout('Yuri Aoto - 2018, 2019')
    toout()
    toout('Current git commit: ' + git_sha)
    toout()
    toout('Directory:\n' + args.wdir)
    toout()
    toout('Command:\n' + args.command)
    toout()
    toout('From command line and environment:')
    toout('External wave function, |extWF>: ' + args.molpro_output)
    if args.WF_templ is not None:
        toout('Template for |extWF>: ' + args.WF_templ)
        use_FCIopt = True
    toout('Orbital basis of |extWF>: ' + args.WF_orb)
    toout('Hartree-Fock orbitals, |min E>: ' + args.HF_orb)
    if args.ini_orb is not None:
        f_out.write('Initial guess: ')
        if args.ini_orb[-4:] == '.npz':
            toout('Zipped numpy file ' + args.ini_orb)
        else:
            toout('Molpro output ' + args.ini_orb)
    if use_FCIopt:
        toout('Using the optimiser to whole FCI (because --WF_templ was given)')
    else:
        toout('Using the optimiser based on the geometry of the Grassmannian. See:')
        toout('   P-A Absil et. al, Acta Applicandae Mathematicae 80, 199-220, 2004.')
    toout('State: ' + args.state)
    toout('Log level: ' + logging.getLevelName(args.loglevel))
    toout('qpy job ID: {:s}'.\
                format(os.environ['QPY_JOB_ID']
                       if 'QPY_JOB_ID' in os.environ
                       else 'not in qpy'))
    toout()
    start_time = time.time()
    toout('Starting at {}'.format(
        time.strftime("%d %b %Y - %H:%M",time.localtime(start_time))))
    toout()
    # ----- loading wave function
    if use_FCIopt:
        ext_wf = FCI.Molpro_FCI_Wave_Function(args.molpro_output,
                                              args.state if args.state else '1.1',
                                              FCI_file_name = args.WF_templ)
    else:
        ext_wf = IntN.Wave_Function_Int_Norm.from_Molpro(args.molpro_output)
        ext_wf.calc_norm()
    if logger.level <= logging.DEBUG:
        logger.debug('|extWF>, loaded coefficients:\n %r', ext_wf)
        to_log = []
        for I in ext_wf.string_indices():
            to_log.append(str(ext_wf[I]) + ' ' + str(I))
        logger.debug('\n'.join(to_log))
    if args.HF_orb != args.WF_orb:
        toout('Using as |min E> a Slater determinant different than |WFref>')
        toout('(the reference of |extWF>). We have:')
        HF_in_basis_of_refWF = orb.Molecular_Orbitals.from_file(args.HF_orb).in_the_basis_of(
            orb.Molecular_Orbitals.from_file(args.WF_orb))
        print_ovlp_D('min E', 'WFref',
                     ovlp_Slater_dets(HF_in_basis_of_refWF,
                                      ext_wf.ref_occ))
    else:
        toout('Using |WFref> (the reference of |extWF>) as |min E>:')
    if isinstance(ext_wf, IntN.Wave_Function_Int_Norm):
        print_ovlp_D('WFref', 'extWF', 1.0/ext_wf.norm)
    if args.ini_orb is not None:
        if args.ini_orb[-4:] == '.npz':
            U = []
            ini_orb = np.load(args.ini_orb)
            for k in ini_orb:
                U.append(ini_orb[k])
        else:
            U = orb.Molecular_Orbitals.from_file(args.ini_orb).in_the_basis_of(
                orb.Molecular_Orbitals.from_file(args.WF_orb))
    else:
        U = []
        for spirrep in ext_wf.spirrep_blocks(restricted=False):
            U.append(np.identity(ext_wf.orb_dim[spirrep]))
        toout('Using the reference determinant (identity) as initial guess for U.')
    if not use_FCIopt:
        for spirrep in ext_wf.spirrep_blocks(restricted=False):
            U[spirrep] = U[spirrep][:,:ext_wf.ref_occ[spirrep]]
    for spirrep, Ui in enumerate(U):
        logger.debug('Initial orbitals for spirrep %s:\n%s',
                     spirrep, Ui)
#    toout('Number of determinants in the external wave function: {0:d}'.\
#          format(len(ext_wf.determinants)))
    toout('Dimension of orbital space: {0:}'.\
          format(ext_wf.orb_dim))
    toout('Number of alpha and beta electrons: {0:d} {1:d}'.\
          format(ext_wf.n_alpha, ext_wf.n_beta))
    toout()
    toout('Starting optimisation')
    toout('-'*30)
    f_out.flush()
    logger.info('Starting optimisation')
    if use_FCIopt:
        res = dGr_optimiser.optimise_distance_to_FCI(
            ext_wf,
            FCI.construct_Jac_Hess,
            FCI.str_Jac_Hess,
            FCI.calc_wf_from_z,
            FCI.transform_wf,
            f_out = f_out,
            ini_U = U)
    else:
        res = dGr_optimiser.optimise_distance_to_CI(
            ext_wf,
            f_out = f_out,
            ini_U = U,
            restricted = False,
            max_iter=20)
    toout('-'*30)
    logger.info('Optimisation completed')
    if isinstance(res.converged, bool):
        converged = res.converged
    else:
        converged = all(res.converged)
    if converged:
        toout('Iterations converged!')
    else:
        toout('WARNING: Iterations did not converge!!!')
        if use_FCIopt:
            toout('Norm of final z vector: {0:8.4e}'.format(res.norm[0]))
            toout('Norm of final Jacobian: {0:8.4e}'.format(res.norm[1]))
        else:
            toout('Norm of final eta vector: {0:8.4e}'.format(res.norm[0]))
            toout('Norm of final C vector: {0:8.4e}'.format(res.norm[1]))
        toout()
    if res.n_pos_H_eigVal is None:
        toout('WARNING: Unknown number of positive eigenvalue(s) of Hessian.')
    else:
        if res.n_pos_H_eigVal > 0:
            toout('WARNING: Found {0:d} positive eigenvalue(s) of Hessian.'.\
                  format(res.n_pos_H_eigVal))
            logger.warning('Found %s positive eigenvalue(s) of Hessian.',
                           res.n_pos_H_eigVal)
        else:
            logger.info('All eigenvalues of Hessian are negative: OK, we are at a maximum!')
    if isinstance(res.f, tuple):
        toout('WARNING: First determinant is not the one with largest coefficient!')
        toout('  Coefficient of first determinant: {:.7f}'.format(res.f[0]))
        toout('  Determinant with largest coefficient: {:s}'.format(res.f[1]))
        final_f = res.f[0]
    else:
        final_f = res.f
    print_ovlp_D('min D', 'extWF', final_f)
    logger.info('Saving U matrices in a .npz file: These make the transformation\n'
                'from the basis used to expand the external wave function\n'
                '(|extWF>) to the one that makes |min D> the first determinant.')
    np.savez(args.basename + '-min_dist_U', *U)
    if args.HF_orb != args.WF_orb:
        print_ovlp_D('refWF', 'min D',
                     ovlp_Slater_dets(res.U,
                                      ext_wf.ref_occ))
        for spirrep, Ui in enumerate(U):
            if Ui.shape[1] > 0:
                res.U[spirrep] = linalg.inv(HF_in_basis_of_refWF[spirrep]) @ Ui
    print_ovlp_D('min E', 'min D',
                 ovlp_Slater_dets(res.U,
                                  ext_wf.ref_occ))
    toout()
    end_time = time.time()
    elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))
    toout('Ending at {}'.format(
        time.strftime("%d %b %Y - %H:%M",time.localtime(end_time))))
    toout('Total time: {}'.format(elapsed_time))
