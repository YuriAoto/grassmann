"""Main function of dGr

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
import dGr_FCI_Molpro as FCI
import dGr_WF_int_norm as IntN
import dGr_optimiser

logger = logging.getLogger(__name__)

def dGr_main(args):
    git_repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
    git_sha = git_repo.head.object.hexsha
    f_out = open(args.basename + '.min_dist' + args.state, 'w')
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
        if isinstance(args.ini_orb, tuple):
            toout('Numpy files ' + str(args.ini_orb[0]) +
                        ' and ' + str(args.ini_orb[1]))
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
    #
    if use_FCIopt:
        ext_wf = FCI.Molpro_FCI_Wave_Function(args.molpro_output,
                                              args.state if args.state else '1.1',
                                              FCI_file_name = args.WF_templ)
    else:
        ext_wf = IntN.Wave_Function_Int_Norm.from_Molpro(args.molpro_output)
        ext_wf.calc_norm()
    # if logger.level <= logging.DEBUG:
    #     logger.debug('|extWF>, loaded coefficients:\n' + str(ext_wf))
    #     S = 0.0
    #     for det in ext_wf.determinants:
    #         S += det[0]**2
    #     logger.debug('Norm of external WF: %f',
    #                  math.sqrt(S))

    if args.HF_orb != args.WF_orb:
        toout('Using as |min E> a Slater determinant different than |WFref>')
        toout('(the reference of |extWF>). We have:')
        Ua_HF_to_WF, Ub_HF_to_WF = FCI.transf_orb_from_to(args.WF_orb, args.HF_orb)
#        print_ovlp_D('min E', 'extWF',
#                     FCI.transform_wf(ext_wf,
#                                      Ua_HF_to_WF, Ub_HF_to_WF,
#                                      just_C0=True).determinants[0][0])
        print_ovlp_D('min E', 'WFref',
                     ovlp_Slater_dets(Ua_HF_to_WF,
                                      Ub_HF_to_WF,
                                      ext_wf.n_alpha,
                                      ext_wf.n_beta))
    else:
        toout('Using |WFref> (the reference of |extWF>) as |min E>:')
#    print_ovlp_D('WFref', 'extWF', ext_wf.determinants[0][0])
    print_ovlp_D('WFref', 'extWF', 1.0/ext_wf.norm)

    if args.ini_orb is not None:
        if isinstance(args.ini_orb, tuple):
            Ua = np.load(args.ini_orb[0])
            Ub = np.load(args.ini_orb[1])[:,:ext_wf.n_beta]
        else:
            Ua, Ub = FCI.transf_orb_from_to(args.WF_orb,
                                            args.ini_orb)
    else:
        Ua, Ub = FCI.get_trans_max_coef(ext_wf)
        toout('Using initial guess for U from determinant with largest coefficient.')
    if not use_FCIopt:
        Ua = Ua[:,:ext_wf.n_alpha]
        Ub = Ub[:,:ext_wf.n_beta]
    logger.debug('Initial U for alpha orbitals:\n' + str(Ua))
    logger.debug('Initial U for beta orbitals:\n' + str(Ub))

#    toout('Number of determinants in the external wave function: {0:d}'.\
#          format(len(ext_wf.determinants)))
    toout('Dimension of orbital space: {0:d}'.\
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
            ini_U = (Ua, Ub))
    else:
        res = dGr_optimiser.optimise_distance_to_CI(
            ext_wf,
            f_out = f_out,
            ini_U = (Ua, Ub),
            restricted = False
        )
    toout('-'*30)
    logger.info('Optimisation completed')

    if res.converged:
        toout('Iterations converged!')
    else:
        toout('WARNING: Iterations not converged!!!')
        if use_FCIopt:
            toout('Norm of final z vector: {0:8.4e}'.format(res.norm[0]))
            toout('Norm of final Jacobian: {0:8.4e}'.format(res.norm[1]))
        else:
            toout('Norm of "residual": {0:8.4e}'.format(res.norm))
        toout()
    if res.n_pos_H_eigVal is None:
        toout('WARNING: Unknown number of positive eigenvalue(s) of Hessian.')
    else:
        if res.n_pos_H_eigVal > 0:
            toout('WARNING: Found {0:d} positive eigenvalue(s) of Hessian.'.\
                  format(res.n_pos_H_eigVal))
            logger.warning('Found {0:d} positive eigenvalue(s) of Hessian.'.\
                           format(res.n_pos_H_eigVal))
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
    Ua, Ub = res.U
    logger.info('Saving U matrices in .npy files: These make the transformation\n'
                'from the basis used to expand the external wave function\n'
                '(|extWF>) to the one that makes |min D> the first determinant.')
    np.save(args.basename + '-min_dist_Ua', Ua)
    np.save(args.basename + '-min_dist_Ub', Ub)

    if args.HF_orb != args.WF_orb:
        print_ovlp_D('refWF', 'min D',
                     ovlp_Slater_dets(Ua, Ub,
                                      ext_wf.n_alpha,
                                      ext_wf.n_beta))
        Ua = np.matmul(linalg.inv(Ua_HF_to_WF), Ua)
        Ub = np.matmul(linalg.inv(Ub_HF_to_WF), Ub)
    print_ovlp_D('min E', 'min D',
                 ovlp_Slater_dets(Ua, Ub,
                                  ext_wf.n_alpha,
                                  ext_wf.n_beta))
    toout()
    end_time = time.time()
    elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))
    toout('Ending at {}'.format(
        time.strftime("%d %b %Y - %H:%M",time.localtime(end_time))))
    toout('Total time: {}'.format(elapsed_time))
    f_out.close()
