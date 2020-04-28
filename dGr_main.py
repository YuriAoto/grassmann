"""Main function of dGr

Functions:
----------
dGr_main
"""
import os
import datetime
import time
import logging

import git
import numpy as np
from scipy import linalg

from dGr_util import dist_from_ovlp, ovlp_Slater_dets, logtime
from dGr_exceptions import dGrValueError
import dGr_orbitals as orb
import dGr_FCI_WF as FCI
import dGr_WF_int_norm as IntN
import dGr_CISD_WF as CISDwf
import dGr_optimiser

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


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
    
    def toout(x=''):
        f_out.write(x + '\n')
    
    def print_ovlp_D(pt1, pt2, ovlp):
        toout('|<{0:s}|{1:s}>| = {2:12.8f} ; D({0:s}, {1:s}) = {3:12.8f}'.
              format(pt1, pt2,
                     abs(ovlp),
                     dist_from_ovlp(ovlp)))
    
    toout('dGr - optimise the distance in the Grassmannian')
    toout('Yuri Aoto - 2018, 2019, 2020')
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
    toout('Orbital basis of |extWF>: ' + args.WF_orb)
    toout('Hartree-Fock orbitals, |min E>: ' + args.HF_orb)
    if args.ini_orb is not None:
        f_out.write('Initial guess: ')
        if args.ini_orb[-4:] == '.npz':
            toout('Zipped numpy file ' + args.ini_orb)
        else:
            toout('Molpro output ' + args.ini_orb)
    if args.algorithm == 'orb_rotations':
        orbRot_opt = True
        toout('Using the optimiser based on orbital rotations.')
    else:
        orbRot_opt = False
        toout('Using the optimiser based on the geometry of the Grassmannian.')
        toout(' See: P-A Absil et. al,')
        toout('      Acta Applicandae Mathematicae 80, 199-220, 2004.')
        if args.algorithm == 'CISD_Absil':
            toout('   Algorithm specific for CISD wave function.')
        else:
            toout('   Using algorithm for general wave function.')
    toout('State: ' + args.state)
    toout('Log level: ' + logging.getLevelName(args.loglevel))
    toout('qpy job ID: {:s}'.
          format(os.environ['QPY_JOB_ID']
                 if 'QPY_JOB_ID' in os.environ else
                 'not in qpy'))
    toout()
    start_time = time.time()
    toout('Starting at {}'.format(
        time.strftime("%d %b %Y - %H:%M", time.localtime(start_time))))
    toout()
    # ----- loading wave function
    if orbRot_opt:
        with logtime('Reading FCI wave function'):
            ext_wf = FCI.Wave_Function_Norm_CI.from_Molpro_FCI(
                args.molpro_output
                if args.WF_templ is None else
                args.WF_templ,
                args.state if args.state else '1.1',
                zero_coefficients=args.WF_templ is not None)
        if args.WF_templ is not None:
            with logtime('Reading wave function coefficients'):
                int_N_WF = IntN.Wave_Function_Int_Norm.from_Molpro(
                    args.molpro_output)
                int_N_WF.calc_norm()
                ext_wf.get_coeff_from_Int_Norm_WF(int_N_WF,
                                                  change_structure=False,
                                                  use_structure=True)
                # ext_wf.get_coeff_from_molpro(args.molpro_output,
                #                              args.state
                #                              if args.state else
                #                              '1.1',
                #                              use_structure=True)
    else:
        with logtime('Reading int. norm. from Molpro output'):
            ext_wf = IntN.Wave_Function_Int_Norm.from_Molpro(
                args.molpro_output)
            ext_wf.calc_norm()
            if args.algorithm == 'CISD_Absil':
                with logtime('Transforming int. norm. WF into CISD wf'):
                    ext_wf = CISDwf.Wave_Function_CISD.from_intNorm(ext_wf)
    logger.debug('External wave function:\n %r', ext_wf)
    if loglevel <= logging.DEBUG and args.algorithm == 'general_Absil':
        x = []
        for I in ext_wf.string_indices():
            x.append(str(I) + ': ' + str(ext_wf[I]))
        logger.debug('The determinants:\n' + '\n'.join(x))
    if args.HF_orb != args.WF_orb:
        toout('Using as |min E> a Slater determinant different than |WFref>')
        toout('(the reference of |extWF>). We have:')
        try:
            HF_in_basis_of_refWF = orb.Molecular_Orbitals.from_file(
                args.HF_orb).in_the_basis_of(
                    orb.Molecular_Orbitals.from_file(args.WF_orb))
        except dGrValueError as e:
            if 'keepspherical' in str(e):
                logger.error(str(e))
                toout('ERROR:\n   '
                      + str(e).replace(':', ':\n  ') + '\n'
                      + '   Ignoring Hartree-Fock orbitals in ' + args.HF_orb)
                args.HF_orb = args.WF_orb
            else:
                raise e
        except Exception as e:
            raise e
        else:
            print_ovlp_D('min E', 'WFref',
                         ovlp_Slater_dets(HF_in_basis_of_refWF,
                                          ext_wf.ref_occ))
    else:
        toout('Using |WFref> (the reference of |extWF>) as |min E>.')
    print_ovlp_D('WFref', 'extWF', ext_wf.C0)
    if args.algorithm == 'CISD_Absil':
        restricted = True
    elif args.algorithm == 'general_Absil':
        restricted = False
    elif args.algorithm == 'orb_rotations':
        restricted = ext_wf.restricted
    if args.ini_orb is not None:
        if args.ini_orb[-4:] == '.npz':
            U = []
            with np.load(args.ini_orb) as ini_orb:
                for k in range(len(ini_orb)):
                    U.append(ini_orb['arr_' + str(k)])
        else:
            U = orb.Molecular_Orbitals.from_file(args.ini_orb).in_the_basis_of(
                orb.Molecular_Orbitals.from_file(args.WF_orb))
    else:
        U = []  # SETO TO NONE
        for spirrep in ext_wf.spirrep_blocks(restricted=restricted):
            U.append(np.identity(ext_wf.orb_dim[spirrep]))
        toout('Using the reference determinant (identity)'
              + ' as initial guess for U.')
    if not orbRot_opt:
        for spirrep in ext_wf.spirrep_blocks(restricted=restricted):
            U[spirrep] = U[spirrep][:, :ext_wf.ref_occ[spirrep]]
#    toout('Number of determinants in the external wave function: {0:d}'.
#          format(len(ext_wf)))
    toout('Dimension of orbital space: {0:}'.
          format(ext_wf.orb_dim))
    toout('Number of alpha and beta electrons: {0:d} {1:d}'.
          format(ext_wf.n_alpha, ext_wf.n_beta))
    toout('Occupation of reference wave function: {0:}'.
          format(ext_wf.ref_occ))
    toout()
    toout('Starting optimisation')
    toout('-' * 30)
    f_out.flush()
    logger.info('Starting optimisation')
    if orbRot_opt:
        res = dGr_optimiser.optimise_overlap_orbRot(
            ext_wf,
            f_out=f_out,
            ini_U=U,
            enable_uphill=False)
    else:
        res = dGr_optimiser.optimise_overlap_Absil(
            ext_wf,
            f_out=f_out,
            ini_U=U)
    toout('-' * 30)
    logger.info('Optimisation completed')
    if isinstance(res.converged, bool):
        converged = res.converged
    else:
        converged = all(res.converged)
    if converged:
        toout('Iterations converged!')
    else:
        toout('WARNING: Iterations did not converge!!!')
        if orbRot_opt:
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
            toout('WARNING: Found {0:d} positive eigenvalue(s) of Hessian.'.
                  format(res.n_pos_H_eigVal))
            logger.warning('Found %s positive eigenvalue(s) of Hessian.',
                           res.n_pos_H_eigVal)
        else:
            logger.info('All eigenvalues of Hessian are negative:'
                        + ' OK, we are at a maximum!')
    if isinstance(res.f, tuple):
        toout('WARNING:'
              + ' First determinant is not the one with largest coefficient!')
        toout('  Coefficient of first determinant: {:.7f}'.format(res.f[0]))
        toout('  Determinant with largest coefficient: {:s}'.format(str(res.f[1])))
        final_f = res.f[0]
    else:
        final_f = res.f
    print_ovlp_D('min D', 'extWF', final_f)
    logger.info("""Saving U matrices in a .npz file:
 These make the transformation from the basis used to expand the
 external wave function (|extWF>) to the one that makes |min D>
 the first determinant.""")
    final_U = (orb.complete_orb_space(res.U, ext_wf.orb_dim)
               if args.save_full_U else
               res.U)
    np.savez(args.basename + '.min_dist_U', *final_U)
    if args.HF_orb != args.WF_orb:
        print_ovlp_D('refWF', 'min D',
                     ovlp_Slater_dets((2 * res.U)
                                      if restricted else
                                      res.U,
                                      ext_wf.ref_occ))
        for spirrep, Ui in enumerate(U):
            if Ui.shape[1] > 0:
                res.U[spirrep] = linalg.inv(HF_in_basis_of_refWF[spirrep]) @ Ui
    print_ovlp_D('min E', 'min D',
                 ovlp_Slater_dets((2 * res.U)
                                  if restricted else
                                  res.U,
                                  ext_wf.ref_occ))
    toout()
    end_time = time.time()
    elapsed_time = str(datetime.timedelta(seconds=(end_time - start_time)))
    toout('Ending at {}'.format(
        time.strftime("%d %b %Y - %H:%M", time.localtime(end_time))))
    toout('Total time: {}'.format(elapsed_time))
