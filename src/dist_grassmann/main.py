"""Minimise the distance to extWF


"""
import os
import logging

import numpy as np
from scipy import linalg

from input_output.log import logger, logtime
from input_output import molpro
from .metric import dist_from_ovlp, ovlp_Slater_dets
from orbitals import orbitals as orb
from . import optimiser
import wave_functions


loglevel = logging.getLogger().getEffectiveLevel()


def main(args, f_out):
    """The main function to calculate the distance to the Grassmannian.
    
    Parameters:
    ---------
    args (argparse.Namespace)
        The parsed information. See parse.parse_cmd_line
    
    f_out (file)
        where the output goes
    """
    
    def toout(x='', add_new_line=True):
        f_out.write(x + ('\n' if add_new_line else ''))
    
    def print_ovlp_D(pt1, pt2, ovlp, with_dist=False, extra=''):
        if with_dist:
            toout(('|<{0:s}|{1:s}>|{4:s} = {2:12.8f} ;'
                   + ' D({0:s}, {1:s}) = {3:12.8f}').
                  format(pt1, pt2,
                         abs(ovlp),
                         dist_from_ovlp(ovlp),
                         extra))
        else:
            toout('|<{0:s}|{1:s}>|{3:s} = {2:12.8f}'.
                  format(pt1, pt2,
                         abs(ovlp),
                         extra))

    toout('From command line and input:')
    toout('External wave function, |extWF>: ' + args.molpro_output)
    if args.WF_templ is not None:
        toout('Template for |extWF>: ' + args.WF_templ)
    toout('Orbital basis of |extWF>: ' + args.WF_orb)
    toout('Hartree-Fock orbitals, |minE>: ' + args.HF_orb)
    if args.ini_orb is not None:
        f_out.write('Initial guess: ')
        if args.ini_orb[-4:] == '.npz':
            toout('Zipped numpy file ' + args.ini_orb)
        else:
            toout('Molpro output ' + args.ini_orb)
    if args.at_ref:
        orbRot_opt = False
        toout('Performing analysis at the reference wave function.')
    elif args.algorithm == 'orb_rotations':
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
    # ----- loading wave function
    with logtime('Reading wave function'):
        ext_wf = molpro.load_wave_function(
            args.molpro_output,
            WF_templ=args.WF_templ,
            use_CISD_norm=not args.at_ref,
            wf_obj_type=('cisd'
                         if args.algorithm == 'CISD_Absil' else
                         ('int_norm'
                          if args.algorithm == 'general_Absil' else
                          'fci')))
    if args.at_ref:
        ext_wf.use_CISD_norm = False
    if (isinstance(ext_wf, wave_functions.norm_ci.NormCI_WaveFunction)
        and 'Absil' in args.algorithm
            and not args.at_ref):
        raise Exception('algorithm CISD_Absil is not compatible with'
                        + 'fci.NormCI_WaveFunction')
    logger.debug('External wave function:\n %r', ext_wf)
    toout('External wave function (|extWF>) is: ' + ext_wf.wf_type)
    if loglevel <= logging.DEBUG and args.algorithm == 'general_Absil':
        x = []
        for Index in ext_wf.string_indices():
            x.append(str(Index) + ': ' + str(ext_wf[Index]))
        logger.debug('The determinants:\n' + '\n'.join(x))
    if args.HF_orb != args.WF_orb:
        toout('Using as |minE> a Slater determinant different than |WFref>')
        toout('(the reference of |extWF>). We have:')
        try:
            HF_in_basis_of_refWF = orb.MolecularOrbitals.from_file(
                args.HF_orb).in_the_basis_of(
                    orb.MolecularOrbitals.from_file(args.WF_orb))
        except ValueError as e:
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
            print_ovlp_D('minE', 'WFref',
                         ovlp_Slater_dets(HF_in_basis_of_refWF,
                                          ext_wf.ref_orb))
    else:
        toout('Using |WFref> (the reference of |extWF>) as |minE>.')
    print_ovlp_D('WFref', 'extWF', ext_wf.C0)
    if args.algorithm == 'general_Absil':
        restricted = False
    else:
        restricted = ext_wf.restricted
    if args.ini_orb is not None:
        if args.ini_orb[-4:] == '.npz':
            U = []
            with np.load(args.ini_orb) as ini_orb:
                for k in range(len(ini_orb)):
                    U.append(ini_orb['arr_' + str(k)])
            if not restricted and len(U) == ext_wf.n_irrep:
                orb.extend_to_unrestricted(U)
        else:
            U = orb.MolecularOrbitals.from_file(args.ini_orb).in_the_basis_of(
                orb.MolecularOrbitals.from_file(args.WF_orb))
    else:
        U = orb.construct_Id_orbitals(ext_wf.ref_orb,
                                      ext_wf.orb_dim,
                                      (1
                                       if restricted else
                                       2) * ext_wf.n_irrep,
                                      full=orbRot_opt)
    toout('Restricted calculation'
          if restricted else
          'Unrestricted calculation')
    toout('Number of alpha and beta electrons: {0:d} {1:d}'.
          format(ext_wf.n_alpha, ext_wf.n_beta))
    toout('Dim. of core orb. space:  {0:}'.
          format(ext_wf.froz_orb))
    toout('Dim. of ref. determinant: {0:}'.
          format(ext_wf.ref_orb))
    toout('Dim. of orbital space:    {0:}'.
          format(ext_wf.orb_dim))
    toout()
    if args.at_ref:
        toout('Calculation at the reference wave function',
              add_new_line=False)
        f_out.flush()
        with logtime('Calculation at the reference wave function',
                     out_stream=f_out,
                     out_fmt=' (elapsed time: {})\n'):
            res = optimiser.optimise_overlap_orbRot(
                ext_wf,
                f_out=None,
                at_reference=True)
        toout('-' * 30)
        if not ('CCD' in ext_wf.WF_type or 'BCCD' in ext_wf.WF_type):
            toout('|J|  = |t_i^a|    = {0:.7f}'.format(res.norm[1]))
            if 'CCSD' in ext_wf.WF_type:
                toout('T1 diagnostic     = {0:.7f}'.format(
                    res.norm[1] / np.sqrt(2 * ext_wf.n_corr_elec)))
            toout('|ΔK| = |H^-1 @ J| = {0:.7f}'.format(res.norm[0]))
            toout('|ΔK|/√(n_corr_el) = {0:.7f}'.format(
                res.norm[0] / np.sqrt(ext_wf.n_corr_elec)))
        ovlp_with_1st_it = ovlp_Slater_dets((2 * res.U)
                                            if restricted else
                                            res.U,
                                            ext_wf.ref_orb)
        print_ovlp_D('refWF', '1st it',
                     ovlp_with_1st_it,
                     with_dist=False)
        print_ovlp_D('refWF', '1st it',
                     ovlp_with_1st_it**(1/ext_wf.n_elec),
                     with_dist=False,
                     extra='^(1/n_el)')
        toout('Hessian (H) has {0:d} positive eigenvalues'.format(
            res.n_pos_H_eigVal))
        toout('-' * 30)
    else:
        toout('Starting optimisation')
        logger.info('Starting optimisation')
        toout('-' * 30)
        f_out.flush()
        if orbRot_opt:
            res = optimiser.optimise_overlap_orbRot(
                ext_wf,
                f_out=f_out,
                ini_U=U,
                max_iter=args.maxiter,
                enable_uphill=False,
                save_all_U_dir=(args.output + '_all_U/'
                                if args.save_all_U else
                                None))
        else:
            res = optimiser.optimise_overlap_Absil(
                ext_wf,
                f_out=f_out,
                max_iter=args.maxiter,
                ini_U=U,
                save_all_U_dir=(args.output + '_all_U/'
                                if args.save_all_U else
                                None))
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
            toout('WARNING:'
                  + ' Unknown number of positive eigenvalue(s) of Hessian.')
        else:
            if res.n_pos_H_eigVal > 0:
                toout('WARNING: '
                      + 'Found {0:d} positive eigenvalue(s) of Hessian.'.
                      format(res.n_pos_H_eigVal))
                logger.warning('Found %s positive eigenvalue(s) of Hessian.',
                               res.n_pos_H_eigVal)
            else:
                if args.at_ref:
                    toout('All eigenvalues of Hessian are negative:'
                          + ' We are closse to a maximum!')
                logger.info('All eigenvalues of Hessian are negative:'
                            + ' OK, we are at a maximum!')
        if isinstance(res.f, tuple):
            toout('WARNING:'
                  + ' Reference determinant is not the one'
                  + ' with largest coefficient!')
            toout('  Coefficient of reference: {:.7f}'.format(res.f[0]))
            toout('  Determinant with largest coefficient: {:s}'.
                  format(str(res.f[1])))
            final_f = res.f[0]
        else:
            final_f = res.f
        print_ovlp_D('minD', 'extWF', final_f)
        logger.info("""Saving U matrices in a .npz file:
 These make the transformation from the basis used to expand the
 external wave function (|extWF>) to the one that makes |minD>
 the first determinant.""")
        final_U = (orb.complete_orb_space(res.U, ext_wf.orb_dim)
                   if args.save_full_U else
                   res.U)
        np.savez(args.output + '_U', *final_U)
        if args.HF_orb != args.WF_orb:
            print_ovlp_D('refWF', 'minD',
                         ovlp_Slater_dets((2 * res.U)
                                          if restricted else
                                          res.U,
                                          ext_wf.ref_orb))
            for spirrep, Ui in enumerate(U):
                if Ui.shape[1] > 0:
                    res.U[spirrep] = np.matmul(
                        linalg.inv(HF_in_basis_of_refWF[spirrep]), Ui)
        print_ovlp_D('minE', 'minD',
                     ovlp_Slater_dets((2 * res.U)
                                      if restricted else
                                      res.U,
                                      ext_wf.ref_orb))
    toout()
