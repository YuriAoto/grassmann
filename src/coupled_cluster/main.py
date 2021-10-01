"""Main caller for Coupled-Cluster


Yuri Aoto, 2021
"""
from input_output.log import logtime, logger
from wave_functions import fci
from input_output.molpro import MolproInputError
from wave_functions.interm_norm import IntermNormWaveFunction
from coupled_cluster.dist_to_fci import (vertical_dist_to_cc_manifold,
                                         calc_dist_to_cc_manifold,
                                         calc_all_distances)
from coupled_cluster import optimiser


def main(args, f_out):
    """Main function for Coupled-Cluster programs"""
    
    found_method = False
    level = 'SD' if 'SD' in args.method else 'D'
    ini_cc_wf = None
    ref_orb = 'maxC' if args.ref_orb is None else args.ref_orb
    if args.ini_cc_wf is not None:
        ini_cc_wf = IntermNormWaveFunction.unrestrict(
            IntermNormWaveFunction.from_Molpro(args.ini_cc_wf))
    if args.method in ('CCD_mani_vert', 'CCSD_mani_vert',
                       'CCD_mani_minD', 'CCSD_mani_minD',
                       'CCD_full_analysis', 'CCSD_full_analysis'):
        with logtime('Loading FCI wave function'):
            fci_wf = fci.FCIWaveFunction.from_Molpro(args.molpro_output,
                                                     ref=ref_orb,
                                                     state=args.state)
        fci_wf.normalise(mode='intermediate')
        logger.debug('FCI wave function, in intermediate norm\n%s', fci_wf)
    
    if args.method in ('CCD_mani_vert', 'CCSD_mani_vert',
                       'CCD_full_analysis', 'CCSD_full_analysis'):
        found_method = True
        with logtime('Running vertical distance to CC manifold'):
            res_vert = vertical_dist_to_cc_manifold(fci_wf,
                                                    level=level)
        logger.info("Results from vertical dist to CC manifold:\n%r", res_vert)
        f_out.write(str(res_vert))
    
    if args.method in ('CCD_mani_minD', 'CCSD_mani_minD',
                       'CCD_full_analysis', 'CCSD_full_analysis'):
        found_method = True
        if ini_cc_wf is None:
            if args.method in ('CCD_full_analysis', 'CCSD_full_analysis'):
                ini_cc_wf = res_vert.wave_function
            else:
                ini_cc_wf = IntermNormWaveFunction.similar_to(fci_wf,
                                                              'CC' + level,
                                                              restricted=False)
        with logtime('Running min dist to CC manifold'):
            res_min_d = calc_dist_to_cc_manifold(fci_wf,
                                                 level=level,
                                                 f_out=f_out,
                                                 ini_wf=ini_cc_wf,
                                                 diag_hess=args.cc_diag_hess)
        logger.info("Results from min dist to CC manifold:\n%r", res_min_d)
        f_out.write(str(res_min_d))
    
    if args.method in ('CCD_full_analysis', 'CCSD_full_analysis'):
        with logtime('Calculating all distances for CC/CI manifolds'):
            try:
                ccwf = IntermNormWaveFunction.unrestrict(
                    IntermNormWaveFunction.from_Molpro(args.cc_wf))
            except (OSError, ValueError, MolproInputError) as exc:
                logger.warning(f'Error when reading cc wave function: {exc}')
                ccwf = None
            try:
                ciwf = IntermNormWaveFunction.unrestrict(
                    IntermNormWaveFunction.from_Molpro(args.ci_wf))
            except (OSError, ValueError, MolproInputError) as exc:
                logger.warning(f'Error when reading ci wave function: {exc}')
                ciwf = None
            res_all_dists = calc_all_distances(
                fci_wf,
                res_vert,
                res_min_d,
                cc_wf=ccwf,
                ci_wf=ciwf,
                level=level)
        f_out.write(str(res_all_dists))
    
    elif args.method in ('CCD', 'CCSD'):
        found_method = True
        level = 'SD' if 'SD' in args.method else 'D'
        optimiser.cc_closed_shell(args.res_hf.energy,
                                  args.res_hf.orbitals,
                                  args.res_hf.integrals, ##TODO: Find where the integrals are!!
                                  wf_ini=None,
                                  preserve_wf_ini=False,
                                  level='SD',
                                  max_inter=20)

    if not found_method:
        raise ValueError(
            f'The CC method {args.method} has not been found')
