"""Main caller for Coupled-Cluster


Yuri Aoto, 2021
"""
from input_output.log import logtime, logger
from wave_functions import fci
from wave_functions.interm_norm import IntermNormWaveFunction
from coupled_cluster.dist_to_fci import (vertical_dist_to_cc_manifold,
                                         calc_dist_to_cc_manifold,
                                         calc_all_distances)


def main(args, f_out):
    """Main function for Coupled-Cluster programs"""
    
    found_method = False
    level = 'SD' if 'SD' in args.method else 'D'
    if args.method in ('CCD_mani_vert', 'CCSD_mani_vert',
                       'CCD_mani_minD', 'CCSD_mani_minD',
                       'CCD_full_analysis', 'CCSD_full_analysis'):
        with logtime('Loading FCI wave function'):
            fci_wf = fci.FCIWaveFunction.from_Molpro_FCI(args.molpro_output)
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
        if args.method in ('CCD_full_analysis', 'CCSD_full_analysis'):
            cc_wf = res_vert.wave_function
        else:
            cc_wf = IntermNormWaveFunction.similar_to(fci_wf,
                                                      'CC' + level,
                                                      restricted=False)
        with logtime('Running min dist to CC manifold'):
            res_min_d = calc_dist_to_cc_manifold(fci_wf,
                                                 level=level,
                                                 f_out=f_out,
                                                 ini_wf=cc_wf,
                                                 diag_hess=args.cc_diag_hess)
        logger.info("Results from min dist to CC manifold:\n%r", res_min_d)
        f_out.write(str(res_min_d))
    
    if args.method in ('CCD_full_analysis', 'CCSD_full_analysis'):
        with logtime('Calculating all distances for CC/CI manifolds'):
            res_all_dists = calc_all_distances(
                fci_wf,
                res_vert,
                res_min_d,
                cc_wf=(None
                       if args.cc_wf is None else
                       IntermNormWaveFunction.unrestrict(
                           IntermNormWaveFunction.from_Molpro(args.cc_wf))),
                ci_wf=(None
                       if args.ci_wf is None else
                       IntermNormWaveFunction.unrestrict(
                           IntermNormWaveFunction.from_Molpro(args.ci_wf))),
                level=level)
        f_out.write(str(res_all_dists))
    
    if not found_method:
        raise ValueError(
            f'The CC method {args.method} has not been found')
