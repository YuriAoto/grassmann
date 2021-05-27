"""Main caller for Coupled-Cluster


Yuri Aoto, 2021
"""
from input_output.log import logtime, logger
from wave_functions import fci
from wave_functions.interm_norm import IntermNormWaveFunction
from coupled_cluster.dist_to_fci import (vertical_dist_to_cc_manifold,
                                         calc_dist_to_cc_manifold)


def main(args, f_out):
    """Main function for Coupled-Cluster programs"""
    
    if args.method in ('CCD_mani_vert', 'CCSD_mani_vert'):
        level = 'SD' if 'SD' in args.method else 'D'
        with logtime('Loading FCI wave function'):
            fci_wf = fci.FCIWaveFunction.from_Molpro_FCI(args.molpro_output)
        fci_wf.normalise(mode='intermediate')
        logger.debug('FCI wave function, in intermediate norm\n%s', fci_wf)
        with logtime('Running CC_manifold analysis'):
            resCC = vertical_dist_to_cc_manifold(fci_wf,
                                                 level=level,
                                                 restore_wf=False)
        logger.info(resCC.wave_function)
        f_out.write(
            f'D_vert(FCI, CC{level} manifold) = {resCC.distance:.8f}\n')
        f_out.write('Number of excitations where the CC manifold\n'
                    + '   curves towards the wave function:\n')
        for rank, n in resCC.right_dir.items():
            f_out.write(f'{rank}: {n[0]} of {n[1]}\n')
    elif args.method in ('CCD_mani_minD', 'CCSD_mani_minD'):
        level = 'SD' if 'SD' in args.method else 'D'
        with logtime('Loading FCI wave function'):
            fci_wf = fci.FCIWaveFunction.from_Molpro_FCI(args.molpro_output)
        fci_wf.normalise(mode='intermediate')
        cc_wf = IntermNormWaveFunction.similar_to(
            fci_wf, 'CC' + level, restricted=False)
        logger.debug('FCI wave function, in intermediate norm\n%s', fci_wf)
        with logtime('Running CC_manifold analysis'):
            resCC = calc_dist_to_cc_manifold(fci_wf,
                                             level=level,
                                             f_out=f_out,
                                             ini_wf=cc_wf,
                                             diag_hess=args.cc_diag_hess)
        logger.info(resCC.wave_function)
        f_out.write(
            f'D(FCI, CC{level} manifold) = {resCC.distance:.8f}\n')
    elif args.method in ('CCD', 'CCSD'):
        level = 'SD' if 'SD' in args.method else 'D'
    else:
        raise ValueError(
            'Only CC method so far implemented is for CC_manifold!')
