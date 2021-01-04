"""Main caller for Coupled-Cluster


Yuri Aoto, 2021
"""
from util import logtime, logger
from wave_functions import fci


def main(args, f_out):
    """Main function for Coupled-Cluster programs"""
    
    if args.method not in ('CCD_manifold', 'CCSD_manifold'):
        raise ValueError('Only CC method so far implemented is CC_manifold!')
    level = 'SD' if 'SD' in args.method else 'D'
    with logtime('Loading FCI wave function'):
        fci_wf = fci.WaveFunctionFCI.from_Molpro_FCI(args.molpro_output)
    fci_wf.normalise(mode='intermediate')
    logger.debug('FCI wave function, in intermediate norm\n%s', fci_wf)
    with logtime('Running CC_manifold analysis'):
        dist, right_dir = fci_wf.compare_to_CC_manifold(level=level, restore_wf=False)
    f_out.write(f'D_vert(FCI, CC{level} manifold) = {dist:.8f}\n')
    f_out.write('Number of excitations where the CC manifold\n'
                + '   curves towards the wave function:\n')
    for rank, n in right_dir.items():
        f_out.write(f'{rank}: {n[0]} of {n[1]}\n')
