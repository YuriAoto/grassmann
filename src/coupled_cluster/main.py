"""Main caller for Coupled-Cluster


Yuri Aoto, 2021
"""
from util import logtime
from wave_functions import fci


def main(args, f_out):
    """Main function for Coupled-Cluster programs"""
    
    if args.method not in ('CCD_manifold', 'CCSD_manifold'):
        raise ValueError('Only CC method so far implemented is CC_manifold!')
    level = 'SD' if 'SD' in args.method else 'D'
    with logtime('Loading FCI wave function'):
        fci_wf = fci.WaveFunctionFCI.from_Molpro_FCI(args.molpro_output)
    with logtime('Running CC_manifold analysis'):
        dist = fci_wf.compare_to_CC_manifold(level=level)
    f_out.write(f'D_vert(FCI, CC{level} manifold) = {dist:.5f}\n')
