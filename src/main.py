"""Main function of Grassmann

Functions:
----------
main_grassmann
"""
from dist_grassmann.main import main as main_dist_gr
from hartree_fock.main import main as main_hf
from coupled_cluster.main import main as main_cc


def main_grassmann(args, f_out):
    """The main function of Grassmann
    
    Parameters:
    ---------
    args (argparse.Namespace)
        The parsed information. See parse (input_output/parse.py) for more information
    
    f_out (file)
        Where the output goes
    """
    if args.method == 'dist_Grassmann':
        main_dist_gr(args, f_out)
    elif (args.method == 'Hartree_Fock'
          or args.method == 'CCSD'
          or args.method == 'CCD'):
        main_hf(args, f_out)
        if args.method == 'CCSD' or args.method == 'CCD':
            main_cc(args, f_out)
    elif args.method in ('CCD_mani_vert', 'CCSD_mani_vert',
                         'CCD_mani_minD', 'CCSD_mani_minD',
                         'CCD_full_analysis', 'CCSD_full_analysis'):
        main_cc(args, f_out)
    else:
        raise ValueError('Unknown method: ' + args.method)
