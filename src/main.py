"""Main function of Grassmann

Functions:
----------
main_grassmann
"""
import os
from datetime import datetime as dt

import git

from input_output.log import logtime
from util import memory
from dist_grassmann.main import main as main_dist_gr
from hartree_fock.main import main as main_hf
from coupled_cluster.main import main as main_cc


def main_grassmann(args, f_out):
    """The main function of Grassmann
    
    Parameters:
    ---------
    args (argparse.Namespace)
        The parsed information. See parse for more information
    
    f_out (file)
        Where the output goes
    """
    git_repo = git.Repo(os.path.dirname(os.path.abspath(__file__)) + '/../')
    git_sha = git_repo.head.object.hexsha
    memory.set_total_memory(args.memory[0], unit=args.memory[1])
    with logtime('Main program') as T:
        def toout(x='', add_new_line=True):
            f_out.write(x + ('\n' if add_new_line else ''))

        toout('Grassmann')
        toout('Exploring the geometry of the electronic wave functions space')
        toout('Yuri Aoto - 2018, 2019, 2020')
        toout()
        toout('Current git commit: ' + git_sha)
        toout()
        toout('Directory:\n' + args.wdir)
        toout()
        toout('Command:\n' + ' '.join(args.sys_argv))
        if args.files_content:
            toout()
            toout(''.join(args.files_content))
        toout()
        toout(f'Starting at {dt.fromtimestamp(T.ini_time):%d %b %Y - %H:%M}')
        toout()
        if args.method == 'dist_Grassmann':
            main_dist_gr(args, f_out)
        elif args.method == 'Hartree_Fock' or args.method == 'CCSD' or args.method == 'CCD':
            args.res_hf = main_hf(args, f_out)
            print(args.res_hf)
            if args.method == 'CCSD' or args.method == 'CCD':
                main_cc(args, f_out)
        elif args.method in ('CCD_mani_vert', 'CCSD_mani_vert',
                             'CCD_mani_minD', 'CCSD_mani_minD'):
            main_cc(args, f_out)
        else:
            raise ValueError('Unknown method: ' + args.method)
    toout()
    toout('Memory usage:')
    toout(memory.show_status())
    toout(f'Ending at {dt.fromtimestamp(T.end_time):%d %b %Y - %H:%M}')
    toout('Total time: {}'.format(T.elapsed_time))
