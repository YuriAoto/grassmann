"""Main function of Grassmann

Functions:
----------
main_grassmann
"""
import os
import time

import git

from util import logtime
import dist_grassmann
import hartree_fock
import memory


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
        toout('Starting at {}'.format(
            time.strftime("%d %b %Y - %H:%M", time.localtime(T.ini_time))))
        toout()
        if args.geometry is not None:
            hartree_fock.main(args, f_out)
        else:
            dist_grassmann.main(args, f_out)
    toout('Memory usage:')
    toout(memory.show_status())
    toout('Ending at {}'.format(
        time.strftime("%d %b %Y - %H:%M", time.localtime(T.end_time))))
    toout('Total time: {}'.format(T.elapsed_time))
