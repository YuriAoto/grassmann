"""Parser to dGr

Functions:
----------

parse_cmd_line
"""
import os
import sys
import re
import argparse
import textwrap
import logging

loglevels = {'critical': logging.CRITICAL,
             'error': logging.ERROR,
             'warning': logging.WARNING,
             'info': logging.INFO,
             'debug': logging.DEBUG,
             'notset': logging.NOTSET}


class ParseError(Exception):
    pass


def __is_molpro_xml_file(file):
    """Return True if file is a Molpro xml file."""
    with open(file, 'r') as f:
        for line in f:
            if ('<molpro xmlns="http://www.molpro.net/schema/molpro-output"'
                    in line):
                return True
    return False


def __is_molpro_output(file):
    """Return True if file is a Molpro output."""
    with open(file, 'r') as f:
        for l in f:
            if '***  PROGRAM SYSTEM MOLPRO  ***' in l:
                return True
    return False


def __assert_molpro_output(file,
                           can_be_xml=False):
    """Raise ParseError if file does not exist or is not Molpro file."""
    if not os.path.isfile(file):
        raise ParseError('File ' + file + ' not found!')
    if not __is_molpro_output(file):
        if can_be_xml:
            if not __is_molpro_xml_file(file):
                raise ParseError('File ' + file
                                 + ' is not a Molpro output!')
        else:
            raise ParseError('File ' + file + ' is not a Molpro output!')


def parse_cmd_line():
    """Parse the command line for dGr, checking if it is all OK.
    
    Returns:
    --------
    An instance of argparse.Namespace, holding the command line arguments
    and some other string attributes:
    basename      the basename of the main Molpro file
    wdir          the working directory
    command       the execution command
    """
    parser = argparse.ArgumentParser(
        description=('Optimise the distance to'
                     + ' the Grassmannian, obtaining |min D>.'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(textwrap.dedent('''\
        dGr read orbitals from Molpro\'s "put" xml files, that can be
        generated by adding the following line in the Molpro input:
        
        {put,xml,<file_name>; nosort; novariables}
        
        <file_name> is the name of the file that Molpro generates,
        and it is what is passed to dGr after --ini_orb, for example.
        ''')))
    parser.add_argument('molpro_output',
                        help='Molpro output with the correlated wave function')
    parser.add_argument('--output',
                        help='Output file name')
    parser.add_argument('--ini_orb',
                        help='initial guess for orbitals'
                        + ' or transformation matrices,'
                        + ' as Molpro\'s "put" xml file or'
                        + ' npz file')
    parser.add_argument('--HF_orb',
                        help='Hartree-Fock orbitals'
                        + ' (as Molpro\'s "put" xml file)')
    parser.add_argument('--WF_orb',
                        help='orbital basis of the wave function'
                        + ' (as Molpro\'s "put" xml file).'
                        + ' If not given, assume  to be the same as'
                        + ' molpro_output')
    parser.add_argument('--WF_templ',
                        help='a Molpro output with a Full CI wave function,'
                        + ' to be used as template')
    parser.add_argument('--maxiter',
                        help='Maximum number of iterations',
                        type=int)
    parser.add_argument('--at_ref',
                        help='Do only one iteration at reference.',
                        action='store_true')
    parser.add_argument('--algorithm',
                        help='the algorithm to be used in the optimisation.'
                        + ' Possible values are: "orb_rotations",'
                        + ' "general_Absil",  and "CISD_Absil".'
                        + ' Default is "CISD_Absil".')
    parser.add_argument('--save_full_U',
                        help='If set, the saved matrix U will contain'
                        + ' also the virtual orbitals of the optimised'
                        + ' point.',
                        action='store_true')
    parser.add_argument('--state',
                        help='desired state, in Molpro notation')
    parser.add_argument('-l', '--loglevel',
                        help='set log level (integer)')
    parser.add_argument('--logfilter',
                        help='regular expression to filter function names'
                        + ' for logging (for debug)')
    cmd_args = parser.parse_args()
    cmd_args.basename = re.sub(r'\.out$', '', cmd_args.molpro_output)
    cmd_args.wdir = os.getcwd()
    cmd_args.command = ''
    for i, arg in enumerate(sys.argv):
        cmd_args.command += arg + ' ' + ('\\\n'
                                         if (i != len(sys.argv) - 1
                                             and (arg[0] != '-'
                                                  or arg == '--save_full_U'
                                                  or arg == '--at_ref'))
                                         else
                                         '')
    __assert_molpro_output(cmd_args.molpro_output)
    if cmd_args.algorithm is None:
        cmd_args.algorithm = 'CISD_Absil'
    elif cmd_args.algorithm not in ['orb_rotations',
                                    'general_Absil',
                                    'CISD_Absil']:
        raise ParseError('Unknown algorithm: ' + cmd_args.algorithm
                         + '. Possible values:\n'
                         + 'orb_rotations (default), '
                         + 'general_Absil, '
                         + 'and CISD_Absil')
    if cmd_args.maxiter is None:
        cmd_args.maxiter = 20
    elif cmd_args.at_ref:
        raise ParseError('--maxiter is not compatible with --at_ref')
    if cmd_args.ini_orb is not None:
        if cmd_args.at_ref:
            raise ParseError('--ini_orb is not compatible with --at_ref')
        if (cmd_args.ini_orb[-4:] == '.npz'
                and os.path.isfile(cmd_args.ini_orb)):
            pass
        else:
            __assert_molpro_output(cmd_args.ini_orb, can_be_xml=True)
    if cmd_args.WF_orb is None:
        cmd_args.WF_orb = cmd_args.molpro_output
    else:
        __assert_molpro_output(cmd_args.WF_orb, can_be_xml=True)
    if cmd_args.HF_orb is None:
        cmd_args.HF_orb = cmd_args.WF_orb
    else:
        __assert_molpro_output(cmd_args.HF_orb, can_be_xml=True)
    if cmd_args.WF_templ is not None:
        __assert_molpro_output(cmd_args.WF_templ)
    if cmd_args.logfilter is not None:
        cmd_args.logfilter = re.compile(cmd_args.logfilter)
    cmd_args.state = cmd_args.state if cmd_args.state is not None else ''
    if cmd_args.loglevel is not None:
        try:
            cmd_args.loglevel = int(cmd_args.loglevel)
        except ValueError:
            try:
                cmd_args.loglevel = loglevels[cmd_args.loglevel.lower()]
            except KeyError:
                raise ParseError('This is not a valid log level: '
                                 + cmd_args.loglevel)
    else:
        cmd_args.loglevel = logging.WARNING
    return cmd_args
