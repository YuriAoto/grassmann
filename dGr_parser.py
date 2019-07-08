"""Parser to dGr


"""
import os
import sys
import re
import argparse
from collections import namedtuple

class ParseError(Exception):
    pass

def __is_molpro_output(file):
    """Return True if file is a Molpro output."""
    with open(file, 'r') as f:
        for l in f:
            if '***  PROGRAM SYSTEM MOLPRO  ***' in l:
                return True
    return False

def __assert_molpro_output(file):
    """Raise ParseError if file does not exist or is not Molpro file."""
    if not os.path.isfile(file):
        raise ParseError('File ' + file + ' not found!')
    if not __is_molpro_output(file):
        raise ParseError('File ' + file + ' is not a Molpro output!')

def parse_cmd_line():
    """Parse the command line for dGr, checking if it is all OK."""
    parser = argparse.ArgumentParser(description='Optimize the Psi_minD.')
    parser.add_argument('molpro_output',
                        help='Molpro output with the wave function')
    parser.add_argument('--ini_orb',
                        help='Initial guess for orbitals or transformation matrices, '
                        + 'as Molpro output or basename of npy files '
                        + '(<basename>_Ua.npy, <basename>_Ub.npy)')
    parser.add_argument('--HF_orb',
                        help='Hartree-Fock orbitals (as Molpro output file)')
    parser.add_argument('--WF_orb',
                        help='Orbital basis of the wave function (as Molpro output file)'
                        + 'If not given, assume to be the same as molpro_output')
    parser.add_argument('--WF_templ',
                        help='A Molpro output with a Full CI wave function,'
                        + ' to be used as template')
    parser.add_argument('--state',
                        help='Desired state, in Molpro notation')
    parser.add_argument('-l', '--loglevel', type=int,
                        help='Set log level (integer)')
    cmd_args = parser.parse_args()
    file_name = cmd_args.molpro_output
    cmd_args.basename = re.sub('\.out$', '', cmd_args.molpro_output)
    cmd_args.wdir = os.getcwd()
    cmd_args.command = ' '.join(sys.argv)
    __assert_molpro_output(cmd_args.molpro_output)
    if cmd_args.ini_orb is not None:
        try:
            __assert_molpro_output(cmd_args.ini_orb)
        except ParseError as e:
            if 'not a Molpro output' in str(e):
                raise e
            else:
                if not os.path.isfile(cmd_args.ini_orb + '_Ua.npy'):
                    raise ParseError('Neither ' + cmd_args.ini_orb
                                     + ' nor ' + cmd_args.ini_orb
                                     + '_Ua.npy exist!')
                if not os.path.isfile(cmd_args.ini_orb + '_Ub.npy'):
                    raise ParseError('Neither ' + cmd_args.ini_orb
                                     + ' nor ' + cmd_args.ini_orb
                                     + '_Ub.npy exist!')
                cmd_args.ini_orb = (cmd_args.ini_orb + '_Ua.npy',
                                    cmd_args.ini_orb + '_Ub.npy')
    if cmd_args.WF_orb is None:
        cmd_args.WF_orb = cmd_args.molpro_output
    else:
        __assert_molpro_output(cmd_args.WF_orb)
    if cmd_args.HF_orb is None:
        cmd_args.HF_orb = cmd_args.WF_orb
    else:
        __assert_molpro_output(cmd_args.HF_orb)
    if cmd_args.WF_templ is not None:
        __assert_molpro_output(cmd_args.WF_templ)
    cmd_args.state = cmd_args.state if cmd_args.state is not None else ''
    return cmd_args
