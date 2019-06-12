"""Parser to dGr


"""
import os
import sys
import re
import argparse
from collections import namedtuple

class ParseError(Exception):
    pass


CmdArgs = namedtuple('CmdArgs',
                     ['basename',
                      'file_name',
                      'file_name_iniU',
                      'file_name_HF',
                      'file_name_FCI',
                      'WF_orb',
                      'state',
                      'loglevel',
                      'wdir',
                      'command'])


def parse_cmd_line():
    """Parse the command line for dGr"""
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
                        help='Orbital basis of the wave function (as Molpro output file)')
    parser.add_argument('--state',
                        help='Desired state, in Molpro notation')
    parser.add_argument('--FCIwf',
                        help='The Full CI wave function, to be used as template')
    parser.add_argument('-l', '--loglevel', type=int,
                        help='Set log level (integer)')
    cmd_args = parser.parse_args()
    file_name = cmd_args.molpro_output
    basename = re.sub('\.out$', '', file_name)
    if cmd_args.ini_orb is None:
        file_name_iniU = None
    else:
        if os.path.isfile(cmd_args.ini_orb):
            file_name_iniU = cmd_args.ini_orb
        else:
            if not os.path.isfile(cmd_args.ini_orb + '_Ua.npy'):
                ParseError('Neither ' + cmd_args.ini_orb
                           + ' nor ' + cmd_args.ini_orb + '_Ua.npy exist!')
            if not os.path.isfile(cmd_args.ini_orb + '_Ub.npy'):
                ParseError('Neither ' + cmd_args.ini_orb
                           + ' nor ' + cmd_args.ini_orb + '_Ub.npy exist!')
            file_name_iniU = (cmd_args.ini_orb + '_Ua.npy',
                              cmd_args.ini_orb + '_Ub.npy')
    if cmd_args.HF_orb is None:
        file_name_HF = None
    else:
        if os.path.isfile(cmd_args.HF_orb):
            file_name_HF = cmd_args.HF_orb
        else:
            ParseError('File ' + cmd_args.HF_orb + ' not found!')
    if cmd_args.WF_orb is None:
        WF_orb = file_name
    else:
        if not os.path.isfile(cmd_args.WF_orb):
            ParseError('File ' + cmd_args.WF_orb + ' not found!')
        WF_orb = cmd_args.WF_orb
    state = cmd_args.state if cmd_args.state is not None else ''
    loglevel = cmd_args.loglevel
    file_name_FCI = cmd_args.FCIwf
    if file_name_FCI is not None and not os.path.isfile(file_name_FCI):
        ParseError('File ' + file_name_FCI + ' not found!')
    return CmdArgs(basename = basename,
                   file_name = file_name,
                   file_name_iniU = file_name_iniU,
                   file_name_HF = file_name_HF,
                   file_name_FCI = file_name_FCI,
                   WF_orb = WF_orb,
                   state = state,
                   loglevel = loglevel,
                   wdir = os.getcwd(),
                   command = ' '.join(sys.argv))
