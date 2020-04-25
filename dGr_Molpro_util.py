"""Useful variables and functions for treating Molpro outputs


"""
import re

import dGr_general_WF as genWF
from dGr_exceptions import dGrMolproInputError

CISD_header = (
    ' PROGRAM * CISD (Closed-shell CI(SD))     '
    + 'Authors: '
    + 'C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n')
UCISD_header = (
    ' PROGRAM * CISD (Unrestricted open-shell CI(SD))     '
    + 'Authors: '
    + 'C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n')
RCISD_header = (
    ' PROGRAM * CISD (Restricted open-shell CI(SD))     '
    + 'Authors: '
    + 'C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n')
CCSD_header = (
    ' PROGRAM * CCSD (Closed-shell coupled cluster)     '
    + 'Authors: '
    + 'C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n')
MRCI_header = (
    ' PROGRAM * CI (Multireference internally contracted CI)     '
    + 'Authors: H.-J. Werner, P.J. Knowles, 1987\n')
FCI_header = ' PROGRAM * FCI (Full CI)     Author: P.J. Knowles, 1984\n'

MRCI_ref_str = (
    ' Reference coefficients greater than 0.0000000\n')
MRCI_sgl_str = (
    ' Coefficients of singly external configurations greater than 0.0000000\n')
MRCI_dbl_str = (
    ' Coefficients of doubly external configurations greater than 0.0000000\n')

CC_sgl_str = 'I         SYM. A    A   T(IA)'
CC_dbl_str = (
    'I         J         SYM. A    SYM. B    A         B      T(IJ, AB)')

CC_sgl_str_2 = ' Singles amplitudes (print threshold =  0.000E+00):\n'
CC_dbl_str_2 = ' Doubles amplitudes (print threshold =  0.000E+00):\n'


def get_orb_info(l, line_number, n_irrep, occ_type):
    """Load orbital info, per irrep, from Molpro's output"""
    re_orb = re.compile(r'.*\s([\d]+)\s*\(([\s\d]*)\)').match(l)
    if re_orb is None:
        raise dGrMolproInputError('Problems reading orbital information',
                                  line=l, line_number=line_number)
    try:
        re_orb = list(map(int, re_orb.group(2).split()))
        if len(re_orb) < n_irrep:
            re_orb += [0] * (n_irrep - len(re_orb))
        elif len(re_orb) > n_irrep:
            re_orb = re_orb[:n_irrep]
        if occ_type == 'F':
            re_orb += re_orb
        return genWF.Orbitals_Sets(re_orb, occ_type=occ_type)
    except Exception:
        raise dGrMolproInputError('Problems reading orbital information.',
                                  line=l, line_number=line_number)
