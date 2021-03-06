"""Useful variables and functions for treating Molpro outputs


"""
import re

from orbitals.orbital_space import OrbitalSpace
from molecular_geometry.symmetry import number_of_irreducible_repr

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
UCCSD_header = (
    ' PROGRAM * CCSD (Unrestricted open-shell coupled cluster)     '
    + 'Authors: '
    + 'C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n')
RCCSD_header = (
    ' PROGRAM * CCSD (Restricted open-shell coupled cluster)     '
    + 'Authors: '
    + 'C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n')
BCCD_header = (
    ' PROGRAM * BCCD (Closed-shell Brueckner coupled-cluster)     '
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

CC_sgl_str_BCC_2 = (
    ' Orbital rotation generators (print threshold =  0.000E+00):\n')
CC_sgl_str_2 = ' Singles amplitudes (print threshold =  0.000E+00):\n'
CC_dbl_str_2 = ' Doubles amplitudes (print threshold =  0.000E+00):\n'


class MolproInputError(Exception):
    def __init__(self, msg, line=None, line_number=None, file_name=None):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.line_number = line_number
        self.file_name = file_name

    def __str__(self):
        return (super().__str__() + '\n'
                + 'at line ' + str(self.line_number)
                + ' of file ' + str(self.file_name) + ':\n'
                + str(self.line))


class MolproLineHasNoPointGroup(Exception):
    pass


def get_point_group_from_line(line, line_number, molpro_output):
    if 'Point group' in line:
        point_group = line.split()[2]
        try:
            number_of_irreducible_repr[point_group]
        except KeyError:
            raise MolproInputError(
                'Unknown point group!',
                line=line,
                line_number=line_number,
                file_name=molpro_output)
        return point_group
    raise MolproLineHasNoPointGroup()


def get_orb_info(line, line_number, n_irrep, occ_type):
    """Load orbital info, per irrep, from Molpro's output"""
    re_orb = re.compile(r'.*\s([\d]+)\s*\(([\s\d]*)\)').match(line)
    if re_orb is None:
        raise MolproInputError('Problems reading orbital information',
                               line=line, line_number=line_number)
    try:
        re_orb = list(map(int, re_orb.group(2).split()))
        if len(re_orb) < n_irrep:
            re_orb += [0] * (n_irrep - len(re_orb))
        elif len(re_orb) > n_irrep:
            re_orb = re_orb[:n_irrep]
        if occ_type == 'F':
            re_orb += re_orb
        return OrbitalSpace(dim=re_orb, orb_type=occ_type)
    except Exception as exc:
        print(exc)
        raise MolproInputError('Problems reading orbital information.',
                               line=line, line_number=line_number)


