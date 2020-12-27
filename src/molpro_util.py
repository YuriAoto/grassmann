"""Useful variables and functions for treating Molpro outputs


"""
import re

from wave_functions import general, fci, int_norm, cisd
import util

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
    'PROGRAM * CCSD (Unrestricted open-shell coupled cluster)     '
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
            util.number_of_irreducible_repr[point_group]
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
        return general.Orbitals_Sets(re_orb, occ_type=occ_type)
    except Exception:
        raise MolproInputError('Problems reading orbital information.',
                               line=line, line_number=line_number)


def load_wave_function(molpro_output,
                       WF_templ=None,
                       state='1.1',
                       method=None,
                       ith=0,
                       use_CISD_norm=True,
                       wf_obj_type='int_norm',
                       _zero_coefficients=False,
                       _change_structure=True,
                       _use_structure=False):
    """Load an arbitrary wave function from Molpro output
    
    Behaviour:
    ----------
    This function loads and return an electronic wave function from
    a Molpro output. It loads that of the i-th method encontered in the
    file, that satisfies the optional arguments state and method.
    
    Parameters:
    ----------
    molpro_output (str, a file name)
        The file with the molpro output, where the wave function will be
        read from
    
    WF_templ (str, a file name, optional, default=None)
        If not None, should be the file name of a Molpro output with
        a wave function (of FCI type), where the structure will be read.
        In this case, the coefficients of the wave function at
        molpro_output will be put in the "template" read in this file.
        If passed, it is assumed that the wave function to be used
        as a template is the first FCI wave function of the file WF_templ.
        That is, options such as state, method, or ith do not affect
        the reading of the template.
    
    state (str, optional, default=None)
        If not None, it should be the designation of a electronic state,
        in Molpro format, '<state_number>.<symmetry>
        (e.g., '1.1', '2.3', etc).
    
    method (str, optional, default=None)
        If not None, it should be a method of electronic structure.
        Possible values:
        'FCI',
        'CISD', 'CS-CISD', 'RCISD', 'UCISD',
        'CCSD', 'CS-CCSD', 'RCCSD', 'UCCSD',
        'BCCD'
        If 'CISD' or 'CCSD' are given, any CISD or CCSD will be considered
        (respectivelly, not considering BCCD).
        On the other hand, if 'CS-CISD' is given, only closed-shell
        CISD will be considered (and the same for 'RCISD', 'UCISD',
        'CS-CCSD', ...).
        To this end, the header in the molpro program is considered.
        Thus, note that asking for CCSD in molpro for a open-shell
        system calls the UCCSD program (but with the "closed-shell"
        header). Thus this would be captured by "CS-CCSD".
    
    ith (int, optional, default=0)
        Loads the ith wave function in the file. If method
        is passed, loads the ith wave function of that method.
    
    wf_obj_type (str, optional, default='intN')
        Indicates the object type to be returned after reading
        a molpro output with a CI/CC wave function in intermediate
        normalisation
        Possible values are: 'int_norm', 'cisd', 'fci'.
    
    wf (general.Wave_Function or None, optional, default=None)
        If given, this object is changed, and nothing is returned.
        Otherwise a new object is created and returned (needed??)
    
    _zero_coefficients (bool, optional, default=False)
        If True, all coefficients (or amplitudes) are set to zero,
        and only the structure of the wave function is obtained
        (for internal use).
    
    _change_structure (bool, optional, default=True)
        If True, admits changes in the structure of the wave function
        (e.g, by adding new determinants)

    _use_structure (bool, optional, default=False)
        If True, uses the structure already present in the object
    
    """
    point_group = None
    if WF_templ is not None:
        wf = load_wave_function(WF_templ,
                                WF_templ=None,
                                state='1.1',
                                method='FCI',
                                ith=0,
                                _zero_coefficients=True,
                                _change_structure=True,
                                _use_structure=False)
    this_ith = 0
    with open(molpro_output, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            try:
                point_group = get_point_group_from_line(
                    line, line_number, molpro_output)
            except MolproLineHasNoPointGroup:
                pass
            if line in (FCI_header,
                        CISD_header, UCISD_header, RCISD_header,
                        CCSD_header, UCCSD_header, RCCSD_header,
                        BCCD_header):
                if method is not None:
                    if method == 'FCI' and line != FCI_header:
                        continue
                    if method == 'CISD' and line not in (CISD_header,
                                                         UCISD_header,
                                                         RCISD_header):
                        continue
                    if method == 'CS-CISD' and line != CISD_header:
                        continue
                    if method == 'RCISD' and line != RCISD_header:
                        continue
                    if method == 'UCISD' and line != UCISD_header:
                        continue
                    if method == 'CCSD' and line not in (CCSD_header,
                                                         UCCSD_header,
                                                         RCCSD_header):
                        continue
                    if method == 'CS-CCSD' and line != CCSD_header:
                        continue
                    if method == 'RCCSD' and line != RCCSD_header:
                        continue
                    if method == 'UCCSD' and line != UCCSD_header:
                        continue
                    if method == 'BCCD' and line != BCCD_header:
                        continue
                if this_ith != ith:
                    this_ith += 1
                    continue
                wf_type = line[11:15]
                if line == FCI_header:
                    if WF_templ is None:
                        wf = fci.Wave_Function_Norm_CI()
                    wf.WF_type = 'FCI'
                    wf.point_group = point_group
                    wf.source = 'From file ' + molpro_output
                    wf.get_coeff_from_molpro(
                        f,
                        start_line_number=line_number-1,
                        point_group=point_group,
                        state=state,
                        zero_coefficients=_zero_coefficients,
                        change_structure=_change_structure,
                        use_structure=_use_structure)
                else:
                    wf_int_norm = int_norm.Wave_Function_Int_Norm.from_Molpro(
                        f, start_line_number=line_number-1,
                        wf_type=wf_type,
                        point_group=point_group)
                    wf_int_norm.use_CISD_norm = use_CISD_norm
                    wf_int_norm.source = 'From file ' + molpro_output
                    if WF_templ is not None:
                        wf.get_coeff_from_int_norm_WF(wf_int_norm,
                                                      change_structure=False,
                                                      use_structure=True)
                    else:
                        wf = wf_int_norm
                        if wf_obj_type == 'cisd':
                            wf = cisd.Wave_Function_CISD.from_int_norm(wf)
                        elif wf_obj_type == 'fci':
                            wf = fci.Wave_Function_Norm_CI.from_int_norm(wf)
                break
    return wf
