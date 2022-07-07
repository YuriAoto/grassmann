"""


"""
from wave_functions import cisd
from wave_functions.interm_norm import IntermNormWaveFunction
from input_output import molpro as molproout

def load_wave_function(molpro_output,
                       state='1.1',
                       method=None,
                       ith=0,
                       use_CISD_norm=True,
                       wf_obj_type='interm_norm',
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
        Possible values are: 'interm_norm', 'cisd', 'fci'.
    
    wf (wave_function.general.WaveFunction or None, optional, default=None)
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
    this_ith = 0
    with open(molpro_output, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            try:
                point_group = molproout.get_point_group_from_line(
                    line, line_number, molpro_output)
            except molproout.MolproLineHasNoPointGroup:
                pass
            if line in (molproout.FCI_header,
                        molproout.CISD_header,
                        molproout.UCISD_header,
                        molproout.RCISD_header,
                        molproout.CCSD_header,
                        molproout.UCCSD_header,
                        molproout.RCCSD_header,
                        molproout.BCCD_header):
                if method is not None:
                    if method == 'FCI' and line != molproout.FCI_header:
                        continue
                    if method == 'CISD' and line not in (molproout.CISD_header,
                                                         molproout.UCISD_header,
                                                         molproout.RCISD_header):
                        continue
                    if method == 'CS-CISD' and line != molproout.CISD_header:
                        continue
                    if method == 'RCISD' and line != molproout.RCISD_header:
                        continue
                    if method == 'UCISD' and line != molproout.UCISD_header:
                        continue
                    if method == 'CCSD' and line not in (molproout.CCSD_header,
                                                         molproout.UCCSD_header,
                                                         molproout.RCCSD_header):
                        continue
                    if method == 'CS-CCSD' and line != molproout.CCSD_header:
                        continue
                    if method == 'RCCSD' and line != molproout.RCCSD_header:
                        continue
                    if method == 'UCCSD' and line != molproout.UCCSD_header:
                        continue
                    if method == 'BCCD' and line != molproout.BCCD_header:
                        continue
                if this_ith != ith:
                    this_ith += 1
                    continue
                wf_type = line[11:15]
                if line == molproout.FCI_header:
                    raise Exception('This was with NormCI_WaveFunction'
                                    ' and has been removed. TODO: implement'
                                    ' with FCIWaveFunction')
                    ##wf = norm_ci.NormCI_WaveFunction()
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
                    wf_interm_norm = IntermNormWaveFunction.from_Molpro(
                            f, start_line_number=line_number-1,
                            wf_type=wf_type,
                            point_group=point_group)
                    wf_interm_norm.use_CISD_norm = use_CISD_norm
                    wf = wf_interm_norm
                    if wf_obj_type == 'cisd':
                        wf = cisd.CISDWaveFunction.from_interm_norm(wf)
                    elif wf_obj_type == 'fci':
                        raise Exception('This was with NormCI_WaveFunction'
                                        ' and has been removed. TODO: implement'
                                        ' with FCIWaveFunction')
                break
    return wf
