"""Functions and variables related to (electronic) systems for testing

This supplies functions to access and iterate over directories
in grassmann/test/inputs_outputs/, where tests can take place.

In particular:

# the generator test_systems can be used to iterate
over these directories, with filters.

# The following functions give the full path for files:
CISD_file, CCSD_file, FCI_file
RHF_file, UHF_file, orbitals_file

The first three accept an optional argument allE, for all electron
results

"""
import os

main_files_dir = ('/home/yuriaoto/Documents/Codes/'
                  + 'grassmann/src/test/inputs_outputs/')

_all_test_systems = [
    'H2__5__sto3g__C1',
    'H2__5__sto3g__D2h',
    'H2__5__631g__C1',
    'H2__5__631g__C2v',
    'H2__5__631g__D2h',
    'H2__5__ccpVDZ__C1',
    'H2__5__ccpVDZ__Cs',
    'H2__5__ccpVDZ__C2v',
    'H2__5__ccpVDZ__D2h',
    'He2__1.5__631g__C2v',
    'He2__1.5__631g__D2h',
    'Li2__5__to2s__C2v',
    'Li2__5__to3s__C2v',
    'Li2__5__sto3g__C1',
    'Li2__5__sto3g__C2v',
    'Li2__5__sto3g__D2h',
    'Li2__5__631g__C1',
    'Li2__5__631g__C2v',
    'Li2__5__631g__D2h',
    'Li2__5__ccpVDZ__C1',
    'Li2__5__ccpVDZ__Cs',
    'Li2__5__ccpVDZ__C2v',
    'Li2__5__ccpVDZ__D2h',
    'Li2__5__ccpVTZ__D2h',
    'Li2__5__ccpVQZ__D2h',
    'N2__3__sto3g__D2h',
    'N2__3__631g__D2h',
    'N2__3__631g__D2h_occ_21101110',
    'N2__3__cc-pVDZ__D2h',
    'HCl__1.5__631g__C1',
    'HCl__1.5__631g__C2v',
    'h2o__Req__sto3g__C2v',
    'h2o__Req__631g__C2v',
    'h2o__1.5__sto3g__C2v',
    'H8_cage__1.5__631g__D2h',
    'He8_cage__1.5__631g__D2h',
    'He8_cage__1.5__ccpVDZ__D2h',
    'Li8_cage__1.5__631g__D2h',
]

_files = ['CISD',
          'CCSD',
          'FCI']

_files_no_allE = ['RHF',
                  'UHF']


def _get_inpout_file(file_name, only_exist=True, only_check=False):
    """Function to get/check file from inputs_outputs
    
    Parameters:
    -----------
    file_name (str)
    
    only_exist (bool, optional, default=True)
    
    only_check (bool, optional, default=False)

    
    """
    full_fname = main_files_dir + file_name
    if only_check:
        return os.path.isfile(full_fname)
    if (not only_exist or os.path.isfile(full_fname)):
        return full_fname
    raise FileNotFoundError('There is no ' + full_fname)


_func_get_file = r"""
def METHODNAME_file(inp_out_dir,
                    allE=False,
                    only_exist=True,
                    only_check=False):
    return _get_inpout_file(inp_out_dir +
                            ('/METHODNAME_allE.out'
                             if allE else
                             '/METHODNAME.out'),
                            only_exist, only_check)
"""

_func_get_file_no_allE = r"""
def METHODNAME_file(inp_out_dir,
                    only_exist=True,
                    only_check=False):
    return _get_inpout_file(inp_out_dir + '/METHODNAME.out',
                            only_exist, only_check)
"""


def orbitals_file(inp_out_dir,
                  only_exist=True,
                  only_check=False):
    return _get_inpout_file(inp_out_dir + '/orbitals.xml',
                            only_exist, only_check)


for f in _files:
    exec(_func_get_file.replace('METHODNAME', f))
    
for f in _files_no_allE:
    exec(_func_get_file_no_allE.replace('METHODNAME', f))


def test_systems(has_method=None,
                 molecule=None,
                 basis=None,
                 symmetry=None,
                 max_number=None):
    """Yield all electronic systems that satisfy condidions
    
    Behaviour:
    -----------
    Every argument should be a str or container of str,
    all are optional, with default=None.
    These are interpreted as conditions:
    Only systems that meet those conditions are yielded.
    
    For has_method, this generator will yield only systems
    that have outputs for all cases given methods given.
    Say: if has_method=('CISD', 'CISD_allE', 'FCI'),
    only the systems that have the output files
    'CISD.out', 'CISD_allE.out', and 'FCI.out' will be yielded.
    
    For the others, systems that satisfy any one of the conditions are
    yielded. So, if molecule='(H2, Li2)', systems for H2 and Li2 will
    be yielded.
        
    Parameters:
    -----------
    has_method  (str or container of str, optional, default=None)
    
    molecule (str or container of str, optional, default=None)
    
    basis (str or container of str, optional, default=None)
    
    symmetry (str or container of str, optional, default=None)
    
    max_number (str or container of str, optional, default=None)
    
    Yield:
    ------
    str for the directories in grassmann/test/inputs_outputs/, that represent
    test_systems systems with molpro outputs where tests can take place.
    
    """
    def does_not_satisfy(x, cond):
        return cond is not None and x not in cond
    if isinstance(has_method, str):
        has_method = (has_method,)
    if isinstance(molecule, str):
        molecule = (molecule,)
    if isinstance(basis, str):
        basis = (basis,)
    if isinstance(symmetry, str):
        symmetry = (symmetry,)
    i = 0
    for s in _all_test_systems:
        s_mol, s_geo, s_basis, s_sym = s.split('__')
        if (does_not_satisfy(s_mol, molecule)
            or does_not_satisfy(s_basis, basis)
                or does_not_satisfy(s_sym, symmetry)):
            continue
        if has_method is not None:
            all_m_found = True
            for m in has_method:
                if not _get_inpout_file(s + '/' + m + '.out',
                                        only_check=True):
                    all_m_found = False
                    break
            if not all_m_found:
                continue
        yield s
        i += 1
        if max_number is not None and i == max_number:
            break
    if i == 0:
        raise ValueError('Did not yield anything!!')
