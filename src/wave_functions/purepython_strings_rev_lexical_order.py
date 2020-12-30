"""Functions to work with strings in the referse lexical order

This is a pure Python implementation, intended to compare performance
to the Cython version

"""
import numpy as np

def next_str(occ):
    """Change occ to the next occupation in reverse lexical order"""
    i_to_be_raised = 0
    while i_to_be_raised <= len(occ)-2:
        if occ[i_to_be_raised] + 1 < occ[i_to_be_raised + 1]:
            break
        i_to_be_raised += 1
    occ[i_to_be_raised] += 1
    occ[:i_to_be_raised] = np.arange(i_to_be_raised, dtype=np.intc)


def generate_graph(nel, norb):
    """Generate the string graph matrix to set the reverse lexical order
    
    Parameters:
    -----------
    nel (int)
        Number of electrons
    
    norb (int)
        Number of orbitals
    
    Return:
    -------
    A np.array with the string graph matrix
    
    Raise:
    ------
    ValueError if nel or orb is not positive, or if nel > norb
    """
    if nel <= 0 or norb <= 0 or nel > norb:
        raise ValueError('nel and norb must be positive and nel > norb')
    str_gr = np.zeros((norb-nel+1, nel),
                      dtype=np.intc)
    str_gr[:,0] = np.arange(norb-nel+1, dtype=np.intc)
    str_gr[1,:] = 1
    for i in range(2, norb-nel+1):
        for j in range(1, nel):
            str_gr[i,j] = str_gr[i-1,j] + str_gr[i,j-1]
    return str_gr


def get_index(occupation, Y):
    """Get the position of a alpha/beta string
    
    This is the inverse of _occupation_from_string_index: The following
    should hold True for any valid str_ind:
    
    _occupation_from_string_index(_get_string_index(occupation, Y), Y)
    
    Parameters:
    -----------
    occ (np.array of int)
        the occupied orbitals
    
    Y (2D np.array)
        The string-graph matrix that allows the calculation of
        the position of occupation in the reverse lexical order
        
    Return:
    -------
    An integer with the position
    
    Raise:
    ------
    ValueError if the position of occ cannot be obtained from Y
    """
    ind = 0
    for i in range(Y.shape[1]):
        ind += Y[occupation[i] - i, i]
    return ind


def occ_from_pos(str_ind, Y):
    """Get the occupied orbitals from its referse lexical order
    
    This is the inverse of _get_string_index: The following
    should hold True for any valid str_ind:
    
    _get_string_index(_occupation_from_string_index(str_ind, Y), Y)
    
    
    Parameters:
    -----------
    str_ind (int)
        the position in the reverse lexical order
    
    Y (2D np.array)
        The string-graph matrix that allows the calculation of
        the position of occupation in the reverse lexical order
        
    Return:
    -------
    A 1D np.array of int, of size Y.shape[1], in ascending order, with the occupied orbitals
    
    Raise:
    ------
    ValueError if the occupation cannot be retrieved from str_ind
    """
    occ = np.arange(Y.shape[1],
                    dtype=np.intc)
    for i in range(str_ind):
        next_str(occ)
    return occ

