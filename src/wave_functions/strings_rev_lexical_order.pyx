"""Functions to work with strings in the referse lexical order



"""
import cython
from libc.math cimport sqrt
import numpy as np

from util.variables import int_dtype

cpdef void next_str(int [:] occ):
    """Change occ to the next occupation in reverse lexical order"""
    cdef int i_to_be_raised = 0
    cdef int len_occ_2 = len(occ) - 2
    cdef int i
    while i_to_be_raised <= len_occ_2:
        if occ[i_to_be_raised] + 1 < occ[i_to_be_raised + 1]:
            break
        i_to_be_raised += 1
    occ[i_to_be_raised] += 1
    for i in range(i_to_be_raised):
        occ[i] = i


def eucl_distance(double [:, :] wf1, double [:, :] wf2):
    """Calculate the euclidean distance between wf1 and wf2"""
    cdef int i, j
    cdef int n1 = wf1.shape[0]
    cdef int n2 = wf1.shape[1]
    cdef double S = 0.0
    with nogil:
        for i in range(n1):
            for j in range(n2):
                S += (wf1[i, j] - wf2[i, j])*(wf1[i, j] - wf2[i, j])
    return sqrt(S)


def generate_graph(int nel, int norb):
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
                      dtype=int_dtype)
    str_gr[:,0] = np.arange(norb-nel+1, dtype=int_dtype)
    str_gr[1,:] = 1
    for i in range(2, norb-nel+1):
        for j in range(1, nel):
            str_gr[i,j] = str_gr[i-1,j] + str_gr[i,j-1]
    return str_gr


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_index(int[:] occupation, int[:, :] Y):
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
    cdef int ind = 0
    cdef int i
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
                    dtype=int_dtype)
    for i in range(str_ind):
        next_str(occ)
    return occ


def sign_relative_to_ref(int[:] holes, int[:] particles, int[:] ref_det_occ):
    """Get sign to adapt to reference Slater determinant
    
    Lehtola's decomposition assume that the orbitals are ordered
    with the reference determinant having first all occupied
    orbitals, followed by all virtual orbitals.
    Due to symmetry adapted orbitals, this might not be tha case
    here. This function find the sign to correct this, namely,
    given the holes and particles associated to an excitation
    relative to a given occupation, give the sign to reorder
    the excited determinant as if all occupied orbitals are
    ordered before the virtuals in the reference
    
    Examples:
    ---------
    holes, particles = [5], [2]
    ref_det_occ = [0, 1, 5, 6]
        The excited determinant would be [0, 1, 2, 6]
        in that convention, but [0, 1, 6, 2] if occ < virtual
    return -1
    
    holes, particles = [1, 5], [2, 3]
    ref_det_occ = [0, 1, 5, 6]
        The excited determinant would be [0, 2, 3, 6]
        in that convention, but [0, 6, 2, 3] if occ < virtual
    return 1
    
    holes, particles = [5], [2]
    ref_det_occ = [0, 1, 5]
        The excited determinant would be [0, 1, 2]
        in that convention, and also [0, 1, 2] if occ < virtual
    return 1
    
    holes, particles = [1, 5], [2, 3]
    ref_det_occ = [0, 1, 5]
        The excited determinant would be [0, 2, 3]
        in that convention, and [0, 2, 3] if occ < virtual
    return 1
    
    Parameters:
    -----------
    hole, particles (1D np.array)
        holes and particles, that is, where the excitation comes from 
        and where goes to
    
    ref_det_occ (1D np.array)
        The occupation of the reference
    
    Return:
    -------
    1 or -1
    
"""
    cdef int n = 0
    cdef int p
    cdef int occ
    for p in particles:
        for occ in ref_det_occ:
            if occ > p and occ not in holes:
                n += 1
    return 1 - (n % 2) * 2
