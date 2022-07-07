"""

"""
import numpy as np

from util.variables import int_dtype


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef bint annihilates(int i, int a, int[:] I):
    cdef int pos
    cdef bint missing_i = True
    for pos in range(len(I)):
        if I[pos] == a:
            return i != a
        if I[pos] == i:
            missing_i = False
    return missing_i



#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef int exc_on_string(int i, int a, int[:] I, int[:] new_I):
    """Obtain the string after the excitation i->a over I
    
    Parameters:
    -----------
    i
        hole index (the orbital where the excitation comes from)
    
    a
        particle index (the orbital where the excitation goes to)
    
    I
        The string that represents the Slater determinant
        where the excitation acts, with a previous sign in the last entry.
    
    new_I
        Is an output:
        This should be allocated by the caller and it is overwritten
        in this function by the indices of occupied   orbitals of the
        new Slater determinant.
    
    Return:
    -------
        If the excitation i->a annihilates the Slater determinant,
        namelly, if i does not belong to I or a does belong, return 0.
        Otherwise, the sign that arises after putting the
        orbitals in the correct order is returned.
    """
    cdef int n = I.shape[0]
    cdef int pos, i_pos, a_pos
    i_pos = -1
    a_pos = 0
    new_I[:] = I[:]
    pos = 0
    if i <= a:
        a_pos = n - 1
        for pos in range(n):
            if I[pos] == a:
                return 1 if i == a else 0
            if I[pos] == i:
                i_pos = pos
            if I[pos] > a:
                a_pos = pos - 1
                break
        if i_pos == -1:
            return 0
        new_I[i_pos: a_pos] = I[i_pos+1: a_pos+1]
        new_I[a_pos] = a
    elif i > a:
        for pos in range(n-1, -1, -1):
            if I[pos] == a:
                return 0
            if I[pos] == i:
                i_pos = pos
            if I[pos] < a:
                a_pos = pos + 1
                break
        if i_pos == -1:
            return 0
        new_I[a_pos+1: i_pos+1] = I[a_pos: i_pos]
        new_I[a_pos] = a
    return 1 - 2*(abs(a_pos - i_pos) % 2)
