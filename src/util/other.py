"""Other useful stuff.

Perhaps find a better place/name

"""

import numpy as np

from util.variables import int_dtype

def int_array(*x):
    """Return an array of the integers in x
    
    If a float is passed in any argument, it is transformed to int
    (not rounded, byt casted)
    
    Parameters:
    -----------
    No parameters: an empty list is returned
    
    A list, tuple, or ndarray: it is transformed to a array with np.array()
    
    Several numbers: an one-dimensional array with these integers is returned,
    as in int_array([*x])
    
    """
    if len(x) == 0:
        return np.array([], dtype=int_dtype)
    if len(x) == 1 and not isinstance(x[0], (int, float)) :
        return np.array(x[0], dtype=int_dtype)
    return np.array(x, dtype=int_dtype)
