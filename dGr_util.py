""" Some useful functions and definitions

"""

import math

from scipy import linalg

sqrt2 = math.sqrt(2.0)

def dist_from_ovlp(x):
    """Convert from overlap to distance."""
    try:
        return sqrt2 * math.sqrt(1 - abs(x))
    except ValueError:
        return 0.0

def ovlp_Slater_dets(Ua, Ub, na, nb):
    """Calculate the overlap between two Slater determinants
    
    Behaviour:
    
    Given the transformation matrices (alpha and beta) between two
    MO basis, calculate the overlap between the first determinant
    associated with each basis. That is, calculates <phi 1|phi 2>,
    where |phi i> are Slater determinants and Ua, Ub (see below)
    are the matrices that transforms the orbitals from a basis B1
    (where |phi 1> is the Slater determinant associated to the
    first orbitals) to a basis B2 (where |phi 2> is the Slater
    determinant associated to the first orbitals)
    
    Parameters:
    
    Ua      transformation matrix for alpha orbitals
    Ub      transformation matrix for beta orbitals
    na      number of alpha orbitals
    nb      number of beta orbitals
    
    Returns:
    
    The overlap between the determinants (float)
    """
    return linalg.det(Ua[:na,:na])*linalg.det(Ub[:nb,:nb])

def str_matrix(X):
    """Return a str of the 2D list or array X."""
    strM = []
    for i in X:
        strI = []
        for j in i:
            strI.append(' {0:10.6f} '.format(j)\
                        if abs(j) > 1.0E-7 else
                        (' ' + '-'*10 + ' '))
        strM.append(''.join(strI))
    return '\n'.join(strM)

def get_I(n, i, a):
    """return range(n).remove(i) + [a]"""
    if isinstance(i, int):
        if not isinstance(a, int):
            raise ValueError('Both i and a must be list or int!')
        return [x for x in range(n) if x != i] + [a]
    else:
        return [x for x in range(n) if x not in i] + a
