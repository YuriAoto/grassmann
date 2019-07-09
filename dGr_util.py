""" Some useful functions and definitions

"""

import math

from scipy import linalg

sqrt2 = math.sqrt(2.0)

def dist_from_ovlp(x):
    """Convert from overlap to distance."""
    return sqrt2 * math.sqrt(1 - abs(x))

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
