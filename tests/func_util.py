"""Useful functions for tests. See package documentation"""
import unittest

import numpy as np
from scipy import linalg

assert_arrays_rtol = 1e-5
assert_arrays_atol = 1e-6

def assert_arrays(array1, array2, msg=None):
    """Asserts arrays. To be used with addTypeEqualityFunc"""
    if array1.shape != array2.shape:
        raise unittest.TestCase.failureException(
            '\n' + str(array1) + '\n!=\n' + str(array2))
    if len(array1) == 0:
        return True
    if array1.dtype != array2.dtype:
        raise unittest.TestCase.failureException(
            '\n' + str(array1) + '\n!=\n' + str(array2))
    if (np.issubdtype(array1.dtype, np.integer)
            and not (array1 == array2).all()):
        raise unittest.TestCase.failureException(
            '\n' + str(array1) + '\n!=\n' + str(array2))
    if (np.issubdtype(array1.dtype, np.floating)
            and not np.allclose(array1, array2,
                                rtol=assert_arrays_rtol,
                                atol=assert_arrays_atol)):
        raise unittest.TestCase.failureException(
            '\n' + str(array1) + '\n!=\n' + str(array2))
    return True


def assert_occupations(occ1, occ2, msg=None):
    """Assert occupations. To be used with addTypeEqualityFunc"""
    raise NotImplementedError('To be done...')


def construct_random_orbitals(n, K, n_irrep, random_state,
                              full=False, orthogonalise=True):
    """Return random coefficients for each irrep
    
    Similar to orbitals.construct_Id_orbitals, but generates
    random coefficients.
    
    Parameters:
    -----------
    See construct_Id_orbitals;
    
    random_state (np.random.random_state)
        the state to call random_sample
    
    orthogonalise (bool, optional, default=True)
        If True, orthogonalise the orbitals
    
    Returns:
    --------
        A list of np.ndarrays, with the orbitals of each (sp)irrep
    """
    U = []
    for irrep in range(n_irrep):
        U.append(random_state.random_sample(
            size=(K[irrep],
                  K[irrep] if full else n[irrep])))
        if orthogonalise:
            if U[-1].shape[0] * U[-1].shape[1] != 0:
                U[-1] = linalg.orth(U[-1])
    return U
