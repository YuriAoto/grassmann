import unittest

import numpy as np
from scipy import linalg


def assert_arrays(array1, array2, msg=None):
    """Asserts arrays. To be used with addTypeEqualityFunc"""
    if array1.shape != array2.shape:
        raise unittest.TestCase.failureException(msg)
    if len(array1) == 0:
        return True
    if array1.dtype != array2.dtype:
        raise unittest.TestCase.failureException(msg)
    if (np.issubdtype(array1.dtype, np.integer)
            and not (array1 == array2).all()):
        raise unittest.TestCase.failureException(msg)
    if (np.issubdtype(array1.dtype, np.floating)
            and not np.allclose(array1, array2, rtol=1e-05, atol=1e-08)):
        raise unittest.TestCase.failureException(msg)
    return True


def assert_occupations(occ1, occ2, msg=None):
    """Assert occupations. To be used with addTypeEqualityFunc"""
    pass


def extend_to_unrestricted(U):
    """Extend U to include all irreps (but with same orbitals)"""
    for i in range(len(U)):
        U.append(np.array(U[i]))


def construct_Id_orbitals(n, K, n_irrep,
                          full=False):
    """Return the identity for each irrep
    
    Parameters:
    -----------
    n (Iterable of int)
        number of electrons in each irrep
    
    K (Iterable of int)
        number of orbitals in each irrep
    
    n_irrep (int)
        number of irreducible representations
    
    full (bool, optional, default=False)
        If True, returns the full set of orbitals,
        even for virtuals
    
    Returns:
    --------
        A list of np.ndarrays, with the orbitals of each (sp)irrep
    
    """
    U = []
    for irrep in range(n_irrep):
        U.append(np.identity(K[irrep]))
        if not full:
            U[-1] = U[-1][:, :n[irrep]]
    return U


def construct_random_orbitals(n, K, n_irrep, random_state,
                              full=False, orthogonalise=True):
    """Return random coefficients for each irrep
    
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
