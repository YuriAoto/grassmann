"""Functions related to the overlap-metrics defined in the proj. space

"""
import math

import numpy as np
from scipy import linalg


def dist_from_ovlp(ovlp,
                   metric='Fubini-Study',
                   norms=(1.0, 1.0),
                   tol=1E-8):
    """Calculate the distance based on the overlap
    
    Given the overlap between two wave functions,
    <Psi|Phi>, calculates the distance between them.
    The following metrics are available
    (normalised wave functions are assumed in the following
    expressions):
    
    Fubini-Study:
        D(Psi, Phi) = arccos |<Psi|Phi>|
    
    DAmico (D'Amico et. al PRL 106 (2011) 050401):
        D(Psi, Phi) = sqrt(2) sqrt(1 - |<Psi|Phi>|)
    
    Benavides-Riveros (Benavides-Riveros et. al PRA 95 (2017) 032507):
        D(Psi, Phi) = 1 - |<Psi|Phi>|^2
    
    Parameters:
    -----------
    ovlp (float)
        The overlap between two wave functions
    
    metric (str, optional, default='Fubini-Study')
        The metric to be used.
    
    norms (2-tuple of floats, optional, default=(1.0, 1.0))
        The norms of both wave functions
    
    tol (float, optional, default=1E-8)
        Tolerance for acepting too large overlap (see Raises below)
    
    Returns:
    --------
    The distance associated to ovlp (float)
    
    Raises:
    -------
    ValueError
        if metric is unknown or if
        ovlp / (norms[0] * norms[1]) > 1 + tol
    """
    absovlp = abs(ovlp / (norms[0] * norms[1]))
    if absovlp > 1.0 + tol:
        raise ValueError(
            '|<Psi|Phi>|/(|Psi||Phi|) > 1: ' + str(absovlp))
    absovlp = min(absovlp, 1.0)
    if metric == 'Fubini-Study':
        return np.arccos(absovlp)
    elif metric == 'DAmico':
        return math.sqrt(2) * math.sqrt(1 - absovlp)
    elif metric == 'Benavides-Riveros':
        return 1 - absovlp**2
    else:
        raise ValueError('Unknown metric in the wave functions space: '
                         + metric)


def ovlp_Slater_dets(U, n):
    """Calculate the overlap between two Slater determinants
    
    Behaviour:
    
    Given the transformation matrices between two
    MO basis, calculate the overlap between the first determinant
    associated with each basis. That is, calculates <phi 1|phi 2>,
    where |phi i> are Slater determinants and U (see below)
    has the matrices that transforms the orbitals from a basis B1
    (where |phi 1> is the Slater determinant associated to the
    first orbitals) to a basis B2 (where |phi 2> is the Slater
    determinant associated to the first orbitals)
    
    Parameters:
    -----------
    U (list of np.ndarray)
        transformation matrices
    n (list of int)
        number of electrons
    
    Returns:
    --------
    
    The overlap between the determinants (float)
    """
    S = 1.0
    for spirrep, Ui in enumerate(U):
        if n[spirrep] > 0:
            S *= linalg.det(Ui[:n[spirrep], :n[spirrep]])
    return S
