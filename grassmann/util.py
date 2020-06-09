"""Some useful functions and definitions

Variables:
----------
zero
sqrt2
irrep_product
number_of_irreducible_repr

Classes:
--------
logtime

Functions:
----------
dist_from_ovlp
ovlp_Slater_dets
get_I

"""
import math
import datetime
import time
import logging

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

zero = 1.0E-10
sqrt2 = math.sqrt(2.0)

irrep_product = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7],
                            [1, 0, 3, 2, 5, 4, 7, 6],
                            [2, 3, 0, 1, 6, 7, 4, 5],
                            [3, 2, 1, 0, 7, 6, 5, 4],
                            [4, 5, 6, 7, 0, 1, 2, 3],
                            [5, 4, 7, 6, 1, 0, 3, 2],
                            [6, 7, 4, 5, 2, 3, 0, 1],
                            [7, 6, 5, 4, 3, 2, 1, 0]],
                           dtype=np.uint8)

number_of_irreducible_repr = {
    'C1': 1,
    'Cs': 2,
    'C2': 2,
    'Ci': 2,
    'C2v': 4,
    'C2h': 4,
    'D2': 4,
    'D2h': 8}


class logtime():
    """A context manager for logging execution time.
    
    Examples:
    ----------
    with logtime('Executing X'):
        # Add time to log (with level INFO)
        
    with logtime('Executing X', log_level=logging.DEBUG):
        # Add time to log (with level DEBUG)
    
    with logtime('Executing X', out_stream=sys.stdout):
        # Add time to sys.stdout as well
    
    with logtime('Executing X',
                 out_stream=sys.stdout,
                 out_fmt="It took {} to run X"):
        # Use out_fmt to write elapsed time to sys.stdout
    
    with logtime('Executing X') as T_X:
        # Save info in object T_X
    print(T_X.elapsed_time)
    
    with logtime('Executing X') as T_X:
        # Save info in object T_X
    with logtime('Executing X') as T_Y:
        # Save info in object T_Y
    print('Time for X and Y: ',
          datetime.timedelta(seconds=(T_Y.end_time - T_X.ini_time)))
    """
    def __init__(self,
                 action_type,
                 log_level=logging.INFO,
                 out_stream=None,
                 out_fmt=None):
        self.action_type = action_type
        self.log_level = log_level
        self.out_stream = out_stream
        self.end_time = None
        self.elapsed_time = None
        if out_fmt is None:
            self.out_fmt = 'Elapsed time for ' + self.action_type + ': {}\n'
        else:
            self.out_fmt = out_fmt
    
    def __enter__(self):
        self.ini_time = time.time()
        logger.log(self.log_level,
                   '%s ...',
                   self.action_type)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = time.time()
        self.elapsed_time = str(datetime.timedelta(seconds=(self.end_time
                                                            - self.ini_time)))
        logger.info('Total time for %s: %s',
                    self.action_type,
                    self.elapsed_time)
        if self.out_stream is not None:
            self.out_stream.write(self.out_fmt.format(self.elapsed_time))

            
def dist_from_ovlp(x):
    """Convert from overlap to distance.
    
    See I. D'Amico et. al PRL 106 (2011) 050401
    """
    try:
        return sqrt2 * math.sqrt(1 - abs(x))
    except ValueError:
        return 0.0

    
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


def get_I(n, i=None, a=None):
    """Return range(n).remove(i) + [a]"""
    if type(i) != type(a):
        raise ValueError('Both i and a must be of same type!')
    if i is None:
        return list(range(n))
    if isinstance(i, int):
        return [x for x in range(n) if x != i] + [a]
    else:
        return [x for x in range(n) if x not in i] + a


def triangular(n):
    r"""The n-th trianglar number = \sum_i^n i"""
    return ((n + 1) * n) // 2


def get_ij_from_triang(n, with_diag=True):
    """Returns (i,j). Inverse of get_n_from_triang"""
    i = math.floor((math.sqrt(1 + 8 * n) - 1) / 2)
    j = n - i * (i + 1) // 2
    if not with_diag:
        i += 1
    return i, j


def get_n_from_triang(i, j, with_diag=True):
    """Return the position in a triangular arrangement (i>=j):
    
    with_diag=True:
    
    0,0                      0
    1,0  1,1                 1  2
    2,0  2,1  2,2            3  4  5
    3,0  3,1  3,2   3,3      6  7  8  9
    ...  i,j
    
    with_diag=False:
    
    1,0               0
    2,0  2,1          1  2
    3,0  3,1  3,2     3  4  5
    ...  i,j
    
    """
    if with_diag:
        return j + triangular(i)
    else:
        return j + triangular(i - 1)


def get_pos_from_rectangular(i, a, n):
    """Returns i*n + a (position in row-major, C order)
    
    i,a                           pos
    
    0,0   0,1   ...   0,n-1       0    1  ...   n-1
    1,0   1,1   ...   1,n-1       n  n+1  ...   2n-1
    2,0   2,1   ...   2,n-1      2n 2n+1  ...   3n-1
    ....        i,a                     i*n + a
    """
    return i * n + a


def get_ia_from_rectangular(pos, n):
    """Returns (i,a). Inverse of get_pos_from_rectangular"""
    return pos // n, pos % n
