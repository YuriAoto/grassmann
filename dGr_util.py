""" Some useful functions and definitions

"""

import math
import datetime
import time
import logging

from scipy import linalg

from dGr_exceptions import *

logger = logging.getLogger(__name__)

sqrt2 = math.sqrt(2.0)


class logtime():
    """A context manager for logging time."""
    
    def __init__(self, action_type, out_stream=None, out_fmt=None):
        self.action_type = action_type
        self.out_stream = out_stream
        self.end_time = None
        self.elapsed_time = None
        if out_fmt is None:
            self.out_fmt = 'Elapsed time for ' + self.action_type + ': {}\n'
        else:
            self.out_fmt = out_fmt
    
    def __enter__(self):
        self.ini_time = time.time()
        logger.info(self.action_type + ' ...')
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = time.time()
        self.elapsed_time = str(datetime.timedelta(seconds=(self.end_time - self.ini_time)))
        logger.info('Total time for {}: {}'.\
                    format(self.action_type, self.elapsed_time))
        if self.out_stream is not None:
            self.out_stream.write(self.out_fmt.format(self.elapsed_time))

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

def get_I(n, i=None, a=None):
    """return range(n).remove(i) + [a]"""
    if type(i) != type(a):
        raise dGrValueError('Both i and a must be of same type!')
    if i is None:
        return list(range(n))
    if isinstance(i, int):
        return [x for x in range(n) if x != i] + [a]
    else:
        return [x for x in range(n) if x not in i] + a
