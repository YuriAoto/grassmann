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
from datetime import timedelta
import logging

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

zero = 1.0E-10
sqrt2 = math.sqrt(2.0)

ANG_to_a0 = 1.8897261246

int_dtype = np.intc

irrep_product = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7],
                            [1, 0, 3, 2, 5, 4, 7, 6],
                            [2, 3, 0, 1, 6, 7, 4, 5],
                            [3, 2, 1, 0, 7, 6, 5, 4],
                            [4, 5, 6, 7, 0, 1, 2, 3],
                            [5, 4, 7, 6, 1, 0, 3, 2],
                            [6, 7, 4, 5, 2, 3, 0, 1],
                            [7, 6, 5, 4, 3, 2, 1, 0]],
                           dtype=int_dtype)

number_of_irreducible_repr = {
    'C1': 1,
    'Cs': 2,
    'C2': 2,
    'Ci': 2,
    'C2v': 4,
    'C2h': 4,
    'D2': 4,
    'D2h': 8}

ATOMS = ['X',
         'H' ,                                                                                'He',
         'Li','Be',                                                  'B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
         'Na','Mg',                                                  'Al','Si','P' ,'S' ,'Cl','Ar',
         'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
         'Rb','Sr','Y' ,'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I' ,'Xe',
         'Cs','Ba',
'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
                        'Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rd',
         'Fr','Ra',
'Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
                        'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']


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
    
    def relative_to(self, other):
        """Return the time from other.ini_time to self.end_time"""
        return str(timedelta(seconds=(self.end_time - other.ini_time)))


class LogFilter():
    """Define a filter for logging, to be attached to all handlers."""
    def __init__(self, logfilter_re):
        """Initialises the class
        
        Parameters:
        -----------
        logfilter_re (str, with a regular expression)
            Only functions that satisfy this RE will be logged
        """
        self.logfilter_re = logfilter_re
    
    def filter(self, rec):
        """Return a boolean, indicating whether rec will be logged or not."""
        if rec.funcName == '__enter__':
            rec.funcName = 'Entering time management'
        elif rec.funcName == '__exit__':
            rec.funcName = 'Finishing time management'
        if self.logfilter_re is not None:
            return self.logfilter_re.search(rec.funcName) is not None
        # Uncomment this to check for possible records to filter
        # print()
        # print(rec)
        # print(rec.__dict__)
        # print()
        return True


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
        return sqrt2 * math.sqrt(1 - absovlp)
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


class Results:
    """Class to store results
    
    Basic class for results of calculations.
    
    Attributes:
    -----------
    kind (str)
        Kind of result
    
    success (bool)
        Indicate success of calculation.
        If a optimisation procedure, should be True if converged
    
    error (str)
        Eventual error messages. Should be None if success == True.
        If success == False, this attribute should indicate what happened
        for the lack of success
    
    warning (str)
        Eventual aarning
    
    """
    
    def __init__(self, kind):
        self.kind = kind
        self.success = None
        self.error = None
        self.warning = None
    
    def __str__(self):
        x = ['Results for ' + self.kind + ':']
        x.append('Success = ' + str(self.success))
        if self.error is not None:
            x.append('error = ' + str(self.error))
        if self.warning is not None:
            x.append('warning = ' + str(self.error))
        return '\n'.join(x)


class OptResults(Results):
    """ Class to store results of a optimisations
    
    Every attribute that is not relevant for this kind
    of calculation should be None.
    
    These are main attributes.
    Add extra attributes to the instances if needed.
    
    Attributes:
    -----------
    energy (float)
        Final energy. Should be None if it is not a energy calculation
    
    distance (float)
        Final distance. Should be None if it is not a energy calculation
    
    wave_function (WaveFunction)
        The final wave function
    
    orbitals (MolecularOrbitals)
        Final molecular orbitals
    
    n_iter (int)
        Number of iterations to reach convergence
    
    conv_norm (float or tuple of floats)
        The norm of vector or vectors that should vanish at convergence
    
    """
    
    def __init__(self, kind):
        super().__init__(kind)
        self.energy = None
        self.distance = None
        self.wave_function = None
        self.orbitals = None
        self.n_iter = None
        self.conv_norm = None
    
    def __str__(self):
        x = [super().__str__()]
        x.append('==========================')
        if self.energy is not None:
            x.append('Energy = ' + str(self.energy))
        if self.distance is not None:
            x.append('Distance = ' + str(self.distance))
        return '\n'.join(x)
