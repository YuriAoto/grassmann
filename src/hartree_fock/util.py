"""General functions for Hartree-Fock

"""
import logging

import numpy as np
from scipy.linalg import solve, svd, lstsq
from . import absil

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

fmt_HF_header = '{0:<5s}  {1:<16s}  {2:<16s}  {3:s}\n'
fmt_HF_iter = '{0:<5d}  {1:<16.12f}  {2:<16.12f}  {3:s}\n'

write_header = \
    'it. \t E \t\t\t |Gradient| \t\t |Restr.| \t step \t\t time in iteration\n'
#    12345678901234567890123456789012345678901234567890123456789012345678901234567890123456

fmt_HF_iter_general = '{0:<5d} \t {1:<16.12f} \t {2:<16.12f} \t          \t {4:6s} \t {5:s}\n'
fmt_HF_iter_gen_lag = '{0:<5d} \t {1:<16.12f} \t {2:<16.12f} \t {3:<.4f} \t {4:6s} \t {5:s}\n'

np.set_printoptions(precision=8,linewidth=10060,suppress=True)


class Diis:
    """A class to compute DIIS.
    
    This has been strongly based on the diis furnished by G. Knizia
    along with his IR-WMME code. Reading his code was very helpful to
    understand DIIS.
    
    Usage:
    ------
    This class should be initiated with the maximum dimension of the
    DIIS subspace. Example:
    
    diis = Diis(5)
    
    To apply the diis, do:
    
    new_p = diis.calculate(e, p)
    
    or:
    
    new_p1, new_p2 = diis.calculate(e, p1, p2)
    
    e are the errors (gradients, residuals), and p, or p1 p2, etc,
    are the parameters where diis will be applied to
    (diis can be simultaneously applied to more parameters,
    although I actually don't know if this is useful ....)
    The object diis will store the required information (namely,
    errors e and parameters p (or p1, p2) from previous iterations.
    
    """
    
    def __init__(self, n_max):
        self.n_max = n_max
        self.parameters_shapes = None
        self.e = None
        self.all_parameters = None
        self.i = 0
        self.n = 1

    def init_matrices(self, e, *parameters):
        self.e = np.empty((self.n_max, np.prod(e.shape)))
        self.all_parameters = [np.empty((self.n_max, np.prod(p.shape))) for p in parameters]

    def add_to_matrices(self, e, *parameters):
        if self.e is None:
            self.init_matrices(e, *parameters)
        self.e[self.i, :] = e.flatten()
        for i, p in enumerate(parameters):
            self.all_parameters[i][self.i, :] = p.flatten()
        logger.debug('DIIS stored e:\n%s', self.e)

    def set_system(self):
        """Create the DIIS linear system
        
        Parameters:
        -----------
        e (np.array)
            Errors of current and n previous steps
        
        """
        B = np.zeros((self.n + 1, self.n + 1))
        B[:self.n, :self.n] = np.einsum('pi,qi->pq',
                                        self.e[:self.n, :],
                                        self.e[:self.n, :])
        fScale = sum([np.log(B[i,i].real) for i in range(self.n)])
        fScale = np.exp(fScale/self.n)
        B[:self.n, self.n] = B[self.n, :self.n] = -1.0
        B[:self.n, :self.n] /= fScale
        a = np.zeros(self.n + 1)
        a[self.n] = -1.0
        return B, a

    def solve(self, B, a):
        """Solve Bw = a, the linear system of DIIS"""
        #  return w = solve(B, a)
        w = lstsq(B, a)
        return w[0]

    def update(self, w):
        """Update the parameters and gradient
        
        Parameters:
        -----------
        w (np.ndarray)
            The solution of the diis system
        """
        self.e[self.i, :] = np.einsum('ji,j->i', self.e[:self.n, :], w[:self.n])
        for p in self.all_parameters:
            p[self.i, :] = np.einsum('ji,j->i', p[:self.n, :], w[:self.n])
    
    def increase(self):
        """Advance self.i, and increase self.n if possible"""
        self.i += 1
        if self.n < self.n_max:
            self.n += 1
        if self.i >= self.n:
            self.i = 0

    def calculate(self, e, *parameters, **kargs):
        """Calculate a DIIS step
        
        Parameters:
        -----------
        e (np.ndarray)
            The errors
        
        *args (np.ndarray)
            The parameters where the DIIS will work on
        
        Returns:
        --------
        
        Does not return anything, but p and g are updated.
        
        TODO:
        -----
        
        This does not give the same result for equivalent
        restricted and unrestricted calculations.
        Investigate why.
        
        This implementation is rather simple, and is not sufficient
        to achieve convergence for benzene. Improve this.
        
        """
        increase_i = kargs['increase_i'] if 'increase_i' in kargs else True
        if self.n_max > 0: self.add_to_matrices(e, *parameters)
        if self.n > 1:
            B, a = self.set_system()
            w = self.solve(B, a)
            self.update(w)
            logger.debug('DIIS:\n\n'
                         'current i=%i; dimension n=%i\n'
                         'B matrix:\n%s\n'
                         'a vector:\n%s\n'
                         'solution w of Bw=a:\n%s\n',
                         self.i, self.n, B, a, w)
            to_return = []
            for i, p in enumerate(parameters):
                p_ = self.all_parameters[i]
                logger.debug('updated by DIIS:\n%s', p_[self.i, :])
                to_return.append(np.copy(p_[self.i, :].reshape(p.shape)))
        else:
            to_return = parameters
        if len(to_return) == 1: to_return = to_return[0]
        if increase_i: self.increase()
        return to_return

    

def geodesic(C, eta, S, Sqrt, invSqrt, t=1):
    """Computes the geodesic of the Grassmannian
    
    
    Parameters:
    -----------
    C (ndarray, of shape (n,N))
        The orbital coefficients (that is, an element at the Stiefel)
    
    eta (ndarray, of shape (n,N))
        The direction at the horizontal space:
        
        C.T @ S @ eta = 0
        
    S (ndarray, shape (n,n))
        The overlap matrix of the basis set    
    
    Sqrt (ndarray, shape (n,n))
        The square root of the overlap matrix S. It is the X matrix of Szabo
    
    invSqrt (ndarray, shape (n,n))
        The inverse of Sqrt
    
    t (float, optional, default=1)
        The step to calculate the geodesic

    
    Returns:
    --------
    A np.array of shape (n, N), with the orbital coefficients

    """
    u, s, v = svd(invSqrt @ eta, full_matrices=False)
    sin, cos = np.diag(np.sin(t*s)), np.diag(np.cos(t*s))
    temp = (C @ v.T @ cos + Sqrt @ u @ sin) @ v
    return absil.gram_schmidt(temp, S)
