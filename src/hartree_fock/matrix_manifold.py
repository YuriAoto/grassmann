"""Functions to deal with a matrix manifold


General notation:
-----------------
Y    a point in the matrix manifold, i.e, a n x p matrix
n    the dimension of the base space (number of rows of Y)
p    the number of columns of Y

Y.shape = n,p

"""
import logging

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

from orbitals import orbitals
from util import array_indices
from util.variables import sqrt2


class MatrixManifold:

    def __init__(self, shape, S=None, invS=None, X=None, invX=None):
        self.shape = shape
        self.S = S
        self.invS = invS
        self.X = X
        self.invX = invX
        self._make_attributes()

    @property
    def n(self):
        return self.shape[0]

    @property
    def p(self):
        return self.shape[1]

    @property
    def d(self):
        return self.shape[0]

    @property
    def N(self):
        return self.shape[1]

    def _make_attributes(self):
        assert self.S is None or self.S.shape == (self.n, self.n)
        self.Id_n = np.eye(self.n)
        self.Id_p = np.eye(self.p)
        self.mat_dim = self.n * self.p
        self.dim = self.n * self.p

    def assert_shape(self, x):
        """Raises ValueError if x.shape != self.shape"""
        if x.shape != self.shape:
            raise ValueError('Matrix has incorrect shape for this Stiefel manifold')

    def rip(self, v1, v2):
        """Calculates the Riemannian internal product in the Stiefel.

        Parameters:
        -----------
        Y (2D arrays of shape n,p) - the point of the Stiefel manifold
        v1 and v2 (2D arrays of shape n,p) - the tangent vectors

        """
        self.assert_shape(v1)
        self.assert_shape(v2)
        return np.trace(v1.T @ v2
                        if self.S is None else
                        v1.T @ self.S @ v2)

    def matvecmul_in_np_space(self, A, b):
        """Make A@b where A is np x np and b is a (n,p) array
        
        This makes the multiplication where A is in the vectorized
        form, b is unvectorizes (thus, b is a nxp matrix) and the result
        is given in the unvectorized form.
        Vectorization means stacking column on top of column
        
        Parameters:
        -----------
        A (2D array of shape (self.mat_dim, self.mat_dim))
        b (2D array of shape self.shape)
        
        Result:
        -------
        A 2D array of shape self.shape

        """
        return np.reshape(A @ np.reshape(b, (self.mat_dim,), 'F'),
                          self.shape, 'F')

