"""Functions to deal with the Stiefel manifold

References:
Edelman, Absil book


General notation:
-----------------
Y    a point in the Stiefel, i.e, a n x p orthogonal matrix
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



def sym(X):
    return (X + X.T)/2


class StiefelManifold:

    def __init__(self, shape, S=None):
        self.shape = shape
        self.S = S
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
        assert self.S.shape == (self.n, self.n)
        self.Id_n = np.eye(self.n)
        self.Id_p = np.eye(self.p)
        self.mat_dim = self.n * self.p
        self.dim = self.mat_dim - self.p * (self.p + 1) // 2
        self.invS = np.linalg.inv(self.S)

    def assert_shape(self, x):
        """Raises ValueError if x.shape != self.shape"""
        if x.shape != self.shape:
            raise ValueError('Matrix has incorrect shape for this Stiefel manifold')
        

    def can_rip(self, Y, v1, v2):
        """Calculates the canonical Riemannian internal product of Stiefel

        Equation  (2.39) of Edelman 

        Parameters:
        -----------
        Y (2D arrays of shape n,p) - the point of the Stiefel manifold
        v1 and v2 (2D arrays of shape n,p) - the tangent vectors
        S (2D array of dimension n,n) - the Gram matrix

        """
        if self.S is not None: raise ValueError("S != None is not implemented.")
        return np.trace(v1.T @ (self.Id_n - Y @ Y.T / 2) @ v2)


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

    def _matvecmul_in_np_space(self, A, b):
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

    def proj_tgspace(self, Y, Z):
        """Calculates the projection of Z into the tangent space of the Stiefel

        Similar to the modified vertion of Equation (2.4) of Edelman, but with Gram matrix.

        Parameters:
        -----------
        Y (2D array of shape d,N) - the point of the Stiefel manifold
        Z (2D array of shape d,N) - the matrix to be projected into the tangent space
        """
        YtZ = (Y.T @ Z
               if self.S is None else
               Y.T @ self.S @ Z)
        return Z - Y @ sym(YtZ)


    def check_in_tgspace(self, Y, v, tol=1.0E-8):
        """Check if v is in the tangent space of Stiefel at Y"""
        Ytv = (Y.T @ v
               if self.S is None else
               Y.T @ self.S @ v)
        is_allclose = np.allclose(Ytv, -Ytv.T, atol=tol)
        logger.info('T(St): ||Y.T @ v + (Y.T @ v).T|| = %s  (Is v in T_Y St: %s)',
                    linalg.norm(Ytv + Ytv.T),
                    is_allclose)
        return is_allclose


    def tg_vecs(self, Y, mode='2.5',
                in_Gr_tgspace=True,
                in_Gr_tgspace_orth=True):
    ####            normalize=True):   implement ??
        """Yield tangent vectors according to Eq. (2.5) or (2.6) of Edelman

        Note that with mode='2.5' this generator yields a basis for the tangent
        space. With mode='2.6' this generator yields a linearly dependent set
        that span the tangent space

        Parameters:
        -----------

        """
        if in_Gr_tgspace_orth:
            for j in range(self.p):
                for i in range(j):
                    A = np.zeros((self.p, self.p))
                    A[i,j] = 1
                    A[j,i] = -1
                    yield f'A_{i}_{j}', Y @ A / sqrt2
        if in_Gr_tgspace:
            if mode == '2.5':
                Y_perp = orbitals.complete_orb_space([Y], [self.n], S=self.S)[0][:,self.p:]
                for _mu in range(self.n-self.p):
                    for i in range(self.p):
                        B = np.zeros((self.n-self.p, self.p))
                        B[_mu, i] = 1
                        yield f'B_{_mu}_{i}', Y_perp @ B
            elif mode == '2.6':
                I_YYt = self.Id_n - (Y @ Y.T
                                     if self.S is None else
                                     Y @ Y.T @ self.S)
                for _mu in range(self.n):
                    for i in range(self.p):
                        C = np.zeros(self.shape)
                        C[_mu, i] = 1
                        yield f'C_{_mu}_{i}', I_YYt @ C
            else:
                raise ValueError('Unknown mode to generate tangent vectors')

    def ith_tg_vecs(self, Y, i, mode='2.5'):
        """Return the i-th tangent vector that is yield by tg_vecs"""
        if i < array_indices.triangular(self.p-1):
            ii, j = array_indices.ij_from_triang(i)
            A = np.zeros((self.p, self.p))
            A[ii,j] = 1
            A[j,ii] = -1
            return Y @ A / sqrt2
        i -= array_indices.triangular(self.p-1)
        if mode == '2.5':
            if i >= (self.n-self.p) * self.p:
                raise ValueError('Too large index for tan vector in mode 2.5')
            Y_perp = orbitals.complete_orb_space([Y], [self.n], S=self.S)[0][:,self.p:]
            _mu, ii = array_indices.ij_from_rect(i, self.p)
            B = np.zeros((self.n-self.p, self.p))
            B[_mu, ii] = 1
            return Y_perp @ B
        elif mode == '2.6':
            if i >= self.n * self.p:
                raise ValueError('Too large index for tan vector in mode 2.6')
            I_YYt = self.Id_n - (Y @ Y.T
                                 if self.S is None else
                                 Y @ Y.T @ self.S)
            _mu, ii = array_indices.ij_from_rect(i, self.p)
            C = np.zeros(self.shape)
            C[_mu, ii] = 1
            return I_YYt @ C
        else:
            raise ValueError('Unknown mode to generate tangent vectors')


    def basis_for_tg_space(self,
                           Y,
                           include_Gr_tgspace=True,
                           include_Gr_tgspace_orth=True):
        basis = []
        for name_v, v in self.tg_vecs(Y,
                                      mode='2.5',
                                      in_Gr_tgspace=include_Gr_tgspace,
                                      in_Gr_tgspace_orth=include_Gr_tgspace_orth):
            basis.append(v)
        return basis

    def tgvec_matrix_to_basis(self, v, basis=None, Y=None):
        """Expand the n x p matrix v in the basis of tangent space at Y
        
        Parameters:
        -----------
        - v (nxN ndarray): the vector in the tangent space of Stiefel.
        - basis (list of nxN ndarray, optional, default=None):
                 a basis calculated by basis_for_tg_space;
                 If None, this funcion is called
        - Y (nxN ndarray): orbitals' coefficients (aka an element in the Stiefel).
        
        Return:
        -------
        a 1D array of len self.dim, that represents v in the basis
        
        """
        if basis is None and Y is None:
            raise ValueError('You must pass basis or Y')
        if basis is None: basis = self.basis_for_tg_space(Y)
        x = np.empty(self.dim)
        for i in range(self.dim):
            x[i] = self.rip(basis[i], v)
        return x


    def tgvec_basis_to_matrix(self, x, basis=None, Y=None):
        """Write the tangent vector x as a n x p matrix. Inverse of tgvec_matrix_to_basis
        
        Parameters:
        -----------
        - x (1D ndarray of len self.dim): the vector in the tangent space of Stiefel
                                          as the coefficients vector in the basis
        - basis (list of nxN ndarray, optional, default=None):
                 a basis calculated by basis_for_tg_space;
                 If None, this funcion is called
        - Y (n x N ndarray, optional, default=None): the element in the Stiefel
        
        Return:
        -------
        a 1D array of len self.dim, that represents v in the basis
        
        """
        if basis is None and Y is None:
            raise ValueError('You must pass basis or Y')
        if basis is None: basis = self.basis_for_tg_space(Y)
        v = np.zeros(self.shape)
        for i in range(self.dim):
            v += x[i] * basis[i]
        return v

    

    def geodesic(self, Y, v, t=1):
        """Computes the geodesic on the Stiefel


        Parameters:
        -----------
        - Y (nxN ndarray): orbitals' coefficients (aka an element in the Stiefel).
        - v (nxN ndarray): the vector in the tangent space of Stiefel.
          It must satisfy
              C.T @ S @ v + (C.T @ S @ v).T = 0.
        - S (nxn ndarray): the overlap matrix of the basis set.
        - t (float, optional, default=1): the size of the step of the geodesic.

        Returns:
        --------
        A nxN ndarray with the new orbitals' coefficients.
        """
        YtSv = (Y.T @ v
                if self.S is None else
                Y.T @ self.S @ v)
        p = self.p
        exp_central = np.empty((2*p, 2*p))
        exp_central[:p, :p] = YtSv
        exp_central[p:, p:] = YtSv
        exp_central[:p, p:] = -v.T @ self.S @ v
        exp_central[p:, :p] = self.Id_p
        exp_central = linalg.expm(t * exp_central)
        return (Y @ exp_central[:p, :p] + v @ exp_central[p:, :p]) @ linalg.expm(- t * YtSv)


    def Hess(self, Y, v, euc_grad, euc_Hess):
        """Calculates the Hessian acting on a tangent vector
        
        Parameters:
        -----------
        Y (2D array, shape self.shape) - point at the Stiefel
        v (2D array, shape self.shape) - the tangent vector
        euc_grad (2D array, shape self.shape) - Euclidean gradient
        euc_Hess (2D array, shape (self.mat_dim, self.mat_dim)) - Euclidean Hessian
        
        Returns:
        --------
        A 2D array of shape self.shape - Hess f(Y)[v]
        
        """
        Hv = self.invS @ self._matvecmul_in_np_space(euc_Hess, v)
        Hv -= v @ sym(Y.T @ euc_grad)
        return self.proj_tgspace(Y, Hv)


    def make_Hess(self, Y, euc_grad, euc_Hess):
        """make the Hessian (on the space of Mat nxd) at Stiefel

        Parameters:
        -----------
        Y (2D array) - point at the Stiefel
        euc_grad (2D array) - Euclidean gradient
        euc_Hess (2D array) - Euclidean Hessian

        """
        Riem_Hess = np.kron(self.Id_p, self.invS) @ euc_Hess
        Riem_Hess -= np.kron(sym(euc_grad.T @ Y), self.Id_n)
        proj = np.kron(self.Id_p, (2*self.Id_n - Y @ Y.T @ self.S))
        Yt_kron_Y = np.kron(Y.T @ self.S, Y)
        # loops below make perm(Yt_kron_Y)
        for i in range(self.p):
            for q in range(self.n):
                proj[:, i*self.n + q] -= Yt_kron_Y[:, q*self.p + i]
        return proj @ Riem_Hess / 2


    def make_Hess_v2(self, Y, euc_grad, euc_Hess):
        """make the Hessian (on the space of Mat nxd) at Stiefel

        Using the fact that
          self.Euc_grad.T @ Y = Y.T @ self.Euc_grad.T
        that is, matrix above is symmetric

        Parameters:
        -----------
        Y (2D array) - point at the Stiefel
        euc_grad (2D array) - Euclidean gradient
        euc_Hess (2D array) - Euclidean Hessian

        """
        Riem_Hess = np.kron(self.Id_p, self.invS) @ euc_Hess
        Riem_Hess -= np.kron(sym(euc_grad.T @ Y), self.Id_n)
        projN = np.kron(self.Id_p, Y @ Y.T @ self.S)
        Yt_kron_Y = np.kron(Y.T @ self.S, Y)
        # loops below make perm(Yt_kron_Y)
        for i in range(self.p):
            for q in range(self.n):
                projN[:, i*self.n + q] -= Yt_kron_Y[:, q*self.p + i]
        Riem_Hess = projN @ Riem_Hess / 2
        proj = np.kron(self.Id_p, (self.Id_n - Y @ Y.T @ self.S))
        Riem_Hess += proj @ (np.kron(self.Id_p, self.invS) @ euc_Hess
                             - np.kron(sym(euc_grad.T @ Y), self.Id_n))
        return Riem_Hess


    def true_Hessian_from_MatHess(self, Y, mat_Hess, basis=None):
        """Construct the matrix representation of Hessian in the basis for T_C St

        Parameters:
        -----------
        Y (2D array) : the element of the Stiefel manifold
        mat_Hess (2D array): the Hessian constructed on Mat(n x d), as coming from
                             make_Hess
        basis (list of nxN ndarray, optional, default=None):
                 a basis calculated by basis_for_tg_space;
                 If None, this funcion is called

        Return:
        -------
        A 2D matrix of dimension dim_stiefel x dim_stiefel
        that represents the Hessian in the basis generated by tg_vecs
        with mode 2.5

        """
        if basis is None: basis = self.basis_for_tg_space(Y)
        true_Hess = np.empty((self.dim, self.dim))
        for i in range(self.dim):
            H_basis_i = self._matvecmul_in_np_space(mat_Hess, basis[i])
            is_in_tgspace = self.check_in_tgspace(Y, H_basis_i)
            if not is_in_tgspace:
                logger.critical('H @ basis[i] is not in tg space for basis element number %s', i)
            true_Hess[:,i] = self.tgvec_matrix_to_basis(H_basis_i, basis=basis, Y=Y)
        return true_Hess


    def true_Hessian_directly(self, Y, euc_grad, euc_Hess, basis=None):
        """Construct the matrix representation of Hessian in the basis for T_C St

        Parameters:
        -----------
        Y (2D array) : the element of the Stiefel manifold
        euc_grad (2D array, shape self.shape) - Euclidean gradient
        euc_Hess (2D array, shape (self.mat_dim, self.mat_dim)) - Euclidean Hessian
        - basis (list of nxN ndarray, optional, default=None):
                 a basis calculated by basis_for_tg_space;
                 If None, this funcion is called

        Return:
        -------
        A 2D matrix of dimension dim_stiefel x dim_stiefel
        that represents the Hessian in the basis generated by tg_vecs
        with mode 2.5

        """
        if basis is None: basis = self.basis_for_tg_space(Y)
        true_Hess = np.empty((self.dim, self.dim))
        for i in range(self.dim):
            H_basis_i = self.Hess(Y, basis[i], euc_grad, euc_Hess)
            true_Hess[:,i] = self.tgvec_matrix_to_basis(H_basis_i, basis=basis, Y=Y)
        return true_Hess


    def make_augmentation(self, Y):
        """Create the augmentation to impose a solution in tg space of Stiefel

        Parameters:
        -----------
        Y (2D array) : the element of the Stiefel manifold

        Return:
        -------
        a N² x Nd matrix A such that

        A @ vec(v) = 0

        if v belongs to the tangent Space of the Stiefel at Y

        """
        YtS = (Y.T
               if self.S is None else
               Y.T @ self.S)
        augmentation = np.kron(self.Id_p, YtS)
        # loops below make perm(Y.T @ S \otimes Id)
        for i in range(self.p):
            s_line = np.s_[i * self.n: (i+1) * self.n]
            for j in range(self.p):
                augmentation[j * self.p + i, s_line] += YtS[j, :]
        return augmentation







def test_HessFDX_with_eta(Y, FYY, eta, F, S=None):
    """This is a test to check if equation below is zero

    F_YY(D, X) + tr(Y.T F D Y.T X)/2

    where D=eta from the prblem in the grassmannian,
    X = YN (N symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix


    """
    logger.debug('Testing FYY_D_X + tr_YtFDX = F_YY(D, X) + tr(Y.T F D Y.T X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False, S=S):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        FYY_D_X = np.trace(eta.T @ FYY_X
                           if S is None else
                           eta.T @ S @ FYY_X)
        tr_YtFDX = np.trace(Y.T @ F @ eta @ YtX)/2
        logger.debug('For X=%s\n'
                     'FYY_D_X = %s\n'
                     'tr_YtFDX = %s\n'
                     'FYY_D_X + tr_YtFDX = %s',
                     nameX,
                     FYY_D_X,
                     tr_YtFDX,
                     FYY_D_X + tr_YtFDX)


def test_HessFDX(Y, FYY, FY, S=None):
    """This is a test to check if equation below is zero

    F_YY(D, X) + tr(Y.T F D Y.T X)/2

    where D=is in the tangent space of Gr from the prblem in the grassmannian,
    X = YN (N skew symmetric) is in its orthogonal complement (within
    the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    F (2D array of shape n,n) - the Fock matrix


    """
    logger.debug('Testing FYY_D_X + tr_YtFDX = F_YY(D, X) + tr(Y.T F D Y.T X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y):  #, in_Gr_tgspace=False):
        YtX = Y.T @ X if S is None else Y.T @ S @ X
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        for nameD, D in tg_vecs(Y):  #, in_Gr_tgspace_orth=False):
            FYY_D_X = np.trace(D.T @ FYY_X)
            tr_YtFDX = np.trace(FY.T @ D @ YtX)
            logger.debug('For X=%s and D=%s: FYY_D_X + tr_YtFDX = %s\n'
                         'FYY_D_X = %s\n'
                         'tr_YtFDX = %s\n',
                         nameX, nameD, FYY_D_X + tr_YtFDX,
                         FYY_D_X,
                         tr_YtFDX
                         )


def test_FYY_0(Y, FYY, S=None):
    """Test to check if equation below is zero

    F_YY(D, X)

    where D is in the tangent space of Gr,
    X is in its orthogonal complement (within the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    FY (2D array of shape n,n) - the euclidean gradient


    """
    logger.debug('Testing FYY_D_X = F_YY(D, X) = 0')
    n, p = Y.shape
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False, S=S):
        FYY_X = FYY @ np.reshape(X, n*p, order='F')
        FYY_X = np.reshape(FYY_X, (n, p), order='F')
        for nameD, D in tg_vecs(Y, in_Gr_tgspace_orth=False, S=S):
            FYY_D_X = np.trace(D.T @ FYY_X
                               if S is None else
                               D.T @ S @ FYY_X)
            logger.debug('For X=%s and D=%s: FYY_D_X = %s\n',
                         nameX, nameD, FYY_D_X
                         )



def log_eta_FYY_X(Y, FYY, eta, S=None):
    """This logs F_YY(eta, X) = eta.T @ S @ F_YY @ X

    for all X in the orthogonal complement of T_Y Gr
    (within the tangent space of Stiefel)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    eta (2D array of shape n,n) - the eta, a vector in T_Y St
    S (2D array of shape n,n) - the overlap matrix


    """
    logger.debug('Calculating X_FYY_eta = F_YY(X, eta)')
    n, p = Y.shape
    FYY_eta = FYY @ np.reshape(eta, n*p, order='F')
    FYY_eta = np.reshape(FYY_eta, (n, p), order='F')
    for nameX, X in tg_vecs(Y, in_Gr_tgspace=False, S=S):
        X_FYY_eta = np.trace(X.T @ FYY_eta
                             if S is None else
                             X.T @ S @ FYY_eta)
        logger.debug('For X=%s: X_FYY_eta = %s\n', nameX, X_FYY_eta)



def log_eta_Fyy_Dn__tr_DnYFy(Y, Fy, riemGrad, Fyy, eta, S=None, restricted=None):
    """This logs F_YY(D, eta) - tr(D.T eta X.T Fy)

    for all D in T_St
    (within the tangent space of Stiefel)

    In the nonorthogonal basis:

    D.T @ S @ Fyy[eta] - tr(D.T S n X.T Fy)

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FY (2D array of shape n,p) - the euclidean Gradient
    Fyy (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    eta (2D array of shape n,n) - the eta, a vector in T_Y St
    S (2D array of shape n,n) - the overlap matrix


    """
    logger.debug('In log_eta_Fyy_Dn__tr_DnYFy')
    n, p = Y.shape
    Fyy_eta = Fyy @ np.reshape(eta, n*p, order='F')
    Fyy_eta__eta_Y_Fy = np.reshape(Fyy_eta, (n, p), order='F')
##    Fyy_eta__eta_Y_Fy -= eta @ Y.T @ Fy
    for nameD, D in tg_vecs(Y, S=S):
        D_Fyy_eta__eta_Y_Fy = np.trace(D.T @ Fyy_eta__eta_Y_Fy
                                       if S is None else
                                       D.T @ S @ Fyy_eta__eta_Y_Fy)
        logger.debug('For D=%s: D_Fyy_eta = %s\n'
                     '          D_grad      = %s\n',
                     nameD,
                     D_Fyy_eta__eta_Y_Fy,
                     np.trace(D.T @ S @ riemGrad))



def test_FYY(Y, FYY, S=None):
    """This is a test funtion to see the solution of Equation (2.60) of Edelman

    Parameters:
    -----------
    Y (2D array of shape n,p) - the point of the Stiefel manifold
    FYY (2D array of shape np,np) - the euclidean Hessian, Equation (2.56) of Edelman
    """
    if S is not None: raise ValueError("S != None is not implemented.")
    logger.debug('Testing finding unique F_YY(D) in tangent space')
    n, p = Y.shape
    for nameD, D in tg_vecs(Y):
#        logger.debug('checking F_YY: D = %s:\n%s', nameD, D)
        logger.debug('checking F_YY: D = %s is in tg space: %s',
                     nameD, check_in_tgspace(Y, D))
        FYYD = FYY @ np.reshape(D, n*p, order='F')
        FYYD = np.reshape(FYYD, (n, p), order='F')
        FYYDproj = proj_tgspace(Y, FYYD, times2=True)
        logger.debug('checking F_YY: FYYDproj is in tg space: %s',
                     check_in_tgspace(Y, FYYDproj))
        for nameX, X in tg_vecs(Y):
            prod1 = np.trace(FYYD.T @ X)
            prod2 = metric(Y, FYYDproj, X)
            logger.debug('  F_YY(D, %s) = %s\n'
                         '<F_YY(D), %s> = %s\n'
                         'Are they equal??  %s',
                         nameX, prod1, nameX, prod2,
                         abs(prod1-prod2)<1.0E-8)
