"""A module with helper functions for Riemannian Hartree--Fock.

    Common parameters:
    ------------------
    C (2D np.ndarray of dimension (d, N_a))
        In the restricted case it's just the Slater determinant while in the
        unrestricted case it's the spin alpha part of the Slater determinant.
        d is the size of the basis set and N_a is the number of spin alpha
        occupied orbitals in the Slater determinant.

    C_b (2D np.ndarray of dimension (d, N_b))
        The spin beta part of the Slater determinant. d is the size of the basis
        set and N_b is the number of spin beta occupied orbitals in the Slater
        determinant.

    P (2D np.ndarray of dimension (d, d))
        Density matrix: 2 * C @ C.T in the restricted case and C @ C.T in the
        unrestricted.

    P_b (2D np.ndarray of dimension (d, d))
        Density matrix for spin beta: C_b @ C_b.T

    proj (2D np.ndarray of dimension (d, N_a))
        Projection matrix to the tangent space of the Grassmannian:
        Id - C @ C.T @ S
    
    proj_b (2D np.ndarray of dimension (d, N_b))
        Projection matrix to the tangent space of the Grassmannian for spin
        beta: Id - C_b @ C_b.T @ S

    C (2D np.ndarray of dimension (n, N_b))
        C_b or C_a, but unspecified in that function

    P_a (2D np.ndarray of dimension (n, n))
        Density matrix for spin alpha: C_a @ C_a.T

    P_b (2D np.ndarray of dimension (n, n))
        Density matrix for spin beta: C_b @ C_b.T

    P (2D np.ndarray of dimension (n, n))
        P_a or P_b, but unspecified in that function

    grad (2D np.ndarray of dimension (n, N_a))
        The gradient of the energy (spin alpha in the unrestricted case).

    grad_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the gradient of the energy.

    h (2D np.ndarray)
        One-electron integral matrix.

    g (3D np.ndarray)
        Two-electron integral matrix.
        Here all functions assume the three indices g,
        that need a further contraction in the first index
        to obtain the actualtwo electron integral that
        depend on four indices

    S (2D np.ndarray)
        Overlap matrix.
"""
from scipy import linalg
import numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def common_blocks(double[:, :] C, double[:, :] P, double[:, :, :] g):
    """Calculate some useful contractions among C, P and g
    
    Parameters:
    -----------
    C, P and g: see module's docstring
    
    Returns:
    --------
    A tuple with the following matrices:
    
    GP (1D array) - the contraction g[:,i,k]P[i,k]
    GC (3D array) - the contraction g[:,:,k]C[k,:]
    GCC (3D array) - the contraction CG[L,i,k]C[i,j] = g[L,i,k']C[k',k]C[i,j]
    """
    cdef int i, j, k, L, T = g.shape[0], n = C.shape[0], N = C.shape[1]
    cdef double[:] GP = np.zeros(T)
    cdef double[:, :, :] GC = np.zeros((T, n, N))
    cdef double[:, :, :] GCC = np.zeros((T, N, N))
    cdef double aux

    for L in range(T):
        for i in range(n):
            for k in range(n):
                aux = g[L, i, k]
                GP[L] += aux * P[i, k]

                for j in range(N):
                    GC[L, i, j] += aux * C[k, j]

            for j in range(N):
                aux = C[i, j]
                for k in range(N):
                    GCC[L, j, k] += GC[L, i, k] * aux

    return GP, GC, GCC


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def fock(double[:] GP_m, double[:] GP_s,
         double[:, :, :] GC_m,
         double[:, :] h, double[:, :, :] g):
    """Fock matrix for (unrestricted) spin m
    
    F_mn = h_mn + sum_rs [ (Pm + Ps) g_mnrs - Pm g_mrsn ]
    
    Parameters:
    -----------
    GP_m, GC_m  - Contractions of g with P and C for spin we are at: m.
                  See common_blocks
    GP_s        - Contraction of g with P for the other spin: s.
                  See common_blocks
    h, g  - molecular integrals, see module's docstring
    
    """
    cdef int i, j, k, L, T = g.shape[0], d = g.shape[1], N = GC_m.shape[2]
    cdef double[:, :] F = np.array(h)

    for L in range(T):
        for i in range(d):
            for j in range(d):
                F[i, j] += g[L, i, j] * (GP_m[L] + GP_s[L])

    for L in range(T):
        for i in range(d):
            for j in range(d):
                for k in range(N):
                    F[i, j] -= GC_m[L, i, k] * GC_m[L, j, k]

    return F


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def fock_restr(double[:] GP, double[:, :, :] GC,
               double[:, :] h, double[:, :, :] g):
    """Fock matrix for restricted orbitals
    
    F_mn = h_mn + sum_rs P [ g_mnrs - g_mrsn/2 ]
    
    Parameters:
    -----------
    GP, GC  - Contractions of g with P and C.
              See common_blocks
    h, g    - molecular integrals, see module's docstring
    
    """
    cdef int i, j, k, L, T = g.shape[0], n = g.shape[1], N = GC.shape[2]
    cdef double[:, :] F = np.array(h)

    for L in range(T):
        for i in range(n):
            for j in range(n):
                F[i, j] += g[L, i, j] * GP[L]

    for L in range(T):
        for i in range(n):
            for j in range(n):
                for k in range(N):
                    F[i, j] -= GC[L, i, k] * GC[L, j, k]

    return F


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def _u_mixed_spin(double[:, :, :] GC, double[:, :, :] GC_b):
    """Construct the whole mixed spin part of the Hessian."""
    cdef int r, s, p, q, L, T = GC.shape[0]
    cdef int d = GC.shape[1], N_a = GC.shape[2], N_b = GC_b.shape[2]
    cdef double[:, :] M = np.zeros((d * N_b, d * N_a))

    for L in range(T):
        for r in range(d):
            for s in range(N_b):
                for p in range(d):
                    for q in range(N_a):
                        M[r + s*d, p + q*d] += 4 * GC[L, p, q] * GC_b[L, r, s]

    return M


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def _u_non_diag_block(double[:, :, :] GC,
                      double[:, :, :] GCC,
                      double[:, :, :] g,
                      int q,
                      int s):
    """Construct the non-diagonal blocks of the hessian given b and j."""
    cdef int p, r, L, T = g.shape[0], d = g.shape[1]
    cdef double[:, :] M = np.zeros((d, d))

    for L in range(T):
        for r in range(d):
            for p in range(d):
                M[r, p] += 2 * (2 * GC[L, p, q] * GC[L, r, s]
                                - GC[L, p, s] * GC[L, r, q]
                                - GCC[L, q, s] * g[L, p, r])

    return M


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def _u_diag_block(double[:, :, :] GC,
                  double[:, :, :] GCC,
                  double[:, :, :] g,
                  int q):
    """Construct the diagonal blocks of the hessian."""
    cdef int p, r, L, T = g.shape[0], d = g.shape[1]
    cdef double[:, :] M = np.zeros((d, d))

    for L in range(T):
        for r in range(d):
            for p in range(r + 1):
                M[r, p] -= 2 * (g[L, p, r] * GCC[L, q, q]
                                - GC[L, p, q] * GC[L, r, q])
                M[p, r] = M[r, p]

    return M


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def hessian(double[:, :] C,
            double[:, :] C_b,
            double[:, :] fock,
            double[:, :] fock_b,
            double[:, :, :] GC,
            double[:, :, :] GC_b,
            double[:, :, :] GCC,
            double[:, :, :] GCC_b,
            double[:, :, :] g):
    cdef int d = C.shape[0], N_a = C.shape[1], N_b = C_b.shape[1]
    cdef int j, b, N = N_a + N_b, T = g.shape[0]
    hess = np.empty((d*N, d*N))

    hess[:d*N_a, :d*N_a] = 2 * np.kron(np.eye(N_a), fock)
    hess[d*N_a:, d*N_a:] = 2 * np.kron(np.eye(N_b), fock_b)
    
    # computational cost: O(N_a * N_b * d^2 * L)
    hess[d*N_a:, :d*N_a] = _u_mixed_spin(GC, GC_b)
    hess[:d*N_a, d*N_a:] = hess[d*N_a:, :d*N_a].T

    # computational cost: O(N_a * d^2 * L)
    for q in range(N_a):
        hess[d*q : d*(q+1), d*q : d*(q+1)] += _u_diag_block(GC, GCC, g, q)

    for q in range(N_b):
        hess[d*(q+N_a) : d*(q+N_a+1), d*(q+N_a) : d*(q+N_a+1)] += \
            _u_diag_block(GC_b, GCC_b, g, q)

    # computational cost: O(d^2 * N_a^2 * L)
    for s in range(N_a - 1):
        for q in range(s + 1):
            hess[d*(s+1) : d*(s+2), d*q : d*(q+1)] = \
                _u_non_diag_block(GC, GCC, g, q, s+1)
            hess[d*q : d*(q+1), d*(s+1) : d*(s+2)] = \
                hess[d*(s+1) : d*(s+2), d*q : d*(q+1)].T

    for s in range(N_b - 1):
        for q in range(s + 1):
            hess[d*(N_a+s+1) : d*(N_a+s+2), d*(q+N_a) : d*(N_a+q+1)] = \
                _u_non_diag_block(GC_b, GCC_b, g, q, s+1)
            hess[d*(q+N_a) : d*(N_a+q+1), d*(N_a+s+1) : d*(N_a+s+2)] = \
                hess[d*(N_a+s+1) : d*(N_a+s+2), d*(N_a+q) : d*(N_a+q+1)].T

    return hess


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def _r_non_diag_block(double[:, :, :] GC,
                      double[:, :, :] GCC,
                      double[:, :, :] g,
                      int q,
                      int s):
    cdef int p, r, L, T = g.shape[0], d = g.shape[1]
    cdef double[:, :] M = np.zeros((d, d))

    for L in range(T):
        for r in range(d):
            for p in range(d):
                M[r, p] += 4 * (4 * GC[L, p, q] * GC[L, r, s]
                                - GC[L, p, s] * GC[L, r, q]
                                - GCC[L, q, s] * g[L, p, r])

    return M


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def _r_diag_block(double[:, :, :] GC,
                  double[:, :, :] GCC,
                  double[:, :, :] g,
                  int q):
    cdef int p, r, L, T = g.shape[0], d = g.shape[1]
    cdef double[:, :] M = np.zeros((d, d))

    for L in range(T):
        for r in range(d):
            for p in range(r + 1):
                M[r, p] += 4 * (3 * GC[L, p, q] * GC[L, r, q]
                                - GCC[L, q, q] * g[L, p, r])
                M[p, r] = M[r, p]

    return M

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def restricted_hessian(double[:, :] C,
                       double[:, :] fock,
                       double[:, :, :] GC,
                       double[:, :, :] GCC,
                       double[:, :, :] g):
    cdef int d = C.shape[0], N = C.shape[1]
    hess = 4 * np.kron(np.eye(N), fock)

    for q in range(N):
        hess[d*q : d*(q+1), d*q : d*(q+1)] += _r_diag_block(GC, GCC, g, q)

    for s in range(N - 1):
        for q in range(s + 1):
            hess[d*(s+1) : d*(s+2), d*q : d*(q+1)] = _r_non_diag_block(GC, GCC, g, q, s+1)
            hess[d*q : d*(q+1), d*(s+1) : d*(s+2)] = hess[d*(s+1) : d*(s+2), d*q : d*(q+1)].T

    return hess


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def dir_proj(double[:, :] C, double[:, :] grad):
    cdef int a, b, n = grad.shape[0], N = grad.shape[1]
    cdef double[:, :] aux = np.transpose(C) @ np.array(grad)
    cdef double[:, :] Id = np.eye(n)
    der = np.zeros((n * N, n * N))

    # computational cost: O(N_a * N_b * n^2 * L)
    hess[:, :] = mixed_spin(GC, GC)

    # computational cost: O(N_a * n^2 * L)
    for b in range(N):
        for a in range(N):
            der[a * n: (a + 1) * n, b * n: (b + 1) * n] = np.multiply(Id, aux[b, a])

    # computational cost: O(n^2 * N_a^2 * L)
    for j in range(N - 1):
        for b in range(j + 1):
            hess[n*(j+1) : n*(j+2), n*b : n*(b+1)] = \
                non_diag_blocks(C, GC, GCC, g, b, j+1)
            hess[n*b : n*(b+1), n*(j+1) : n*(j+2)] = \
                hess[n*(j+1) : n*(j+2), n*b : n*(b+1)].T

    return hess




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def dir_proj(double[:, :] C, double[:, :] grad):
    cdef int a, b, n = grad.shape[0], N = grad.shape[1]
    cdef double[:, :] aux = np.transpose(C) @ np.array(grad)
    cdef double[:, :] Id = np.eye(n)
    der = np.zeros((n*N, n*N))

    for b in range(N):
        for a in range(N):
            der[a*n : (a+1)*n, b*n : (b+1)*n] = np.multiply(Id, aux[b, a])

    return der


# DEPRECATED FUNCTIONS
cdef int getindex(int i, int j, int k, int l):
    """Convert the indexes of the two-electron integrals."""
    cdef int ij, kl, ijkl

    ij = j + i * (i + 1) // 2 if i >= j else i + j * (j + 1) // 2
    kl = l + k * (k + 1) // 2 if k >= l else k + l * (l + 1) // 2
    ijkl = (kl + ij * (ij + 1) // 2 if ij >= kl else ij + kl * (kl + 1) // 2)

    return ijkl


def energy(double[:, :] C_a, double[:, :] C_b,
           double[:, :] xxt, double[:, :] yyt,
           double[:, :] h, double[:] g):
    """Compute the energy.

    Parameters:
    -----------
    C_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_a the number of spin alpha orbitals in the
        Slater determinant.

    C_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_b the number of spin beta orbitals in the
        Slater determinant.

    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, C_a @ C_a.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (C_b @ C_b.T).

    h (2D np.ndarray)
        One-electron integral matrix.

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    Three doubles: the total, the one-electron and the two-electron energies.
    """
    cdef int j, k, p, q, r, s
    cdef int n = C_a.shape[0], N_a = C_a.shape[1], N_b = C_b.shape[1]
    cdef double one_elec = 0, two_elec = 0

    for p in range(n):
        for q in range(n):
            one_elec += (xxt[p, q] + yyt[p, q]) * h[p, q]
            for r in range(n):
                for s in range(n):
                    two_elec += (0.5 * (xxt[p, r] * xxt[q, s] + yyt[p, r] * yyt[q, s])
                                 * (g[getindex(p, r, q, s)]
                                    - g[getindex(p, s, q, r)]))
                    two_elec += xxt[p, r] * yyt[q, s] * g[getindex(p, r, q, s)]

    return one_elec + two_elec, one_elec, two_elec


def grad_one(double[:, :] C_a, double[:, :] C_b, double[:, :] h):
    """Compute the one-electron gradient.

    The formula for the one-electron gradient is: grad = h @ C_a, being h the
    one-electron integrals and C_a the Slater determinant that you want to compute
    the gradient at. Here we have C_a and C_b because C_a is for alpha spin and C_b for
    beta spin.

    Parameters:
    -----------
    C_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_a is the number of spin alpha orbitals in the
        Slater determinant.

    C_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_b the number of spin beta orbitals in the
        Slater determinant.

    h (2D np.ndarray)
        One-electron integral matrix.

    Returns:
    --------
    A 2D np.ndarray of dimension (n, N_a + N_b)).
    """
    cdef int n = C_a.shape[0], N_a = C_a.shape[1], N_b = C_b.shape[1]
    cdef double[:, :] grad = np.zeros((n, N_a + N_b))

    grad[:, :N_a] = np.array(h) @ np.array(C_a)
    grad[:, N_a:] = np.array(h) @ np.array(C_b)

    return np.array(grad)


def aux_grad_two(double[:, :] W, double[:, :] wwt,
                 double[:, :] zzt, double[:] g):
    """Compute the two-electron gradient.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_a or N_b.

    wwt (2D np.ndarray of dimension (n, n))
        Density matrix, ie, W @ W.T.

    zzt (2D np.ndarray of dimension (n, n))
        Density matrix, but for the other spin (Z @ Z.T).

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n, M).
    """
    cdef int a, b, p, q, s, n = W.shape[0], M = W.shape[1]
    cdef double[:, :] grad = np.array((n, M))

    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        grad[a, b] += (W[p, b] * wwt[q, s]
                                       * (2 * g[getindex(a, p, q, s)]
                                          - g[getindex(a, s, q, p)]
                                          - g[getindex(p, s, q, a)]))
                        grad[a, b] += (2 * W[p, b] * zzt[q, s]
                                       * g[getindex(a, p, q, s)])

    return grad


def gradtwo(double[:, :] C_a, double[:, :] C_b,
            double[:, :] xxt, double[:, :] yyt,
            double[:] g):
    """Build the two-electron gradient matrix.

    Parameters:
    -----------
    C_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_a the number of spin alpha orbitals in the
        Slater determinant.

    C_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_b the number of spin beta orbitals in the
        Slater determinant.

    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, C_a @ C_a.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (C_b @ C_b.T).

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n, N_a+N_b).
    """
    cdef int n = C_a.shape[0], N_a = C_a.shape[1], N_b = C_b.shape[1]
    cdef double[:, :] grad = np.zeros((n, N_a + N_b))

    grad[:, :N_a] = aux_grad_two(N_a, n, C_a, xxt, yyt, g)
    grad[:, N_a:] = aux_grad_two(N_b, n, C_b, yyt, xxt, g)

    return np.array(grad)


def gradient_three(double[:, :] W, double[:, :] Z,
                   double[:, :] wwt, double[:, :] zzt,
                   double[:, :] h, double[:, :, :] g):
    """Compute gradient of the energy.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_a or N_b.

    Z (2D np.ndarray of dimension (n, M))
        The other part of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_a or N_b.

    wwt (2D np.ndarray of dimension (n, n))
        Density matrix, ie, W @ W.T.

    zzt (2D np.ndarray of dimension (n, n))
        Density matrix, but for the other spin (Z @ Z.T).

    h (2D np.ndarray)
        One-electron integral matrix.

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n, M).
    """
    cdef int a, b, p, q, s, ko
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    cdef double[:, :] grad = 2 * (np.array(h) @ np.array(W))

    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        for L in range(g.shape[0]):
                            grad[a, b] += (2 * W[p, b] * zzt[q, s]
                                           * g[L, a, p] * g[L, q, s])
                            grad[a, b] += (W[p, b] * wwt[q, s]
                                           * (2 * g[L, a, p] * g[L, q, s]
                                              - g[L, a, s] * g[L, q, p]
                                              - g[L, p, s] * g[L, q, a]))

    return np.array(grad)


def gradient(double[:, :] W, double[:, :] Z,
             double[:, :] wwt, double[:, :] zzt,
             double[:, :] h, double[:] g):
    """Compute gradient of the energy.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_a or N_b.

    Z (2D np.ndarray of dimension (n, M))
        The other part of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_a or N_b.

    wwt (2D np.ndarray of dimension (n, n))
        Density matrix, ie, W @ W.T.

    zzt (2D np.ndarray of dimension (n, n))
        Density matrix, but for the other spin (Z @ Z.T).

    h (2D np.ndarray)
        One-electron integral matrix.

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n, M).
    """
    cdef int a, b, p, q, s, k
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    cdef double[:, :] grad = 2 * (np.array(h) @ np.array(W))

    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        grad[a, b] += (2 * W[p, b] * zzt[q, s]
                                       * g[getindex(a, p, q, s)])
                        grad[a, b] += (W[p, b] * wwt[q, s]
                                       * (2 * g[getindex(a, p, q, s)]
                                          - g[getindex(a, s, q, p)]
                                          - g[getindex(p, s, q, a)]))

    return np.array(grad)


def grad_fock_three(double[:, :] W, double[:, :] F, double[:, :] wwt, double[:] g):
    cdef double[:, :] grad = 2 * (np.array(F) @ np.array(W))
    cdef int i, j, k, n = W.shape[0], N = W.shape[1]

    for a in range(n):
        for b in range(N):
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        grad[a, b] += (W[i, b] * wwt[j, k]
                                       * (g[getindex(a, k, j, i)]
                                          - g[getindex(a, j, k, i)]))

    return np.array(grad)


def grad_fock(double[:, :] W, double[:, :] F, double[:, :] wwt, double[:, :, :] g):
    cdef double[:, :] grad = 2 * (np.array(F) @ np.array(W))
    cdef int i, j, k, n = W.shape[0], N = W.shape[1]

    for a in range(n):
        for b in range(N):
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for L in range(g.shape[0]):
                            grad[a, b] += (W[i, b] * wwt[j, k]
                                           * (g[L, a, k] * g[L, j, i]
                                              - g[L, a, j] * g[L, k, i]))

    return np.array(grad)


def g_eta(double[:, :] W, double[:, :] wwt, double[:, :] zzt,
          double[:, :] etaW, double[:, :] etaWwt, double[:, :] etaZzt,
          double[:, :] h, double[:] g, double[:, :] Z, double[:, :] etaZ):
    """Compute the derivative of the gradient in the eta direction.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_a or N_b.

    wwt (2D np.ndarray of dimension (n, n))
        Density matrix, ie, W @ W.T.

    zzt (2D np.ndarray of dimension (n, n))
        Density matrix, but for the other spin (Z @ Z.T).

    etaW (2D np.ndarray of dimension (n, M))
        The part of the eta direction that pairs with W.

    etaWwt (2D np.ndarray of dimension (n, n))
        etaW @ W.T.

    etaZzt (2D np.ndarray of dimension (n, n))
        etaZ @ Z.T (etaZ being the part of the eta direction that pairs with Z, ie,
        with the other spin).

    h (2D np.ndarray)
        One-electron integral matrix.

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n, M).
    """
    cdef int a, b, p, q, s, n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    cdef double[:, :] grad_eta = 2 * (np.array(h) @ np.array(etaW))

    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        grad_eta[a, b] += ((etaW[p, b] * wwt[q, s]
                                            + W[p, b] * etaWwt[q, s]
                                            + W[p, b] * etaWwt[s, q])
                                           * (2 * g[getindex(a, p, q, s)]
                                              - g[getindex(a, s, q, p)]
                                              - g[getindex(p, s, q, a)]))
                        grad_eta[a, b] += (2 * (etaW[p, b] * zzt[q, s]
                                                + W[p, b] * etaZzt[q, s]
                                                + W[p, b] * etaZzt[s, q])
                                           * g[getindex(a, p, q, s)])

    return np.array(grad_eta)


def directional_derivative(double[:, :] C_a, double[:, :] C_b,
                           double[:, :] xxt, double[:, :] yyt,
                           double[:, :] projC_a, double[:, :] projC_b,
                           double[:, :] gradC_a, double[:, :] gradC_b,
                           double[:, :] invS, double[:, :] h, double[:] g):
    """Compute the matrix of the Levi-Civita connection, ie, Absil's LHS.

    Parameters:
    -----------
    C_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_a the number of spin alpha orbitals in the
        Slater determinant.

    C_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_b the number of spin beta orbitals in the
        Slater determinant.

    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, C_a @ C_a.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (C_b @ C_b.T).

    projC_a (2D np.ndarray of dimension (n, N_a))
        Projection matrix at H_C_a: Id - C_a @ inv(C_a.T @ S @ C_a) @ C_a.T @ S

    projC_b (2D np.ndarray of dimension (n, N_b))
        Projection matrix at H_C_b: Id - C_b @ inv(C_b.T @ S @ C_b) @ C_b.T @ S

    gradC_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the gradient of the energy.

    gradC_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the gradient of the energy.

    invS (2D np.ndarray of dimension (n, n))
        The inverse of the overlap matrix.

    h (2D np.ndarray)
        One-electron integral matrix.

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n*N, n*N).
    """
    cdef int n = C_a.shape[0], N_a = C_a.shape[1], N_b = C_b.shape[1]
    cdef int N = N_a + N_b, col = 0
    cdef double[:, :] C_atransp = np.transpose(C_a), C_btransp = np.transpose(C_b)
    cdef double[:, :] gC_a = np.empty((n, N_a)), gC_b = np.empty((n, N_b))
    cdef double[:, :] E = np.zeros((n, N_a)), F = np.zeros((n, N_b))
    cdef double[:, :] zero = np.zeros((n, n))
    tmp, D = np.zeros((n, N)), np.zeros((n * N, n * N))

    for j in range(N_a):
        for i in range(n):
            E[i, j] = 1.0
            gC_a = g_eta(C_a, xxt, yyt, E, np.array(E) @ C_atransp, zero, h, g, C_b, F)
            gC_b = g_eta(C_b, yyt, xxt, F, zero, np.array(E) @ C_atransp, h, g, C_a, E)
            tmp[:, :N_a] = projC_a @ (invS @ np.array(gC_a)
                                      - np.array(E) @ C_atransp @ gradC_a)
            tmp[:, N_a:] = projC_b @ (invS @ np.array(gC_b))
            # tmp[:, :N_a] = gC_a
            # tmp[:, N_a:] = gC_b
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            E[i, j] = 0.0
            col += 1
    for j in range(N_b):
        for i in range(n):
            F[i, j] = 1.0
            gC_a = g_eta(C_a, xxt, yyt, E, zero, np.array(F) @ C_btransp, h, g, C_b, F)
            gC_b = g_eta(C_b, yyt, xxt, F, np.array(F) @ C_btransp, zero, h, g, C_a, E)
            tmp[:, :N_a] = projC_a @ (invS @ np.array(gC_a))
            tmp[:, N_a:] = projC_b @ (invS @ np.array(gC_b)
                                      - np.array(F) @ C_btransp @ gradC_b)
            # tmp[:, :N_a] = gC_a
            # tmp[:, N_a:] = gC_b
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            F[i, j] = 0.0
            col += 1

    return D


def verifica_g_eta(double[:, :] W, double[:, :] Z, double[:, :] etaW,
                   double[:, :] etaZ, double[:, :] g_eta, double[:, :] h,
                   double[:] g, double t):
    """(grad{f}(C_a+t\\eta) - grad{f}(C_a-t\\eta)) / 2t"""
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]

    tmpW = np.array(W) + t * np.array(etaW)
    tmpZ = np.array(Z) + t * np.array(etaZ)
    ini = gradient(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    tmpW -= 2 * t * np.array(etaW)
    tmpZ -= 2 * t * np.array(etaZ)
    ini -= gradient(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    ini /= 2 * t

    return ini - g_eta
