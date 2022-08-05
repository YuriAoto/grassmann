"""A module with helper functions for Absil's Hartree-Fock.

    Common parameters:
    ------------------
    C_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_a is the number of the spin alpha occupied orbitals
        in the Slater determinant.

    C_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_b is the number of the spin beta occupied orbitals
        in the Slater determinant.

    P_a (2D np.ndarray of dimension (n, n))
        Density matrix for spin alpha: C_a @ C_a.T

    P_b (2D np.ndarray of dimension (n, n))
        Density matrix for spin beta: C_b @ C_b.T
    
    proj_a (2D np.ndarray of dimension (n, N_a))
        Projection matrix at H_{C_a}: Id - C_a @ C_a.T @ S

    proj_b (2D np.ndarray of dimension (n, N_b))
        Projection matrix at H_{C_b}: Id - C_b @ C_b.T @ S
    
    grad_a (2D np.ndarray of dimension (n, N_a))
        The spin alpha part of the gradient of the energy.

    grad_b (2D np.ndarray of dimension (n, N_b))
        The spin beta part of the gradient of the energy.

    h (2D np.ndarray)
        One-electron integral matrix.

    g (3D np.ndarray)
        Two-electron integral matrix.

    S (2D np.ndarray)
        Overlap matrix.
"""
cimport cython

import numpy as np
from scipy import linalg


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def common_blocks(double[:, :] C_a, double[:, :] C_b,
                  double[:, :] P_a, double[:, :] P_b,
                  double[:, :, :] g):
    cdef int i, j, k, L, T = g.shape[0], n = C_a.shape[0]
    cdef int N_a = C_a.shape[1], N_b = C_b.shape[1]
    cdef double[:] GP_a = np.zeros(T)
    cdef double[:] GP_b = np.zeros(T)
    cdef double[:, :, :] GC_a = np.zeros((T, n, N_a))
    cdef double[:, :, :] GC_b = np.zeros((T, n, N_b))
    cdef double[:, :, :] GCC_a = np.zeros((T, N_a, N_a))
    cdef double[:, :, :] GCC_b = np.zeros((T, N_b, N_b))
    cdef double aux

    for L in range(T):
        for i in range(n):
            for k in range(n):
                aux = g[L, i, k]
                GP_a[L] += aux * P_a[i, k]
                GP_b[L] += aux * P_b[i, k]

                for j in range(N_a):
                    GC_a[L, i, j] += aux * C_a[k, j]

                for j in range(N_b):
                    GC_b[L, i, j] += aux * C_b[k, j]

            for j in range(N_a):
                aux = C_a[i, j]
                for k in range(N_a):
                    GCC_a[L, j, k] += GC_a[L, i, k] * aux

            for j in range(N_b):
                aux = C_b[i, j]
                for k in range(N_b):
                    GCC_b[L, j, k] += GC_b[L, i, k] * aux

    return GP_a, GP_b, GC_a, GC_b, GCC_a, GCC_b

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def fock(double[:] GP_m, double[:] GP_s,
         double[:, :, :] GC_m, double[:, :, :] GC_s,
         double[:, :] h, double[:, :, :] g):
    cdef int i, j, k, L, T = g.shape[0], n = g.shape[1], N = GC_m.shape[2]
    cdef double[:, :] F = np.array(h)

    for L in range(T):
        for i in range(n):
            for j in range(n):
                F[i, j] += g[L, i, j] * (GP_m[L] + GP_s[L])

    for L in range(T):
        for i in range(n):
            for j in range(n):
                for k in range(N):
                    F[i, j] -= GC_m[L, i, k] * GC_m[L, j, k]

    return F

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def mixed_spin(double[:, :, :] GC_a, double[:, :, :] GC_b):
    """Construct the whole mixed spin part of the hessian."""
    cdef int a, b, i, j, L, T = GC_a.shape[0]
    cdef int N_a = GC_a.shape[2], N_b = GC_b.shape[2], n = GC_a.shape[1]
    cdef double[:, :] M = np.zeros((n*N_b, n*N_a))

    for L in range(T):
        for i in range(n):
            for j in range(N_b):
                for a in range(n):
                    for b in range(N_a):
                        M[i + j*n, a + b*n] += 2 * GC_a[L, a, b] * GC_b[L, i, j]

    return M

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def non_diag_blocks(double[:, :] C, double[:, :, :] GC,
                    double[:, :, :] GCC, double[:, :, :] g,
                    int b, int j):
    """Construct the non-diagonal blocks of the hessian given b and j."""
    cdef int a, i, L
    cdef int n = C.shape[0], T = g.shape[0]
    cdef double[:, :] M = np.zeros((n, n))

    for L in range(T):
        for i in range(n):
            for a in range(n):
                M[i, a] += (2 * GC[L, a, b] * GC[L, i, j]
                            - GC[L, a, j] * GC[L, i, b]
                            - g[L, i, a] * GCC[L, j, b])

    return M

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def diag_blocks(double[:, :] C, double[:, :, :] GC, double[:, :, :] GCC,
                double[:, :] fock, double[:, :, :] g, int b):
    """Construct the diagonal blocks of the hessian."""
    cdef int a, i, L
    cdef int n = C.shape[0], T = g.shape[0]
    cdef double[:, :] M = np.array(fock)

    for L in range(T):
        for i in range(n):
            for a in range(i + 1):
                M[i, a] -= (g[L, i, a] * GCC[L, b, b]
                            - GC[L, a, b] * GC[L, i, b])
                M[a, i] = M[i, a]

    return M

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def hessian(double[:, :] C_a, double[:, :] C_b,
            double[:, :] fock_a, double[:, :] fock_b,
            double[:, :, :] GC_a, double[:, :, :] GC_b,
            double[:, :, :] GCC_a, double[:, :, :] GCC_b,
            double[:, :, :] g):
    cdef int n = C_a.shape[0], N_a = C_a.shape[1], N_b = C_b.shape[1]
    cdef int j, b, N = N_a + N_b, T = g.shape[0]
    hess = np.empty((n*N, n*N))

    # computational cost: N_a * N_b * n^2 * L
    hess[n*N_a :, : n*N_a] = mixed_spin(GC_a, GC_b)
    hess[: n*N_a, n*N_a :] = hess[n*N_a :, : n*N_a].T

    # computational cost: N_a * n^2 * L
    for b in range(N_a):
        hess[n*b : n*(b+1), n*b : n*(b+1)] = \
            diag_blocks(C_a, GC_a, GCC_a, fock_a, g, b)

    for b in range(N_b):
        hess[n*(b+N_a) : n*(b+N_a+1), n*(b+N_a) : n*(b+N_a+1)] = \
            diag_blocks(C_b, GC_b, GCC_b, fock_b, g, b)

    # computational cost: n^2 * N_a^2 * L
    for j in range(N_a - 1):
        for b in range(j + 1):
            hess[n*(j+1) : n*(j+2), n*b : n*(b+1)] = \
                non_diag_blocks(C_a, GC_a, GCC_a, g, b, j+1)
            hess[n*b : n*(b+1), n*(j+1) : n*(j+2)] = \
                hess[n*(j+1) : n*(j+2), n*b : n*(b+1)].T

    for j in range(N_b - 1):
        for b in range(j + 1):
            hess[n*(N_a+j+1) : n*(N_a+j+2), n*(b+N_a) : n*(N_a+b+1)] = \
                non_diag_blocks(C_b, GC_b, GCC_b, g, b, j+1)
            hess[n*(b+N_a) : n*(N_a+b+1), n*(N_a+j+1) : n*(N_a+j+2)] = \
                hess[n*(N_a+j+1) : n*(N_a+j+2), n*(N_a+b) : n*(N_a+b+1)].T

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

def normalize(double[:] v, double[:, :] S):
    """Normalize a vector with respect to an arbitrary basis.

    Parameters:
    -----------
    v (1D np.ndarray of dimension (n, 1))
        The vector to be normalized

    S (2D np.ndarray of dimension (n, n))
        The Gram matrix associated with the basis of interest.

    Returns:
    --------
    A 1D np.ndarray of dimension (n, 1)
    """
    return v / np.sqrt(np.transpose(v).T @ S @ v)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def gram_schmidt(M, double[:, :] S):
    """Gram-Schmidt algorithm in an arbitrary basis.

    Parameters:
    -----------
    M (2D np.ndarray of dimension (n, k))
        Matrix containing the k vectors to be orthonormalized. Each vector is a
        column of the matrix.

    S (2D np.ndarray of dimension (n, n))
        The Gram matrix associated with the basis of interest.

    Returns:
    --------
    A 2D np.ndarray of dimension (n, k) containing the orthonormalized vectors.
    """
    cdef int i, j

    M[:, 0] = normalize(M[:, 0], S)
    for i in range(1, M.shape[1]):
        Mi = M[:, i]
        for j in range(i):
            Mj = M[:, j]
            t = Mi.T @ S @ Mj
            Mi = Mi - t * Mj
            M[:, i] = normalize(Mi, S)

    return M





### DEPRECATED FUNCTIONS


cdef int getindex(int i, int j, int k, int l):
    """Convert the indexes of the two-electron integrals."""
    cdef int ij, kl, ijkl

    ij = j + i*(i + 1) // 2 if i >= j else i + j*(j + 1) // 2
    kl = l + k*(k + 1) // 2 if k >= l else k + l*(l + 1) // 2
    ijkl = (kl + ij*(ij + 1) // 2 if ij >= kl else ij + kl*(kl + 1) // 2)

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
                    two_elec += (0.5 * (xxt[p, r]*xxt[q, s] + yyt[p, r]*yyt[q, s])
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
    cdef double[:, :] grad = np.zeros((n, N_a+N_b))

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
    cdef double[:, :] grad = np.zeros((n, N_a+N_b))

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

def grad_fock_three(double[:,:] W, double[:,:] F, double[:,:] wwt, double[:] g):
    cdef double[:,:] grad = 2 * (np.array(F) @ np.array(W))
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

def grad_fock(double[:,:] W, double[:,:] F, double[:,:] wwt, double[:, :, :] g):
    cdef double[:,:] grad = 2 * (np.array(F) @ np.array(W))
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
    tmp, D = np.zeros((n, N)), np.zeros((n*N, n*N))

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
    """(grad{f}(C_a+t\eta) - grad{f}(C_a-t\eta)) / 2t"""
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    
    tmpW = np.array(W) + t * np.array(etaW)
    tmpZ = np.array(Z) + t * np.array(etaZ)
    ini = gradient(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    tmpW -= 2 * t * np.array(etaW)
    tmpZ -= 2 * t * np.array(etaZ)
    ini -= gradient(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    ini /= 2 * t

    return ini - g_eta
