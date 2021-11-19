#cython: profile=True
"""A module with helper functions for Absil's Hartree-Fock."""

cimport cython

import numpy as np
from scipy import linalg


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def mixed_spins(double[:, :] X, double[:, :] Y , double[:, :, :] g):
    cdef int a, b, i, j, p, q, L
    cdef int n = X.shape[0], N_a = X.shape[1], N_b = Y.shape[1], T = g.shape[0]
    cdef double[:, :] M = np.zeros((n*N_b, n*N_a))
    cdef double[:, :, :] tmp1 = np.zeros((T, n, N_a))
    cdef double[:, :, :] tmp2 = np.zeros((T, n, N_b))
    cdef double aux

    for L in range(T):
        for a in range(n):
            for p in range(n):
                aux = g[L, a, p]
                for b in range(N_a):
                    tmp1[L, a, b] += aux * X[p, b]
                for b in range(N_b):
                    tmp2[L, a, b] += aux * Y[p, b]

    for L in range(T):
        for i in range(n):
            for a in range(n):
                for j in range(N_b):
                    for b in range(N_a):
                        M[i + j*n, a + b*n] += 4 * tmp1[L, a, b] * tmp2[L, i, j]

    return np.array(M)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def non_diag_blocks(double[:, :] W, double[:, :, :] g, int b, int j):
    cdef int a, i, p, L
    cdef int n = W.shape[0], T = g.shape[0]
    cdef double[:, :] M = np.zeros((n, n))
    cdef double[:, :] tmp1 = np.zeros((T, n)), tmp2 = np.zeros((T, n))
    cdef double[:] tmp3 = np.zeros(T)

    for L in range(T):
        for a in range(n):
            for p in range(n):
                tmp1[L, a] += g[L, a, p] * W[p, b]
                tmp2[L, a] += g[L, a, p] * W[p, j]
            tmp3[L] += tmp1[L, a] * W[a, j]

    for L in range(T):
        for i in range(n):
            for a in range(n):
                M[i, a] += 2 * (2 * tmp1[L, a] * tmp2[L, i]
                            - tmp2[L, a] * tmp1[L, i]
                            - g[L, a, i] * tmp3[L])

    return np.array(M)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def diag_blocks(double[:, :] W, double[:, :, :] g, double[:, :] fock, int b):
    cdef int a, i, q, s
    cdef int n = W.shape[0], T = g.shape[0]
    cdef double[:, :] M = np.zeros((n, n))
    cdef double[:, :] tmp1 = np.zeros((T, n))
    cdef double[:, :] tmp2 = np.zeros((T, n))
    cdef double[:] tmp3 = np.zeros(T)
    cdef double aux

    for L in range(T):
        for a in range(n):
            for s in range(n):
                aux = W[s, b]
                tmp1[L, a] += g[L, a, s] * aux
                tmp2[L, a] += g[L, s, a] * aux
            tmp3[L] += tmp1[L, a] * W[a, b]

    for L in range(T):
        for i in range(n):
            for a in range(i + 1):
                M[i, a] += (g[L, a, i]*tmp3[L] - tmp1[L, a]*tmp2[L, i])
                M[a, i] = M[i, a]

    return fock - np.array(M)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def hessian_f(double[:, :] X, double[:, :] Y, double[:, :, :] g,
              double[:, :] fock_a, double[:, :] fock_b):
    cdef int n = X.shape[0], N_a = X.shape[1], N_b = Y.shape[1], T = g.shape[0]
    cdef int j, b, N = N_a + N_b
    hess = np.empty((n*N, n*N))

    # custo N_a * N_b * n^2 * L
    hess[n*N_a :, : n*N_a] = mixed_spins(X, Y, g)
    hess[: n*N_a, n*N_a :] = hess[n*N_a :, : n*N_a].T

    # custo N_a * n^2 * L
    for b in range(N_a):
        hess[n*b : n * (b+1), n*b : n * (b+1)] = \
            2 * diag_blocks(X, g, fock_a, b)

    for b in range(N_b):
        hess[n * (b+N_a) : n * (b+N_a+1), n * (b+N_a) : n * (b+N_a+1)] = \
            2 * diag_blocks(Y, g, fock_b, b)

    # custo n^2 * N_a^2 * L
    for j in range(N_a - 1):
        for b in range(j + 1):
            hess[n*(j + 1) : n*(j + 2), n*b : n * (b+1)] = \
                non_diag_blocks(X, g, b, j+1)
            hess[n*b : n * (b+1), n * (j+1) : n * (j+2)] = \
                hess[n*(j + 1) : n*(j + 2), n*b : n*(b + 1)].T

    for j in range(N_b - 1):
        for b in range(j + 1):
            hess[n*(N_a+j+1) : n*(N_a+j+2), n*(b+N_a) : n*(N_a+b+1)] = \
                non_diag_blocks(Y, g, b, j+1)
            hess[n*(b+N_a) : n*(N_a+b+1), n*(N_a+j+1) : n*(N_a+j+2)] = \
                hess[n*(N_a+j+1) : n*(N_a+j+2), n*(N_a+b) : n*(N_a+b+1)].T

    return hess

def dir_proj(double[:, :] WT, double[:, :] grad):
    # if grad.shape[1] < 1:
    #     return np.empty((0, 0))

    cdef int a, b, n = grad.shape[0], N = grad.shape[1]
    cdef double[:, :] aux = np.array(WT) @ np.array(grad)
    cdef double[:, :] Id = np.eye(n)
    der = np.zeros((n*N, n*N))

    for a in range(N):
        for b in range(N):
            der[a*n : (a+1)*n, b*n : (b+1)*n] = np.multiply(Id, aux[b, a])

    return der



def g_eta(double[:, :] W, double[:, :] wwt, double[:, :] zzt,
          double[:, :] etaW, double[:, :] etaWwt, double[:, :] etaZzt,
          double[:, :] h, double[:] g, double[:, :] Z, double[:, :] etaZ):
    """Compute the derivative of the gradient in the eta direction.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.
    
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

def directional_derivative(double[:, :] X, double[:, :] Y,
                           double[:, :] xxt, double[:, :] yyt,
                           double[:, :] projX, double[:, :] projY,
                           double[:, :] gradX, double[:, :] gradY,
                           double[:, :] invS, double[:, :] h, double[:] g):
    """Compute the matrix of the Levi-Civita connection, ie, Absil's LHS.

    Parameters:
    -----------
    X (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_alpha the number of spin alpha orbitals in the
        Slater determinant.

    Y (2D np.ndarray of dimension (n, N_beta))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_beta the number of spin beta orbitals in the
        Slater determinant.

    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, X @ X.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (Y @ Y.T).
    
    projX (2D np.ndarray of dimension (n, N_alpha))
        Projection matrix at H_X: Id - X @ inv(X.T @ S @ X) @ X.T @ S

    projY (2D np.ndarray of dimension (n, N_beta))
        Projection matrix at H_Y: Id - Y @ inv(Y.T @ S @ Y) @ Y.T @ S
    
    gradX (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the gradient of the energy.

    gradY (2D np.ndarray of dimension (n, N_beta))
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
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef int N = N_alpha + N_beta, col = 0
    cdef double[:, :] Xtransp = np.transpose(X), Ytransp = np.transpose(Y)
    cdef double[:, :] gX = np.empty((n, N_alpha)), gY = np.empty((n, N_beta))
    cdef double[:, :] E = np.zeros((n, N_alpha)), F = np.zeros((n, N_beta))
    cdef double[:, :] zero = np.zeros((n, n))
    tmp, D = np.zeros((n, N)), np.zeros((n*N, n*N))

    for j in range(N_alpha):
        for i in range(n):
            E[i, j] = 1.0
            gX = g_eta(X, xxt, yyt, E, np.array(E) @ Xtransp, zero, h, g, Y, F)
            gY = g_eta(Y, yyt, xxt, F, zero, np.array(E) @ Xtransp, h, g, X, E)
            tmp[:, :N_alpha] = projX @ (invS @ np.array(gX)
                                        - np.array(E) @ Xtransp @ gradX)
            tmp[:, N_alpha:] = projY @ (invS @ np.array(gY))
            # tmp[:, :N_alpha] = gX
            # tmp[:, N_alpha:] = gY
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            E[i, j] = 0.0
            col += 1
    for j in range(N_beta):
        for i in range(n):
            F[i, j] = 1.0
            gX = g_eta(X, xxt, yyt, E, zero, np.array(F) @ Ytransp, h, g, Y, F)
            gY = g_eta(Y, yyt, xxt, F, np.array(F) @ Ytransp, zero, h, g, X, E)
            tmp[:, :N_alpha] = projX @ (invS @ np.array(gX))
            tmp[:, N_alpha:] = projY @ (invS @ np.array(gY)
                                        - np.array(F) @ Ytransp @ gradY)
            # tmp[:, :N_alpha] = gX
            # tmp[:, N_alpha:] = gY
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            F[i, j] = 0.0
            col += 1

    return D

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
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

    return np.array(M)


def verifica_g_eta(double[:, :] W, double[:, :] Z, double[:, :] etaW,
                   double[:, :] etaZ, double[:, :] g_eta, double[:, :] h,
                   double[:] g, double t):
    """(grad{f}(X+t\eta) - grad{f}(X-t\eta)) / 2t"""
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    
    tmpW = np.array(W) + t * np.array(etaW)
    tmpZ = np.array(Z) + t * np.array(etaZ)
    ini = gradient(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    tmpW -= 2 * t * np.array(etaW)
    tmpZ -= 2 * t * np.array(etaZ)
    ini -= gradient(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    ini /= 2 * t
    
    return ini - g_eta

def fock_four(double [:,:] P_s, double[:,:] P_t, double[:,:] h, double[:] g):
    cdef double[:,:] F = np.array(h)
    cdef int i, j, k, l, n = h.shape[0]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    F[i, j] += P_s[k, l] * (g[getindex(i, j, l, k)]
                                            - g[getindex(i, k, l, j)])
                    F[i, j] += P_t[k, l] * g[getindex(i, j, l, k)]

    return np.array(F)

def fock_three(double[:,:] P_s, double[:,:] P_t, double[:,:] h, double[:,:,:] g):
    cdef double[:,:] F = np.array(h)
    cdef int i, j, k, l, L, n = h.shape[0]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    for L in range(g.shape[0]):
                        F[i, j] += P_s[k, l] * (g[L, i, j] * g[L, l, k]
                                                - g[L, i, k] * g[L, l, j])
                        F[i, j] += P_t[k, l] * g[L, i, j] * g[L, l, k]

    return np.array(F)

def fock_three_2(double[:,:] P_s, double[:,:] P_t, double[:,:] h, double[:,:,:] g):
    cdef double[:,:] F = np.array(h)
    cdef int i, j, k, l, L, n = h.shape[0]
    cdef double ijlk, iklj

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    ijlk = iklj = 0.0
                    for L in range(g.shape[0]):
                        ijlk += g[L, i, j] * g[L, l, k]
                        iklj += g[L, i, k] * g[L, l, j]
                    F[i, j] += P_s[k, l] * (ijlk - iklj)
                    F[i, j] += P_t[k, l] * ijlk

    return np.array(F)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def fock_three_3(double[:,:] P_s, double[:,:] P_t, double[:,:] h, double[:,:,:] g):
    cdef double[:,:] F = np.array(h)
    cdef int i, j, k, l, L, n = h.shape[0]
    cdef double[:] tmp = np.zeros(g.shape[0], dtype=np.float64)
    cdef double[:] tmp2 = np.zeros(g.shape[0], dtype=np.float64)
    cdef double[:,:,:] tmp3 = np.zeros((g.shape[0], n, n), dtype=np.float64)

    for L in range(g.shape[0]):
        for k in range(n):
            for l in range(n):
                tmp[L] += g[L, l, k] * P_s[k, l]

    for L in range(g.shape[0]):
        for i in range(n):
            for j in range(n):
                F[i, j] += tmp[L] * g[L, i, j]

    for L in range(g.shape[0]):
        for k in range(n):
            for l in range(n):
                tmp2[L] += g[L, l, k] * P_t[k, l]

    for L in range(g.shape[0]):
        for i in range(n):
            for j in range(n):
                F[i, j] += tmp2[L] * g[L, i, j]

    for L in range(g.shape[0]):
        for k in range(n):
            for l in range(n):
                for i in range(n):
                    tmp3[L, k, i] += g[L, i, l] * P_s[l, k]

    for L in range(g.shape[0]):
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    F[i, j] -= tmp3[L, k, i] * g[L, k, j]

    return np.array(F)


def hessian(double[:, :] W, double[:, :] Z, double[:, :] wwt,
            double[:, :] zzt, double[:, :] h, double[:, :, :] g):
    # double[:,:] projW, double [:,:] invS):
    """Compute the derivative of the gradient in the eta direction.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.
    
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
    cdef int a, b, i, j, p, q, s, L, col = 0
    cdef int n = W.shape[0], N_w = W.shape[1], N_z = Z.shape[1], N = N_w + N_z
    cdef double[:, :] aux = np.empty((n, N))
    hess = np.zeros((n*N, n*N_w))

    for a in range(n):
        for b in range(N_w):
            for j in range(N_w):
                for i in range(n):
                    if j != b:
                        for L in range(g.shape[0]):
                            for p in range(n):
                                for q in range(n):
                                    aux[i, j] += (2 * W[p, b] * W[q, j]
                                                  * (2 * g[L, a, p] * g[L, q, i]
                                                     - g[L, a, i] * g[L, q, p]
                                                     - g[L, a, q] * g[L, i, p]))
                    else:
                        aux[i, j] += 2 * h[a, i]
                        for L in range(g.shape[0]):
                            for q in range(n):
                                for s in range(n):
                                    aux[i, j] += (2 * W[q, b] * W[s, j]
                                                  * (2 * g[L, a, q] * g[L, s, i]
                                                     - g[L, a, i] * g[L, s, q]
                                                     - g[L, a, s] * g[L, i, q]))
                                    aux[i, j] += (2 * zzt[q, s]
                                                  * g[L, a, i] * g[L, q, s])
                                    aux[i, j] += (2 * wwt[q, s]
                                                  * (g[L, a, i] * g[L, q, s]
                                                     - g[L, a, s] * g[L, q, i]))
            for j in range(N_z):
                for i in range(n):
                    for L in range(g.shape[0]):
                        for p in range(n):
                            for q in range(n):
                                aux[i, N_w+j] += (2 * W[p, b] * Z[q, j]
                                                  * g[L, a, p] * g[L, i, q])
            # aux = np.array(projW) @ np.array(invS) @ aux
            hess[:, col] = np.reshape(aux, (n*N,), 'F')
            col += 1

    return hess

def directional_derivative_2(double[:, :] X, double[:, :] Y,
                             double[:, :] projX, double[:, :] projY,
                             double[:, :] gradX, double[:, :] gradY,
                             double[:, :] fock_a, double[:, :] fock_b,
                             double[:, :] invS, double[:, :, :] g):
    """Compute the matrix of the Levi-Civita connection, ie, Absil's LHS.

    Parameters:
    -----------
    X (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_alpha the number of spin alpha orbitals in the
        Slater determinant.

    Y (2D np.ndarray of dimension (n, N_beta))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_beta the number of spin beta orbitals in the
        Slater determinant.

    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, X @ X.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (Y @ Y.T).
    
    projX (2D np.ndarray of dimension (n, N_alpha))
        Projection matrix at H_X: Id - X @ inv(X.T @ S @ X) @ X.T @ S

    projY (2D np.ndarray of dimension (n, N_beta))
        Projection matrix at H_Y: Id - Y @ inv(Y.T @ S @ Y) @ Y.T @ S
    
    gradX (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the gradient of the energy.

    gradY (2D np.ndarray of dimension (n, N_beta))
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
    cdef int n = X.shape[0], N_a = X.shape[1], N_b = Y.shape[1], N = N_a + N_b
    dir_proj_a = np.zeros((n*N, n*N_a))
    dir_proj_b = np.zeros((n*N, n*N_b))
    
    lS = [invS for _ in range(N)]
    inv = np.block(lS)
    lS = [[inv] for _ in range(N)]
    inv = np.block(lS)

    lPX = [projX for _ in range(N)]
    PX = np.block(lPX)
    lPX = [[PX] for _ in range(N)]
    PX = np.block(lPX)

    lPY = [projY for _ in range(N)]
    PY = np.block(lPY)
    lPY = [[PY] for _ in range(N)]
    PY = np.block(lPY)

    hess = inv @ hessian_f(X, Y, g, fock_a, fock_b)
    dir_proj_a[: n*N_a, :] = dir_proj(np.transpose(X), gradX)
    dir_proj_b[n*N_a :, :] = dir_proj(np.transpose(Y), gradY)
    hess[:, : n*N_a] -= dir_proj_a
    hess[:, n*N_a :] -= dir_proj_b
    hess[:, : n*N_a] = lPX @ hess[:, : n*N_a]
    hess[:, n*N_a :] = lPY @ hess[:, n*N_a :]

    return hess

def g_eta_2(double[:, :] W, double[:, :] wwt, double[:, :] zzt,
            double[:, :] etaW, double[:, :] etaWwt, double[:, :] etaZzt,
            double[:, :] h, double[:, :, :] g):
    """Compute the derivative of the gradient in the eta direction.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.
    
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
    cdef int a, b, p, q, s, L, n = W.shape[0], M = W.shape[1]
    cdef double[:, :] hess = 2 * (np.array(h) @ np.array(etaW))

    for a in range(n):
        for b in range(M):
            for p in range(n):
                for L in range(g.shape[0]):
                    for q in range(n):
                        for s in range(n):
                            hess[a, b] += ((etaW[p, b] * wwt[q, s]
                                            + W[p, b] * etaWwt[q, s]
                                            + W[p, b] * etaWwt[s, q])
                                           * (2 * g[L, a, p] * g[L, q, s]
                                              - g[L, a, s] * g[L, q, p]
                                              - g[L, p, s] * g[L, q, a]))
                            hess[a, b] += (2 * (etaW[p, b] * zzt[q, s]
                                                + W[p, b] * etaZzt[q, s]
                                                + W[p, b] * etaZzt[s, q])
                                           * g[L, a, p] * g[L, q, s])

    return np.array(hess)

def directional_derivative_3(double[:, :] X, double[:, :] Y,
                             double[:, :] xxt, double[:, :] yyt,
                             double[:, :] projX, double[:, :] projY,
                             double[:, :] gradX, double[:, :] gradY,
                             double[:, :] invS, double[:, :] h, double[:, :, :] g):
    """Compute the matrix of the Levi-Civita connection, ie, Absil's LHS.

    Parameters:
    -----------
    X (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_alpha the number of spin alpha orbitals in the
        Slater determinant.

    Y (2D np.ndarray of dimension (n, N_beta))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_beta the number of spin beta orbitals in the
        Slater determinant.

    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, X @ X.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (Y @ Y.T).
    
    projX (2D np.ndarray of dimension (n, N_alpha))
        Projection matrix at H_X: Id - X @ inv(X.T @ S @ X) @ X.T @ S

    projY (2D np.ndarray of dimension (n, N_beta))
        Projection matrix at H_Y: Id - Y @ inv(Y.T @ S @ Y) @ Y.T @ S
    
    gradX (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the gradient of the energy.

    gradY (2D np.ndarray of dimension (n, N_beta))
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
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef int N = N_alpha + N_beta, col = 0
    cdef double[:, :] Xtransp = np.transpose(X), Ytransp = np.transpose(Y)
    cdef double[:, :] gX = np.empty((n, N_alpha)), gY = np.empty((n, N_beta))
    cdef double[:, :] E = np.zeros((n, N_alpha)), F = np.zeros((n, N_beta))
    cdef double[:, :] zero = np.zeros((n, n))
    tmp, D = np.empty((n, N)), np.empty((n*N, n*N))

    for j in range(N_alpha):
        for i in range(n):
            E[i, j] = 1.0
            gX = g_eta_2(X, xxt, yyt, E, np.array(E) @ Xtransp, zero, h, g)
            gY = g_eta_2(Y, yyt, xxt, F, zero, np.array(E) @ Xtransp, h, g)
            tmp[:, :N_alpha] = projX @ (invS @ np.array(gX)
                                        - np.array(E) @ Xtransp @ gradX)
            tmp[:, N_alpha:] = projY @ (invS @ np.array(gY))
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            E[i, j] = 0.0
            col += 1
    for j in range(N_beta):
        for i in range(n):
            F[i, j] = 1.0
            gX = g_eta_2(X, xxt, yyt, E, zero, np.array(F) @ Ytransp, h, g)
            gY = g_eta_2(Y, yyt, xxt, F, np.array(F) @ Ytransp, zero, h, g)
            tmp[:, :N_alpha] = projX @ (invS @ np.array(gX))
            tmp[:, N_alpha:] = projY @ (invS @ np.array(gY)
                                        - np.array(F) @ Ytransp @ gradY)
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            F[i, j] = 0.0
            col += 1

    return D





### DEPRECATED FUNCTIONS


cdef int getindex(int i, int j, int k, int l):
    """Convert the indexes of the two-electron integrals."""
    cdef int ij, kl, ijkl

    ij = j + i*(i + 1) // 2 if i >= j else i + j*(j + 1) // 2
    kl = l + k*(k + 1) // 2 if k >= l else k + l*(l + 1) // 2
    ijkl = (kl + ij*(ij + 1) // 2 if ij >= kl else ij + kl*(kl + 1) // 2)

    return ijkl


def energy(double[:, :] X, double[:, :] Y,
           double[:, :] xxt, double[:, :] yyt,
           double[:, :] h, double[:] g):
    """Compute the energy.

    Parameters:
    -----------
    X (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_alpha the number of spin alpha orbitals in the
        Slater determinant.
    
    Y (2D np.ndarray of dimension (n, N_beta))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_beta the number of spin beta orbitals in the
        Slater determinant.
    
    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, X @ X.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (Y @ Y.T).

    h (2D np.ndarray)
        One-electron integral matrix.

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    Three doubles: the total, the one-electron and the two-electron energies.
    """
    cdef int j, k, p, q, r, s
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
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


def grad_one(double[:, :] X, double[:, :] Y, double[:, :] h):
    """Compute the one-electron gradient.

    The formula for the one-electron gradient is: grad = h @ X, being h the
    one-electron integrals and X the Slater determinant that you want to compute
    the gradient at. Here we have X and Y because X is for alpha spin and Y for
    beta spin.

    Parameters:
    -----------
    X (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_alpha is the number of spin alpha orbitals in the
        Slater determinant.

    Y (2D np.ndarray of dimension (n, N_beta))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_beta the number of spin beta orbitals in the
        Slater determinant.

    h (2D np.ndarray)
        One-electron integral matrix.

    Returns:
    --------
    A 2D np.ndarray of dimension (n, N_alpha + N_beta)).
    """
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double[:, :] grad = np.zeros((n, N_alpha+N_beta))

    grad[:, :N_alpha] = np.array(h) @ np.array(X)
    grad[:, N_alpha:] = np.array(h) @ np.array(Y)

    return np.array(grad)


def aux_grad_two(double[:, :] W, double[:, :] wwt,
                 double[:, :] zzt, double[:] g):
    """Compute the two-electron gradient.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.

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


def gradtwo(double[:, :] X, double[:, :] Y,
            double[:, :] xxt, double[:, :] yyt,
            double[:] g):
    """Build the two-electron gradient matrix.

    Parameters:
    -----------
    X (2D np.ndarray of dimension (n, N_alpha))
        The spin alpha part of the Slater determinant. n is the size of the
        basis set and N_alpha the number of spin alpha orbitals in the
        Slater determinant.
    
    Y (2D np.ndarray of dimension (n, N_beta))
        The spin beta part of the Slater determinant. n is the size of the
        basis set and N_beta the number of spin beta orbitals in the
        Slater determinant.
    
    xxt (2D np.ndarray of dimension (n, n))
        Density matrix of the spin alpha part, ie, X @ X.T.

    yyt (2D np.ndarray of dimension (n, n))
        Density matrix, but for spin beta (Y @ Y.T).

    g (1D np.ndarray)
        Two-electron integral "matrix".

    Returns:
    --------
    A 2D np.ndarray of dimension (n, N_alpha+N_beta).
    """
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double[:, :] grad = np.zeros((n, N_alpha+N_beta))

    grad[:, :N_alpha] = aux_grad_two(N_alpha, n, X, xxt, yyt, g)
    grad[:, N_alpha:] = aux_grad_two(N_beta, n, Y, yyt, xxt, g)

    return np.array(grad)


def gradient_three(double[:, :] W, double[:, :] Z,
                   double[:, :] wwt, double[:, :] zzt,
                   double[:, :] h, double[:, :, :] g):
    """Compute gradient of the energy.

    Parameters:
    -----------
    W (2D np.ndarray of dimension (n, M))
        One of the parts of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.

    Z (2D np.ndarray of dimension (n, M))
        The other part of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.
    
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
        caller). n is the size of the basis set and M is N_alpha or N_beta.

    Z (2D np.ndarray of dimension (n, M))
        The other part of the Slater determinant (alpha or beta, depend of the
        caller). n is the size of the basis set and M is N_alpha or N_beta.
    
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
