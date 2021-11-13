#cython: profile=True
"""A module with helper functions for Absil's Hartree-Fock."""

cimport cython

import numpy as np
from scipy import linalg
# from cython.parallel import prange


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


def g_eta(double[:, :] W, double[:, :] wwt, double[:, :] zzt,
          double[:, :] etaW, double[:, :] etaWwt, double[:, :] etaZzt,
          double[:, :] h, double[:] g):
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
    cdef int a, b, p, q, s, n = W.shape[0], M = W.shape[1]
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
    tmp, D = np.empty((n, N)), np.empty((n*N, n*N))

    for j in range(N_alpha):
        for i in range(n):
            E[i, j] = 1.0
            gX = g_eta(X, xxt, yyt, E, np.array(E) @ Xtransp, zero, h, g)
            gY = g_eta(Y, yyt, xxt, F, zero, np.array(E) @ Xtransp, h, g)
            tmp[:, :N_alpha] = projX @ (invS @ np.array(gX)
                                        - np.array(E) @ Xtransp @ gradX)
            tmp[:, N_alpha:] = projY @ (invS @ np.array(gY))
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            E[i, j] = 0.0
            col += 1
    for j in range(N_beta):
        for i in range(n):
            F[i, j] = 1.0
            gX = g_eta(X, xxt, yyt, E, zero, np.array(F) @ Ytransp, h, g)
            gY = g_eta(Y, yyt, xxt, F, np.array(F) @ Ytransp, zero, h, g)
            tmp[:, :N_alpha] = projX @ (invS @ np.array(gX))
            tmp[:, N_alpha:] = projY @ (invS @ np.array(gY)
                                        - np.array(F) @ Ytransp @ gradY)
            D[:, col] = np.reshape(tmp, (n * N,), 'F')
            F[i, j] = 0.0
            col += 1

    return D


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

def verifica_grad(double[:, :] X, double[:, :] Y, double[:, :] gradX,
                 double[:, :] gradY, double[:, :] h, double[:] g, double t):
    cdef int i, j, n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double[:, :] M = np.zeros((n, N_alpha + N_beta))
    cdef double[:, :] xxt = X @ np.transpose(X), yyt = Y @ np.transpose(Y)

    # mudei o tamanho do grad, adaptar
    
    for i in range(n):
        for j in range(N_alpha):
            X[i, j] += t
            xxt = X @ np.transpose(X)
            energyp = energy(X, Y, xxt, yyt, h, g)
            X[i, j] -= 2 * t
            xxt = X @ np.transpose(X)
            energym = energy(X, Y, xxt, yyt, h, g)
            M[i, j] = (energyp - energym) / (2 * t) - gradX[i, j]
            X[i, j] += t

    xxt = X @ np.transpose(X)
    for i in range(n):
        for j in range(N_beta):
            Y[i, j] += t
            yyt = Y @ np.transpose(Y)
            energyp = energy(X, Y, xxt, yyt, h, g)
            Y[i, j] -= 2 * t
            yyt = Y @ np.transpose(Y)
            energym = energy(X, Y, xxt, yyt, h, g)
            M[i+n, j+N_alpha] = (energyp - energym) / (2 * t) - gradY[i,j]
            Y[i, j] += t
            
    return np.array(M)

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
                                    aux[i, j] += (W[q, b] * W[s, j]
                                                  * (2 * g[L, a, q] * g[L, s, i]
                                                     - g[L, a, i] * g[L, s, q]
                                                     - g[L, a, s] * g[L, i, q]))
                                    aux[i, j] += (2 * zzt[q, s]
                                                  * g[L, a, i] * g[L, q, s])
                                    aux[i, j] += (wwt[q, s]
                                                  * (2 * g[L, a, i] * g[L, q, s]
                                                     - g[L, a, s] * g[L, q, i]
                                                     - g[L, a, q] * g[L, s, i]))
            for j in range(N_z):
                for i in range(n):
                    for L in range(g.shape[0]):
                        for p in range(n):
                            for q in range(n):
                                aux[i, N_w+j] += (2 * W[p, b] * Z[q, j]
                                                  * g[L, a, p] * g[L, q, i])
            # aux = np.array(projW) @ np.array(invS) @ aux
            hess[:, col] = np.reshape(aux, (n*N,), 'F')
            col += 1

    return hess

def dir_proj(double[:, :] WT, double[:, :] grad, double[:, :] projW):
    if grad.shape[1] < 1:
        return np.empty((0, 0))

    cdef int a, b, n = grad.shape[0], N = grad.shape[1]
    cdef double[:, :] aux = np.array(WT) @ np.array(grad)
    cdef double[:, :] Id = np.eye(n)
    cdef double aux2
    der = np.empty((n * N, n * N))

    for a in range(N):
        for b in range(N):
            aux2 = aux[b, a] / Id[0, 0]
            der[a*n : (a+1)*n, b*n : (b+1)*n] = np.multiply(Id, aux2)

    return np.kron(np.eye(N), np.array(projW)) @ der

def directional_derivative_2(double[:, :] X, double[:, :] Y,
                             double[:, :] xxt, double[:, :] yyt,
                             double[:, :] projX, double[:, :] projY,
                             double[:, :] gradX, double[:, :] gradY,
                             double[:, :] invS, double[:, :] h,
                             double[:, :, :] g):
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
    D = np.empty((n*N, n*N))
    dir_proj_a = np.zeros((n*N, n*N_a))
    dir_proj_b = np.zeros((n*N, n*N_b))

    hess_a = hessian(X, Y, xxt, yyt, h, g, projX, invS)
    hess_b = hessian(Y, X, yyt, xxt, h, g, projY, invS)
    dir_proj_a[:n*N_a, :] = dir_proj(np.transpose(X), gradX, projX)
    dir_proj_b[:n*N_b, :] = dir_proj(np.transpose(Y), gradY, projY)
    D[:, :n*N_a] = hess_a - dir_proj_a
    D[:, n*N_a:] = hess_b - dir_proj_b

    return D

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
