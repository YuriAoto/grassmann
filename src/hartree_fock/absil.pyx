cimport cython

import numpy as np
from scipy import linalg
# from cython.parallel import prange


cdef int getindex(int i, int j, int k, int l):
    cdef int ij, kl, ijkl
    
    ij = j + i * (i + 1) // 2 if i >= j else i + j * (j + 1) // 2
    kl = l + k * (k + 1) // 2 if k >= l else k + l * (l + 1) // 2
    ijkl = (kl + ij * (ij + 1) // 2 if ij >= kl else ij + kl * (kl + 1) // 2)
    
    return ijkl

def grad_one(double[:,:] X, double[:,:] Y, double[:,:] h):
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double[:,:] grad = np.zeros((n, N_alpha + N_beta))
    
    grad[:,:N_alpha] = np.array(h) @ np.array(X)
    grad[:,N_alpha:] = np.array(h) @ np.array(Y)
    
    return np.array(grad)

def aux_grad_two(double[:,:] W, double[:,:] wwt, double[:,:] zzt, double[:] g):
    cdef int a, b, p, q, s, n = W.shape[0], M = W.shape[1]
    cdef double[:,:] grad = np.array((n, M))
    
    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        grad[a,b] += (0.5 * W[p,b] * wwt[q,s]
                                          * (2 * g[getindex(a,p,q,s)]
                                             - g[getindex(a,s,q,p)]
                                             - g[getindex(p,s,q,a)]))
                        grad[a,b] += W[p,b] * zzt[q,s] * g[getindex(a,p,q,s)]

    return grad
                        
def gradtwo(double[:,:] X,
            double[:,:] Y,
            double[:,:] xxt,
            double[:,:] yyt,
            double[:] g):
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double[:,:] grad = np.zeros((n, N_alpha + N_beta))
    
    grad[:,:N_alpha] = aux_grad_two(N_alpha, n, X, xxt, yyt, g)
    grad[:,N_alpha:] = aux_grad_two(N_beta, n, Y, yyt, xxt, g)
    
    return np.array(grad)

def energyfinal(double[:,:] X,
                double[:,:] Y,
                double[:,:] xxt,
                double[:,:] yyt,
                double[:,:] h,
                double[:] g):
    cdef int j, k, p, q, r, s
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double one_elec = 0, two_elec = 0

    for p in range(n):
        for q in range(n):
            one_elec += (xxt[p,q] + yyt[p,q]) * h[p,q]
            for r in range(n):
                for s in range(n):
                    two_elec += (0.5 * (xxt[p,r]*xxt[q,s] + yyt[p,r]*yyt[q,s])
                                 * (g[getindex(p,r,q,s)] - g[getindex(p,s,q,r)]))
                    two_elec += xxt[p,r] * yyt[q,s] * g[getindex(p,r,q,s)]
    
    return one_elec + two_elec

def gradfinal(double[:,:] W,
              double[:,:] Z,
              double[:,:] wwt,
              double[:,:] zzt,
              double[:,:] h,
              double[:] g):
    cdef int a, b, p, q, s, k
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    cdef double[:,:] grad = 2 * np.array(h) @ np.array(W)
    
    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        grad[a,b] += (W[p,b] * zzt[q,s]
                                      * (g[getindex(a,p,q,s)]
                                         + g[getindex(p,a,q,s)]))
                        grad[a,b] += (W[p,b] * wwt[q,s]
                                      * (2 * g[getindex(a,p,q,s)]
                                         - g[getindex(a,s,q,p)]
                                         - g[getindex(p,s,q,a)]))
    return np.array(grad)

def g_eta(double[:,:] W,
          double[:,:] wwt,
          double[:,:] zzt,
          double[:,:] etaW,
          double[:,:] etaWwt,
          double[:,:] etaZzt,
          double[:,:] h,
          double[:] g):
    cdef int a, b, p, q, s, n = W.shape[0], M = W.shape[1]
    cdef double[:,:] grad_eta = 2 * np.array(h) @ np.array(etaW)
    
    for a in range(n):
        for b in range(M):
            for p in range(n):
                for q in range(n):
                    for s in range(n):
                        grad_eta[a,b] += ((etaW[p,b] * wwt[q,s]
                                           + W[p,b] * etaWwt[q,s]
                                           + W[p,b] * etaWwt[s,q])
                                          * (2 * g[getindex(a,p,q,s)]
                                             - g[getindex(a,s,q,p)]
                                             - g[getindex(p,s,q,a)]))
                        grad_eta[a,b] += (2 * (etaW[p,b] * zzt[q,s]
                                               + W[p,b] * etaZzt[q,s]
                                               + W[p,b] * etaZzt[s,q])
                                          * g[getindex(a,p,q,s)])

    return np.array(grad_eta)

def direc_derivative(double[:,:] X,
                     double[:,:] Y,
                     double[:,:] xxt,
                     double[:,:] yyt,
                     double[:,:] projX,
                     double[:,:] projY,
                     double[:,:] gradX,
                     double[:,:] gradY,
                     double[:,:] invS,
                     double[:,:] h,
                     double[:] g):
    cdef int n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef int N = N_alpha + N_beta, col = 0
    cdef double[:,:] Xtransp = np.transpose(X), Ytransp = np.transpose(Y)
    cdef double[:,:] gX = np.empty((n, N_alpha)), gY = np.empty((n, N_beta))
    cdef double[:,:] E = np.zeros((n, N_alpha)), F = np.zeros((n, N_beta))
    cdef double[:,:] zero = np.zeros((n, n))
    tmp, D = np.empty((n, N)), np.empty((n * N, n * N))
    
    for j in range(N_alpha):
        for i in range(n):
            E[i][j] = 1.0
            gX = g_eta(X, xxt, yyt, E, np.array(E) @ Xtransp, zero, h, g)
            gY = g_eta(Y, yyt, xxt, F, zero, np.array(E) @ Xtransp, h, g)
            tmp[:,:N_alpha] = projX @ (invS @ np.array(gX)
                                        - np.array(E) @ Xtransp @ gradX)
            tmp[:,N_alpha:] = projY @ (invS @ np.array(gY))
            D[:,col] = np.reshape(tmp, (n * N,), 'F')
            E[i][j] = 0.0
            col += 1
    for j in range(N_beta):
        for i in range(n):
            F[i][j] = 1.0
            gX = g_eta(X, xxt, yyt, E, zero, np.array(F) @ Ytransp, h, g)
            gY = g_eta(Y, yyt, xxt, F, np.array(F) @ Ytransp, zero, h, g)
            tmp[:,:N_alpha] = projX @ (invS @ np.array(gX))
            tmp[:,N_alpha:] = projY @ (invS @ np.array(gY)
                                        - np.array(F) @ Ytransp @ gradY)
            D[:,col] = np.reshape(tmp, (n * N,), 'F')
            F[i][j] = 0.0
            col += 1
            
    return D

def verifica_g_eta(double[:,:] W,
                   double[:,:] Z,
                   double[:,:] etaW,
                   double[:,:] etaZ,
                   double[:,:] g_eta,
                   double[:,:] h,
                   double[:] g,
                   double t):
    """(grad{f}(X+t\eta) - grad{f}(X-t\eta)) / 2t"""
    cdef int n = W.shape[0], M = W.shape[1], N = Z.shape[1]
    
    tmpW = np.array(W) + t * np.array(etaW)
    tmpZ = np.array(Z) + t * np.array(etaZ)
    ini = gradfinal(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    tmpW -= 2 * t * np.array(etaW)
    tmpZ -= 2 * t * np.array(etaZ)
    ini -= gradfinal(tmpW, tmpZ, tmpW @ tmpW.T, tmpZ @ tmpZ.T, h, g)
    ini /= 2 * t
    
    return ini - g_eta

def verificagrad(double[:,:] X,
                 double[:,:] Y,
                 double[:,:] gradX,
                 double[:,:] gradY,
                 double[:,:] h,
                 double[:] g,
                 double t):
    cdef int i, j, n = X.shape[0], N_alpha = X.shape[1], N_beta = Y.shape[1]
    cdef double[:,:] M = np.zeros((n, N_alpha + N_beta))
    cdef double[:,:] xxt = X @ np.transpose(X), yyt = Y @ np.transpose(Y)

    # mudei o tamanho do grad, adaptar
    
    for i in range(n):
        for j in range(N_alpha):
            X[i,j] += t
            xxt = X @ np.transpose(X)
            energyp = energyfinal(X, Y, xxt, yyt, h, g)
            X[i,j] -= 2 * t
            xxt = X @ np.transpose(X)
            energym = energyfinal(X, Y, xxt, yyt, h, g)
            M[i,j] = (energyp - energym) / (2 * t) - gradX[i,j]
            X[i,j] += t

    xxt = X @ np.transpose(X)
    for i in range(n):
        for j in range(N_beta):
            Y[i,j] += t
            yyt = Y @ np.transpose(Y)
            energyp = energyfinal(X, Y, xxt, yyt, h, g)
            Y[i,j] -= 2 * t
            yyt = Y @ np.transpose(Y)
            energym = energyfinal(X, Y, xxt, yyt, h, g)
            M[i+n,j+N_alpha] = (energyp - energym) / (2 * t) - gradY[i,j]
            Y[i,j] += t
            
    return np.array(M)

def normalize(double[:] v, double[:,:] S):
    return v / np.sqrt(np.transpose(v).T @ S @ v)

def gs(M, double[:,:] S):
    cdef int i, j

    M[:,0] = normalize(M[:,0], S)
    for i in range(1, M.shape[1]):
        Mi = M[:, i]
        for j in range(i):
            Mj = M[:, j]
            t = Mi.T @ S @ Mj
            Mi = Mi - t * Mj
            M[:, i] = normalize(Mi, S)

    return np.array(M)
