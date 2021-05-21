cimport cython

import numpy as np
from scipy import linalg

cdef int getindex(int i, int j, int k, int l) nogil:
    cdef int ij, kl, ijkl
    
    ij = j + i * (i + 1) // 2 if i >= j else i + j * (j + 1) // 2
    kl = l + k * (k + 1) // 2 if k >= l else k + l * (l + 1) // 2
    ijkl = (kl + ij * (ij + 1) // 2
            if ij >= kl else
            ij + kl * (kl + 1) // 2)
    
    return ijkl


def energy(int N_alpha, int N_beta, int n, double[:,:] Z, double[:] g, double[:,:] h):
    cdef int p, q, r, s, j, k
    cdef double one_elec = 0, two_elec = 0, energy
    cdef double[:,:] X = Z[:n,:N_alpha]
    cdef double[:,:] Y = Z[n:,N_alpha:]

    for p in range(n):
        for q in range(n):
            for i in range(N_alpha):
                one_elec += X[p,i] * X[q,i] * h[p,q]
            for i in range(N_beta):
                one_elec += Y[p,i] * Y[q,i] * h[p,q]
            for r in range(n):
                for s in range(n):
                    for j in range(N_alpha):
                        for k in range(j+1, N_alpha):
                            two_elec += (X[p,j] * X[q,k] * X[r,j] * X[s,k]
                                          * (g[getindex(p,r,q,s)]
                                             - g[getindex(p,s,q,r)]))
                                    
                        for k in range(N_beta):
                            two_elec += (X[p,j] * Y[q,k] * X[r,j] * Y[s,k]
                                          * (g[getindex(p,r,q,s)]))
                                        
                    for j in range(N_beta):
                        for k in range(j+1,N_beta):
                            two_elec += (Y[p,j] * Y[q,k] * Y[r,j] * Y[s,k]
                                          * (g[getindex(p,r,q,s)]
                                             - g[getindex(p,s,q,r)]))

    energy = one_elec + two_elec
    return energy, one_elec, two_elec

def gradone(int N_alpha, int N_beta, int n, double[:,:] X, double[:,:] Y, double[:,:] h):
    cdef double[:,:] Gh_alpha = np.zeros((n,N_alpha))
    cdef double[:,:] Gh_beta = np.zeros((n,N_beta))
    
    for a in range(n):
        for p in range(n):
            for b in range(N_alpha):
                Gh_alpha[a,b] += X[p,b] * (h[a,p] + h[p,a])
            for b in range(N_beta):
                Gh_beta[a,b] += Y[p,b] * (h[a,p] + h[p,a])

    return np.array(Gh_alpha), np.array(Gh_beta)

def gradonet(int N_alpha, int N_beta, int n, double[:,:] X, double[:,:] Y, double[:,:] h):
    cdef double[:,:] Gh = np.zeros((2*n,N_alpha+N_beta))
    cdef double[:,:] Z = np.zeros((2*n,N_alpha+N_beta))
    Z[:n,:N_alpha] = X
    Z[n:,N_alpha:] = Y
    
    for a in range(2*n):
        for p in range(2*n):
            for b in range(N_alpha+N_beta):
                Gh[a,b] += Z[p,b] * (h[a%n, p%n] + h[p%n, a%n])
                
    return np.array(Gh)

def gradtwo(int N_alpha, int N_beta, int n, double[:,:] X, double[:,:] Y, double[:] g):
    cdef double[:,:] Gg = np.zeros((2*n,N_alpha+N_beta))
    cdef int a, b, q, s, k

    for b in range(N_alpha):
        for a in range(n):
            for q in range(n):
                for s in range(n):
                    for k in range(N_alpha):
                        if k != b:
                            Gg[a,b] += (2 * X[a,b] * X[q,k] * X[s,k]
                                        * (g[getindex(a,a,q,s)]
                                          - g[getindex(a,s,q,a)]))
                            for p in range(n):
                                if p != a:
                                    Gg[a,b] += (X[p,b] * X[q,k] * X[s,k]
                                                * (2 * g[getindex(a,p,q,s)]
                                                   - g[getindex(a,s,q,p)]
                                                   - g[getindex(p,s,q,a)]))
                                    Gg[a+n,b] += (X[p,b] * X[q,k] * X[s,k]
                                                  * (2 * g[getindex(a,p,q,s)]
                                                     - g[getindex(a,s,q,p)]
                                                     - g[getindex(p,s,q,a)]))
                    for k in range(N_beta):
                        Gg[a,b] += (2*X[a,b]*Y[q,k]*Y[s,k]
                                    *(g[getindex(a,a,q,s)]
                                      - g[getindex(a,s,q,a)]))
                        for p in range(n):
                            if p != a:
                                Gg[a,b] += (X[p,b]*Y[q,k]*Y[s,k]
                                              *(2*g[getindex(a,p,q,s)]
                                                - g[getindex(a,s,q,p)]
                                                - g[getindex(p,s,q,a)]))
                                Gg[a+n,b] += (X[p,b]*Y[q,k]*Y[s,k]
                                              *(2*g[getindex(a,p,q,s)]
                                                - g[getindex(a,s,q,p)]
                                                - g[getindex(p,s,q,a)]))

    for b in range(N_beta):
        for a in range(n):
            for q in range(n):
                for s in range(n):
                    for k in range(N_alpha):
                        Gg[a+n,b+N_alpha] += (2*Y[a,b]*X[q,k]*X[s,k]
                                              *(g[getindex(a,a,q,s)]
                                                - g[getindex(a,s,q,a)]))
                        for p in range(n):
                            if p != a:
                                Gg[a+n,b+N_alpha] += (Y[p,b]*X[q,k]*X[s,k]
                                                      *(2*g[getindex(a,p,q,s)]
                                                        - g[getindex(a,s,q,p)]
                                                        - g[getindex(p,s,q,a)]))
                                Gg[a,b+N_alpha] += (Y[p,b]*X[q,k]*X[s,k]
                                                    *(2*g[getindex(a,p,q,s)]
                                                      - g[getindex(a,s,q,p)]
                                                      - g[getindex(p,s,q,a)]))
                    for k in range(N_beta):
                        if k != b:
                            Gg[a+n,b+N_alpha] += (2*Y[a,b]*Y[q,k]*Y[s,k]
                                                  *(g[getindex(a,a,q,s)]
                                                    - g[getindex(a,s,q,a)]))
                            for p in range(n):
                                if p != a:
                                    Gg[a+n,b+N_alpha] += (Y[p,b]*Y[q,k]*Y[s,k]
                                                          *(2*g[getindex(a,p,q,s)]
                                                            - g[getindex(a,s,q,p)]
                                                            - g[getindex(p,s,q,a)]))
                                    Gg[a,b+N_alpha] += (Y[p,b]*Y[q,k]*Y[s,k]
                                                        *(2*g[getindex(a,p,q,s)]
                                                          - g[getindex(a,s,q,p)]
                                                          - g[getindex(p,s,q,a)]))

    return np.array(Gg)

def gradtwot(int N_alpha, int N_beta, int n, double[:,:] X, double[:,:] Y, double[:] g):
    cdef int N = N_alpha + N_beta
    cdef double[:,:] Gg = np.zeros((2*n,N))
    cdef double[:,:] Z = np.zeros((2*n,N))
    Z[:n,:N_alpha] = X
    Z[n:,N_alpha:] = Y

    for a in range(2*n):
        for b in range(N):
            for q in range(2*n):
                for s in range(2*n):
                    for k in range(N):
                        if k != b:
                            Gg[a,b] += (2 * Z[a,b] * Z[q,k] * Z[s,k]
                                        * (g[getindex(a%n, a%n, q%n, s%n)]
                                          - g[getindex(a%n, s%n, q%n, a%n)]))
                            for p in range(2*n):
                                if p != a:
                                    Gg[a,b] += (Z[p,b] * Z[q,k] * Z[s,k]
                                                * (2 * g[getindex(a%n, p%n, q%n, s%n)]
                                                  - g[getindex(a%n, s%n, q%n, p%n)]
                                                  - g[getindex(p%n, s%n, q%n, a%n)]))

    return np.array(Gg)

def verificagradone(int n, int N_alpha, int N_beta, double[:] g, double [:,:] h,
                 double [:,:] grad, double[:,:] Z, double[:,:] C):
    cdef int N = N_alpha + N_beta
    cdef double energyplus, energyminus
    cdef double[:,:] M = np.zeros((2*n,N))
    
    for i in range(2*n):
        for j in range(N):
            Z[i,j] += 0.001
            x, energyplus, z = energy(N_alpha, N_beta, n, Z, g, h)
            Z[i,j] -= 0.002
            x, energyminus, z = energy(N_alpha, N_beta, n, Z, g, h)
            M[i,j] = (energyplus - energyminus) / 0.002 - grad[i,j]
            Z[i,j] += 0.001
            
    return np.array(M)

def verificagrad(int n, int N_alpha, int N_beta, double[:] g, double [:,:] h,
                 double [:,:] grad, double[:,:] Z, double[:,:] C):
    cdef int N = N_alpha + N_beta
    cdef double energyplus, energyminus
    cdef double[:,:] M = np.zeros((2*n,N))
    
    for i in range(2*n):
        for j in range(N):
            Z[i,j] += 0.001
            energyplus, y, z = energy(N_alpha, N_beta, n, Z, g, h)
            Z[i,j] -= 0.002
            energyminus, y, z = energy(N_alpha, N_beta, n, Z, g, h)
            M[i,j] = (energyplus - energyminus) / 0.002 - grad[i,j]
            Z[i,j] += 0.001
            
    return np.array(M)

def printateste(double[:] g, double[:,:] Z):

    print((2*Z[0,0]*(Z[2,1]*Z[2,1]*(g[getindex(0,0,0,0)] - g[getindex(0,0,0,0)])
		     + Z[2,1]*Z[3,1]*(g[getindex(0,0,0,1)] - g[getindex(0,1,0,0)])
		     + Z[3,1]*Z[2,1]*(g[getindex(0,0,1,0)] - g[getindex(0,0,1,0)])
		     + Z[3,1]*Z[3,1]*(g[getindex(0,0,1,1)] - g[getindex(0,1,1,0)]))
           # + Z[0,0]*Z[2,1]*Z[2,1]*(2*g[getindex(0,0,0,0)] - g[getindex(0,0,0,0)] - g[getindex(0,0,0,0)])
           + Z[1,0]*Z[2,1]*Z[2,1]*(2*g[getindex(0,1,0,0)] - g[getindex(0,0,0,1)] - g[getindex(1,0,0,0)])
           # + Z[0,0]*Z[2,1]*Z[3,1]*(2*g[getindex(0,0,0,1)] - g[getindex(0,1,0,0)] - g[getindex(0,1,0,0)])
           + Z[1,0]*Z[2,1]*Z[3,1]*(2*g[getindex(0,1,0,1)] - g[getindex(0,1,0,1)] - g[getindex(1,1,0,0)])
           # + Z[0,0]*Z[3,1]*Z[2,1]*(2*g[getindex(0,0,1,0)] - g[getindex(0,0,1,0)] - g[getindex(0,0,1,0)])
           + Z[1,0]*Z[3,1]*Z[2,1]*(2*g[getindex(0,1,1,0)] - g[getindex(0,0,1,1)] - g[getindex(1,0,1,0)])
           # + Z[0,0]*Z[3,1]*Z[3,1]*(2*g[getindex(0,0,1,1)] - g[getindex(0,1,1,0)] - g[getindex(0,1,1,0)])
           + Z[1,0]*Z[3,1]*Z[3,1]*(2*g[getindex(0,1,1,1)] - g[getindex(0,1,1,1)] - g[getindex(0,1,1,1)])))
