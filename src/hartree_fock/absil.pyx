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
    cdef double[:,:] Gh_alpha = np.zeros((n, N_alpha))
    cdef double[:,:] Gh_beta = np.zeros((n, N_beta))
    cdef double[:,:] Gh = np.zeros((2*n, N_alpha+N_beta))
    
    for a in range(n):
        for p in range(n):
            for b in range(N_alpha):
                Gh_alpha[a,b] += X[p,b] * (h[a,p] + h[p,a])
            for b in range(N_beta):
                Gh_beta[a,b] += Y[p,b] * (h[a,p] + h[p,a])

    Gh[:n,:N_alpha] = Gh_alpha
    Gh[n:,N_alpha:] = Gh_beta
    return np.array(Gh)

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
    cdef double[:,:] Gg_alpha = np.zeros((n, N_alpha))
    cdef double[:,:] Gg_beta = np.zeros((n, N_beta))
    cdef double[:,:] Gg = np.zeros((2*n, N_alpha + N_beta))
    cdef int a, b, q, s, k

    for a in range(n):
        for b in range(N_alpha):
            for q in range(n):
                for s in range(n):
                    for k in range(N_alpha):
                        if k != b:
                            Gg_alpha[a,b] += (2 * X[a,b] * X[q,k] * X[s,k]
                                              * (g[getindex(a, a, q, s)]
                                                 - g[getindex(a, s, q, a)]))
                            for p in range(n):
                                if p != a:
                                    Gg_alpha[a,b] += (X[p,b] * X[q,k] * X[s,k]
                                                      * (2 * g[getindex(a, p, q, s)]
                                                        - g[getindex(a, s, q, p)]
                                                         - g[getindex(p, s, q, a)]))
                    for k in range(N_beta):
                        Gg_alpha[a,b] += (2 * X[a,b] * Y[q,k] * Y[s,k]
                                          * g[getindex(a, a, q, s)])
                        for p in range(n):
                            if p != a:
                                Gg_alpha[a,b] += (2 * X[p,b] * Y[q,k] * Y[s,k]
                                                  * g[getindex(a, p, q, s)])
            
        for b in range(N_beta):
            for q in range(n):
                for s in range(n):
                    for k in range(N_beta):
                        if k != b:
                            Gg_beta[a,b] += (2 * Y[a,b] * Y[q,k] * Y[s,k]
                                             * (g[getindex(a, a, q, s)]
                                                - g[getindex(a, s, q, a)]))
                            for p in range(n):
                                if p != a:
                                    Gg_beta[a,b] += (Y[p,b] * Y[q,k] * Y[s,k]
                                                     * (2 * g[getindex(a, p, q, s)]
                                                        - g[getindex(a, s, q, p)]
                                                        - g[getindex(p, s, q, a)]))
                    for k in range(N_alpha):
                        Gg_beta[a,b] += (2 * Y[a,b] * X[q,k] * X[s,k]
                                         * g[getindex(a, a, q, s)])
                        for p in range(n):
                            if p != a:
                                Gg_beta[a,b] += (2 * Y[p,b] * X[q,k] * X[s,k]
                                                 * g[getindex(a, p, q, s)])

    Gg[:n,:N_alpha] = Gg_alpha
    Gg[n:,N_alpha:] = Gg_beta
    return np.array(Gg)

def directionalderivative(int n, int N_alpha, int N_beta, double[:] g,
                          double [:,:] h, double[:,:] X, double[:,:] Y):
    cdef int N = N_alpha + N_beta
    cdef double[:,:] D = np.zeros((2*n*N, 2*n*N))
    cdef double[:,:] GEij
    
    for col in range(2*n*N):
        GEij = np.zeros((2*n,N))
        for a in range(n):
            for b in range(N_alpha):
                GEij[a,b] += 2 * h[a,a]
                GEij[a+n,b] += 2 * h[a,a]
                for q in range(n):
                    for s in range(n):
                        for k in range(N_alpha):
                            if k != b:
                                GEij[a,b] += (2 * X[q,k] * X[s,k]
                                              * (g[getindex(a, a, q, s)]
                                                 - g[getindex(a, s, q, a)]))
                                GEij[a+n,b] += (2 * X[q,k] * X[s,k]
                                                * g[getindex(a, a, q, s)])
                        for k in range(N_beta):
                            GEij[a,b] += (2 * Y[q,k] * Y[s,k]
                                          * g[getindex(a, a, q, s)])
                            GEij[a+n,b] += (2 * Y[q,k] * Y[s,k]
                                            * (g[getindex(a, a, q, s)]
                                               - g[getindex(a, s, q, a)]))
                for i in range(n):
                    if i != a:
                        GEij[i,b] += h[i,a] + h[a,i]
                        GEij[i+n,b] += h[a,i] + h[i,a]
                        for q in range(0,n):
                            for s in range(0,n):
                                for k in range(N_alpha):
                                    if k != b:
                                        GEij[i,b] += (X[q,k] * X[s,k]
                                                      * (2 * g[getindex(a, i, q, s)]
                                                         - g[getindex(a, s, q, i)]
                                                         - g[getindex(a, q, s, i)]))
                                        GEij[i+n,b] += (2 * X[q,k] * X[s,k]
                                                        * g[getindex(a, i, q, s)]) 
                                for k in range(N_beta):
                                    GEij[i,b] += (2 * Y[q,k] * Y[s,k]
                                                  * g[getindex(a, i, q, s)])
                                    GEij[i+n,b] += (Y[q,k] * Y[s,k]
                                                    * (2 * g[getindex(a, i, q, s)]
                                                       - g[getindex(a, s, q, i)]
                                                       - g[getindex(a, q, s, i)]))
                    for j in range(N_alpha):
                        if j != b:
                            GEij[a,j] += (4 * X[a,b] * X[i,j]
                                          * (g[getindex(a, a, i, i)]
                                             - g[getindex(a, i, i, a)]))
                            for q in range(n):
                                if q != i:
                                    GEij[i,j] += (2 * X[a,b] * X[q,j]
                                                  * (2 * g[getindex(a, a, i, q)]
                                                     - g[getindex(a, q, i, a)]
                                                     - g[getindex(a, i, q, a)]))
                                for p in range(n):
                                    if p != a:
                                        GEij[i,j] += (2 * X[p,b] * X[q,j]
                                                      * (2 * g[getindex(a, p, i, q)]
                                                         - g[getindex(a, q, i, p)]
                                                         - g[getindex(a, i, q, p)]))
                                    GEij[i+n,j] -= (2 * X[p,b] * X[q,j]
                                                    * g[getindex(a, i, q, p)])
                    for j in range(N_beta):
                        GEij[i+n,j+N_alpha] += (4 * X[a,b] * Y[i,j]
                                                * g[getindex(a, a, i, i)])
                        for q in range(n):
                            if q != i:
                                GEij[i+n,j+N_alpha] += (4 * X[a,b] * Y[q,j]
                                                        * g[getindex(a, a, i, q)])
                            for p in range(n):
                                if p != a:
                                    GEij[i+n,j+N_alpha] += (4 * X[p,b] * Y[q,j]
                                                            * (g[getindex(a, p, i, q)]
                                                               - g[getindex(a, p, q, i)]))
                                GEij[i,j+N_alpha] -= (2 * X[p,b] * Y[q,j]
                                                      * g[getindex(a, q, i, p)])
            for b in range(N_beta):
                GEij[a,b+N_alpha] += 2 * h[a,a]
                GEij[a+n,b+N_alpha] += 2 * h[a,a]
                for q in range(n):
                    for s in range(n):
                        for k in range(N_alpha):
                            GEij[a,b+N_alpha] += (2 * X[q,k] * X[s,k]
                                                  * (g[getindex(a, a, q, s)]
                                                     - g[getindex(a, s, q, a)]))
                            GEij[a+n,b+N_alpha] += (2 * X[q,k] * X[s,k]
                                                    * g[getindex(a, a, q, s)])
                        for k in range(N_beta):
                            if k != b:
                                GEij[a,b+N_alpha] += (2 * Y[q,k] * Y[s,k]
                                                      * g[getindex(a, a, q, s)])
                                GEij[a+n,b+N_alpha] += (2 * Y[q,k] * Y[s,k]
                                                       * (g[getindex(a, a, q, s)]
                                                          - g[getindex(a, s, q, a)]))
                for i in range(n):
                    GEij[i+n,b+N_alpha] += h[a,i] + h[i,a]
                    if i != a:
                        GEij[i,b] += h[a,i] + h[i,a]
                        for q in range(n):
                            for s in range(n):
                                for k in range(N_alpha):
                                    GEij[i,b+N_alpha] += (X[q,k] * X[s,k]
                                                          * (2 * g[getindex(a, i, q, s)]
                                                             - g[getindex(a, s, q, i)]
                                                             - g[getindex(a, q, s, i)]))
                                    GEij[i+n,b+N_alpha] += (2 * X[q,k] * X[s,k]
                                                            * g[getindex(a, i, q, s)])
                                for k in range(N_beta):
                                    if k != b:
                                        GEij[i,b+N_alpha] += (2 * Y[q,k] * Y[s,k]
                                                              * g[getindex(a, i, q, s)])
                                        GEij[i+n,b+N_alpha] += (Y[q,k] * Y[s,k]
                                                            * (2 * g[getindex(a, i, q, s)]
                                                               - g[getindex(a, s, q, i)]
                                                               - g[getindex(a, q, s, i)]))
                    for j in range(N_alpha):
                        GEij[i,j] += 4 * Y[a,b] * X[i,j] * g[getindex(a, a, i, i)]
                        for q in range(n):
                            if q != i:
                                GEij[i,j] += 4 * Y[a,b] * X[q,j] * g[getindex(a, a, i, q)]
                            for p in range(n):
                                if p != a:
                                    GEij[i,j] += (2 * Y[p,b] * X[q,j]
                                                  * (g[getindex(a, p, i, q)]
                                                     - g[getindex(a, p, q, i)]))
                                GEij[i+n,j] -= (2 * Y[p,b] * X[q,j]
                                                * g[getindex(a, q, i, p)])
                    for j in range(N_beta):
                        if j != b:
                            GEij[i+n,j+N_alpha] += (4 * Y[a,b] * Y[i,j]
                                                    * (g[getindex(a, a, i, i)]
                                                       - g[getindex(a, i, i, a)]))
                            for q in range(n):
                                if q != i:
                                    GEij[i+n,j+N_alpha] += (2 * Y[a,b] * Y[q,j]
                                                            * (g[getindex(a, a, i, q)]
                                                               - g[getindex(a, q, i, a)]
                                                               - g[getindex(a, i, q, a)]))
                                for p in range(n):
                                    if p != a:
                                        GEij[i+n,j+N_alpha] += (2 * Y[p,b] * Y[q,j]
                                                                * (2 * g[getindex(a, p, i, q)]
                                                                   - g[getindex(a, q, i, p)]
                                                                   - g[getindex(a, i, q, p)]))
                                    GEij[i,j+N_alpha] -= (2 * Y[p,b] * Y[q,j]
                                                          * g[getindex(a, i, q, p)])
        GEij = np.reshape(GEij, (2*n*N,1), 'F')
        D[:,col] = GEij

        return D
                        
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
                 double [:,:] grad, double[:,:] Z):
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
