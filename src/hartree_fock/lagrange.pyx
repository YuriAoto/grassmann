cimport cython

import numpy as np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def grad_restr(double[:, :] C, double[:, :] S, Jg):
    cdef int j, n = C.shape[0], N = C.shape[1]
    cdef double[:, :] aux = np.transpose(C) @ S
    cdef double[:, :] e_j = np.zeros((N, 1))

    for j in range(N):
        e_j[j] = 1.0
        Jg[:, n*j : n*(j+1)] = np.kron(aux, e_j)
        Jg[N*j : N*(j+1), n*j : n*(j+1)] += aux
        e_j[j] = 0.0

    return Jg
