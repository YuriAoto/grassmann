"""Cython functions for integrals

"""
import cython
from cython.parallel import prange

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
def from_2e_atomic_to_molecular_cy(double[:] mo_integrals,
                                double[:,:] molecular,
                                double[:] atomic,
                                int n_func):
    cdef int i,j,k,l,ij,kl,ijkl,p,q,r,s,pq,rs,pqrs
    cdef double at_int

    ij=-1
    with nogil:
        for i in range(n_func):
            for j in range(n_func):
                if i < j:
                    continue
                ij += 1
                kl = -1
                for k in range(n_func):
                    for l in range(n_func):
                        if k < l:
                            continue
                        kl += 1
                        if ij < kl:
                            continue
                        ijkl = (kl + ij * (ij + 1) // 2
                                if ij >= kl else
                                ij + kl * (kl + 1) // 2)
                        pq=-1
                        for p in range(n_func):
                            for q in range(n_func):
                                if p < q:
                                    continue
                                pq += 1
                                rs = -1
                                for r in range(n_func):
                                    for s in range(n_func):
                                        if r < s:
                                            continue
                                        rs += 1
                                        if pq < rs:
                                            continue
                                        pqrs = (rs + pq * (pq + 1) // 2
                                                if pq >= rs else
                                                pq + rs * (rs + 1) // 2)
                                        ##TODO: Add F2e case
                                        at_int = atomic[pqrs]
                                        mo_integrals[ijkl]+= molecular[p,i]*molecular[q,j]*molecular[r,k]*molecular[s,l]*at_int
                                        if p != q and r != s and ( q != s or p != r ):
                                            mo_integrals[ijkl]+= molecular[q,i]*molecular[p,j]*molecular[r,k]*molecular[s,l]*at_int
                                            mo_integrals[ijkl]+= molecular[p,i]*molecular[q,j]*molecular[s,k]*molecular[r,l]*at_int
                                            mo_integrals[ijkl]+= molecular[q,i]*molecular[p,j]*molecular[s,k]*molecular[r,l]*at_int
                                            mo_integrals[ijkl]+= molecular[r,i]*molecular[s,j]*molecular[p,k]*molecular[q,l]*at_int
                                            mo_integrals[ijkl]+= molecular[r,i]*molecular[s,j]*molecular[q,k]*molecular[p,l]*at_int
                                            mo_integrals[ijkl]+= molecular[s,i]*molecular[r,j]*molecular[p,k]*molecular[q,l]*at_int
                                            mo_integrals[ijkl]+= molecular[s,i]*molecular[r,j]*molecular[q,k]*molecular[p,l]*at_int
                                        elif p != q and r != s: 
                                            mo_integrals[ijkl]+= molecular[q,i]*molecular[p,j]*molecular[r,k]*molecular[s,l]*at_int
                                            mo_integrals[ijkl]+= molecular[p,i]*molecular[q,j]*molecular[s,k]*molecular[r,l]*at_int
                                            mo_integrals[ijkl]+= molecular[q,i]*molecular[p,j]*molecular[s,k]*molecular[r,l]*at_int
                                        elif p != q:           
                                            mo_integrals[ijkl]+= molecular[q,i]*molecular[p,j]*molecular[r,k]*molecular[s,l]*at_int
                                            mo_integrals[ijkl]+= molecular[r,i]*molecular[s,j]*molecular[p,k]*molecular[q,l]*at_int
                                            mo_integrals[ijkl]+= molecular[r,i]*molecular[s,j]*molecular[q,k]*molecular[p,l]*at_int
                                        elif r != s:          
                                            mo_integrals[ijkl]+= molecular[p,i]*molecular[q,j]*molecular[s,k]*molecular[r,l]*at_int
                                            mo_integrals[ijkl]+= molecular[r,i]*molecular[s,j]*molecular[p,k]*molecular[q,l]*at_int
                                            mo_integrals[ijkl]+= molecular[s,i]*molecular[r,j]*molecular[p,k]*molecular[q,l]*at_int
                                        elif p != r or q != s:
                                            mo_integrals[ijkl]+= molecular[r,i]*molecular[s,j]*molecular[p,k]*molecular[q,l]*at_int

    return mo_integrals


def from_1e_atomic_to_molecular_cy(double[:,:] mo_integrals,
                                double[:,:] molecular,
                                double[:,:] atomic,
                                int n_func):
    cdef int i,j,p,q
    cdef double at_int

    with nogil:
        for i in prange(n_func):
            for j in range(n_func):
                for p in range(n_func):
                    for q in range(n_func):
                        mo_integrals[i,j]+= molecular[p,i]*molecular[q,j]*atomic[p,q]

    return mo_integrals

def from_1e_atomic_to_molecular_sym_cy(double[:] mo_integrals,
                                double[:,:] molecular,
                                double[:] atomic,
                                int n_func):
    cdef int i,j,ij,p,q,pq
    cdef double at_int

    with nogil:
        for i in range(n_func):
            for j in range(n_func):
                if i < j:
                    continue
                ij = i+j*(j+1)//2 ##Check if it's right
                for p in range(n_func):
                    for q in range(n_func):
                        if p < q:
                            continue
                        pq = p+q*(q+1)//2
                        at_int = atomic[pq]
                        mo_integrals[ij]+= molecular[p,i]*molecular[q,j]*atomic[pq]
                        if p != q:
                            mo_integrals[ij]+= molecular[q,i]*molecular[p,j]*atomic[pq]

    return mo_integrals
