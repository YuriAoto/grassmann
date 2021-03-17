"""Core functions to optimise the distance to the CC manifold

The term <\Psi - \Psi_cc | \tau_\rho | \Psi_cc>

Where \tau_\rho may be up to quadruple excitations

The parameters for these functions are:

SingleExc single_exc, DoubleExc double_exc
    TripleExc triple_exc, or QuadrupleExc quadruple_exc
    The excitation

double[:, :] wf
    The wave function, as a matrix of alpha and beta strings

double[:, :] wf_cc
    The coupled cluster wave function, as a matrix of alpha
    and beta strings

int[:, :] alpha_string_graph
int[:, :] beta_string_graph
int[:, :] string_graph
    String graphs, associated to alpha, beta, or the only kind
    of orbitals associated to the excitation

The functions always return a float (double C)

"""
import numpy as np

from util.variables import int_dtype
from wave_functions.strings_rev_lexical_order cimport get_index, next_str, ini_str
from coupled_cluster.exc_on_string cimport exc_on_string


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_a(SingleExc exc,
                    double[:, :] wf,
                    double[:, :] wf_cc,
                    int[:, :] string_graph,
                    int[:] occ,
                    int[:] exc_occ):
    """(alpha->alpha)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_beta):
                S += (sign
                      * (wf_cc[i_exc_occ, i]
                         - wf[i_exc_occ, i])
                      * wf_cc[i_occ, i])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_b(SingleExc exc,
                    double[:, :] wf,
                    double[:, :] wf_cc,
                    int[:, :] string_graph,
                    int[:] occ,
                    int[:] exc_occ):
    """(alpha->alpha)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_alpha):
                S += (sign
                      * (wf_cc[i, i_exc_occ]
                         - wf[i, i_exc_occ])
                      * wf_cc[i, i_occ])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_aa(DoubleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ):
    """(alpha->alpha)(alpha->alpha)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.j, exc.b, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_beta):
                S += (sign
                      * (wf_cc[i_exc_occ, i]
                         - wf[i_exc_occ, i])
                      * wf_cc[i_occ, i])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_bb(DoubleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ):
    """(beta->beta)(beta->beta)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.j, exc.b, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_alpha):
                S += (sign
                      * (wf_cc[i, i_exc_occ]
                         - wf[i, i_exc_occ])
                      * wf_cc[i, i_occ])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_aaa(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ):
    """(alpha->alpha)(alpha->alpha)(alpha->alpha)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.j, exc.b, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.k, exc.c, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_beta):
                S += (sign
                      * (wf_cc[i_exc_occ, i]
                         - wf[i_exc_occ, i])
                      * wf_cc[i_occ, i])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_bbb(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ):
    """(beta->beta)(beta->beta)(beta->beta)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.j, exc.b, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.k, exc.c, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_alpha):
                S += (sign
                      * (wf_cc[i, i_exc_occ]
                         - wf[i, i_exc_occ])
                      * wf_cc[i, i_occ])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_aaaa(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ):
    """(alpha->alpha)(alpha->alpha)(alpha->alpha)(alpha->alpha)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.j, exc.b, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.k, exc.c, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.l, exc.d, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_beta):
                S += (sign
                      * (wf_cc[i_exc_occ, i]
                         - wf[i_exc_occ, i])
                      * wf_cc[i_occ, i])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_bbbb(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ):
    """(beta->beta)(beta->beta)(beta->beta)(beta->beta)"""
    cdef int nel = occ.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ, i_exc_occ, i, sign
    cdef double S = 0.0
    ini_str(occ)
    for i_occ in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.i, exc.a, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.j, exc.b, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.k, exc.c, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.l, exc.d, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_occ = get_index(exc_occ, string_graph)
        with nogil:
            for i in range(nstr_alpha):
                S += (sign
                      * (wf_cc[i, i_exc_occ]
                         - wf[i, i_exc_occ])
                      * wf_cc[i, i_occ])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_ab(DoubleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] alpha_string_graph,
                     int[:, :] beta_string_graph,
                     int[:] occ_a,
                     int[:] exc_occ_a,
                     int[:] occ_b,
                     int[:] exc_occ_b):
    """(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.j, exc.b, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_aab(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(alpha->alpha)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.j, exc.b, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.k, exc.c, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_abb(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(alpha->alpha)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.j, exc.b, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.k, exc.c, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_aaab(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """(alpha->alpha)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.j, exc.b, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.k, exc.c, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.l, exc.d, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_aabb(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """(alpha->alpha)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.j, exc.b, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.k, exc.c, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.l, exc.d, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_abbb(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """(alpha->alpha)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.j, exc.b, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.k, exc.c, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.l, exc.d, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_baa(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(beta->beta)(alpha->alpha)(alpha->alpha)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.j, exc.b, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.k, exc.c, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.i, exc.a, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_bab(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(beta->beta)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.j, exc.b, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.i, exc.a, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.k, exc.c, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_bbab(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """(beta->beta)(beta->beta)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.k, exc.c, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.i, exc.a, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.j, exc.b, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.l, exc.d, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double term1_abab(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """(alpha->alpha)(beta->beta)(alpha->alpha)(beta->beta)"""
    cdef int nalpha = occ_a.shape[0]
    cdef int nbeta = occ_b.shape[0]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_occ_a, i_exc_occ_a, sign_a
    cdef int i_occ_b, i_exc_occ_b, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_occ_a in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.i, exc.a, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.k, exc.c, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_occ_a = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_occ_b in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.j, exc.b, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.l, exc.d, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_occ_b = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * (wf_cc[i_exc_occ_a, i_exc_occ_b]
                     - wf[i_exc_occ_a, i_exc_occ_b])
                  * wf_cc[i_occ_a, i_occ_b])
    return S


