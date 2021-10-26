""" Core functions to optimise the distance to the CC manifold

The term <\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>


term2_diag terms




term2 terms


"""
import numpy as np
import cython

from coupled_cluster.exc_on_string cimport exc_on_string, annihilates
from wave_functions.strings_rev_lexical_order cimport get_index
from wave_functions.strings_rev_lexical_order cimport next_str, ini_str
from util.variables import int_dtype


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_diag_a(SingleExc exc,
                         double[:, :] wf,
                         int[:] occ):
    """(alpha->alpha)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef double S = 0.0
    cdef int i_alpha
    cdef int i_beta
    ini_str(occ)
    for i_alpha in range(nstr_alpha):
        next_str(occ)
        if annihilates(exc.i, exc.a, occ):
            continue
        with nogil:
            for i_beta in range(nstr_beta):
                S += wf[i_alpha, i_beta] * wf[i_alpha, i_beta]
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_diag_b(SingleExc exc,
                         double[:, :] wf,
                         int[:] occ):
    """(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef double S = 0.0
    cdef int i_alpha
    cdef int i_beta
    ini_str(occ)
    for i_beta in range(nstr_beta):
        next_str(occ)
        if annihilates(exc.i, exc.a, occ):
            continue
        with nogil:
            for i_alpha in range(nstr_alpha):
                S += wf[i_alpha, i_beta] * wf[i_alpha, i_beta]
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_diag_aa(DoubleExc exc,
                          double[:, :] wf,
                          int[:] occ):
    """(alpha->alpha)(alpha->alpha)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef double S = 0.0
    cdef int i_alpha
    cdef int i_beta
    ini_str(occ)
    for i_alpha in range(nstr_alpha):
        next_str(occ)
        if (annihilates(exc.i, exc.a, occ)
            or annihilates(exc.j, exc.b, occ)):
            continue
        with nogil:
            for i_beta in range(nstr_beta):
                S += wf[i_alpha, i_beta] * wf[i_alpha, i_beta]
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_diag_bb(DoubleExc exc,
                           double[:, :] wf,
                           int[:] occ):
    """(beta->beta)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef double S = 0.0
    cdef int i_alpha
    cdef int i_beta
    ini_str(occ)
    for i_beta in range(nstr_beta):
        next_str(occ)
        if (annihilates(exc.i, exc.a, occ)
            or annihilates(exc.j, exc.b, occ)):
            continue
        with nogil:
            for i_alpha in range(nstr_alpha):
                S += wf[i_alpha, i_beta] * wf[i_alpha, i_beta]
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_diag_ab(DoubleExc exc,
                           double[:, :] wf,
                           int[:] occ_a,
                           int[:] occ_b):
    """(alpha->alpha)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha
    cdef int i_beta
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        if annihilates(exc.i, exc.a, occ_a):
            continue
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            if annihilates(exc.j, exc.b, occ_b):
                continue
            S += wf[i_alpha, i_beta] * wf[i_alpha, i_beta]
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_aa(DoubleExc exc,
                     double[:, :] wf,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ):
    """(alpha->alpha)^+ (alpha->alpha)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha
    cdef int i_beta
    cdef int sign
    cdef double S = 0.0
    ini_str(occ)
    for i_alpha in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.j, exc.b, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.a, exc.i, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_alpha = get_index(exc_occ, string_graph)
        with nogil:
            for i_beta in range(nstr_beta):
                S += (sign
                      * wf[i_exc_alpha, i_beta]
                      * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_bb(DoubleExc exc,
                     double[:, :] wf,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ):
    """(beta->beta)^+ (beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha
    cdef int i_beta, i_exc_beta
    cdef int sign
    cdef double S = 0.0
    ini_str(occ)
    for i_beta in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.j, exc.b, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.a, exc.i, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_beta = get_index(exc_occ, string_graph)
        with nogil:
            for i_alpha in range(nstr_alpha):
                S += (sign
                      * wf[i_alpha, i_exc_beta]
                      * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_aaa(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ):
    """(alpha->alpha)^+ (alpha->alpha)(alpha->alpha)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha
    cdef int i_beta
    cdef int sign
    cdef double S = 0.0
    ini_str(occ)
    for i_alpha in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.j, exc.b, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.k, exc.c, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.a, exc.i, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_alpha = get_index(exc_occ, string_graph)
        with nogil:
            for i_beta in range(nstr_beta):
                S += (sign
                      * wf[i_exc_alpha, i_beta]
                      * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_bbb(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ):
    """(beta->beta)^+ (beta->beta)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha
    cdef int i_beta, i_exc_beta
    cdef int sign
    cdef double S = 0.0
    ini_str(occ)
    for i_beta in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.j, exc.b, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.k, exc.c, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.a, exc.i, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_beta = get_index(exc_occ, string_graph)
        with nogil:
            for i_alpha in range(nstr_beta):
                S += (sign
                      * wf[i_alpha, i_exc_beta]
                      * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_aaaa(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ):
    """{(alpha->alpha)(alpha->alpha)}^+ (alpha->alpha)(alpha->alpha)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha
    cdef int i_beta
    cdef int sign
    cdef double S = 0.0
    ini_str(occ)
    for i_alpha in range(nstr_alpha):
        next_str(occ)
        sign = exc_on_string(exc.k, exc.c, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.l, exc.d, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.b, exc.j, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.a, exc.i, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_alpha = get_index(exc_occ, string_graph)
        with nogil:
            for i_beta in range(nstr_beta):
                S += (sign
                      * wf[i_exc_alpha, i_beta]
                      * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_bbbb(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ):
    """{(beta->beta)(beta->beta)}^+ (beta->beta)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha
    cdef int i_beta, i_exc_beta
    cdef int sign
    cdef double S = 0.0
    ini_str(occ)
    for i_beta in range(nstr_beta):
        next_str(occ)
        sign = exc_on_string(exc.k, exc.c, occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.l, exc.d, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.b, exc.j, exc_occ, exc_occ)
        if sign == 0:
            continue
        sign *= exc_on_string(exc.a, exc.i, exc_occ, exc_occ)
        if sign == 0:
            continue
        i_exc_beta = get_index(exc_occ, string_graph)
        with nogil:
            for i_alpha in range(nstr_alpha):
                S += (sign
                      * wf[i_alpha, i_exc_beta]
                      * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_ab(DoubleExc exc,
                     double[:, :] wf,
                     int[:, :] alpha_string_graph,
                     int[:, :] beta_string_graph,
                     int[:] occ_a,
                     int[:] exc_occ_a,
                     int[:] occ_b,
                     int[:] exc_occ_b):
    """(alpha->alpha)^+ (beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.a, exc.i, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.j, exc.b, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_aab(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(alpha->alpha)^+ (alpha->alpha)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.j, exc.b, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.a, exc.i, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.k, exc.c, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_abb(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(alpha->alpha)^+ (beta->beta)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.a, exc.i, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.j, exc.b, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.k, exc.c, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_aaab(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """{(alpha->alpha)(alpha->alpha)}^+ (alpha->alpha)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.k, exc.c, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.a, exc.i, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.b, exc.j, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.l, exc.d, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_aabb(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """{(alpha->alpha)(alpha->alpha)}^+ (beta->beta)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.a, exc.i, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.b, exc.j, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.k, exc.c, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.l, exc.d, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_abbb(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """{(alpha->alpha)(beta->beta)}^+ (beta->beta)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.a, exc.i, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.b, exc.j, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.k, exc.c, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.l, exc.d, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_baa(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(beta->beta)^+ (alpha->alpha)(alpha->alpha)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.j, exc.b, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.k, exc.c, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.a, exc.i, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_bab(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b):
    """(beta->beta)^+ (alpha->alpha)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.j, exc.b, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.k, exc.c, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.a, exc.i, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_bbab(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """{(beta->beta)(beta->beta)}^+ (alpha->alpha)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.k, exc.c, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.l, exc.d, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.a, exc.i, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.b, exc.j, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef double term2_abab(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b):
    """{(alpha->alpha)(beta->beta)}^+ (alpha->alpha)(beta->beta)"""
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int i_alpha, i_exc_alpha, sign_a
    cdef int i_beta, i_exc_beta, sign_b
    cdef double S = 0.0
    ini_str(occ_a)
    for i_alpha in range(nstr_alpha):
        next_str(occ_a)
        sign_a = exc_on_string(exc.k, exc.c, occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        sign_a *= exc_on_string(exc.a, exc.i, exc_occ_a, exc_occ_a)
        if sign_a == 0:
            continue
        i_exc_alpha = get_index(exc_occ_a, alpha_string_graph)
        ini_str(occ_b)
        for i_beta in range(nstr_beta):
            next_str(occ_b)
            sign_b = exc_on_string(exc.l, exc.d, occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            sign_b *= exc_on_string(exc.b, exc.j, exc_occ_b, exc_occ_b)
            if sign_b == 0:
                continue
            i_exc_beta = get_index(exc_occ_b, beta_string_graph)
            S += (sign_a * sign_b
                  * wf[i_exc_alpha, i_exc_beta]
                  * wf[i_alpha, i_beta])
    return S
