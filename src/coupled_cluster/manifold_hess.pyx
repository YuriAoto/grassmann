"""Core functions to optimise the distance to the CC manifold

The elements of the Hessian,

<\Psi_cc | \tau_\sigma^\dagger \tau_\rho | \Psi_cc>
   -  <\Psi - \Psi_cc | \tau_\sigma \tau_\rho | \Psi_cc>

are calculated here, except for the diagonal elements.
Each function calculates a row of the Hessian matrix, such as:

H[x, x+1:],

The functions are divided in the cases where x corresponds to
the following excitations:

alpha->alpha               calc_H_a
beta->beta                 calc_H_b
alpha,alpha->alpha,alpha   calc_H_aa
beta,beta->beta,beta       calc_H_bb
alpha,beta->alpha,beta     calc_H_ab

The parameters for these functions are:

double [:] H
    The row of the Hessian matrix, from x to the end of the row:
    H[x, x:]
    Note that the diagonal elements, namely the first element of
    this part of the row, the H[x,x], is not calculated here.

SingleExc single_exc or DoubleExc double_exc
    The excitation associated to that row

double[:, :] wf
    The wave function, as a matrix of alpha and beta strings

double[:, :] wf_cc
    The coupled cluster wave function, as a matrix of alpha
    and beta strings

int pos_ini
    The position associated to the excitation, namely, x as above

Similar to min_dist_jac_hess

"""

import numpy as np
import cython

from molecular_geometry.symmetry import irrep_product
from coupled_cluster.manifold_term1 cimport (
    term1_a, term1_b,
    term1_aa, term1_bb,
    term1_aaa, term1_bbb,
    term1_aaaa, term1_bbbb,
    term1_ab,
    term1_aab, term1_abb,
    term1_aaab, term1_aabb, term1_abbb,
    term1_baa, term1_bab,
    term1_bbab, term1_abab
)
from coupled_cluster.manifold_term2 cimport (
    term2_aa, term2_bb,
    term2_aaa, term2_bbb,
    term2_aaaa, term2_bbbb,
    term2_ab,
    term2_aab, term2_abb,
    term2_aaab, term2_aabb, term2_abbb,
    term2_baa, term2_bab,
    term2_bbab, term2_abab
)
from orbitals.occ_orbitals cimport OccOrbital
from util.variables import int_dtype
from util.array_indices cimport n_from_rect
from wave_functions.fci import FCIWaveFunction
from wave_functions.fci cimport FCIWaveFunction
from orbitals.orbital_space cimport OrbitalSpace
from orbitals.orbital_space import OrbitalSpace


cdef int calc_H_a(double [:] H,
                  SingleExc single_exc,
                  FCIWaveFunction wf,
                  FCIWaveFunction wf_cc,
                  int pos_ini) except -1:
    """H[alpha->alpha, alpha->alpha:]"""
    cdef int pos
    pos = calc_H_block_aa(H, single_exc, wf, wf_cc, pos_ini)
    pos += calc_H_block_ab(H[pos:], single_exc, wf, wf_cc)
    pos += calc_H_block_aaa(H[pos:], single_exc, wf, wf_cc)
    pos += calc_H_block_abb(H[pos:], single_exc, wf, wf_cc)
    pos += calc_H_block_aab(H[pos:], single_exc, wf, wf_cc)
    return pos


cdef int calc_H_b(double [:] H,
                  SingleExc single_exc,
                  FCIWaveFunction wf,
                  FCIWaveFunction wf_cc,
                  int pos_ini) except -1:
    """H[beta->beta, beta->beta:]"""
    cdef int pos
    pos = calc_H_block_bb(H, single_exc, wf, wf_cc, pos_ini)
    pos += calc_H_block_baa(H[pos:], single_exc, wf, wf_cc)
    pos += calc_H_block_bbb(H[pos:], single_exc, wf, wf_cc)
    pos += calc_H_block_bab(H[pos:], single_exc, wf, wf_cc)
    return pos


cdef int calc_H_aa(double [:] H,
                   DoubleExc double_exc,
                   FCIWaveFunction wf,
                   FCIWaveFunction wf_cc,
                   int pos_ini) except -1:
    """H[(alpha,alpha)->(alpha,alpha), (alpha,alpha)->(alpha,alpha):]"""
    cdef int pos
    pos = calc_H_block_aaaa(H, double_exc, wf, wf_cc, pos_ini)
    pos += calc_H_block_aabb(H[pos:], double_exc, wf, wf_cc)
    pos += calc_H_block_aaab(H[pos:], double_exc, wf, wf_cc)
    return pos


cdef int calc_H_bb(double [:] H,
                   DoubleExc double_exc,
                   FCIWaveFunction wf,
                   FCIWaveFunction wf_cc,
                   int pos_ini) except -1:
    """H[(beta,beta)->(beta,beta), (beta,beta)->(beta,beta):]"""
    cdef int pos
    pos = calc_H_block_bbbb(H, double_exc, wf, wf_cc, pos_ini)
    pos += calc_H_block_bbab(H[pos:], double_exc, wf, wf_cc)
    return pos


cdef int calc_H_ab(double [:] H,
                   DoubleExc double_exc,
                   FCIWaveFunction wf,
                   FCIWaveFunction wf_cc,
                   int pos_ini) except -1:
    """H[(alpha,beta)->(alpha,beta), (alpha,beta)->(alpha,beta):]"""
    return calc_H_block_abab(H, double_exc, wf, wf_cc, pos_ini)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_aa(double [:] H,
                         SingleExc single_exc,
                         FCIWaveFunction wf,
                         FCIWaveFunction wf_cc,
                         int pos_ini) except -1:
    """H[alpha->alpha, alpha->alpha]"""
    cdef int pos = -pos_ini
    cdef int j, b, irrep, spirrep
    cdef bint occ_differ
    cdef DoubleExc double_exc
    cdef int [:] occ_buff = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    double_exc.i = single_exc.i
    double_exc.a = single_exc.a
    for irrep in range(wf.n_irrep):
        spirrep = irrep + wf.n_irrep
        double_exc.j = wf.orbspace.orbs_before[irrep]
        for j in range(wf.orbspace.corr[spirrep]):
            occ_differ = double_exc.i != double_exc.j
            double_exc.b = wf.orbspace.first_virtual(spirrep)
            for b in range(wf.orbspace.virt[spirrep]):
                if pos > 0:
                    H[pos] = term2_aa(double_exc,
                                      wf_cc.coefficients,
                                      wf.alpha_string_graph,
                                      occ_buff, exc_occ_buff)
                    if occ_differ and double_exc.a != double_exc.b:
                        H[pos] += term1_aa(double_exc,
                                           wf.coefficients,
                                           wf_cc.coefficients,
                                           wf.alpha_string_graph,
                                           occ_buff, exc_occ_buff)
                pos += 1
                double_exc.b += 1
            double_exc.j += 1
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_bb(double [:] H,
                         SingleExc single_exc,
                         FCIWaveFunction wf,
                         FCIWaveFunction wf_cc,
                         int pos_ini) except -1:
    """H[beta->beta, beta->beta]"""
    cdef int pos = -pos_ini
    cdef int j, b, irrep, spirrep
    cdef bint occ_differ
    cdef DoubleExc double_exc
    cdef int [:] occ_buff = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    double_exc.i = single_exc.i
    double_exc.a = single_exc.a
    for irrep in range(wf.n_irrep):
        spirrep = irrep + wf.n_irrep
        double_exc.j = wf.orbspace.orbs_before[irrep]
        for j in range(wf.orbspace.corr[spirrep]):
            occ_differ = double_exc.i != double_exc.j
            double_exc.b = wf.orbspace.first_virtual(spirrep)
            for b in range(wf.orbspace.virt[spirrep]):
                if pos > 0:
                    H[pos] = term2_bb(double_exc,
                                      wf_cc.coefficients,
                                      wf.beta_string_graph,
                                      occ_buff, exc_occ_buff)
                    if occ_differ and double_exc.a != double_exc.b:
                        H[pos] += term1_bb(double_exc,
                                           wf.coefficients,
                                           wf_cc.coefficients,
                                           wf.beta_string_graph,
                                           occ_buff, exc_occ_buff)
                pos += 1
                double_exc.b += 1
            double_exc.j += 1
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_aaa(double [:] H,
                          SingleExc single_exc,
                          FCIWaveFunction wf,
                          FCIWaveFunction wf_cc) except -1:
    """H[alpha->alpha,  (alpha,alpha)->(alpha,alpha)]"""
    cdef int pos = 0, nvirt_1
    cdef int b, b_irrep, b_spirrep
    cdef int c, c_irrep, c_spirrep
    cdef bint occ_differ
    cdef TripleExc triple_exc
    cdef OccOrbital j, k
    cdef int [:] occ_buff = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    triple_exc.i = single_exc.i
    triple_exc.a = single_exc.a
    j = OccOrbital(wf.orbspace, True)
    k = OccOrbital(wf.orbspace, True)
    k.next_()
    triple_exc.j = j.orb
    triple_exc.k = k.orb
    while k.alive:
        occ_differ = (triple_exc.i != triple_exc.j
                      and triple_exc.i != triple_exc.k)
        for b_irrep in range(wf.n_irrep):
            c_irrep = irrep_product[irrep_product[j.spirrep, k.spirrep], b_irrep]
            b_spirrep = b_irrep
            c_spirrep = c_irrep
            triple_exc.b = wf.orbspace.first_virtual(b_spirrep)
            if b_irrep <= c_irrep:
                for b in range(wf.orbspace.virt[b_spirrep]):
                    nvirt_1 = wf.orbspace.virt[b_spirrep] - 1
                    triple_exc.c = wf.orbspace.first_virtual(c_spirrep)
                    for c in range(wf.orbspace.virt[c_spirrep]):
                        if b_irrep < c_irrep or b < c:
                            H[pos] = term2_aaa(triple_exc,
                                               wf_cc.coefficients,
                                               wf.alpha_string_graph,
                                               occ_buff, exc_occ_buff)
                            if (occ_differ
                                and triple_exc.a != triple_exc.b
                                and triple_exc.a != triple_exc.c
                            ):
                                H[pos] += term1_aaa(triple_exc,
                                                    wf.coefficients,
                                                    wf_cc.coefficients,
                                                    wf.alpha_string_graph,
                                                    occ_buff, exc_occ_buff)
                            pos += 1
                        triple_exc.c += 1
                    triple_exc.b += 1
        if j.pos_in_occ == k.pos_in_occ - 1:
            k.next_()
            j.rewind()
            triple_exc.k = k.orb
        else:
            j.next_()
        triple_exc.j = j.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_bbb(double [:] H,
                          SingleExc single_exc,
                          FCIWaveFunction wf,
                          FCIWaveFunction wf_cc) except -1:
    """H[beta->beta,  (beta,beta)->(beta,beta)]"""
    cdef int pos = 0, nvirt_1
    cdef int b, b_irrep, b_spirrep
    cdef int c, c_irrep, c_spirrep
    cdef bint occ_differ
    cdef TripleExc triple_exc
    cdef OccOrbital j, k
    cdef int j_irrep, k_irrep
    cdef int [:] occ_buff = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    triple_exc.i = single_exc.i
    triple_exc.a = single_exc.a
    j = OccOrbital(wf.orbspace, False)
    k = OccOrbital(wf.orbspace, False)
    k.next_()
    triple_exc.j = j.orb
    triple_exc.k = k.orb
    while k.alive:
        j_irrep = j.spirrep - wf.n_irrep
        k_irrep = k.spirrep - wf.n_irrep
        occ_differ = (triple_exc.i != triple_exc.j
                      and triple_exc.i != triple_exc.k)
        for b_irrep in range(wf.n_irrep):
            c_irrep = irrep_product[irrep_product[j_irrep, k_irrep], b_irrep]
            b_spirrep = b_irrep
            c_spirrep = c_irrep
            triple_exc.b = wf.orbspace.first_virtual(b_spirrep)
            if b_irrep <= c_irrep:
                for b in range(wf.orbspace.virt[b_spirrep]):
                    nvirt_1 = wf.orbspace.virt[b_spirrep] - 1
                    triple_exc.c = wf.orbspace.first_virtual(c_spirrep)
                    for c in range(wf.orbspace.virt[c_spirrep]):
                        if b_irrep < c_irrep or b < c:
                            H[pos] = term2_bbb(triple_exc,
                                               wf_cc.coefficients,
                                               wf.beta_string_graph,
                                               occ_buff, exc_occ_buff)
                            if (occ_differ
                                and triple_exc.a != triple_exc.b
                                and triple_exc.a != triple_exc.c
                            ):
                                H[pos] += term1_bbb(triple_exc,
                                                    wf.coefficients,
                                                    wf_cc.coefficients,
                                                    wf.beta_string_graph,
                                                    occ_buff, exc_occ_buff)
                            pos += 1
                        triple_exc.c += 1
                    triple_exc.b += 1
        if j.pos_in_occ == k.pos_in_occ - 1:
            k.next_()
            j.rewind()
            triple_exc.k = k.orb
        else:
            j.next_()
        triple_exc.j = j.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_aaaa(double [:] H,
                           DoubleExc double_exc,
                           FCIWaveFunction wf,
                           FCIWaveFunction wf_cc,
                           int pos_ini) except -1:
    """H[(alpha,alpha)->(alpha,alpha), (alpha,alpha)->(alpha,alpha)]"""
    cdef int pos = -pos_ini
    cdef int c, c_irrep
    cdef int d, d_irrep
    cdef bint occ_differ, cd_same_irrep
    cdef QuadrupleExc quadruple_exc
    cdef OccOrbital k, l
    cdef int [:] occ_buff = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    quadruple_exc.i = double_exc.i
    quadruple_exc.a = double_exc.a
    quadruple_exc.j = double_exc.j
    quadruple_exc.b = double_exc.b
    k = OccOrbital(wf.orbspace, True)
    l = OccOrbital(wf.orbspace, True)
    l.next_()
    quadruple_exc.k = k.orb
    quadruple_exc.l = l.orb
    while l.alive:
        occ_differ = (quadruple_exc.i != quadruple_exc.k
                      and quadruple_exc.i != quadruple_exc.l
                      and quadruple_exc.j != quadruple_exc.k
                      and quadruple_exc.j != quadruple_exc.l)
        for c_irrep in range(wf.n_irrep):
            d_irrep = irrep_product[irrep_product[k.spirrep, l.spirrep], c_irrep]
            quadruple_exc.c = wf.orbspace.first_virtual(c_irrep)
            if c_irrep <= d_irrep:
                for c in range(wf.orbspace.virt[c_irrep]):
                    quadruple_exc.d = wf.orbspace.first_virtual(d_irrep)
                    for d in range(wf.orbspace.virt[d_irrep]):
                        if c_irrep < d_irrep or c < d:
                            if pos > 0:
                                H[pos] = term2_aaaa(quadruple_exc,
                                                    wf_cc.coefficients,
                                                    wf.alpha_string_graph,
                                                    occ_buff, exc_occ_buff)
                                if (occ_differ
                                    and quadruple_exc.a != quadruple_exc.c
                                    and quadruple_exc.a != quadruple_exc.d
                                    and quadruple_exc.b != quadruple_exc.c
                                    and quadruple_exc.b != quadruple_exc.d
                                ):
                                    H[pos] += term1_aaaa(quadruple_exc,
                                                         wf.coefficients,
                                                         wf_cc.coefficients,
                                                         wf.alpha_string_graph,
                                                         occ_buff, exc_occ_buff)
                            pos += 1
                        quadruple_exc.d += 1
                    quadruple_exc.c += 1
        if k.pos_in_occ == l.pos_in_occ - 1:
            l.next_()
            k.rewind()
            quadruple_exc.l = l.orb
        else:
            k.next_()
        quadruple_exc.k = k.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_bbbb(double [:] H,
                           DoubleExc double_exc,
                           FCIWaveFunction wf,
                           FCIWaveFunction wf_cc,
                           int pos_ini) except -1:
    """H[(beta,beta)->(beta,beta),  (beta,beta)->(beta,beta)]"""
    cdef int pos = -pos_ini
    cdef int c, c_irrep, c_spirrep
    cdef int d, d_irrep, d_spirrep
    cdef bint occ_differ
    cdef QuadrupleExc quadruple_exc
    cdef OccOrbital k, l
    cdef int k_irrep, l_irrep
    cdef int [:] occ_buff = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    quadruple_exc.i = double_exc.i
    quadruple_exc.a = double_exc.a
    quadruple_exc.j = double_exc.j
    quadruple_exc.b = double_exc.b
    k = OccOrbital(wf.orbspace, False)
    l = OccOrbital(wf.orbspace, False)
    l.next_()
    quadruple_exc.k = k.orb
    quadruple_exc.l = l.orb
    while l.alive:
        k_irrep = k.spirrep - wf.n_irrep
        l_irrep = l.spirrep - wf.n_irrep
        occ_differ = (quadruple_exc.i != quadruple_exc.k
                      and quadruple_exc.i != quadruple_exc.l
                      and quadruple_exc.j != quadruple_exc.k
                      and quadruple_exc.j != quadruple_exc.l)
        for c_irrep in range(wf.n_irrep):
            d_irrep = irrep_product[irrep_product[k_irrep, l_irrep], c_irrep]
            c_spirrep = c_irrep + wf.n_irrep
            d_spirrep = d_irrep + wf.n_irrep
            quadruple_exc.c = wf.orbspace.first_virtual(c_spirrep)
            if c_irrep <= d_irrep:
                for c in range(wf.orbspace.virt[c_spirrep]):
                    quadruple_exc.d = wf.orbspace.first_virtual(d_spirrep)
                    for d in range(wf.orbspace.virt[d_spirrep]):
                        if c_irrep < d_irrep or c < d:
                            if pos > 0:
                                H[pos] = term2_bbbb(quadruple_exc,
                                                    wf_cc.coefficients,
                                                    wf.beta_string_graph,
                                                    occ_buff, exc_occ_buff)
                                if (occ_differ
                                    and quadruple_exc.a != quadruple_exc.c
                                    and quadruple_exc.a != quadruple_exc.d
                                    and quadruple_exc.b != quadruple_exc.c
                                    and quadruple_exc.b != quadruple_exc.d
                                ):
                                    H[pos] += term1_bbbb(quadruple_exc,
                                                         wf.coefficients,
                                                         wf_cc.coefficients,
                                                         wf.beta_string_graph,
                                                         occ_buff, exc_occ_buff)
                            pos += 1
                        quadruple_exc.d += 1
                    quadruple_exc.c += 1
        if k.pos_in_occ == l.pos_in_occ - 1:
            l.next_()
            k.rewind()
            quadruple_exc.l = l.orb
        else:
            k.next_()
        quadruple_exc.k = k.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_ab(double [:] H,
                         SingleExc single_exc,
                         FCIWaveFunction wf,
                         FCIWaveFunction wf_cc) except -1:
    """H[alpha->alpha, beta->beta]"""
    cdef int pos = 0
    cdef int j, b, irrep, spirrep
    cdef DoubleExc double_exc
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    double_exc.i = single_exc.i
    double_exc.a = single_exc.a
    for irrep in range(wf.n_irrep):
        spirrep = irrep + wf.n_irrep
        double_exc.j = wf.orbspace.orbs_before[irrep]
        for j in range(wf.orbspace.corr[spirrep]):
            double_exc.b = wf.orbspace.first_virtual(spirrep)
            for b in range(wf.orbspace.virt[spirrep]):
                H[pos] = term2_ab(double_exc,
                                  wf_cc.coefficients,
                                  wf.alpha_string_graph,
                                  wf.beta_string_graph,
                                  occ_buff_a, exc_occ_buff_a,
                                  occ_buff_b, exc_occ_buff_b)
                H[pos] += term1_ab(double_exc,
                                   wf.coefficients,
                                   wf_cc.coefficients,
                                   wf.alpha_string_graph,
                                   wf.beta_string_graph,
                                   occ_buff_a, exc_occ_buff_a,
                                   occ_buff_b, exc_occ_buff_b)
                pos += 1
                double_exc.b += 1
            double_exc.j += 1
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_abb(double [:] H,
                          SingleExc single_exc,
                          FCIWaveFunction wf,
                          FCIWaveFunction wf_cc) except -1:
    """H[alpha->alpha,  (beta,beta)->(beta,beta)]"""
    cdef int pos = 0, nvirt_1
    cdef int b, b_irrep, b_spirrep
    cdef int c, c_irrep, c_spirrep
    cdef TripleExc triple_exc
    cdef OccOrbital j, k
    cdef int j_irrep, k_irrep
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    triple_exc.i = single_exc.i
    triple_exc.a = single_exc.a
    j = OccOrbital(wf.orbspace, False)
    k = OccOrbital(wf.orbspace, False)
    k.next_()
    triple_exc.j = j.orb
    triple_exc.k = k.orb
    while k.alive:
        j_irrep = j.spirrep - wf.n_irrep
        k_irrep = k.spirrep - wf.n_irrep
        for b_irrep in range(wf.n_irrep):
            c_irrep = irrep_product[irrep_product[j_irrep, k_irrep], b_irrep]
            b_spirrep = b_irrep + wf.n_irrep
            c_spirrep = c_irrep + wf.n_irrep
            triple_exc.b = wf.orbspace.first_virtual(b_spirrep)
            if b_irrep <= c_irrep:
                for b in range(wf.orbspace.virt[b_spirrep]):
                    triple_exc.c = wf.orbspace.first_virtual(c_spirrep)
                    for c in range(wf.orbspace.virt[c_spirrep]):
                        if b_irrep < c_irrep or b < c:
                            H[pos] = term2_abb(triple_exc,
                                               wf_cc.coefficients,
                                               wf.alpha_string_graph,
                                               wf.beta_string_graph,
                                               occ_buff_a, exc_occ_buff_a,
                                               occ_buff_b, exc_occ_buff_b)
                            H[pos] += term1_abb(triple_exc,
                                                wf.coefficients,
                                                wf_cc.coefficients,
                                                wf.alpha_string_graph,
                                                wf.beta_string_graph,
                                                occ_buff_a, exc_occ_buff_a,
                                                occ_buff_b, exc_occ_buff_b)
                            pos += 1
                        triple_exc.c += 1
                    triple_exc.b += 1
        if j.pos_in_occ == k.pos_in_occ - 1:
            k.next_()
            j.rewind()
            triple_exc.k = k.orb
        else:
            j.next_()
        triple_exc.j = j.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_baa(double [:] H,
                          SingleExc single_exc,
                          FCIWaveFunction wf,
                          FCIWaveFunction wf_cc) except -1:
    """H[beta->beta,  (alpha,alpha)->(alpha,alpha)]"""
    cdef int pos = 0
    cdef int b, b_irrep, b_spirrep
    cdef int c, c_irrep, c_spirrep
    cdef TripleExc triple_exc
    cdef OccOrbital j, k
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    triple_exc.i = single_exc.i
    triple_exc.a = single_exc.a
    j = OccOrbital(wf.orbspace, True)
    k = OccOrbital(wf.orbspace, True)
    k.next_()
    triple_exc.j = j.orb
    triple_exc.k = k.orb
    while k.alive:
        for b_irrep in range(wf.n_irrep):
            c_irrep = irrep_product[irrep_product[j.spirrep, k.spirrep], b_irrep]
            b_spirrep = b_irrep
            c_spirrep = c_irrep
            triple_exc.b = wf.orbspace.first_virtual(b_spirrep)
            if b_irrep <= c_irrep:
                for b in range(wf.orbspace.virt[b_spirrep]):
                    triple_exc.c = wf.orbspace.first_virtual(c_spirrep)
                    for c in range(wf.orbspace.virt[c_spirrep]):
                        if b_irrep < c_irrep or b < c:
                            H[pos] = term2_baa(triple_exc,
                                               wf_cc.coefficients,
                                               wf.alpha_string_graph,
                                               wf.beta_string_graph,
                                               occ_buff_a, exc_occ_buff_a,
                                               occ_buff_b, exc_occ_buff_b)
                            H[pos] += term1_baa(triple_exc,
                                                wf.coefficients,
                                                wf_cc.coefficients,
                                                wf.alpha_string_graph,
                                                wf.beta_string_graph,
                                                occ_buff_a, exc_occ_buff_a,
                                                occ_buff_b, exc_occ_buff_b)
                            pos += 1
                        triple_exc.c += 1
                    triple_exc.b += 1
        if j.pos_in_occ == k.pos_in_occ - 1:
            k.next_()
            j.rewind()
            triple_exc.k = k.orb
        else:
            j.next_()
        triple_exc.j = j.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_aabb(double [:] H,
                           DoubleExc double_exc,
                           FCIWaveFunction wf,
                           FCIWaveFunction wf_cc) except -1:
    """H[(alpha,alpha)->(alpha,alpha),  (beta,beta)->(beta,beta)]"""
    cdef int pos = 0
    cdef int c, c_irrep, c_spirrep
    cdef int d, d_irrep, d_spirrep
    cdef QuadrupleExc quadruple_exc
    cdef OccOrbital k, l
    cdef int k_irrep, l_irrep
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    quadruple_exc.i = double_exc.i
    quadruple_exc.a = double_exc.a
    quadruple_exc.j = double_exc.j
    quadruple_exc.b = double_exc.b
    k = OccOrbital(wf.orbspace, False)
    l = OccOrbital(wf.orbspace, False)
    l.next_()
    quadruple_exc.k = k.orb
    quadruple_exc.l = l.orb
    while l.alive:
        k_irrep = k.spirrep - wf.n_irrep
        l_irrep = l.spirrep - wf.n_irrep
        for c_irrep in range(wf.n_irrep):
            d_irrep = irrep_product[irrep_product[k_irrep, l_irrep], c_irrep]
            c_spirrep = c_irrep + wf.n_irrep
            d_spirrep = d_irrep + wf.n_irrep
            quadruple_exc.c = wf.orbspace.first_virtual(c_spirrep)
            if c_irrep <= d_irrep:
                for c in range(wf.orbspace.virt[c_spirrep]):
                    quadruple_exc.d = wf.orbspace.first_virtual(d_spirrep)
                    for d in range(wf.orbspace.virt[d_spirrep]):
                        if c_irrep < d_irrep or c < d:
                            H[pos] = term2_aabb(quadruple_exc,
                                                wf_cc.coefficients,
                                                wf.alpha_string_graph,
                                                wf.beta_string_graph,
                                                occ_buff_a, exc_occ_buff_a,
                                                occ_buff_b, exc_occ_buff_b)
                            H[pos] += term1_aabb(quadruple_exc,
                                                 wf.coefficients,
                                                 wf_cc.coefficients,
                                                 wf.alpha_string_graph,
                                                 wf.beta_string_graph,
                                                 occ_buff_a, exc_occ_buff_a,
                                                 occ_buff_b, exc_occ_buff_b)
                            pos += 1
                        quadruple_exc.d += 1
                    quadruple_exc.c += 1
        if k.pos_in_occ == l.pos_in_occ - 1:
            l.next_()
            k.rewind()
            quadruple_exc.l = l.orb
        else:
            k.next_()
        quadruple_exc.k = k.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_aab(double [:] H,
                          SingleExc single_exc,
                          FCIWaveFunction wf,
                          FCIWaveFunction wf_cc) except -1:
    """H[alpha->alpha, (alpha,beta)->(alpha,beta)]"""
    cdef int pos = 0
    cdef int b, b_irrep, b_spirrep
    cdef int c, c_irrep, c_spirrep
    cdef TripleExc triple_exc
    cdef OccOrbital j, k
    cdef int k_irrep
    cdef bint occ_differ
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    triple_exc.i = single_exc.i
    triple_exc.a = single_exc.a
    j = OccOrbital(wf.orbspace, True)
    k = OccOrbital(wf.orbspace, False)
    triple_exc.j = j.orb
    triple_exc.k = k.orb
    while k.alive:
        k_irrep = k.spirrep - wf.n_irrep
        occ_differ = triple_exc.i != triple_exc.j
        for b_irrep in range(wf.n_irrep):
            c_irrep = irrep_product[irrep_product[j.spirrep, k_irrep], b_irrep]
            b_spirrep = b_irrep
            c_spirrep = c_irrep + wf.n_irrep
            triple_exc.b = wf.orbspace.first_virtual(b_spirrep)
            for b in range(wf.orbspace.virt[b_spirrep]):
                triple_exc.c = wf.orbspace.first_virtual(c_spirrep)
                for c in range(wf.orbspace.virt[c_spirrep]):
                    H[pos] = term2_aab(triple_exc,
                                       wf_cc.coefficients,
                                       wf.alpha_string_graph,
                                       wf.beta_string_graph,
                                       occ_buff_a, exc_occ_buff_a,
                                       occ_buff_b, exc_occ_buff_b)
                    if occ_differ and triple_exc.a != triple_exc.b:
                        H[pos] += term1_aab(triple_exc,
                                            wf.coefficients,
                                            wf_cc.coefficients,
                                            wf.alpha_string_graph,
                                            wf.beta_string_graph,
                                            occ_buff_a, exc_occ_buff_a,
                                            occ_buff_b, exc_occ_buff_b)
                    pos += 1
                    triple_exc.c += 1
                triple_exc.b += 1
        j.next_()
        if not j.alive:
            k.next_()
            j.rewind()
            triple_exc.k = k.orb
        triple_exc.j = j.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_bab(double [:] H,
                          SingleExc single_exc,
                          FCIWaveFunction wf,
                          FCIWaveFunction wf_cc) except -1:
    """H[beta->beta,  (alpha,beta)->(alpha,beta)]"""
    cdef int pos = 0
    cdef int b, b_irrep, b_spirrep
    cdef int c, c_irrep, c_spirrep
    cdef TripleExc triple_exc
    cdef OccOrbital j, k
    cdef int k_irrep
    cdef bint occ_differ
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    triple_exc.i = single_exc.i
    triple_exc.a = single_exc.a
    j = OccOrbital(wf.orbspace, True)
    k = OccOrbital(wf.orbspace, False)
    triple_exc.j = j.orb
    triple_exc.k = k.orb
    while k.alive:
        k_irrep = k.spirrep - wf.n_irrep
        occ_differ = triple_exc.i != triple_exc.k
        for b_irrep in range(wf.n_irrep):
            c_irrep = irrep_product[irrep_product[j.spirrep, k_irrep], b_irrep]
            b_spirrep = b_irrep
            c_spirrep = c_irrep + wf.n_irrep
            triple_exc.b = wf.orbspace.first_virtual(b_spirrep)
            for b in range(wf.orbspace.virt[b_spirrep]):
                triple_exc.c = wf.orbspace.first_virtual(c_spirrep)
                for c in range(wf.orbspace.virt[c_spirrep]):
                    H[pos] = term2_bab(triple_exc,
                                       wf_cc.coefficients,
                                       wf.alpha_string_graph,
                                       wf.beta_string_graph,
                                       occ_buff_a, exc_occ_buff_a,
                                       occ_buff_b, exc_occ_buff_b)
                    if occ_differ and triple_exc.a != triple_exc.c:
                        H[pos] += term1_bab(triple_exc,
                                            wf.coefficients,
                                            wf_cc.coefficients,
                                            wf.alpha_string_graph,
                                            wf.beta_string_graph,
                                            occ_buff_a, exc_occ_buff_a,
                                            occ_buff_b, exc_occ_buff_b)
                    pos += 1
                    triple_exc.c += 1
                triple_exc.b += 1
        j.next_()
        if not j.alive:
            k.next_()
            j.rewind()
            triple_exc.k = k.orb
        triple_exc.j = j.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_aaab(double [:] H,
                           DoubleExc double_exc,
                           FCIWaveFunction wf,
                           FCIWaveFunction wf_cc) except -1:
    """H[(alpha,alpha)->(alpha,alpha),  (alpha,beta)->(alpha,beta)]"""
    cdef int pos = 0
    cdef int c, c_irrep, c_spirrep
    cdef int d, d_irrep, d_spirrep
    cdef bint occ_differ
    cdef QuadrupleExc quadruple_exc
    cdef OccOrbital k, l
    cdef int l_irrep
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    quadruple_exc.i = double_exc.i
    quadruple_exc.a = double_exc.a
    quadruple_exc.j = double_exc.j
    quadruple_exc.b = double_exc.b
    k = OccOrbital(wf.orbspace, True)
    l = OccOrbital(wf.orbspace, False)
    quadruple_exc.k = k.orb
    quadruple_exc.l = l.orb
    while l.alive:
        l_irrep = l.spirrep - wf.n_irrep
        occ_differ = (quadruple_exc.i != quadruple_exc.k
                      and quadruple_exc.j != quadruple_exc.k)
        for c_irrep in range(wf.n_irrep):
            d_irrep = irrep_product[irrep_product[k.spirrep, l_irrep], c_irrep]
            c_spirrep = c_irrep
            d_spirrep = d_irrep + wf.n_irrep
            quadruple_exc.c = wf.orbspace.first_virtual(c_spirrep)
            for c in range(wf.orbspace.virt[c_spirrep]):
                quadruple_exc.d = wf.orbspace.first_virtual(d_spirrep)
                for d in range(wf.orbspace.virt[d_spirrep]):
                    H[pos] = term2_aaab(quadruple_exc,
                                        wf_cc.coefficients,
                                        wf.alpha_string_graph,
                                        wf.beta_string_graph,
                                        occ_buff_a, exc_occ_buff_a,
                                        occ_buff_b, exc_occ_buff_b)
                    if (occ_differ
                        and quadruple_exc.a != quadruple_exc.c
                        and quadruple_exc.b != quadruple_exc.c
                    ):
                        H[pos] += term1_aaab(quadruple_exc,
                                             wf.coefficients,
                                             wf_cc.coefficients,
                                             wf.alpha_string_graph,
                                             wf.beta_string_graph,
                                             occ_buff_a, exc_occ_buff_a,
                                             occ_buff_b, exc_occ_buff_b)
                    pos += 1
                    quadruple_exc.d += 1
                quadruple_exc.c += 1
        k.next_()
        if not k.alive:
            l.next_()
            k.rewind()
            quadruple_exc.l = l.orb
        quadruple_exc.k = k.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_bbab(double [:] H,
                           DoubleExc double_exc,
                           FCIWaveFunction wf,
                           FCIWaveFunction wf_cc) except -1:
    """H[(beta,beta)->(beta,beta),  (alpha,beta)->(alpha,beta)]"""
    cdef int pos = 0
    cdef int c, c_irrep, c_spirrep
    cdef int d, d_irrep, d_spirrep
    cdef bint occ_differ
    cdef QuadrupleExc quadruple_exc
    cdef OccOrbital k, l
    cdef int l_irrep
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    quadruple_exc.i = double_exc.i
    quadruple_exc.a = double_exc.a
    quadruple_exc.j = double_exc.j
    quadruple_exc.b = double_exc.b
    k = OccOrbital(wf.orbspace, True)
    l = OccOrbital(wf.orbspace, False)
    quadruple_exc.k = k.orb
    quadruple_exc.l = l.orb
    while l.alive:
        l_irrep = l.spirrep - wf.n_irrep
        occ_differ = (quadruple_exc.i != quadruple_exc.l
                      and quadruple_exc.j != quadruple_exc.l)
        for c_irrep in range(wf.n_irrep):
            d_irrep = irrep_product[irrep_product[k.spirrep, l_irrep], c_irrep]
            c_spirrep = c_irrep
            d_spirrep = d_irrep + wf.n_irrep
            quadruple_exc.c = wf.orbspace.first_virtual(c_spirrep)
            for c in range(wf.orbspace.virt[c_spirrep]):
                quadruple_exc.d = wf.orbspace.first_virtual(d_spirrep)
                for d in range(wf.orbspace.virt[d_spirrep]):
                    H[pos] = term2_bbab(quadruple_exc,
                                        wf_cc.coefficients,
                                        wf.alpha_string_graph,
                                        wf.beta_string_graph,
                                        occ_buff_a, exc_occ_buff_a,
                                        occ_buff_b, exc_occ_buff_b)
                    if (occ_differ
                        and quadruple_exc.a != quadruple_exc.d
                        and quadruple_exc.b != quadruple_exc.d
                    ):
                        H[pos] += term1_bbab(quadruple_exc,
                                             wf.coefficients,
                                             wf_cc.coefficients,
                                             wf.alpha_string_graph,
                                             wf.beta_string_graph,
                                             occ_buff_a, exc_occ_buff_a,
                                             occ_buff_b, exc_occ_buff_b)
                    pos += 1
                    quadruple_exc.d += 1
                quadruple_exc.c += 1
        k.next_()
        if not k.alive:
            l.next_()
            k.rewind()
            quadruple_exc.l = l.orb
        quadruple_exc.k = k.orb
    return pos


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef int calc_H_block_abab(double [:] H,
                           DoubleExc double_exc,
                           FCIWaveFunction wf,
                           FCIWaveFunction wf_cc,
                           int pos_ini) except -1:
    """H[(alpha,beta)->(alpha,beta), (alpha,beta)->(alpha,beta)]"""
    cdef int pos = -pos_ini
    cdef int c, c_irrep, c_spirrep
    cdef int d, d_irrep, d_spirrep
    cdef bint occ_differ
    cdef QuadrupleExc quadruple_exc
    cdef OccOrbital k, l
    cdef int l_irrep
    cdef int [:] occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_a = np.empty(wf.alpha_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    cdef int [:] exc_occ_buff_b = np.empty(wf.beta_string_graph.shape[1], dtype=int_dtype)
    quadruple_exc.i = double_exc.i
    quadruple_exc.a = double_exc.a
    quadruple_exc.j = double_exc.j
    quadruple_exc.b = double_exc.b
    k = OccOrbital(wf.orbspace, True)
    l = OccOrbital(wf.orbspace, False)
    quadruple_exc.k = k.orb
    quadruple_exc.l = l.orb
    while l.alive:
        l_irrep = l.spirrep - wf.n_irrep
        occ_differ = (quadruple_exc.i != quadruple_exc.k
                      and quadruple_exc.j != quadruple_exc.l)
        for c_irrep in range(wf.n_irrep):
            d_irrep = irrep_product[irrep_product[k.spirrep, l_irrep], c_irrep]
            c_spirrep = c_irrep
            d_spirrep = d_irrep + wf.n_irrep
            quadruple_exc.c = wf.orbspace.first_virtual(c_spirrep)
            for c in range(wf.orbspace.virt[c_spirrep]):
                quadruple_exc.d = wf.orbspace.first_virtual(d_spirrep)
                for d in range(wf.orbspace.virt[d_spirrep]):
                    if pos > 0:
                        H[pos] = term2_abab(quadruple_exc,
                                            wf_cc.coefficients,
                                            wf.alpha_string_graph,
                                            wf.beta_string_graph,
                                            occ_buff_a, exc_occ_buff_a,
                                            occ_buff_b, exc_occ_buff_b)
                        if (occ_differ
                            and quadruple_exc.a != quadruple_exc.c
                            and quadruple_exc.b != quadruple_exc.d
                        ):
                            H[pos] += term1_abab(quadruple_exc,
                                                 wf.coefficients,
                                                 wf_cc.coefficients,
                                                 wf.alpha_string_graph,
                                                 wf.beta_string_graph,
                                                 occ_buff_a, exc_occ_buff_a,
                                                 occ_buff_b, exc_occ_buff_b)
                    pos += 1
                    quadruple_exc.d += 1
                quadruple_exc.c += 1
        k.next_()
        if not k.alive:
            l.next_()
            k.rewind()
            quadruple_exc.l = l.orb
        quadruple_exc.k = k.orb
    return pos
