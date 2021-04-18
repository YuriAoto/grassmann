""" Function to optimise distance to the CC manifold

"""
import cython
import numpy as np

from orbitals.occ_orbitals cimport OccOrbital
# from orbitals.occ_orbitals import OccOrbital
from util.array_indices cimport n_from_rect
from util.variables import int_dtype
from molecular_geometry.symmetry import irrep_product
from coupled_cluster.manifold_util cimport SingleExc, DoubleExc
from coupled_cluster.manifold_term1 cimport (
    term1_a, term1_b, term1_aa, term1_bb, term1_ab)
from coupled_cluster.manifold_term2 cimport (
    term2_diag_a, term2_diag_b, term2_diag_aa, term2_diag_bb, term2_diag_ab)
from coupled_cluster.manifold_hess cimport (
    calc_H_a, calc_H_b, calc_H_aa, calc_H_bb, calc_H_ab)
from wave_functions.fci import FCIWaveFunction


def min_dist_jac_hess(double[:, :] wf,
                      double[:, :] wf_cc,
                      int n_ampl,
                      int[:] orbs_before,
                      int[:] corr_orb,
                      int[:] virt_orb,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      bint diag_hess,
                      level='SD'):
    """Calculate the Jacobian and Hessian
    
    Parameters:
    -----------
    wf
        The wave function, as a matrix of alpha and beta strings
    
    wf_cc
        The coupled cluster wave function, as a matrix of alpha
        and beta strings
    
    n_ampl
        The total number of amplitudes
    
    orbs_before
        The number of orbitals before each irreducible representation,
        in the order "all orbitals of first irrep, then all orbitals of
        second irrep, ...".
        
        It must have one element more than the number of irreps
        its size is used to get the n_irrep!
        
        Thus, orbs_before[0] = 0
              orbs_before[1] = total number of orbitals in irrep 0
                                (occ and virt)
              orbs_before[2] = total number of orbitals in irrep 0 and 1
                                (occ and virt)
    
    corr_orb
        The number of correlatad orbitals in each spirrep.
        Its size must be twice n_irrep = orbs_before.size - 1
        Currently, the wave function is assumed to be of unrestricted type
    
    virt_orb
        The number of orbitals in the external space (number of virtuals)
        for each spirrep
        Its size must be twice n_irrep = orbs_before.size - 1
        Currently, the wave function is assumed to be of unrestricted type
    
    alpha_string_graph
        The graph to obtain the alpha strings (associated to the first
        index of wf and wf_cc) in the reverse lexical order
    
    beta_string_graph
        The graph to obtain the beta strings (associated to the second
        index of wf and wf_cc) in the reverse lexical order
    
    diag_hess
        If True, use approximate diagonal Hessian
    
    level (str, optional, default='SD')
        Coupled cluster level
    
    Return:
    -------
    If diag_hess:
        The z vector (that updates the amplitudes),
        and the norm of the Jacobian
    else:
        The Jacobian and the Hessian matrices
    
    """
    cdef double[:] J
    cdef double[:, :] H
    cdef int pos = 0, pos2 = 0, nvirt_1 = 0
    cdef int n_alpha, n_beta
    cdef int n_irrep, spirrep, irrep, a_irrep, b_irrep, a, b, ii, jj
    cdef SingleExc single_exc
    cdef DoubleExc double_exc
    cdef int[8] pos_ini  # assuming no more than 8 irreps in a point group!
    cdef int pos_ini_exc_type
    cdef OccOrbital i, j
    cdef int[:] occ_buff_a = np.empty(alpha_string_graph.shape[1],
                                      dtype=int_dtype)
    cdef int[:] exc_occ_buff_a = np.empty(alpha_string_graph.shape[1],
                                          dtype=int_dtype)
    cdef int[:] occ_buff_b = np.empty(beta_string_graph.shape[1],
                                      dtype=int_dtype)
    cdef int[:] exc_occ_buff_b = np.empty(beta_string_graph.shape[1],
                                          dtype=int_dtype)
    J = np.zeros(n_ampl)
    if diag_hess:
        H = np.zeros((1, 1))
    else:
        H = np.zeros((n_ampl, n_ampl))
    n_alpha = alpha_string_graph.shape[1]
    n_beta = beta_string_graph.shape[1]
    n_irrep = orbs_before.size - 1
    if level == 'SD':
        # --- alpha -> alpha
        for irrep in range(n_irrep):
            spirrep = irrep
            single_exc.i = orbs_before[irrep]
            for ii in range(corr_orb[spirrep]):
                single_exc.a = orbs_before[irrep] + corr_orb[spirrep]
                for a in range(virt_orb[spirrep]):
                    J[pos] = term1_a(single_exc,
                                     wf,
                                     wf_cc,
                                     alpha_string_graph,
                                     occ_buff_a, exc_occ_buff_a)
                    diag = term2_diag_a(single_exc,
                                        wf_cc,
                                        occ_buff_a)
                    if diag_hess:
                        H[0, 0] += J[pos]**2
                        J[pos] /= diag
                    else:
                        H[pos, pos] = diag
                        calc_H_a(H[pos, pos:],
                                 single_exc,
                                 wf,
                                 wf_cc,
                                 pos,
                                 n_irrep,
                                 orbs_before,
                                 corr_orb,
                                 virt_orb,
                                 alpha_string_graph,
                                 beta_string_graph)
                    pos += 1
                    single_exc.a += 1
                single_exc.i += 1
        # --- beta -> beta
        pos_ini_exc_type = pos
        for irrep in range(n_irrep):
            spirrep = irrep + n_irrep
            single_exc.i = orbs_before[irrep]
            for ii in range(corr_orb[spirrep]):
                single_exc.a = orbs_before[irrep] + corr_orb[spirrep]
                for a in range(virt_orb[spirrep]):
                    J[pos] = term1_b(single_exc,
                                     wf,
                                     wf_cc,
                                     beta_string_graph,
                                     occ_buff_b, exc_occ_buff_b)
                    diag = term2_diag_b(single_exc,
                                        wf_cc,
                                        occ_buff_b)
                    if diag_hess:
                        H[0, 0] += J[pos]**2
                        J[pos] /= diag
                    else:
                        H[pos, pos] = diag
                        calc_H_b(H[pos, pos:],
                                 single_exc,
                                 wf,
                                 wf_cc,
                                 pos - pos_ini_exc_type,
                                 n_irrep,
                                 orbs_before,
                                 corr_orb,
                                 virt_orb,
                                 alpha_string_graph,
                                 beta_string_graph)
                    pos += 1
                    single_exc.a += 1
                single_exc.i += 1
    # --- alpha, alpha -> alpha, alpha
    pos_ini_exc_type = pos
    i = OccOrbital(corr_orb, orbs_before, True)
    j = OccOrbital(corr_orb, orbs_before, True)
    j.next_()
    double_exc.i = i.orb
    double_exc.j = j.orb
    while j.alive:
        for a_irrep in range(n_irrep):
            pos_ini[a_irrep] = pos
            b_irrep = irrep_product[
                irrep_product[i.spirrep, j.spirrep], a_irrep]
            a_spirrep = a_irrep
            b_spirrep = b_irrep
            double_exc.a = (orbs_before[a_irrep]
                            + corr_orb[a_spirrep])
            if a_irrep <= b_irrep:
                for a in range(virt_orb[a_spirrep]):
                    nvirt_1 = virt_orb[a_spirrep] - 1
                    double_exc.b = orbs_before[b_irrep] + corr_orb[b_spirrep]
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or a < b:
                            J[pos] = term1_aa(double_exc,
                                              wf,
                                              wf_cc,
                                              alpha_string_graph,
                                              occ_buff_a, exc_occ_buff_a)
                            diag = term2_diag_aa(double_exc, wf_cc, occ_buff_a)
                            if diag_hess:
                                H[0, 0] += J[pos]**2
                                J[pos] /= diag
                            else:
                                H[pos, pos] = diag
                                calc_H_aa(H[pos, pos:],
                                          double_exc,
                                          wf,
                                          wf_cc,
                                          pos - pos_ini_exc_type,
                                          n_irrep,
                                          orbs_before,
                                          corr_orb,
                                          virt_orb,
                                          alpha_string_graph,
                                          beta_string_graph)
                        elif a > b:
                            J[pos] = -J[pos - (a-b)*nvirt_1]
                            if not diag_hess:
                                for pos2 in range(pos+1, n_ampl):
                                    H[pos, pos2] = -H[pos - (a-b)*nvirt_1, pos2]
                        pos += 1
                        double_exc.b += 1
                    double_exc.a += 1
            else:  # and a_irrep > b_irrep
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        J[pos] = -J[pos_ini[b_irrep]
                                    + n_from_rect(b, a, virt_orb[a_spirrep])]
                        pos += 1
        if i.pos_in_occ == j.pos_in_occ - 1:
            j.next_()
            i.rewind()
            double_exc.j = j.orb
        else:
            i.next_()
        double_exc.i = i.orb
    # --- beta, beta -> beta, beta
    pos_ini_exc_type = pos
    i = OccOrbital(corr_orb, orbs_before, False)
    j = OccOrbital(corr_orb, orbs_before, False)
    j.next_()
    double_exc.i = i.orb
    double_exc.j = j.orb
    while j.alive:
        for a_irrep in range(n_irrep):
            pos_ini[a_irrep] = pos
            b_irrep = irrep_product[
                irrep_product[i.spirrep - n_irrep,
                              j.spirrep - n_irrep], a_irrep]
            a_spirrep = a_irrep + n_irrep
            b_spirrep = b_irrep + n_irrep
            double_exc.a = (orbs_before[a_irrep]
                            + corr_orb[a_spirrep])
            if a_irrep <= b_irrep:
                for a in range(virt_orb[a_spirrep]):
                    nvirt_1 = virt_orb[a_spirrep] - 1
                    double_exc.b = orbs_before[b_irrep] + corr_orb[b_spirrep]
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or a < b:
                            J[pos] = term1_bb(double_exc,
                                              wf,
                                              wf_cc,
                                              beta_string_graph,
                                              occ_buff_b, exc_occ_buff_b)
                            diag = term2_diag_bb(double_exc, wf_cc, occ_buff_b)
                            if diag_hess:
                                H[0, 0] += J[pos]**2
                                J[pos] /= diag
                            else:
                                H[pos, pos] = diag
                                calc_H_bb(H[pos, pos:],
                                          double_exc,
                                          wf,
                                          wf_cc,
                                          pos - pos_ini_exc_type,
                                          n_irrep,
                                          orbs_before,
                                          corr_orb,
                                          virt_orb,
                                          alpha_string_graph,
                                          beta_string_graph)
                        elif a > b:
                            J[pos] = -J[pos - (a-b)*nvirt_1]
                            if not diag_hess:
                                for pos2 in range(pos+1, n_ampl):
                                    H[pos, pos2] = -H[pos - (a-b)*nvirt_1, pos2]
                        pos += 1
                        double_exc.b += 1
                    double_exc.a += 1
            else:  # and a_irrep > b_irrep
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        J[pos] = -J[pos_ini[b_irrep]
                                    + n_from_rect(b, a, virt_orb[a_spirrep])]
                        pos += 1
        if i.pos_in_occ == j.pos_in_occ - 1:
            j.next_()
            i.rewind()
            double_exc.j = j.orb
        else:
            i.next_()
        double_exc.i = i.orb
    # --- alpha, beta -> alpha, beta
    pos_ini_exc_type = pos
    i = OccOrbital(corr_orb, orbs_before, True)
    j = OccOrbital(corr_orb, orbs_before, False)
    double_exc.i = i.orb
    double_exc.j = j.orb
    while j.alive:
        for a_irrep in range(n_irrep):
            b_irrep = irrep_product[
                irrep_product[i.spirrep, j.spirrep - n_irrep], a_irrep]
            a_spirrep = a_irrep
            b_spirrep = b_irrep + n_irrep
            double_exc.a = orbs_before[a_irrep] + corr_orb[a_spirrep]
            for a in range(virt_orb[a_spirrep]):
                double_exc.b = orbs_before[b_irrep] + corr_orb[b_spirrep]
                for b in range(virt_orb[b_spirrep]):
                    J[pos] = term1_ab(double_exc,
                                      wf,
                                      wf_cc,
                                      alpha_string_graph,
                                      beta_string_graph,
                                      occ_buff_a, exc_occ_buff_a,
                                      occ_buff_b, exc_occ_buff_b)
                    diag = term2_diag_ab(double_exc, wf_cc, occ_buff_a, occ_buff_b)
                    if diag_hess:
                        H[0, 0] += J[pos]**2
                        J[pos] /= diag
                    else:
                        H[pos, pos] = diag
                        calc_H_ab(H[pos, pos:],
                                  double_exc,
                                  wf,
                                  wf_cc,
                                  pos - pos_ini_exc_type,
                                  n_irrep,
                                  orbs_before,
                                  corr_orb,
                                  virt_orb,
                                  alpha_string_graph,
                                  beta_string_graph)
                    pos += 1
                    double_exc.b += 1
                double_exc.a += 1
        i.next_()
        if not i.alive:
            j.next_()
            i.rewind()
            double_exc.j = j.orb
        double_exc.i = i.orb
    if pos != n_ampl:
        raise Exception(str(pos) + ' = pos != n_ampl = ' + str(n_ampl))
    if not diag_hess:
        for pos in range(n_ampl):
            for pos2 in range(pos):
                H[pos, pos2] = H[pos2, pos]
    return J, H


def min_dist_jac_hess_num(wf,
                          cc_wf,
                          int n_singles,
                          int n_ampl,
                          int[:] orbs_before,
                          int[:] corr_orb,
                          int[:] virt_orb,
                          eps=0.001):
    """Calculate the Jacobian and Hessian numerically
    
    The Jacobian and Hessian are calculated numerically, by the finite
    differences approach
    
    Paramaters:
    -----------
    wf (FCIWaveFunction)
        The wave function whose distance to the CC manifold is to be measured
    
    cc_wf (IntermNormWaveFunction)
        The wave function at the CC manifold, where the jacobian will
        be calculated
    
    eps (float, optional, default=0.001)
        The step to be used in the finite differences approximation to the
        derivative:
        df(x0)/dx \approx (f(x0+eps) - f(x0-eps))/(2*eps)
    
    Return:
    -------
    Two numpy arrays: one 1D of the same lenght as the amplitudes,
    with the jacobian, and another 2D, square, with the hessian.
    These are exactly the hessian of the function "square of the distance".
    Note that what is returned from min_dist_jac_hess is half of this.
    
    """
    cdef int a_irrep, b_irrep, a_spirrep, b_spirrep
    cdef int nvirt_1 = 0, pos = 0, pos_transp
    cdef int[8] pos_ini  # assuming no more than 8 irreps in a point group!
    cdef double[:] f_p, f_m
    cdef double[:] jac
    cdef double[:, :] hess
    cdef double f, f_pp, f_mm
    cdef int n_irrep_or_0
    wf.set_max_coincidence_orbitals()
    
    def Func(x):
        """The square of distance from x and wf
        
        Parameters:
        -----------
        x (IntermNormWaveFunction)
            The wave function whose distance to wf is to be measured
        
        Observation:
        ------------
        x will be transformed to a FCI wave function with the orbitals
        ordered as maximum coincidence with the reference, as it is wf
        in the main function. Calculations can be carried out too
        with wf having the convention of ordered orbitals. In such case,
        the following line should be added after contructing wf_as_fci:
            wf_as_fci.set_ordered_orbitals()
        """
        wf_as_fci = FCIWaveFunction.from_int_norm(x)
        d = wf.dist_to(wf_as_fci,
                       metric='IN',
                       normalise=False)**2
        return d
    
    f = Func(cc_wf)
    f_p = np.zeros(len(cc_wf))
    f_m = np.zeros(len(cc_wf))
    jac = np.zeros(len(cc_wf))
    hess = np.zeros((len(cc_wf), len(cc_wf)))
    n_irrep = orbs_before.size - 1
    pos = 0
    # --- alpha -> alpha and beta -> beta
    while pos < n_singles:
        cc_wf.amplitudes[pos] += eps
        f_p[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos] -= 2*eps
        f_m[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos] += eps
        pos += 1
    # --- alpha, alpha -> alpha, alpha, and beta, beta -> beta, beta
    for is_alpha in [True, False]:
        i = OccOrbital(corr_orb, orbs_before, is_alpha)
        j = OccOrbital(corr_orb, orbs_before, is_alpha)
        j.next_()
        n_irrep_or_0 = 0 if is_alpha else n_irrep
        while j.alive:
            for a_irrep in range(n_irrep):
                if a_irrep == 0:
                    pos_ini[a_irrep] = pos
                else:
                    b_irrep = irrep_product[
                        irrep_product[i.spirrep - n_irrep_or_0,
                                      j.spirrep - n_irrep_or_0], a_irrep]
                    a_spirrep = a_irrep + n_irrep_or_0
                    b_spirrep = b_irrep + n_irrep_or_0
                    pos_ini[a_irrep] = (pos_ini[a_irrep-1]
                                        + virt_orb[a_spirrep-1]*virt_orb[b_spirrep-1])
            for a_irrep in range(n_irrep):
                b_irrep = irrep_product[
                    irrep_product[i.spirrep - n_irrep_or_0,
                                  j.spirrep - n_irrep_or_0], a_irrep]
                a_spirrep = a_irrep + n_irrep_or_0
                b_spirrep = b_irrep + n_irrep_or_0
                if a_irrep <= b_irrep:
                    for a in range(virt_orb[a_spirrep]):
                        nvirt_1 = virt_orb[a_spirrep] - 1
                        for b in range(virt_orb[b_spirrep]):
                            if a_irrep < b_irrep or a < b:
                                pos_transp = (pos + (b-a)*nvirt_1
                                              if a_irrep == b_irrep else
                                              pos_ini[b_irrep] + n_from_rect(
                                                  b, a, virt_orb[a_spirrep]))
                                cc_wf.amplitudes[pos] += eps
                                cc_wf.amplitudes[pos_transp] -= eps
                                f_p[pos] = Func(cc_wf)
                                cc_wf.amplitudes[pos] -= 2*eps
                                cc_wf.amplitudes[pos_transp] += 2*eps
                                f_m[pos] = Func(cc_wf)
                                cc_wf.amplitudes[pos] += eps
                                cc_wf.amplitudes[pos_transp] -= eps
                            elif a > b:
                                pos_transp = pos - (a-b)*nvirt_1
                                f_p[pos] = f_m[pos_transp]
                                f_m[pos] = f_p[pos_transp]
                            pos += 1
                else:  # and a_irrep > b_irrep
                    for a in range(virt_orb[a_spirrep]):
                        for b in range(virt_orb[b_spirrep]):
                            pos_transp = (pos_ini[b_irrep]
                                          + n_from_rect(
                                              b, a, virt_orb[a_spirrep]))
                            f_p[pos] = f_m[pos_transp]
                            f_m[pos] = f_p[pos_transp]
                            pos += 1
            if i.pos_in_occ == j.pos_in_occ - 1:
                j.next_()
                i.rewind()
            else:
                i.next_()
    # alpha, beta -> alpha, beta
    while pos < n_ampl:
        cc_wf.amplitudes[pos] += eps
        f_p[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos] -= 2*eps
        f_m[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos] += eps
        pos += 1
    if pos != n_ampl:
        raise Exception(str(pos) + ' = pos != n_ampl = ' + str(n_ampl))
    
    # for pos in range(len(cc_wf)):
    #     for pos_2 in range(len(cc_wf)):
    #         if pos == pos_2:
    #             hess[pos, pos] = (f_p[pos]
    #                                     - 2*f
    #                                     + f_m[pos]) / eps**2
    #         elif pos_2 > pos:
    #             cc_wf.amplitudes[pos] += eps
    #             cc_wf.amplitudes[pos_2] += eps
    #             f_pp = Func(cc_wf)
    #             cc_wf.amplitudes[pos] -= 2*eps
    #             cc_wf.amplitudes[pos_2] -= 2*eps
    #             f_mm = Func(cc_wf)
    #             cc_wf.amplitudes[pos] += eps
    #             cc_wf.amplitudes[pos_2] += eps
    #             hess[pos, pos_2] = (f_pp
    #                                       - f_p[pos] - f_p[pos_2]
    #                                       + 2*f
    #                                       - f_m[pos] - f_m[pos_2]
    #                                       + f_mm) / (2 * eps**2)
    #         else:
    #             hess[pos, pos_2] = hess[pos_2, pos]
    for pos in range(len(cc_wf)):
        jac[pos] = (f_p[pos] - f_m[pos])/(2*eps)
    return jac, hess
