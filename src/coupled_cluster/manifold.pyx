""" Function to optimise distance to the CC manifold

"""
import cython
import numpy as np

from libc.math cimport sqrt

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
                      double[:] J,
                      double[:, :] H,
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
        It should be with ordered orbitals.
    
    wf_cc
        The coupled cluster wave function, as a matrix of alpha
        and beta strings
        It should be with ordered orbitals.
    
    J
        Will be filled with the Jacobian, or with the z vector (diag_hess=True)
    
    H
        Will be filled with the Hessian, or its first entry with the norm of
        the Jacobian (diag_hess=True)
    
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
    Does not return anything, but the arguments J and H are filled up:
    If diag_hess:
        J recieves the z vector (that updates the amplitudes),
        and H[0,0] the norm of the Jacobian
    else:
        J and H receive the Jacobian and the Hessian matrices
    
    """
    cdef int pos = 0, pos2 = 0
    cdef int n_alpha, n_beta, n_indep_ampl
    cdef int n_irrep, spirrep, irrep, a_irrep, b_irrep, a, b, ii, jj
    cdef SingleExc single_exc
    cdef DoubleExc double_exc
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
    n_indep_ampl = J.shape[0]
    n_alpha = alpha_string_graph.shape[1]
    n_beta = beta_string_graph.shape[1]
    n_irrep = orbs_before.size - 1
    if diag_hess:
        H[0, 0] = 0.0
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
                        J[pos] /= -diag
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
                        J[pos] /= -diag
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
            b_irrep = irrep_product[irrep_product[i.spirrep, j.spirrep], a_irrep]
            a_spirrep = a_irrep
            b_spirrep = b_irrep
            double_exc.a = orbs_before[a_irrep] + corr_orb[a_spirrep]
            if a_irrep <= b_irrep:
                for a in range(virt_orb[a_spirrep]):
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
                                J[pos] /= -diag
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
                            pos += 1
                        double_exc.b += 1
                    double_exc.a += 1
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
            b_irrep = irrep_product[irrep_product[i.spirrep - n_irrep,
                                                  j.spirrep - n_irrep], a_irrep]
            a_spirrep = a_irrep + n_irrep
            b_spirrep = b_irrep + n_irrep
            double_exc.a = orbs_before[a_irrep] + corr_orb[a_spirrep]
            if a_irrep <= b_irrep:
                for a in range(virt_orb[a_spirrep]):
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
                                J[pos] /= -diag
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
                            pos += 1
                        double_exc.b += 1
                    double_exc.a += 1
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
            b_irrep = irrep_product[irrep_product[i.spirrep,
                                                  j.spirrep - n_irrep], a_irrep]
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
                        J[pos] /= -diag
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
    if pos != n_indep_ampl:
        raise Exception(str(pos) + ' = pos != n_indep_ampl = ' + str(n_indep_ampl))
    if diag_hess:
        H[0, 0] = sqrt(H[0, 0])
    else:
        for pos in range(n_indep_ampl):
            for pos2 in range(pos):
                H[pos, pos2] = H[pos2, pos]


cdef inline int _transp(int a,
                        int b,
                        int a_irrep,
                        int b_irrep,
                        int pos_ampl,
                        int a_virt,
                        int b_pos_ini):
    """The position of the transpose of t^ab (t^ba, that is -t^ab)
    
    The position is relative to the "full" amplitudes array, that includes both
    t^ab an t^ba
    
    Parameters:
    -----------
    a, b
        The indices of virtual orbitals
    
    a_irrep, b_irrep,
        Irreps of a and b
    
    pos_ampl
        Position of t_ij^ab in the amplitudes array
    
    a_virt
        Number of virtual orbitals of same spirrep of a
    
    b_pos_ini
        Initial position of irrep associated to b (see _fill_pos_ini)
    """
    return (pos_ampl + (b - a)*(a_virt - 1)
            if a_irrep == b_irrep else
            b_pos_ini + n_from_rect(b, a, a_virt))


cdef _fill_pos_ini(int n_irrep,
                   int i_spirrep,
                   int j_spirrep,
                   int pos_ampl,
                   int n_irrep_or_0,
                   int [:] pos_ini,
                   int [:] virt_orb):
    """Fill in the initial position for each irrep block
    
    The array pos_ini is filled, with the position (in the full
    amplitudes array) of the first element of each irrep for a fixed
    pair of occupied orbitals (not given explicitly)
    
    Paramaters:
    -----------
    n_irrep
        Number of irreps
    
    i_spirrep, j_spirrep
        The spirrep of both occupied orbitals
    
    pos_ampl
        The position of the first amplitude for that pair of occupied
        orbitals
    
    n_irrep_or_0
        n_irrep if beta/beta or 0 if alpha/alpha
    
    pos_ini
        The array that will be filled up with the initial positions
    
    virt_orb
        The number of virtual orbitals for each spirrep
    """
    cdef int a_irrep, b_irrep, a_spirrep, b_spirrep
    for a_irrep in range(n_irrep):
        if a_irrep == 0:
            pos_ini[a_irrep] = pos_ampl
        else:
            b_irrep = irrep_product[
                irrep_product[i_spirrep - n_irrep_or_0,
                              j_spirrep - n_irrep_or_0], a_irrep]
            a_spirrep = a_irrep + n_irrep_or_0
            b_spirrep = b_irrep + n_irrep_or_0
            pos_ini[a_irrep] = (pos_ini[a_irrep - 1]
                                + virt_orb[a_spirrep - 1]
                                * virt_orb[b_spirrep - 1])


cpdef inline (int, int, int) _sp_irreps(int a_irrep,
                                        int i_spirrep,
                                        int j_spirrep,
                                        int n_irrep_or_0):
    """Return b_irrep, a_spirrep and b_spirrep
    
    For a (non-vanishing-by-symmetry) amplitude t_ij^ab,
    return the irrep of b and the spirreps of a and b
    
    Parameters:
    -----------
    a_irrep
        Irrep of orbital a
    
    i_spirrep, j_spirrep
        Spirreps of i and j
    
    n_irrep_or_0
        n_irrep if beta/beta or 0 if alpha/alpha
    
    TODO:
    -----
    This might go well in WaveFunction too
    """
    b_irrep = irrep_product[irrep_product[i_spirrep - n_irrep_or_0,
                                          j_spirrep - n_irrep_or_0], a_irrep]
    return (b_irrep,
            a_irrep + n_irrep_or_0,
            b_irrep + n_irrep_or_0)

def update_indep_amplitudes(double [:] amplitudes,
                            double [:] z,
                            int n_irrep,
                            int n_singles,
                            int n_ampl,
                            int n_indep_ampl,
                            int [:] orbs_before,
                            int [:] corr_orb,
                            int [:] virt_orb):
    """Does amplitudes += z considering only independent amplitudes in z
    
    The amplitudes are updated by z, for the case where the array amplitudes
    stores both t_ij^ab and t_ij^ba for alpha/alpha and beta/beta excitations,
    whereas z store only one (with a<b).
    
    Paramaters:
    -----------
    amplitudes
        The amplitudes as stored in IntermNormWaveFunction, with
        (for alpha/alpha and beta/beta) both t_ij^ab and t_ij^ba = -t_ij^ba
        This array will be changed.
    
    z
        The update, with (for alpha/alpha and beta/beta) only t_ij^ab with a<b
    
    n_irrep
        Number of irreps
    
    n_singles
        Number of singles
    
    n_ampl
        Number of amplitudes
    
    n_indep_ampl
        Number of independent amplitudes
    
    orbs_before, corr_orb, virt_orb
        See min_dist_jac_hess
    
    Return:
    -------
    Does not return anything, but the array amplitudes is updated.
    """
    cdef int pos = 0, pos_ampl
    cdef int a, a_irrep, a_spirrep
    cdef int b, b_irrep, b_spirrep
    cdef int n_irrep_or_0
    cdef bint is_alpha
    cdef int[8] pos_ini
    cdef OccOrbital i, j
    pos = 0
    # singles
    while pos < n_singles:
        amplitudes[pos] += z[pos]
        pos += 1
    pos_ampl = pos
    # alpha/alpha and beta/beta
    for is_alpha in [True, False]:
        i = OccOrbital(corr_orb, orbs_before, is_alpha)
        j = OccOrbital(corr_orb, orbs_before, is_alpha)
        j.next_()
        n_irrep_or_0 = 0 if is_alpha else n_irrep
        while j.alive:
            _fill_pos_ini(n_irrep,
                          i.spirrep,
                          j.spirrep,
                          pos_ampl,
                          n_irrep_or_0,
                          pos_ini,
                          virt_orb)
            for a_irrep in range(n_irrep):
                b_irrep, a_spirrep, b_spirrep = _sp_irreps(a_irrep,
                                                           i.spirrep,
                                                           j.spirrep,
                                                           n_irrep_or_0)
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or (a_irrep == b_irrep and a < b):
                            amplitudes[pos_ampl] += z[pos]
                            pos += 1
                        elif a_irrep > b_irrep or a > b:
                            pos_transp = _transp(a, b, a_irrep, b_irrep,
                                                 pos_ampl,
                                                 virt_orb[a_spirrep],
                                                 pos_ini[b_irrep])
                            amplitudes[pos_ampl] = -amplitudes[pos_transp]
                        pos_ampl += 1
            if i.pos_in_occ == j.pos_in_occ - 1:
                j.next_()
                i.rewind()
            else:
                i.next_()
    # alpha/beta
    while pos < n_indep_ampl:
        amplitudes[pos_ampl] += z[pos]
        pos += 1
        pos_ampl += 1
    if pos != n_indep_ampl:
        raise Exception(str(pos) + ' = pos != n_indep_ampl = ' + str(n_indep_ampl))
    if pos_ampl != n_ampl:
        raise Exception(str(pos_ampl) + ' = pos_ampl != n_ampl = ' + str(n_ampl))


cdef inline double _df2_dxdy(f_pp, f_mm, f_p, f_m, f, pos, pos_2, eps_2_2):
    return (f_pp - f_p[pos] - f_p[pos_2] + 2*f - f_m[pos] - f_m[pos_2] + f_mm) / eps_2_2


cdef _compare_jac(int pos, double [:] J, double [:] Janal):
    """Helper to min_dist_jac_hess_num: raise Exception if entries of Jacobian differ"""
    if abs(Janal[pos] - J[pos]) > 1.0E-6:
        raise Exception('Diff Jacobian: pos = ' + str(pos)
                        + '\n anal: ' + str(Janal[pos])
                        + '\n num:  ' + str(J[pos]))


cdef _compare_hess(int pos, int pos2, double [:, :] H, double [:, :] Hanal):
    """Helper to min_dist_jac_hess_num: raise Exception if entries of Hessian differ"""
    if abs(Hanal[pos, pos2] - H[pos, pos2]) > 1.0E-6:
        raise Exception('Diff Hessian: pos, pos2 = ' + str(pos) + ', ' + str(pos2)
                        + '\n anal: ' + str(Hanal[pos, pos2])
                        + '\n num:  ' + str(H[pos, pos2]))


def min_dist_jac_hess_num(wf,
                          cc_wf,
                          int[:] orbs_before,
                          int[:] corr_orb,
                          int[:] virt_orb,
                          Janal, Hanal,
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
    
    orbs_before, corr_orb, virt_orb
        See min_dist_jac_hess
    
    Janal (1D np.array)
        The analytic Jacobian. Exception is raised imediatly when an entry of
        the numeric Jacobian differs to the corresponding entry of Janal
    
    Hanal (2D np.array)
        The analytic Hessian. Exception is raised imediatly when an entry of
        the numeric Hessian differs to the corresponding entry of Hanal
    
    eps (float, optional, default=0.001)
        The step to be used in the finite differences approximation to the
        derivative:
        df(x0)/dx \approx (f(x0+eps) - f(x0-eps))/(2*eps)
    
    Side Effects:
    -------------
    wf is put in the orbital convention of maximum coincidence with the reference.
    
    Return:
    -------
    Two numpy arrays: one 1D of the same lenght as the amplitudes,
    with the jacobian, and another 2D, square, with the hessian.
    These are exactly the hessian of the function "square of the distance".
    Note that what is returned from min_dist_jac_hess is half of this.
    
    Raise:
    ------
    Raises an exception if a calculated entry of the Jacobian or Hessian
    does not match the corresponding entry of Janal or Hanal.
    
    """
    cdef int a, b, a2, b2
    cdef int a_irrep, b_irrep, a_spirrep, b_spirrep
    cdef int a2_irrep, b2_irrep, a2_spirrep, b2_spirrep
    cdef int n_irrep_or_0, n_irrep_or_02
    cdef int pos, pos2  # For the position in J or H
    cdef int pos_ampl, pos2_ampl  # For the position in cc_wf.amplitudes
    cdef int pos_transp, pos2_transp  # For the position of transpose of pos_ampl, pos2_ampl
    cdef int[8] pos_ini, pos2_ini  # assuming no more than 8 irreps in a point group!
    cdef int n_singles, n_ampl, n_indep_ampl
    cdef double[:] f_p, f_m
    cdef double[:] J
    cdef double[:, :] H
    cdef double f, f_pp, f_mm
    cdef OccOrbital i, j
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
    
    n_singles = cc_wf.ini_blocks_D[0, 0]
    n_ampl = len(cc_wf)
    n_indep_ampl = cc_wf.n_indep_ampl
    f = Func(cc_wf)
    f_p = np.zeros(n_indep_ampl)
    f_m = np.zeros(n_indep_ampl)
    J = np.zeros(n_indep_ampl)
    H = np.zeros((n_indep_ampl, n_indep_ampl))
    n_irrep = orbs_before.size - 1
    pos = 0
    pos_ampl = 0
    # --- alpha -> alpha and beta -> beta
    while pos < n_singles:
        cc_wf.amplitudes[pos_ampl] += eps
        f_p[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos_ampl] -= 2*eps
        f_m[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos_ampl] += eps
        J[pos] = (f_p[pos] - f_m[pos])/(2*eps)
        _compare_jac(pos, J, Janal)
        pos += 1
        pos_ampl += 1
    # --- alpha, alpha -> alpha, alpha, and beta, beta -> beta, beta
    for is_alpha in [True, False]:
        i = OccOrbital(corr_orb, orbs_before, is_alpha)
        j = OccOrbital(corr_orb, orbs_before, is_alpha)
        j.next_()
        n_irrep_or_0 = 0 if is_alpha else n_irrep
        while j.alive:
            _fill_pos_ini(n_irrep,
                          i.spirrep,
                          j.spirrep,
                          pos_ampl,
                          n_irrep_or_0,
                          pos_ini,
                          virt_orb)
            for a_irrep in range(n_irrep):
                b_irrep, a_spirrep, b_spirrep = _sp_irreps(a_irrep,
                                                           i.spirrep,
                                                           j.spirrep,
                                                           n_irrep_or_0)
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or (a_irrep == b_irrep and a < b):
                            pos_transp = _transp(a, b, a_irrep, b_irrep,
                                                 pos_ampl,
                                                 virt_orb[a_spirrep],
                                                 pos_ini[b_irrep])
                            cc_wf.amplitudes[pos_ampl] += eps
                            cc_wf.amplitudes[pos_transp] -= eps
                            f_p[pos] = Func(cc_wf)
                            cc_wf.amplitudes[pos_ampl] -= 2*eps
                            cc_wf.amplitudes[pos_transp] += 2*eps
                            f_m[pos] = Func(cc_wf)
                            cc_wf.amplitudes[pos_ampl] += eps
                            cc_wf.amplitudes[pos_transp] -= eps
                            J[pos] = (f_p[pos] - f_m[pos])/(2*eps)
                            _compare_jac(pos, J, Janal)
                            pos += 1
                        pos_ampl += 1
            if i.pos_in_occ == j.pos_in_occ - 1:
                j.next_()
                i.rewind()
            else:
                i.next_()
    # alpha, beta -> alpha, beta
    while pos < n_indep_ampl:
        cc_wf.amplitudes[pos_ampl] += eps
        f_p[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos_ampl] -= 2*eps
        f_m[pos] = Func(cc_wf)
        cc_wf.amplitudes[pos_ampl] += eps
        J[pos] = (f_p[pos] - f_m[pos])/(2*eps)
        _compare_jac(pos, J, Janal)
        pos += 1
        pos_ampl += 1
    if pos != n_indep_ampl:
        raise Exception(str(pos) + ' = pos != n_indep_ampl = ' + str(n_indep_ampl))
    if pos_ampl != n_ampl:
        raise Exception(str(pos_ampl) + ' = pos_ampl != n_ampl = ' + str(n_ampl))
    # =================
    # The Hessian:
    pos = 0
    pos_ampl = 0
    eps22 = 2 * eps**2
    # --- pos: alpha -> alpha and beta -> beta
    while pos < n_singles:
        H[pos, pos] = (f_p[pos] - 2*f + f_m[pos]) / eps**2
        pos2 = pos + 1
        pos2_ampl = pos_ampl + 1
        # --- pos2: alpha -> alpha and beta -> beta
        while pos2 < n_singles:
            cc_wf.amplitudes[pos_ampl] += eps
            cc_wf.amplitudes[pos2_ampl] += eps
            f_pp = Func(cc_wf)
            cc_wf.amplitudes[pos_ampl] -= 2*eps
            cc_wf.amplitudes[pos2_ampl] -= 2*eps
            f_mm = Func(cc_wf)
            cc_wf.amplitudes[pos_ampl] += eps
            cc_wf.amplitudes[pos2_ampl] += eps
            H[pos, pos2] = _df2_dxdy(f_pp, f_mm,
                                     f_p, f_m,
                                     f,
                                     pos, pos2, eps22)
            _compare_hess(pos, pos2, H, Hanal)
            pos2 += 1
            pos2_ampl += 1
        # --- pos2: alpha, alpha -> alpha, alpha, and beta, beta -> beta, beta
        for is_alpha2 in [True, False]:
            i2 = OccOrbital(corr_orb, orbs_before, is_alpha2)
            j2 = OccOrbital(corr_orb, orbs_before, is_alpha2)
            j2.next_()
            n_irrep_or_02 = 0 if is_alpha2 else n_irrep
            while j2.alive:
                _fill_pos_ini(n_irrep,
                              i2.spirrep,
                              j2.spirrep,
                              pos2_ampl,
                              n_irrep_or_02,
                              pos2_ini, virt_orb)
                for a2_irrep in range(n_irrep):
                    b2_irrep, a2_spirrep, b2_spirrep = _sp_irreps(a2_irrep,
                                                                  i2.spirrep,
                                                                  j2.spirrep,
                                                                  n_irrep_or_02)
                    for a2 in range(virt_orb[a2_spirrep]):
                        for b2 in range(virt_orb[b2_spirrep]):
                            if a2_irrep < b2_irrep or (a2_irrep == b2_irrep and a2 < b2):
                                pos2_transp = _transp(a2, b2, a2_irrep, b2_irrep,
                                                      pos2_ampl,
                                                      virt_orb[a2_spirrep],
                                                      pos2_ini[b2_irrep])
                                cc_wf.amplitudes[pos_ampl] += eps
                                cc_wf.amplitudes[pos2_ampl] += eps
                                cc_wf.amplitudes[pos2_transp] -= eps
                                f_pp = Func(cc_wf)
                                cc_wf.amplitudes[pos_ampl] -= 2*eps
                                cc_wf.amplitudes[pos2_ampl] -= 2*eps
                                cc_wf.amplitudes[pos2_transp] += 2*eps
                                f_mm = Func(cc_wf)
                                cc_wf.amplitudes[pos_ampl] += eps
                                cc_wf.amplitudes[pos2_ampl] += eps
                                cc_wf.amplitudes[pos2_transp] -= eps
                                H[pos, pos2] = _df2_dxdy(f_pp, f_mm,
                                                         f_p, f_m,
                                                         f,
                                                         pos, pos2, eps22)
                                _compare_hess(pos, pos2, H, Hanal)
                                pos2 += 1
                            pos2_ampl += 1
                if i2.pos_in_occ == j2.pos_in_occ - 1:
                    j2.next_()
                    i2.rewind()
                else:
                    i2.next_()
        # --- pos2: alpha, beta -> alpha, beta
        while pos2 < n_indep_ampl:
            cc_wf.amplitudes[pos_ampl] += eps
            cc_wf.amplitudes[pos2_ampl] += eps
            f_pp = Func(cc_wf)
            cc_wf.amplitudes[pos_ampl] -= 2*eps
            cc_wf.amplitudes[pos2_ampl] -= 2*eps
            f_mm = Func(cc_wf)
            cc_wf.amplitudes[pos_ampl] += eps
            cc_wf.amplitudes[pos2_ampl] += eps
            H[pos, pos2] = _df2_dxdy(f_pp, f_mm,
                                     f_p, f_m,
                                     f,
                                     pos, pos2, eps22)
            _compare_hess(pos, pos2, H, Hanal)
            pos2 += 1
            pos2_ampl += 1
        if pos2_ampl != n_ampl:
            raise Exception(str(pos2_ampl) + ' = pos2_ampl != n_ampl = ' + str(n_ampl))
        pos += 1
        pos_ampl += 1
    ini_pos_aabb = pos
    # --- pos: alpha, alpha -> alpha, alpha, and beta, beta -> beta, beta
    for is_alpha in [True, False]:
        i = OccOrbital(corr_orb, orbs_before, is_alpha)
        j = OccOrbital(corr_orb, orbs_before, is_alpha)
        j.next_()
        n_irrep_or_0 = 0 if is_alpha else n_irrep
        while j.alive:
            _fill_pos_ini(n_irrep,
                          i.spirrep,
                          j.spirrep,
                          pos_ampl,
                          n_irrep_or_0,
                          pos_ini,
                          virt_orb)
            for a_irrep in range(n_irrep):
                b_irrep, a_spirrep, b_spirrep = _sp_irreps(a_irrep,
                                                           i.spirrep,
                                                           j.spirrep,
                                                           n_irrep_or_0)
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or (a_irrep == b_irrep and a < b):
                            pos_transp = _transp(a, b, a_irrep, b_irrep,
                                                 pos_ampl,
                                                 virt_orb[a_spirrep],
                                                 pos_ini[b_irrep])
                            H[pos, pos] = (f_p[pos] - 2*f + f_m[pos]) / eps**2
                            _compare_hess(pos, pos, H, Hanal)
                            pos2 = ini_pos_aabb
                            pos2_ampl = ini_pos_aabb
                            # --- pos2: alpha, alpha -> alpha, alpha, and beta, beta -> beta, beta
                            for is_alpha2 in [True, False]:
                                i2 = OccOrbital(corr_orb, orbs_before, is_alpha2)
                                j2 = OccOrbital(corr_orb, orbs_before, is_alpha2)
                                j2.next_()
                                n_irrep_or_02 = 0 if is_alpha2 else n_irrep
                                while j2.alive:
                                    _fill_pos_ini(n_irrep,
                                                  i2.spirrep,
                                                  j2.spirrep,
                                                  pos2_ampl,
                                                  n_irrep_or_02,
                                                  pos2_ini,
                                                  virt_orb)
                                    for a2_irrep in range(n_irrep):
                                        b2_irrep, a2_spirrep, b2_spirrep = _sp_irreps(a2_irrep,
                                                                                      i2.spirrep,
                                                                                      j2.spirrep,
                                                                                      n_irrep_or_02)
                                        for a2 in range(virt_orb[a2_spirrep]):
                                            for b2 in range(virt_orb[b2_spirrep]):
                                                if a2_irrep < b2_irrep or (a2_irrep == b2_irrep and a2 < b2):
                                                    if pos2 > pos:
                                                        pos2_transp = _transp(a2, b2, a2_irrep, b2_irrep,
                                                                              pos2_ampl,
                                                                              virt_orb[a2_spirrep],
                                                                              pos2_ini[b2_irrep])
                                                        cc_wf.amplitudes[pos_ampl] += eps
                                                        cc_wf.amplitudes[pos_transp] -= eps
                                                        cc_wf.amplitudes[pos2_ampl] += eps
                                                        cc_wf.amplitudes[pos2_transp] -= eps
                                                        f_pp = Func(cc_wf)
                                                        cc_wf.amplitudes[pos_ampl] -= 2*eps
                                                        cc_wf.amplitudes[pos_transp] += 2*eps
                                                        cc_wf.amplitudes[pos2_ampl] -= 2*eps
                                                        cc_wf.amplitudes[pos2_transp] += 2*eps
                                                        f_mm = Func(cc_wf)
                                                        cc_wf.amplitudes[pos_ampl] += eps
                                                        cc_wf.amplitudes[pos_transp] -= eps
                                                        cc_wf.amplitudes[pos2_ampl] += eps
                                                        cc_wf.amplitudes[pos2_transp] -= eps
                                                        H[pos, pos2] = _df2_dxdy(f_pp, f_mm,
                                                                                 f_p, f_m,
                                                                                 f,
                                                                                 pos, pos2, eps22)
                                                        _compare_hess(pos, pos2, H, Hanal)
                                                    pos2 += 1
                                                pos2_ampl += 1
                                    if i2.pos_in_occ == j2.pos_in_occ - 1:
                                        j2.next_()
                                        i2.rewind()
                                    else:
                                        i2.next_()
                            # --- pos2: alpha, beta -> alpha, beta
                            while pos2 < n_indep_ampl:
                                cc_wf.amplitudes[pos_ampl] += eps
                                cc_wf.amplitudes[pos_transp] -= eps
                                cc_wf.amplitudes[pos2_ampl] += eps
                                f_pp = Func(cc_wf)
                                cc_wf.amplitudes[pos_ampl] -= 2*eps
                                cc_wf.amplitudes[pos_transp] += 2*eps
                                cc_wf.amplitudes[pos2_ampl] -= 2*eps
                                f_mm = Func(cc_wf)
                                cc_wf.amplitudes[pos_ampl] += eps
                                cc_wf.amplitudes[pos_transp] -= eps
                                cc_wf.amplitudes[pos2_ampl] += eps
                                H[pos, pos2] = _df2_dxdy(f_pp, f_mm,
                                                         f_p, f_m,
                                                         f,
                                                         pos, pos2, eps22)
                                _compare_hess(pos, pos2, H, Hanal)
                                pos2 += 1
                                pos2_ampl += 1
                            if pos2_ampl != n_ampl:
                                raise Exception(str(pos2_ampl) + ' = pos2_ampl != n_ampl = ' + str(n_ampl))
                            pos += 1
                        pos_ampl += 1
            if i.pos_in_occ == j.pos_in_occ - 1:
                j.next_()
                i.rewind()
            else:
                i.next_()
    # --- pos: alpha, beta -> alpha, beta
    while pos < n_indep_ampl:
        H[pos, pos] = (f_p[pos] - 2*f + f_m[pos]) / eps**2
        _compare_hess(pos, pos, H, Hanal)
        pos2 = pos + 1
        pos2_ampl = pos_ampl + 1
        # --- pos2: alpha, beta -> alpha, beta
        while pos2 < n_indep_ampl:
            cc_wf.amplitudes[pos_ampl] += eps
            cc_wf.amplitudes[pos2_ampl] += eps
            f_pp = Func(cc_wf)
            cc_wf.amplitudes[pos_ampl] -= 2*eps
            cc_wf.amplitudes[pos2_ampl] -= 2*eps
            f_mm = Func(cc_wf)
            cc_wf.amplitudes[pos_ampl] += eps
            cc_wf.amplitudes[pos2_ampl] += eps
            H[pos, pos2] = _df2_dxdy(f_pp, f_mm,
                                     f_p, f_m,
                                     f,
                                     pos, pos2, eps22)
            _compare_hess(pos, pos2, H, Hanal)
            pos2 += 1
            pos2_ampl += 1
        if pos2_ampl != n_ampl:
            raise Exception(str(pos2_ampl) + ' = pos2_ampl != n_ampl = ' + str(n_ampl))
        pos += 1
        pos_ampl += 1
    if pos_ampl != n_ampl:
        raise Exception(str(pos_ampl) + ' = pos_ampl != n_ampl = ' + str(n_ampl))
    for pos in range(n_indep_ampl):
        for pos2 in range(pos):
            H[pos, pos2] = H[pos2, pos]
    return J, H
