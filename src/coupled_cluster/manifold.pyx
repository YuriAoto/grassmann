""" Core functions to optimise distance to the CC manifold


"""
import cython
import numpy as np


from wave_functions.strings_rev_lexical_order import get_index
from wave_functions.strings_rev_lexical_order cimport next_str
from util.variables import int_dtype
from util.array_indices cimport get_pos_from_rectangular
from molecular_geometry.symmetry import irrep_product

cdef int EXC_TYPE_A = 0
cdef int EXC_TYPE_B = 1
cdef int EXC_TYPE_AA = 2
cdef int EXC_TYPE_AB = 3
cdef int EXC_TYPE_BB = 4


cdef class OccOrbital:
    """Class to run over occupied orbitals

    Usage:
    ------
    # alpha and beta correlated orbitals
    corr_orb = np.array([5,2,2,0, 4,2,2,0], dtype=int_dtype)
    # number of orbitals before that irrep
    n_orb_before = np.array([0,10,15,20])

    i = OccOrbital(corr_orb, n_orb_before, True)
    i.pos_in_occ  # 0
    i.orb  # 0
    i.spirrep  # 0
    i.alive  # True

    i.next_()

    i.pos_in_occ  # 1
    i.orb  # 1
    i.spirrep  # 0

    for k in range(3):
       i.next_()

    i.pos_in_occ  # 4
    i.orb  # 4
    i.spirrep  # 0

    i.next_()

    i.pos_in_occ  # 5
    i.orb  # 0
    i.spirrep  # 1

    for k in range(4):
       i.next_()

    i.pos_in_occ   # 9 > n of occupied orb
    i.alive  # False

    i.rewind()
    i.pos_in_occ  # 0
    i.orb  # 0
    i.spirrep  # 0
    i.alive  # True


    """
    def __init__(self,
                  int[:] corr_orb,
                  int[:] n_orb_before,
                  bint is_alpha):
        cdef int irrep, spirrep
        cdef int n_irrep = n_orb_before.shape[0]
        self.is_alpha = is_alpha
        self._corr_orb = corr_orb
        self._n_orb_before = n_orb_before
        self._n_occ = 0
        self.spirrep = 0 if is_alpha else n_irrep
        spirrep = self.spirrep
        self.orb = -1
        for irrep in range(n_irrep):
            self._n_occ += self._corr_orb[spirrep]
            if self.orb == -1 and self._corr_orb[spirrep] > 0:
                self.orb = self._n_orb_before[irrep]
                self.spirrep = spirrep
            spirrep += 1
        if self._n_occ == 0:
            self.alive = False
            return
        self.alive = True
        self.pos_in_occ = 0

    cdef rewind(self):
        cdef int irrep = 0
        if self._n_occ == 0:
            self.alive = False
            return
        self.alive = True
        self.spirrep = 0 if self.is_alpha else self._n_orb_before.shape[0]
        while self._corr_orb[self.spirrep] == 0:
            self.spirrep += 1
            irrep += 1
        self.orb = self._n_orb_before[irrep]
        self.pos_in_occ = 0

    cdef next_(self):
        self.pos_in_occ += 1
        if self.pos_in_occ < self._n_occ:
            self.orb += 1
            if self.orb < (self._n_orb_before[self.spirrep
                                              % self._n_orb_before.size]
                           + self._corr_orb[self.spirrep]):
                return
            self.spirrep += 1
            while self._corr_orb[self.spirrep] == 0:
                self.spirrep += 1
            self.orb = self._n_orb_before[self.spirrep
                                          % self._n_orb_before.size]
        else:
            self.alive = False


cdef struct SingleExc:
    int i, a

cdef struct DoubleExc:
    int i, j, a, b


def min_dist_app_hess(double[:, :] wf,
                      double[:, :] wf_cc,
                      int n_ampl,
                      int[:] n_orb_before,
                      int[:] corr_orb,
                      int[:] virt_orb,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      level='SD'):
    """Calculate the Jacobian and approximate the Hessian
    
    Parameters:
    -----------
    wf
        The wave function, as a matrix of alpha and beta strings
    
    wf_cc
        The coupled cluster wave function, as a matrix of alpha
        and beta strings
    
    n_ampl
        The total number of amplitudes
    
    n_orb_before
        The number of orbitals before each irreducible representation,
        in the order "all orbitals of first irrep, then all orbitals of
        second irrep, ...".
        
        It must have the same number of elements as n_irrep: its size
        is used to get the n_irrep!
        
        Thus, n_orb_before[0] = 0
              n_orb_before[1] = total number of orbitals in irrep 0
                                (occ and virt)
              n_orb_before[2] = total number of orbitals in irrep 0 and 1
                                (occ and virt)
    
    corr_orb
        The number of correlatad orbitals in each spirrep.
        Its size must be twice n_irrep = n_orb_before.size
        Currently, the wave function is assumed to be of unrestricted type
    
    virt_orb
        The number of orbitals in the external space (number of virtuals)
        for each spirrep
        Its size must be twice n_irrep = n_orb_before.size
        Currently, the wave function is assumed to be of unrestricted type
    
    alpha_string_graph
        The graph to obtain the alpha strings (associated to the first
        index of wf and wf_cc) in the reverse lexical order
    
    beta_string_graph
        The graph to obtain the beta strings (associated to the second
        index of wf and wf_cc) in the reverse lexical order
    
    level (str, optional, default='SD')
        Coupled cluster level
    
    Return:
    -------
    The z vector (that updates the amplitudes),
    and the norm of the Jacobian
    
    """
    cdef double normJac = 0.0
    cdef double[:] z = np.zeros(n_ampl)
    cdef int pos = 0, n_alpha, n_beta
    cdef int n_irrep, spirrep, irrep, a_irrep, b_irrep, i, j, a, b
    cdef SingleExc single_exc
    cdef DoubleExc double_exc
    cdef int[8] pos_ini  ## assuming no more than 8 irreps in a point group!
    n_alpha = alpha_string_graph.shape[1]
    n_beta = beta_string_graph.shape[1]
    n_irrep = n_orb_before.size
    if level == 'SD':
        # --- alpha -> alpha
        for irrep in range(n_irrep):
            spirrep = irrep
            single_exc.i = n_orb_before[irrep]
            for i in range(corr_orb[spirrep]):
                single_exc.a = n_orb_before[irrep] + corr_orb[spirrep]
                for a in range(virt_orb[spirrep]):
                    J = _term1_a(single_exc,
                                 wf,
                                 wf_cc,
                                 alpha_string_graph)
                    normJac += J**2
                    z[pos] = J/_term2_diag_a(single_exc,
                                             wf_cc,
                                             n_alpha)
                    pos += 1
                    single_exc.a += 1
                single_exc.i += 1
        # --- beta -> beta
        for irrep in range(n_irrep):
            spirrep = irrep + n_irrep
            single_exc.i = n_orb_before[irrep]
            for i in range(corr_orb[spirrep]):
                single_exc.a = n_orb_before[irrep] + corr_orb[spirrep]
                for a in range(virt_orb[spirrep]):
                    J = _term1_b(single_exc,
                                 wf,
                                 wf_cc,
                                 beta_string_graph)
                    normJac += J**2
                    z[pos] = J/_term2_diag_b(single_exc,
                                             wf_cc,
                                             n_beta)
                    pos += 1
                    single_exc.a += 1
                single_exc.i += 1
    # --- alpha, alpha -> alpha, alpha
    i = OccOrbital(corr_orb, n_orb_before, True)
    j = OccOrbital(corr_orb, n_orb_before, True)
    j.next_()
    while j.alive:
        double_exc.i = i.orb
        double_exc.j = j.orb
        for a_irrep in range(n_irrep):
            pos_ini[a_irrep] = pos
            b_irrep = irrep_product[
                irrep_product[i.irrep, j.irrep], a_irrep]
            a_spirrep = a_irrep
            b_spirrep = b_irrep
            double_exc.a = (n_orb_before[a_irrep]
                            + corr_orb[a_spirrep])
            if a_irrep <= b_irrep:
                for a in range(virt_orb[a_spirrep]):
                    if a_irrep == b_irrep:
                        nvirt_1 = virt_orb[a_spirrep] - 1
                        double_exc.b = double_exc.a
                    else:
                        double_exc.b = (n_orb_before[b_irrep]
                                        + corr_orb[b_spirrep])
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or a < b:
                            J = _term1_aa(double_exc,
                                          wf,
                                          wf_cc,
                                          alpha_string_graph)
                            normJac += J**2
                            z[pos] = J/_term2_diag_aa(double_exc,
                                                      wf_cc,
                                                      n_alpha)
                        elif a > b:
                            z[pos] = -z[pos - (a-b)*nvirt_1]
                        pos += 1
                        double_exc.b += 1
                    double_exc.a += 1
            else:  # and a_irrep > b_irrep
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        z[pos] = -z[pos_ini[b_irrep]
                                    + get_pos_from_rectangular(
                                        b, a, virt_orb[a_spirrep])]
                        pos += 1
        if i.pos_in_occ == j.pos_in_occ - 1:
            j.next_()
            i.rewind()
            double_exc.j = j.orb
        else:
            i.next_()
        double_exc.i = i.orb
    # --- beta, beta -> beta, beta
    i = OccOrbital(corr_orb, n_orb_before, False)
    j = OccOrbital(corr_orb, n_orb_before, False)
    j.next_()
    while j.alive:
        double_exc.i = i.orb
        double_exc.j = j.orb
        for a_irrep in range(n_irrep):
            pos_ini[a_irrep] = pos
            b_irrep = irrep_product[
                irrep_product[i.spirrep - n_irrep,
                              j.spirrep - n_irrep], a_irrep]
            a_spirrep = a_irrep + n_irrep
            b_spirrep = b_irrep + n_irrep
            double_exc.a = (n_orb_before[a_irrep]
                            + corr_orb[a_spirrep])
            if a_irrep <= b_irrep:
                for a in range(virt_orb[a_spirrep]):
                    if a_irrep == b_irrep:
                        nvirt_1 = virt_orb[a_spirrep] - 1
                        double_exc.b = double_exc.a
                    else:
                        double_exc.b = (n_orb_before[b_irrep]
                                        + corr_orb[b_spirrep])
                    for b in range(virt_orb[b_spirrep]):
                        if a_irrep < b_irrep or a < b:
                            J = _term1_bb(double_exc,
                                          wf,
                                          wf_cc,
                                          beta_string_graph)
                            normJac += J**2
                            z[pos] = J/_term2_diag_bb(double_exc,
                                                      wf_cc,
                                                      n_beta)
                        elif a > b:
                            z[pos] = -z[pos - (a-b)*nvirt_1]
                        pos += 1
                        double_exc.b += 1
                    double_exc.a += 1
            else:  # and a_irrep > b_irrep
                for a in range(virt_orb[a_spirrep]):
                    for b in range(virt_orb[b_spirrep]):
                        z[pos] = -z[pos_ini[b_irrep]
                                    + get_pos_from_rectangular(
                                        b, a, virt_orb[a_spirrep])]
                        pos += 1
        if i.pos_in_occ == j.pos_in_occ - 1:
            j.next_()
            i.rewind()
            double_exc.j = j.orb
        else:
            i.next_()
        double_exc.i = i.orb
    # --- alpha, beta -> alpha, beta
    i = OccOrbital(corr_orb, n_orb_before, True)
    j = OccOrbital(corr_orb, n_orb_before, False)
    double_exc.i = i.orb
    double_exc.j = j.orb
    while j.alive:
        for a_irrep in range(n_irrep):
            b_irrep = irrep_product[
                irrep_product[i.spirrep, j.spirrep - n_irrep], a_irrep]
            a_spirrep = a_irrep
            b_spirrep = b_irrep + n_irrep
            double_exc.a = (n_orb_before[a_irrep]
                            + corr_orb[a_spirrep])
            for a in range(virt_orb[a_spirrep]):
                double_exc.b = (n_orb_before[b_irrep]
                                + corr_orb[b_spirrep])
                for b in range(virt_orb[b_spirrep]):
                    J = _term1_ab(double_exc,
                                  wf,
                                  wf_cc,
                                  alpha_string_graph,
                                  beta_string_graph)
                    normJac += J**2
                    z[pos] = J/_term2_diag_ab(double_exc,
                                              wf_cc,
                                              n_alpha,
                                              n_beta)
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
    return z, np.sqrt(normJac)


def min_dist_jac_hess():
    raise NotImplementedError('do it')
    

cdef double _term1(int[:] exc,
                   int exc_type,
                   double[:, :] wf,
                   double[:, :] wf_cc,
                   int[:, :] alpha_string_graph,
                   int[:, :] beta_string_graph):
    """The term <\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    We recomment to use the functions _term1_a, _term1_b,
    _term1_aa, _term1_bb, and _term1_ab directly, since they
    are cdef functions and strongly typed.
    
    Parameters:
    -----------
    exc
        the excitation, with [i, a] for single and [i, a, j, b] for double
    
    exc_type
        the excitation type. Possible values are EXC_TYPE_A
        and EXC_TYPE_B for singles, and EXC_TYPE_AA, EXC_TYPE_AB
        and EXC_TYPE_BB for doubles.
        This determines the "\rho"
    
    wf
        The wave function, as a coefficients matrix of alpha and beta
        strings. This determines the "\Psi".
    
    wf_cc
        The coupled cluster wave function, as a coefficients matrix
        of alpha and beta strings. This determines the "\Psi_cc".
    
    {alpha,beta}_string_graph
        The string graph that determines the reverse lexical order
        for the {alpha,beta} strings
    
    Return:
    -------
    A float (C double)
    
    """
    cdef DoubleExc double_exc
    cdef SingleExc single_exc
    if exc_type == EXC_TYPE_A:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return _term1_a(single_exc, wf, wf_cc,
                        alpha_string_graph)
    if exc_type == EXC_TYPE_AA:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return _term1_aa(double_exc, wf, wf_cc,
                         alpha_string_graph)
    if exc_type == EXC_TYPE_B:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return _term1_b(single_exc, wf, wf_cc,
                        beta_string_graph)
    if exc_type == EXC_TYPE_BB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return _term1_bb(double_exc, wf, wf_cc,
                         beta_string_graph)
    if exc_type == EXC_TYPE_AB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return _term1_ab(double_exc, wf, wf_cc,
                         alpha_string_graph,
                         beta_string_graph)


cdef double _term2_diag(int[:] exc,
                        int exc_type,
                        double[:, :] wf,
                        int alpha_nel,
                        int beta_nel):
    """The term <\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    Parameters:
    -----------
    See _term1 (here wf is \Psi_cc)
    
    Return:
    -------
    A float (C double)
    
    """
    cdef DoubleExc double_exc
    cdef SingleExc single_exc
    if exc_type == EXC_TYPE_A:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return _term2_diag_a(single_exc, wf, alpha_nel)
    if exc_type == EXC_TYPE_AA:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return _term2_diag_aa(double_exc, wf, alpha_nel)
    if exc_type == EXC_TYPE_B:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return _term2_diag_b(single_exc, wf, beta_nel)
    if exc_type == EXC_TYPE_BB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return _term2_diag_bb(double_exc, wf, beta_nel)
    if exc_type == EXC_TYPE_AB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return _term2_diag_ab(double_exc, wf, alpha_nel, beta_nel)


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_a(SingleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph):
    """<\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    for \rho=a_{\alpha}^{\alpha}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the first dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] I = np.arange(nel, dtype=int_dtype)
    cdef int[:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, i
    cdef double S = 0.0
    for I_i in range(nstr_alpha):
        if (exc.i in I) and (exc.a not in I):
            I_exc = _exc_on_string(exc.i, exc.a, I)
            I_exc_i = get_index(I_exc[:nel], string_graph)
            with nogil:
                for i in range(nstr_beta):
                    S += (I_exc[nel]
                          * (wf[I_exc_i, i] - wf_cc[I_exc_i, i])
                          * wf_cc[I_i, i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_b(SingleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph):
    """<\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    for \rho=a_{\beta}^{\beta}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the second dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] I = np.arange(nel, dtype=int_dtype)
    cdef int[:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, i
    cdef double S = 0.0
    for I_i in range(nstr_beta):
        if (exc.i in I) and (exc.a not in I):
            I_exc = _exc_on_string(exc.i, exc.a, I)
            I_exc_i = get_index(I_exc[:nel], string_graph)
            for i in range(nstr_alpha):
                S += (I_exc[nel]
                      * (wf[i, I_exc_i] - wf_cc[i, I_exc_i])
                      * wf_cc[i, I_i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_aa(DoubleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] string_graph):
    """<\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    for \rho=a_{\alpha\alpha}^{\alpha\alpha}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the first dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] I = np.arange(nel, dtype=int_dtype)
    cdef int[:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, sign, i
    cdef double S = 0.0
    for I_i in range(nstr_alpha):
        if ((exc.i in I) and (exc.a not in I)
                and (exc.j in I) and (exc.b not in I)):
            I_exc = _exc_on_string(exc.i, exc.a, I)
            sign = I_exc[nel]
            I_exc = _exc_on_string(exc.j, exc.b, I_exc[:nel])
            I_exc_i = get_index(I_exc[:nel], string_graph)
            for i in range(nstr_beta):
                S += (I_exc[nel] * sign
                      * (wf[I_exc_i, i] - wf_cc[I_exc_i, i])
                      * wf_cc[I_i, i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_bb(DoubleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] string_graph):
    """<\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    for \rho=a_{\beta\beta}^{\beta\beta}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the second dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] I = np.arange(nel, dtype=int_dtype)
    cdef int[:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, sign, i
    cdef double S = 0.0
    for I_i in range(nstr_beta):
        if ((exc.i in I) and (exc.a not in I)
                and (exc.j in I) and (exc.b not in I)):
            I_exc = _exc_on_string(exc.i, exc.a, I)
            sign = I_exc[nel]
            I_exc = _exc_on_string(exc.j, exc.b, I_exc[:nel])
            I_exc_i = get_index(I_exc[:nel], string_graph)
            for i in range(nstr_alpha):
                S += (I_exc[nel]
                      * (wf[i, I_exc_i] - wf_cc[i, I_exc_i])
                      * wf_cc[i, I_i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_ab(DoubleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph):
    """<\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    for \rho=a_{\alpha\beta}^{\alpha\beta}
    
    Parameters:
    -----------
    See _term1
    
    Return:
    -------
    A float (C double)
    """
    cdef int nalpha = alpha_string_graph.shape[1]
    cdef int nbeta = beta_string_graph.shape[1]
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] Ia = np.arange(nalpha, dtype=int_dtype)
    cdef int[:] Ib = np.arange(nbeta, dtype=int_dtype)
    cdef int[:] Ia_exc = np.empty(nalpha + 1, dtype=int_dtype)
    cdef int[:] Ib_exc = np.empty(nbeta + 1, dtype=int_dtype)
    cdef int Ia_i, Ib_i, Ia_exc_i, Ib_exc_i
    cdef double S = 0.0
    for Ia_i in range(nstr_alpha):
        if (exc.i in Ia) and (exc.a not in Ia):
            Ia_exc = _exc_on_string(exc.i, exc.a, Ia)
            Ia_exc_i = get_index(Ia_exc[:nalpha],
                             alpha_string_graph)
            Ib = np.arange(nbeta, dtype=int_dtype)
            for Ib_i in range(nstr_beta):
                if (exc.j in Ib) and (exc.b not in Ib):
                    Ib_exc = _exc_on_string(exc.j, exc.b, Ib)
                    Ib_exc_i = get_index(Ib_exc[:nbeta],
                                     beta_string_graph)
                    S += (Ia_exc[nalpha] * Ib_exc[nbeta]
                          * (wf[Ia_exc_i, Ib_exc_i]
                             - wf_cc[Ia_exc_i, Ib_exc_i])
                          * wf_cc[Ia_i, Ib_i])
                next_str(Ib)
        next_str(Ia)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_a(SingleExc exc,
                          double[:, :] wf,
                          int nel):
    """<\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    for \rho=a_{\alpha}^{\alpha}
    
    Parameters:
    -----------
    See _term2_diag
    
    Return:
    -------
    A float (C double)
    """
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] Ia = np.arange(nel, dtype=int_dtype)
    cdef double S = 0.0
    cdef int Ia_i, Ib_i
    for Ia_i in range(nstr_alpha):
        if (exc.i in Ia) and (exc.a not in Ia):
            with nogil:
                for Ib_i in range(nstr_beta):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ia)
    return S

##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_b(SingleExc exc,
                          double[:, :] wf,
                          int nel):
    """<\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    for \rho=a_{\beta}^{\beta}
    
    Parameters:
    -----------
    See _term2_diag
    
    Return:
    -------
    A float (C double)
    """
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] Ib = np.arange(nel, dtype=int_dtype)
    cdef double S = 0.0
    for Ib_i in range(nstr_beta):
        if (exc.i in Ib) and (exc.a not in Ib):
            with nogil:
                for Ia_i in range(nstr_alpha):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ib)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_aa(DoubleExc exc,
                           double[:, :] wf,
                           int nel):
    """<\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    for \rho=a_{\alpha\alpha}^{\alpha\alpha}
    
    Parameters:
    -----------
    See _term2_diag
    
    Return:
    -------
    A float (C double)
    """
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] Ia = np.arange(nel, dtype=int_dtype)
    cdef double S = 0.0
    cdef int Ia_i, Ib_i
    for Ia_i in range(nstr_alpha):
        if ((exc.i in Ia) and (exc.a not in Ia)
                and (exc.j in Ia) and (exc.b not in Ia)):
            with nogil:
                for Ib_i in range(nstr_beta):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ia)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_bb(DoubleExc exc,
                           double[:, :] wf,
                           int nel):
    """<\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    for \rho=a_{\beta\beta}^{\beta\beta}
    
    Parameters:
    -----------
    See _term2_diag
    
    Return:
    -------
    A float (C double)
    """
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] Ib = np.arange(nel, dtype=int_dtype)
    cdef int Ia_i, Ib_i
    cdef double S = 0.0
    for Ib_i in range(nstr_beta):
        if ((exc.i in Ib) and (exc.a not in Ib)
                and (exc.j in Ib) and (exc.b not in Ib)):
            with nogil:
                for Ia_i in range(nstr_alpha):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ib)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_ab(DoubleExc exc,
                           double[:, :] wf,
                           int alpha_nel,
                           int beta_nel):
    """<\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    for \rho=a_{\alpha\beta}^{\alpha\beta}
    
    Parameters:
    -----------
    See _term2_diag
    
    Return:
    -------
    A float (C double)
    """
    cdef int nstr_alpha = wf.shape[0]
    cdef int nstr_beta = wf.shape[1]
    cdef int[:] Ia = np.arange(alpha_nel, dtype=int_dtype)
    cdef int[:] Ib
    cdef int Ia_i, Ib_i
    cdef double S = 0.0
    for Ia_i in range(nstr_alpha):
        if (exc.i in Ia) and (exc.a not in Ia):
            Ib = np.arange(beta_nel, dtype=int_dtype)
            for Ib_i in range(nstr_beta):
                if (exc.j in Ib) and (exc.b not in Ib):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
                next_str(Ib)
        next_str(Ia)
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef int[:] _exc_on_string(int i, int a, int[:] I):
    """Obtain the string after the excitation i->a over I
    
    Parameters:
    -----------
    i
        hole index (the orbital where the excitation comes from)
    
    a
        particle index (the orbital where the excitation goes to)
    
    I
        The string that represents the Slater determinant
        where the excitation acts
    
    Return:
    -------
    A new array, with one more entry than I:
        The first entries are with the new Slater determinant,
        and the last has the sign that arises after putting the
        orbitals in the correct order
    
    """
    cdef int n = I.shape[0]
    cdef int[:] new_I = np.empty(n+1, dtype=int_dtype)
    cdef int pos, i_pos, a_pos
    i_pos = 0
    a_pos = 0
    new_I[:n] = I[:]
    pos = 0
    if i < a:
        a_pos = n - 1
        for pos in range(n):
            if I[pos] == i:
                i_pos = pos
            if I[pos] > a:
                a_pos = pos - 1
                break
        new_I[i_pos: a_pos] = I[i_pos+1: a_pos+1]
        new_I[a_pos] = a
    elif i > a:
        for pos in range(n-1, -1, -1):
            if I[pos] == i:
                i_pos = pos
            if I[pos] < a:
                a_pos = pos + 1
                break
        new_I[a_pos+1: i_pos+1] = I[a_pos: i_pos]
        new_I[a_pos] = a
    new_I[n] = 1 - 2*(abs(a_pos - i_pos) % 2)
    return new_I
