"""Core functions to optimise distance to the CC manifold


"""
import cython
import numpy as np


from wave_functions.strings_rev_lexical_order import get_index
from wave_functions.strings_rev_lexical_order cimport next_str
from util import irrep_product, int_dtype

cdef int EXC_TYPE_A = 0
cdef int EXC_TYPE_B = 1
cdef int EXC_TYPE_AA = 2
cdef int EXC_TYPE_AB = 3
cdef int EXC_TYPE_BB = 4


def min_dist_app_hess(double [:, :] wf,
                      double [:, :] wf_cc,
                      int n_ampl,
                      int n_irrep,
                      int [:] n_orb_before,
                      int [:] n_corr_orb,
                      int [:] n_ext,
                      int [:, :] alpha_string_graph,
                      int [:, :] beta_string_graph,
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
    
    n_irrep
        The number of irreducible representations
    
    n_orb_before
        The number of orbitals before each irreducible representation,
        in the order "all orbitals of first irrep, then all orbitals of
        second irrep, ...".
        Thus, n_orb_before[0] = 0
              n_orb_before[1] = total number of orbitals in irrep 0
                                (occ and virt)
              n_orb_before[2] = total number of orbitals in irrep 0 and 1
                                (occ and virt)
    
    n_corr_orb
        The number of correlatad orbitals in each irrep
    
    n_ext
        The number of orbitals in the external space (number of virtuals)
        for each irrep
    
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
    cdef double [:] z = np.zeros(n_ampl)
    cdef int pos = 0, exc_type
    cdef int spirrep, irrep, i_irrep, j_irrep, a_irrep, b_irrep, i, j, a, b
    cdef int [:] single_exc = np.zeros(2, dtype=int_dtype)  # [i, a]
    cdef int [:] double_exc = np.zeros(4, dtype=int_dtype)  # [i, a, j, b]
    if level == 'SD':
        for spirrep in range(2 * n_irrep):
            irrep = spirrep % n_irrep
            exc_type = EXC_TYPE_A if spirrep < n_irrep else EXC_TYPE_B
            single_exc[0] = n_orb_before[irrep]
            for i in range(n_corr_orb[spirrep]):
                single_exc[1] = n_orb_before[irrep] + n_corr_orb[spirrep]
                for a in range(n_ext[spirrep]):
                    J = _term1(single_exc,
                               exc_type,
                               wf,
                               wf_cc,
                               alpha_string_graph,
                               beta_string_graph)
                    normJac += J**2
                    z[pos] = J/_term2_diag(
                        single_exc,
                        exc_type,
                        wf_cc,
                        alpha_string_graph.shape[1],
                        beta_string_graph.shape[1])
                    pos += 1
                    single_exc[1] += 1
                single_exc[0] += 1
    for i_spirrep in range(2 * n_irrep):
        alpha_exc1 = i_spirrep < n_irrep
        i_irrep = i_spirrep % n_irrep
        for j_spirrep in range(i_spirrep, 2 * n_irrep):
            alpha_exc2 = j_spirrep < n_irrep
            j_irrep = j_spirrep % n_irrep
            exc_type = (EXC_TYPE_AA
                        if alpha_exc2 else
                        (EXC_TYPE_AB
                         if alpha_exc1 else
                         EXC_TYPE_BB))
            double_exc[0] = n_orb_before[i_irrep]
            for i in range(n_corr_orb[i_spirrep]):
                double_exc[2] = n_orb_before[j_irrep]
                for j in range(n_corr_orb[j_spirrep]):
                    if i_spirrep < j_spirrep or i < j:
                        for a_irrep in range(n_irrep):
                            a_spirrep = a_irrep
                            if not alpha_exc1:
                                a_spirrep + n_irrep
                            b_irrep = irrep_product[
                                irrep_product[i_irrep, j_irrep],a_irrep]
                            b_spirrep = b_irrep
                            if not alpha_exc2:
                                b_spirrep + n_irrep
                            if b_spirrep < a_spirrep:
                                pos += n_ext[b_spirrep] * n_ext[a_spirrep]
                                continue
                            double_exc[1] = (n_orb_before[a_irrep]
                                             + n_corr_orb[a_spirrep])
                            for a in range(n_ext[a_spirrep]):
                                if a_irrep == b_irrep:
                                    double_exc[3] = double_exc[1]
                                else:
                                    double_exc[3] = (n_orb_before[b_irrep]
                                                     + n_corr_orb[b_spirrep])
                                for b in range(n_ext[b_spirrep]):
                                    if a_spirrep < b_spirrep or a < b:
                                        J = _term1(double_exc,
                                                   exc_type,
                                                   wf,
                                                   wf_cc,
                                                   alpha_string_graph,
                                                   beta_string_graph)
                                        normJac += J**2
                                        z[pos] = J/_term2_diag(
                                            double_exc,
                                            exc_type,
                                            wf_cc,
                                            alpha_string_graph.shape[1],
                                            beta_string_graph.shape[1])
                                    pos += 1
                                    double_exc[3] += 1  # b++
                                double_exc[1] += 1  # a++
                    double_exc[2] += 1  # j++
                double_exc[0] += 1  # i++
    if pos != n_ampl:
        raise Exception(str(pos) + ' = pos != n_ampl = ' + str(n_ampl))
    return z, np.sqrt(normJac)


def min_dist_jac_hess():
    raise NotImplementedError('do it')
    

cdef double _term1(int [:] exc,
                   int exc_type,
                   double [:, :] wf,
                   double [:, :] wf_cc,
                   int [:, :] alpha_string_graph,
                   int [:, :] beta_string_graph):
    """The term <\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    Parameters:
    -----------
    exc
        the excitation: [i, a] for single
                        [i, j, a, b] for double
    
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
    if exc_type == EXC_TYPE_A:
        return _term1_a(exc, wf, wf_cc,
                        alpha_string_graph)
    if exc_type == EXC_TYPE_AA:
        return _term1_aa(exc, wf, wf_cc,
                         alpha_string_graph)
    if exc_type == EXC_TYPE_B:
        return _term1_b(exc, wf, wf_cc,
                        beta_string_graph)
    if exc_type == EXC_TYPE_BB:
        return _term1_bb(exc, wf, wf_cc,
                         beta_string_graph)
    if exc_type == EXC_TYPE_AB:
        return _term1_ab(exc, wf, wf_cc,
                         alpha_string_graph,
                         beta_string_graph)


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_a(int [:] exc,  # [i, a]
                     double [:, :] wf,
                     double [:, :] wf_cc,
                     int [:, :] string_graph):
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
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, i
    cdef double S = 0.0
    for I_i in range(nstr_alpha):
        if (exc[0] in I) and (exc[1] not in I):
            I_exc = _exc_on_string(exc[0], exc[1], I)
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
cdef double _term1_b(int [:] exc,  # [i, a]
                     double [:, :] wf,
                     double [:, :] wf_cc,
                     int [:, :] string_graph):
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
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, i
    cdef double S = 0.0
    for I_i in range(nstr_beta):
        if (exc[0] in I) and (exc[1] not in I):
            I_exc = _exc_on_string(exc[0], exc[1], I)
            I_exc_i = get_index(I_exc[:nel], string_graph)
            for i in range(nstr_alpha):
                S += (I_exc[nel]
                      * (wf[i, I_exc_i] - wf_cc[i, I_exc_i])
                      * wf_cc[i, I_i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_aa(int [:] exc,  # [i, a, j, b]
                      double [:, :] wf,
                      double [:, :] wf_cc,
                      int [:, :] string_graph):
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
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, sign, i
    cdef double S = 0.0
    for I_i in range(nstr_alpha):
        if ((exc[0] in I) and (exc[1] not in I)
                and (exc[2] in I) and (exc[3] not in I)):
            I_exc = _exc_on_string(exc[0], exc[1], I)
            sign = I_exc[nel]
            I_exc = _exc_on_string(exc[2], exc[3], I_exc[:nel])
            I_exc_i = get_index(I_exc[:nel], string_graph)
            for i in range(nstr_beta):
                S += (I_exc[nel] * sign
                      * (wf[I_exc_i, i] - wf_cc[I_exc_i, i])
                      * wf_cc[I_i, i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_bb(int [:] exc,  # [i, a, j, b]
                      double [:, :] wf,
                      double [:, :] wf_cc,
                      int [:, :] string_graph):
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
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_i, I_exc_i, sign, i
    cdef double S = 0.0
    for I_i in range(nstr_beta):
        if ((exc[0] in I) and (exc[1] not in I)
                and (exc[2] in I) and (exc[3] not in I)):
            I_exc = _exc_on_string(exc[0], exc[1], I)
            sign = I_exc[nel]
            I_exc = _exc_on_string(exc[2], exc[3], I_exc[:nel])
            I_exc_i = get_index(I_exc[:nel], string_graph)
            for i in range(nstr_alpha):
                S += (I_exc[nel]
                      * (wf[i, I_exc_i] - wf_cc[i, I_exc_i])
                      * wf_cc[i, I_i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_ab(int [:] exc,  # [i, a, j, b]
                      double [:, :] wf,
                      double [:, :] wf_cc,
                      int [:, :] alpha_string_graph,
                      int [:, :] beta_string_graph):
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
    cdef int [:] Ia = np.arange(nalpha, dtype=int_dtype)
    cdef int [:] Ib = np.arange(nbeta, dtype=int_dtype)
    cdef int [:] Ia_exc = np.empty(nalpha + 1, dtype=int_dtype)
    cdef int [:] Ib_exc = np.empty(nbeta + 1, dtype=int_dtype)
    cdef int Ia_i, Ib_i, Ia_exc_i, Ib_exc_i
    cdef double S = 0.0
    for Ia_i in range(nstr_alpha):
        if (exc[0] in Ia) and (exc[1] not in Ia):
            Ia_exc = _exc_on_string(exc[0], exc[1], Ia)
            Ia_exc_i = get_index(Ia_exc[:nalpha],
                             alpha_string_graph)
            Ib = np.arange(nbeta, dtype=int_dtype)
            for Ib_i in range(nstr_beta):
                if (exc[2] in Ib) and (exc[3] not in Ib):
                    Ib_exc = _exc_on_string(exc[2], exc[3], Ib)
                    Ib_exc_i = get_index(Ib_exc[:nbeta],
                                     beta_string_graph)
                    S += (Ia_exc[nalpha] * Ib_exc[nbeta]
                          * (wf[Ia_exc_i, Ib_exc_i]
                             - wf_cc[Ia_exc_i, Ib_exc_i])
                          * wf_cc[Ia_i, Ib_i])
                next_str(Ib)
        next_str(Ia)
    return S


cdef double _term2_diag(int [:] exc,
                        int exc_type,
                        double [:, :] wf,
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
    if exc_type == EXC_TYPE_A:
        return _term2_diag_a(exc, wf, alpha_nel)
    if exc_type == EXC_TYPE_AA:
        return _term2_diag_aa(exc, wf, alpha_nel)
    if exc_type == EXC_TYPE_B:
        return _term2_diag_b(exc, wf, beta_nel)
    if exc_type == EXC_TYPE_BB:
        return _term2_diag_bb(exc, wf, beta_nel)
    if exc_type == EXC_TYPE_AB:
        return _term2_diag_ab(exc, wf, alpha_nel, beta_nel)


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_a(int [:] exc,  # [i, a]
                          double [:, :] wf,
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
    cdef int [:] Ia = np.arange(nel, dtype=int_dtype)
    cdef double S = 0.0
    cdef int Ia_i, Ib_i
    for Ia_i in range(nstr_alpha):
        if (exc[0] in Ia) and (exc[1] not in Ia):
            with nogil:
                for Ib_i in range(nstr_beta):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ia)
    return S

##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_b(int [:] exc,  # [i, a]
                          double [:, :] wf,
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
    cdef int [:] Ib = np.arange(nel, dtype=int_dtype)
    cdef double S = 0.0
    for Ib_i in range(nstr_beta):
        if (exc[0] in Ib) and (exc[1] not in Ib):
            with nogil:
                for Ia_i in range(nstr_alpha):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ib)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_aa(int [:] exc,  # [i, a, j, b]
                           double [:, :] wf,
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
    cdef int [:] Ia = np.arange(nel, dtype=int_dtype)
    cdef double S = 0.0
    cdef int Ia_i, Ib_i
    for Ia_i in range(nstr_alpha):
        if ((exc[0] in Ia) and (exc[1] not in Ia)
                and (exc[2] in Ia) and (exc[3] not in Ia)):
            with nogil:
                for Ib_i in range(nstr_beta):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ia)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_bb(int [:] exc,  # [i, a, j, b]
                           double [:, :] wf,
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
    cdef int [:] Ib = np.arange(nel, dtype=int_dtype)
    cdef int Ia_i, Ib_i
    cdef double S = 0.0
    for Ib_i in range(nstr_beta):
        if ((exc[0] in Ib) and (exc[1] not in Ib)
                and (exc[2] in Ib) and (exc[3] not in Ib)):
            with nogil:
                for Ia_i in range(nstr_alpha):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
        next_str(Ib)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_diag_ab(int [:] exc,  # [i, a, j, b]
                           double [:, :] wf,
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
    cdef int [:] Ia = np.arange(alpha_nel, dtype=int_dtype)
    cdef int [:] Ib
    cdef int Ia_i, Ib_i
    cdef double S = 0.0
    for Ia_i in range(nstr_alpha):
        if (exc[0] in Ia) and (exc[1] not in Ia):
            Ib = np.arange(beta_nel, dtype=int_dtype)
            for Ib_i in range(nstr_beta):
                if (exc[2] in Ib) and (exc[3] not in Ib):
                    S += wf[Ia_i, Ib_i]*wf[Ia_i, Ib_i]
                next_str(Ib)
        next_str(Ia)
    return S


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef int [:] _exc_on_string(int i, int a, int [:] I):
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
    cdef int [:] new_I = np.empty(n+1, dtype=int_dtype)
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
