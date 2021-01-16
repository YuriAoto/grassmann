"""Core functions to optimise distance to the CC manifold


"""
import cython
import numpy as np


from wave_functions.strings_rev_lexical_order import next_str, get_index
from util import irrep_product

cdef int EXC_TYPE_A = 0
cdef int EXC_TYPE_B = 1
cdef int EXC_TYPE_AA = 2
cdef int EXC_TYPE_AB = 3
cdef int EXC_TYPE_BB = 4

int_dtype = np.intc


def min_dist_app_hess(wf, wf_cc, int n_ampl, level='SD'):
    """Calculate the Jacobian and approximate the Hessian
    
    Parameters:
    -----------
    wf (WaveFunctionFCI)
        The wave function
    
    wf_cc (2D np.array)
        The coupled cluster wave function
    
    n_ampl (int)
        The total number of amplitudes
    
    level (str, optional, default='SD')
        Coupled cluster level
    
    Return:
    -------
    
    
    """
    cdef double normJac = 0.0
    cdef double [:] z = np.zeros(n_ampl)
    cdef int pos = 0, exc_type
    cdef int spirrep, irrep, i_irrep, j_irrep, a_irrep, b_irrep, i, j, a, b
    cdef int [:] single_exc = np.zeros(2)
    cdef int [:] double_exc = np.zeros(4)
    for spirrep in range(2 * wf.n_irrep):
        irrep = spirrep % wf.n_irrep
        exc_type = EXC_TYPE_A if spirrep < wf.n_irrep else EXC_TYPE_B
        single_exc[0] = wf.n_orb_before[irrep]
        for i in range(wf.n_corr_orb[spirrep]):
            single_exc[0] += 1
            single_exc[1] = wf.n_orb_before[irrep]
            for a in range(wf.n_ext[spirrep]):
                single_exc[1] += 1
                J = _term1(single_exc,
                           exc_type,
                           wf._coeffients,
                           wf_cc._coeffients,
                           wf._alpha_string_graph,
                           wf._beta_string_graph)
                normJac += J**2
                z[pos] = J/_term2(single_exc,
                                  exc_type,
                                  wf_cc._coeffients,
                                  wf._alpha_string_graph,
                                  wf._beta_string_graph)
                pos += 1
    for i_irrep in range(2 * wf.n_irrep):
        alpha_exc1 = i_irrep < wf.n_irrep
        for j_irrep in range(i_irrep, 2 * wf.n_irrep):
            alpha_exc2 = j_irrep < wf.n_irrep
            exc_type = (EXC_TYPE_AA
                        if alpha_exc1 and alpha_exc2 else
                        (EXC_TYPE_AB
                         if alpha_exc1 or alpha_exc2 else
                         EXC_TYPE_BB))
            double_exc[0] = wf.n_orb_before[i_irrep]
            for i in range(wf.n_corr_orb[i_irrep]):
                double_exc[0] += 1
                double_exc[1] = wf.n_orb_before[i_irrep]
                for j in range(wf.n_corr_orb[j_irrep]):
                    if j <= i:
                        continue
                    double_exc[1] += 1
                    for a_irrep in range(wf.n_irrep):
                        double_exc[2] = wf.n_orb_before[a_irrep]
                        for a in range(wf.n_ext[a_irrep
                                                + (0
                                                   if alpha_exc1 else
                                                   wf.n_irrep)]):
                            double_exc[2] += 1
                            b_irrep = (irrep_product[i_irrep]
                                       * irrep_product[j_irrep]
                                       * irrep_product[a_irrep])
                            double_exc[3] = wf.n_orb_before[b_irrep]
                            for b in range(wf.n_ext[b_irrep
                                                    + (0
                                                       if alpha_exc2 else
                                                       wf.n_irrep)]):
                                double_exc[3] += 1
                                J = _term1(double_exc,
                                           exc_type,
                                           wf._coeffients,
                                           wf_cc._coeffients,
                                           wf._alpha_string_graph,
                                           wf._beta_string_graph)
                                normJac += J**2
                                z[pos] = J/_term2(double_exc,
                                                  exc_type,
                                                  wf_cc._coeffients,
                                                  wf._alpha_string_graph,
                                                  wf._beta_string_graph)
                pos += 1
    if pos != n_ampl:
        raise Exception('pos != n_ampl')
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


cdef double _term2(int [:] exc,
                   int exc_type,
                   double [:, :] wf_cc,
                   int [:, :] alpha_string_graph,
                   int [:, :] beta_string_graph):
    """The term <\Psi_cc | \tau_\rho | \Psi_cc>
    
    Parameters:
    -----------
    See _term1
    
    Return:
    -------
    A float (C double)
    
    """
    if exc_type == EXC_TYPE_A:
        return _term2_a(exc, wf_cc,
                        alpha_string_graph)
    if exc_type == EXC_TYPE_AA:
        return _term2_aa(exc, wf_cc,
                         alpha_string_graph)
    if exc_type == EXC_TYPE_B:
        return _term2_b(exc, wf_cc,
                        beta_string_graph)
    if exc_type == EXC_TYPE_BB:
        return _term2_bb(exc, wf_cc,
                         beta_string_graph)
    if exc_type == EXC_TYPE_AB:
        return _term2_ab(exc, wf_cc,
                         alpha_string_graph,
                         beta_string_graph)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_a(int [:] exc, # [i, a]
                     double [:, :] wf,
                     double [:, :] wf_cc,
                     int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\alpha}^{\alpha}
    
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
    cdef int I_index, I_exc_index, i
    cdef double S = 0.0
    for I_index in range(nstr_alpha):
        if (exc[0] not in I) or (exc[1] in I):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        I_exc_index = get_index(I_exc[:nel], string_graph)
        with nogil:
            for i in range(nstr_beta):
                S += (I_exc[nel]
                      * (wf[I_exc_index, i] - wf_cc[I_exc_index, i])
                      * wf_cc[I_index, i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_b(int [:] exc, # [i, a]
                     double [:, :] wf,
                     double [:, :] wf_cc,
                     int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\beta}^{\beta}
    
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
    cdef int I_index, I_exc_index, i
    cdef double S = 0.0
    for I_index in range(nstr_beta):
        if (exc[0] not in I) or (exc[1] in I):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        I_exc_index = get_index(I_exc[:nel], string_graph)
        for i in range(nstr_alpha):
            S += (I_exc[nel]
                  * (wf[i, I_exc_index] - wf_cc[i, I_exc_index])
                  * wf_cc[i, I_index])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_aa(int [:] exc, # [i, a, j, b]
                      double [:, :] wf,
                      double [:, :] wf_cc,
                      int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\alpha\alpha}^{\alpha\alpha}
    
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
    cdef int I_index, I_exc_index, sign, i
    cdef double S = 0.0
    for I_index in range(nstr_alpha):
        if ((exc[0] not in I) or (exc[1] in I)
                or (exc[2] not in I) or (exc[3] in I)):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        sign = I_exc[nel]
        I_exc = _exc_on_string(exc[2], exc[3], I_exc[:nel])
        I_exc_index = get_index(I_exc[:nel], string_graph)
        for i in range(nstr_beta):
            S += (I_exc[nel] * sign
                  * (wf[I_exc_index, i] - wf_cc[I_exc_index, i])
                  * wf_cc[I_index, i])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_bb(int [:] exc, # [i, a, j, b]
                      double [:, :] wf,
                      double [:, :] wf_cc,
                      int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\beta\beta}^{\beta\beta}
    
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
    cdef int I_index, I_exc_index, sign, i
    cdef double S = 0.0
    for I_index in range(nstr_beta):
        if ((exc[0] not in I) or (exc[1] in I)
                or (exc[2] not in I) or (exc[3] in I)):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        sign = I_exc[nel]
        I_exc = _exc_on_string(exc[2], exc[3], I_exc[:nel])
        I_exc_index = get_index(I_exc[:nel], string_graph)
        for i in range(nstr_alpha):
            S += (I_exc[nel]
                  * (wf[i, I_exc_index] - wf_cc[i, I_exc_index])
                  * wf_cc[i, I_index])
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term1_ab(int [:] exc, # [i, a, j, b]
                      double [:, :] wf,
                      double [:, :] wf_cc,
                      int [:, :] alpha_string_graph,
                      int [:, :] beta_string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\alpha\beta}^{\alpha\beta}
    
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
    cdef int Ia_index, Ib_index, Ia_exc_index, Ib_exc_index
    cdef double S = 0.0
    for Ia_index in range(nstr_alpha):
        if (exc[0] not in Ia) or (exc[1] in Ia):
            continue
        Ia_exc = _exc_on_string(exc[0], exc[1], Ia)
        Ia_exc_index = get_index(Ia_exc[:nalpha], alpha_string_graph)
        for Ib_index in range(nstr_beta):
            if (exc[2] not in Ib) or (exc[3] in Ib):
                continue
            Ib_exc = _exc_on_string(exc[2], exc[3], Ib)
            Ib_exc_index = get_index(Ib_exc[:nbeta], beta_string_graph)
            S += (Ia_exc[nalpha] * Ib_exc[nbeta]
                  * (wf[Ia_exc_index, Ib_exc_index]
                     - wf_cc[Ia_exc_index, Ib_exc_index])
                  * wf_cc[Ia_index, Ib_index])
            next_str(Ib)
        next_str(Ia)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_a(int [:] exc, # [i, a]
                     double [:, :] wf_cc,
                     int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\alpha}^{\alpha}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the first dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr = wf_cc.shape[0]
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_index, I_exc_index
    cdef double S = 0.0
    for I_index in range(nstr):
        if (exc[0] not in I) or (exc[1] in I):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        I_exc_index = get_index(I_exc[:nel], string_graph)
        S += (I_exc[nel]
              * np.dot(wf_cc[I_exc_index, :],
                       wf_cc[I_index, :]))
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_b(int [:] exc, # [i, a]
                     double [:, :] wf_cc,
                     int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\beta}^{\beta}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the second dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr = wf_cc.shape[0]
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_index, I_exc_index
    cdef double S = 0.0
    for I_index in range(nstr):
        if (exc[0] not in I) or (exc[1] in I):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        I_exc_index = get_index(I_exc[:nel], string_graph)
        S += (I_exc[nel]
              * np.dot(wf_cc[:, I_exc_index],
                       wf_cc[:, I_index]))
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_aa(int [:] exc, # [i, a, j, b]
                      double [:, :] wf_cc,
                      int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\alpha\alpha}^{\alpha\alpha}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the first dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr = wf_cc.shape[0]
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_index, I_exc_index, sign
    cdef double S = 0.0
    for I_index in range(nstr):
        if ((exc[0] not in I) or (exc[1] in I)
                or (exc[2] not in I) or (exc[3] in I)):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        sign = I_exc[nel]
        I_exc = _exc_on_string(exc[2], exc[3], I_exc[:nel])
        I_exc_index = get_index(I_exc[:nel], string_graph)
        S += (I_exc[nel] * sign
              * np.dot(wf_cc[I_exc_index, :],
                       wf_cc[I_index, :]))
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_bb(int [:] exc, # [i, a, j, b]
                      double [:, :] wf_cc,
                      int [:, :] string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\beta\beta}^{\beta\beta}
    
    Parameters:
    -----------
    See _term1
    
    string_graph should be associated to the second dimension of wf and wf_cc
    
    Return:
    -------
    A float (C double)
    """
    cdef int nel = string_graph.shape[1]
    cdef int nstr = wf_cc.shape[0]
    cdef int [:] I = np.arange(nel, dtype=int_dtype)
    cdef int [:] I_exc = np.empty(nel+1, dtype=int_dtype)
    cdef int I_index, I_exc_index, sign
    cdef double S = 0.0
    for I_index in range(nstr):
        if ((exc[0] not in I) or (exc[1] in I)
                or (exc[2] not in I) or (exc[3] in I)):
            continue
        I_exc = _exc_on_string(exc[0], exc[1], I)
        sign = I_exc[nel]
        I_exc = _exc_on_string(exc[2], exc[3], I_exc[:nel])
        I_exc_index = get_index(I_exc[:nel], string_graph)
        S += (I_exc[nel] * sign
              * np.dot(wf_cc[:, I_exc_index],
                       wf_cc[:, I_index]))
        next_str(I)
    return S


##@cython.boundscheck(False)  # Deactivate bounds checking
##@cython.wraparound(False)   # Deactivate negative indexing
cdef double _term2_ab(int [:] exc, # [i, a, j, b]
                      double [:, :] wf_cc,
                      int [:, :] alpha_string_graph,
                      int [:, :] beta_string_graph):
    """<\Psi_cc | \tau_\rho | \Psi_cc> for \rho=a_{\alpha\beta}^{\alpha\beta}
    
    Parameters:
    -----------
    See _term1
    
    Return:
    -------
    A float (C double)
    """
    cdef int nalpha = alpha_string_graph.shape[1]
    cdef int nbeta = beta_string_graph.shape[1]
    cdef int nstr_alpha = wf_cc.shape[0]
    cdef int nstr_beta = wf_cc.shape[1]
    cdef int [:] Ia = np.arange(nalpha, dtype=int_dtype)
    cdef int [:] Ib = np.arange(nbeta, dtype=int_dtype)
    cdef int [:] Ia_exc = np.empty(nalpha + 1, dtype=int_dtype)
    cdef int [:] Ib_exc = np.empty(nbeta + 1, dtype=int_dtype)
    cdef int Ia_index, Ib_index, Ia_exc_index, Ib_exc_index
    cdef double S = 0.0
    for Ia_index in range(nstr_alpha):
        if (exc[0] not in Ia) or (exc[1] in Ia):
            continue
        Ia_exc = _exc_on_string(exc[0], exc[1], Ia)
        Ia_exc_index = get_index(Ia_exc[:nalpha], alpha_string_graph)
        for Ib_index in range(nstr_beta):
            if (exc[2] not in Ib) or (exc[3] in Ib):
                continue
            Ib_exc = _exc_on_string(exc[2], exc[3], Ib)
            Ib_exc_index = get_index(Ib_exc[:nbeta], beta_string_graph)
            S += (Ia_exc[nalpha] * Ib_exc[nbeta]
                  * wf_cc[Ia_exc_index, Ib_exc_index]
                  * wf_cc[Ia_index, Ib_index])
            next_str(Ib)
        next_str(Ia)
    return S


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
    cdef int pos, i_pos, a_pos, orb
    i_pos = 0
    a_pos = 0
    new_I[:n] = I[:]
    pos = 0
    if i < a:
        for orb in I:
            if orb == i:
                i_pos = pos
            if orb > a:
                a_pos = pos-1
                break
            pos += 1
        new_I[i_pos: a_pos] = I[i_pos+1: a_pos+1]
        new_I[a_pos] = a
    elif i > a:
        for orb in I:
            if orb == i:
                i_pos = pos
                break
            if orb > a:
                a_pos = pos
            pos += 1
        new_I[a_pos+1: i_pos+1] = I[a_pos: i_pos]
        new_I[a_pos] = a
    new_I[-1] = 1 - 2*(abs(a_pos - i_pos) % 2)
    return new_I
