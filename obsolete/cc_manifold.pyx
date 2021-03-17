
#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
cdef int[:] exc_on_string(int i, int a, int[:] I):
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


cdef double term1(int[:] exc,
                  int exc_type,
                  double[:, :] wf,
                  double[:, :] wf_cc,
                  int[:, :] alpha_string_graph,
                  int[:, :] beta_string_graph):
    """The term <\Psi - \Psi_cc | \tau_\rho | \Psi_cc>
    
    We recommend to use the functions _term1_a, _term1_b,
    _term1_aa, _term1_bb, and _term1_ab directly, since they
    are cdef functions and strongly typed.
    
    Parameters:
    -----------
    exc
        the excitation, with [i, a] for single and [i, a, j, b] for double
    
    exc_type
        the excitation type.
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
        return term1_a(single_exc, wf, wf_cc,
                       alpha_string_graph)
    if exc_type == EXC_TYPE_AA:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return term1_aa(double_exc, wf, wf_cc,
                        alpha_string_graph)
    if exc_type == EXC_TYPE_B:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return term1_b(single_exc, wf, wf_cc,
                       beta_string_graph)
    if exc_type == EXC_TYPE_BB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return term1_bb(double_exc, wf, wf_cc,
                        beta_string_graph)
    if exc_type == EXC_TYPE_AB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return term1_ab(double_exc, wf, wf_cc,
                        alpha_string_graph,
                        beta_string_graph)


cdef double term2_diag(int[:] exc,
                        int exc_type,
                        double[:, :] wf,
                        int alpha_nel,
                        int beta_nel):
    """The term <\Psi_cc | \tau_\rho^\dagger \tau_\rho | \Psi_cc>
    
    Parameters:
    -----------
    See term1 (here wf is \Psi_cc)
    
    Return:
    -------
    A float (C double)
    
    """
    cdef DoubleExc double_exc
    cdef SingleExc single_exc
    if exc_type == EXC_TYPE_A:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return term2_diag_a(single_exc, wf, alpha_nel)
    if exc_type == EXC_TYPE_AA:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return term2_diag_aa(double_exc, wf, alpha_nel)
    if exc_type == EXC_TYPE_B:
        single_exc.i = exc[0]
        single_exc.a = exc[1]
        return term2_diag_b(single_exc, wf, beta_nel)
    if exc_type == EXC_TYPE_BB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return term2_diag_bb(double_exc, wf, beta_nel)
    if exc_type == EXC_TYPE_AB:
        double_exc.i = exc[0]
        double_exc.a = exc[1]
        double_exc.j = exc[2]
        double_exc.b = exc[3]
        return term2_diag_ab(double_exc, wf, alpha_nel, beta_nel)
