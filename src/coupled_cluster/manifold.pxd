cdef int EXC_TYPE_A
cdef int EXC_TYPE_B
cdef int EXC_TYPE_AA
cdef int EXC_TYPE_AB
cdef int EXC_TYPE_BB

cdef double _term1(int [:] exc,
                   int exc_type,
                   double [:, :] wf,
                   double [:, :] wf_cc,
                   int [:, :] alpha_string_graph,
                   int [:, :] beta_string_graph)

cdef double _term2_diag(int [:] exc,
                        int exc_type,
                        double [:, :] wf_cc,
                        int alpha_nel,
                        int beta_nel)

cdef int [:] _exc_on_string(int i, int a, int [:] I)
