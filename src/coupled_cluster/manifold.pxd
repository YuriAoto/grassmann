cdef struct SingleExc:
    int i, a

cdef struct DoubleExc:
    int i, j, a, b

cdef double _term1(int[:] exc,
                   int exc_type,
                   double[:, :] wf,
                   double[:, :] wf_cc,
                   int[:, :] alpha_string_graph,
                   int[:, :] beta_string_graph)

cdef double _term2_diag(int[:] exc,
                        int exc_type,
                        double[:, :] wf_cc,
                        int alpha_nel,
                        int beta_nel)

cdef double _term1_a(SingleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph)

cdef int[:] _exc_on_string(int i, int a, int[:] I)
