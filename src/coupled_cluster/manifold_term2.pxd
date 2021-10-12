from coupled_cluster.manifold_util cimport SingleExc, DoubleExc, TripleExc, QuadrupleExc

cdef double term2_diag_a(SingleExc exc,
                         double[:, :] wf,
                         int[:] occ)

cdef double term2_diag_b(SingleExc exc,
                         double[:, :] wf,
                         int[:] occ)

cdef double term2_diag_aa(DoubleExc exc,
                          double[:, :] wf,
                          int[:] occ)

cdef double term2_diag_bb(DoubleExc exc,
                           double[:, :] wf,
                           int[:] occ)

cdef double term2_diag_ab(DoubleExc exc,
                           double[:, :] wf,
                           int[:] occ_a,
                           int[:] occ_b)

cdef double term2_aa(DoubleExc exc,
                     double[:, :] wf,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ)

cdef double term2_bb(DoubleExc exc,
                     double[:, :] wf,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ)

cdef double term2_aaa(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ)

cdef double term2_bbb(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ)

cdef double term2_aaaa(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ)

cdef double term2_bbbb(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ)

cdef double term2_ab(DoubleExc exc,
                     double[:, :] wf,
                     int[:, :] alpha_string_graph,
                     int[:, :] beta_string_graph,
                     int[:] occ_a,
                     int[:] exc_occ_a,
                     int[:] occ_b,
                     int[:] exc_occ_b)

cdef double term2_aab(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term2_abb(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term2_aaab(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term2_aabb(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term2_abbb(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term2_baa(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term2_bab(TripleExc exc,
                      double[:, :] wf,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term2_bbab(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term2_abab(QuadrupleExc exc,
                       double[:, :] wf,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)
