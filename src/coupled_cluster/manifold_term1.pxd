from coupled_cluster.manifold_util cimport SingleExc, DoubleExc, TripleExc, QuadrupleExc


cdef double term1_a(SingleExc exc,
                    double[:, :] wf,
                    double[:, :] wf_cc,
                    int[:, :] string_graph,
                    int[:] occ,
                    int[:] exc_occ)

cdef double term1_b(SingleExc exc,
                    double[:, :] wf,
                    double[:, :] wf_cc,
                    int[:, :] string_graph,
                    int[:] occ,
                    int[:] exc_occ)

cdef double term1_aa(DoubleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ)

cdef double term1_bb(DoubleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] string_graph,
                     int[:] occ,
                     int[:] exc_occ)

cdef double term1_aaa(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ)

cdef double term1_bbb(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] string_graph,
                      int[:] occ,
                      int[:] exc_occ)

cdef double term1_aaaa(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ)

cdef double term1_bbbb(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] string_graph,
                       int[:] occ,
                       int[:] exc_occ)

cdef double term1_ab(DoubleExc exc,
                     double[:, :] wf,
                     double[:, :] wf_cc,
                     int[:, :] alpha_string_graph,
                     int[:, :] beta_string_graph,
                     int[:] occ_a,
                     int[:] exc_occ_a,
                     int[:] occ_b,
                     int[:] exc_occ_b)

cdef double term1_aab(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term1_abb(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term1_aaab(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term1_aabb(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term1_abbb(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term1_baa(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term1_bab(TripleExc exc,
                      double[:, :] wf,
                      double[:, :] wf_cc,
                      int[:, :] alpha_string_graph,
                      int[:, :] beta_string_graph,
                      int[:] occ_a,
                      int[:] exc_occ_a,
                      int[:] occ_b,
                      int[:] exc_occ_b)

cdef double term1_bbab(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

cdef double term1_abab(QuadrupleExc exc,
                       double[:, :] wf,
                       double[:, :] wf_cc,
                       int[:, :] alpha_string_graph,
                       int[:, :] beta_string_graph,
                       int[:] occ_a,
                       int[:] exc_occ_a,
                       int[:] occ_b,
                       int[:] exc_occ_b)

