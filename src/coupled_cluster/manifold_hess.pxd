from coupled_cluster.manifold_util cimport (
    SingleExc, DoubleExc, TripleExc, QuadrupleExc)


cdef int calc_H_a(double [:] H,
                  SingleExc single_exc,
                  double[:, :] wf,
                  double[:, :] wf_cc,
                  int pos_ini,
                  int n_irrep,
                  int[:] orbs_before,
                  int[:] corr_orb,
                  int[:] virt_orb,
                  int[:, :] alpha_string_graph,
                  int[:, :] beta_string_graph)


cdef int calc_H_b(double [:] H,
                  SingleExc single_exc,
                  double[:, :] wf,
                  double[:, :] wf_cc,
                  int pos_ini,
                  int n_irrep,
                  int[:] orbs_before,
                  int[:] corr_orb,
                  int[:] virt_orb,
                  int[:, :] alpha_string_graph,
                  int[:, :] beta_string_graph)


cdef int calc_H_aa(double [:] H,
                   DoubleExc double_exc,
                   double[:, :] wf,
                   double[:, :] wf_cc,
                   int pos_ini,
                   int n_irrep,
                   int[:] orbs_before,
                   int[:] corr_orb,
                   int[:] virt_orb,
                   int[:, :] alpha_string_graph,
                   int[:, :] beta_string_graph)


cdef int calc_H_bb(double [:] H,
                   DoubleExc double_exc,
                   double[:, :] wf,
                   double[:, :] wf_cc,
                   int pos_ini,
                   int n_irrep,
                   int[:] orbs_before,
                   int[:] corr_orb,
                   int[:] virt_orb,
                   int[:, :] alpha_string_graph,
                   int[:, :] beta_string_graph)


cdef int calc_H_ab(double [:] H,
                   DoubleExc double_exc,
                   double[:, :] wf,
                   double[:, :] wf_cc,
                   int pos_ini,
                   int n_irrep,
                   int[:] orbs_before,
                   int[:] corr_orb,
                   int[:] virt_orb,
                   int[:, :] alpha_string_graph,
                   int[:, :] beta_string_graph)
