import numpy as np

from util.variables import int_dtype
from input_output import log
from coupled_cluster.manifold_term1 cimport (term1_ab, term1_ab_v2)
from coupled_cluster.manifold_util cimport DoubleExc


def testspeed(i,j,a,b,
              double[:, :] wf,
              double[:, :] wf_cc,
              int[:, :] alpha_string_graph,
              int[:, :] beta_string_graph):
    cdef DoubleExc exc
    exc.i = i
    exc.j = j
    exc.a = a
    exc.b = b
    return term1_ab(exc, wf, wf_cc, alpha_string_graph, beta_string_graph)


def testspeed_2(i,j,a,b,
                double[:, :] wf,
                double[:, :] wf_cc,
                int[:, :] alpha_string_graph,
                int[:, :] beta_string_graph):
    cdef int[:] I_buff, I_exc_buff, I_exc2_buff
    cdef int nalpha, nbeta
    cdef DoubleExc exc
    nalpha = alpha_string_graph.shape[1]
    nbeta = beta_string_graph.shape[1]
    exc.i = i
    exc.j = j
    exc.a = a
    exc.b = b
    Ia_buff = np.empty(nalpha + 1, dtype=int_dtype)
    Ia_exc_buff = np.empty(nalpha + 1, dtype=int_dtype)
    Ib_buff = np.empty(nbeta + 1, dtype=int_dtype)
    Ib_exc_buff = np.empty(nbeta + 1, dtype=int_dtype)
    return term1_ab_v2(exc, wf, wf_cc, alpha_string_graph, beta_string_graph,
                       Ia_buff, Ia_exc_buff,
                       Ib_buff, Ib_exc_buff)


