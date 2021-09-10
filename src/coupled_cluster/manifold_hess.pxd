from coupled_cluster.manifold_util cimport (
    SingleExc, DoubleExc, TripleExc, QuadrupleExc)
from wave_functions.fci import FCIWaveFunction
from wave_functions.fci cimport FCIWaveFunction


cdef int calc_H_a(double [:] H,
                  SingleExc single_exc,
                  FCIWaveFunction wf,
                  FCIWaveFunction wf_cc,
                  int pos_ini) except -1


cdef int calc_H_b(double [:] H,
                  SingleExc single_exc,
                  FCIWaveFunction wf,
                  FCIWaveFunction wf_cc,
                  int pos_ini) except -1


cdef int calc_H_aa(double [:] H,
                   DoubleExc double_exc,
                   FCIWaveFunction wf,
                   FCIWaveFunction wf_cc,
                   int pos_ini) except -1


cdef int calc_H_bb(double [:] H,
                   DoubleExc double_exc,
                   FCIWaveFunction wf,
                   FCIWaveFunction wf_cc,
                   int pos_ini) except -1


cdef int calc_H_ab(double [:] H,
                   DoubleExc double_exc,
                   FCIWaveFunction wf,
                   FCIWaveFunction wf_cc,
                   int pos_ini) except -1
