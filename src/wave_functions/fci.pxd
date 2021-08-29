from wave_functions.general cimport WaveFunction


cdef class FCIWaveFunction(WaveFunction):
    cdef double [:,:] coefficients
    cdef int [:,:] _alpha_string_graph
    cdef int [:,:] _beta_string_graph
    cdef bint n_alpha_str_init
    cdef bint n_beta_str_init
    cdef int _n_alpha_str
    cdef int _n_beta_str
    cdef readonly object ref_det
    cpdef (int, int) index(self,  det) except *
