from wave_functions.slater_det cimport SlaterDet
from wave_functions.general cimport WaveFunction
from orbitals.orbital_space cimport FullOrbitalSpace, OrbitalSpace

cdef class FCIWaveFunction(WaveFunction):
    cdef double [:,:] coefficients
    cdef int [:,:] alpha_string_graph
    cdef int [:,:] beta_string_graph
    cdef bint n_alpha_str_init
    cdef bint n_beta_str_init
    cdef int _n_alpha_str
    cdef int _n_beta_str
    cdef bint _ordered_orbs
    cdef object _normalisation
    cdef object _sign_change_orbs
    cdef readonly SlaterDet ref_det
    cpdef (int, int) index(self,  det) except *
    cdef (int, int) set_slater_det(self, SlaterDet det) except *
    cdef int sign_change_orb_from(self, FCIWaveFunction wf) except -1
    cdef int initialize_coeff_matrix(self) except -1
