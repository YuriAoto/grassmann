from orbitals.orbital_space cimport FullOrbitalSpace
from orbitals.orbital_space import FullOrbitalSpace

cdef class WaveFunction:
    cdef public object wf_type
    cdef public object source
    cdef public object point_group
    cdef public bint restricted
    cdef public int _irrep
    cdef public double Ms
    cdef readonly FullOrbitalSpace orbspace
    cdef public double mem
#    cdef public void _set_memory(self, object destination=*, object calc_args=*)
#    cdef public double calc_memory(self, calc_args)
