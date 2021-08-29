

cdef class OrbitalSpace:
    cdef int[16] _dim_per_irrep
    cdef str _type
    cdef int _n_irrep


cdef class FullOrbitalSpace:
    cdef readonly int n_irrep
    cdef readonly OrbitalSpace full, froz, ref, virt, corr, act
    cdef readonly int n_orb, n_orb_nofrozen
    cdef readonly int[9] orbs_before
    cdef readonly int[17] corr_orbs_before
    cpdef set_n_irrep(self, int n)
    cpdef set_full(self, OrbitalSpace other, bint update=*)
    cpdef add_to_full(self, OrbitalSpace other, bint update=*)
    cpdef set_froz(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef add_to_froz(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef set_ref(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef add_to_ref(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef set_act(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef add_to_act(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef calc_remaining(self)
    cpdef get_attributes_from(self, FullOrbitalSpace other)
