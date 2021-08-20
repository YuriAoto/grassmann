

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
    cdef set_n_irrep(self, int n)
    cdef set_full(self, OrbitalSpace other, bint update=*)
    cdef add_to_full(self, OrbitalSpace other, bint update=*)
    cdef set_froz(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef add_to_froz(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef set_ref(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef add_to_ref(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef set_act(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef add_to_act(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef calc_remaining(self)
    cdef get_attributes_from(self, FullOrbitalSpace other)
