

cdef class OrbitalSpace:
    cdef int[16] _dim_per_irrep
    cdef str _type
    cdef int n_irrep


cdef class FullOrbitalSpace:
    cdef readonly int n_irrep
    cdef readonly OrbitalSpace full, froz, ref, virt, corr, act
    cdef readonly int n_orb, n_orb_nofrozen
    cdef readonly int[9] orbs_before
    cdef readonly int[17] corr_orbs_before
    cpdef set_n_irrep(self, int n)
    cpdef set_full(self, OrbitalSpace other, bint update=*)
    cpdef add_to_full(self, OrbitalSpace other, bint update=*)
    cpdef set_froz(self, OrbitalSpace other, bint update=*, bint add_to_full=*, bint add_to_ref=*)
    cpdef add_to_froz(self, OrbitalSpace other, bint update=*, bint add_to_full=*, bint add_to_ref=*)
    cpdef set_ref(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef add_to_ref(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef set_act(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cpdef add_to_act(self, OrbitalSpace other, bint update=*, bint add_to_full=*)
    cdef calc_remaining(self)
    cpdef get_attributes_from(self, FullOrbitalSpace other)
    cdef inline int first_virtual(self, int spirrep)
    cdef int get_orb_irrep(self, int orb) except -1
    cdef (int, int) get_local_index(self,
                                    int p,
                                    bint alpha_orb) except *
    cdef int get_absolute_index(self,
                                int p,
                                int irrep,
                                bint occupied,
                                bint alpha_orb) except -1
    cdef int get_abs_corr_index(self,
                                int p,
                                int irrep,
                                bint alpha_orb) except -1
