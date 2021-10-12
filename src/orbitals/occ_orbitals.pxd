cdef class OccOrbital:
    cdef readonly int pos_in_occ, orb, orbirp, spirrep
    cdef int _n_occ, _n_irrep
    cdef int[:] _corr_orb, _orbs_before
    cdef bint is_alpha
    cdef readonly bint alive
    cpdef rewind(self)
    cpdef next_(self)
