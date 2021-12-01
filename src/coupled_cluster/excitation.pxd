cdef class SDExcitation:
    cdef int alpha_rank, beta_rank
    cdef int[2] alpha_h, alpha_p
    cdef int[2] beta_h, beta_p
    cdef inline int rank(self)
    cdef void add_alpha_hp(self, int i, int a)
    cdef void add_beta_hp(self, int i, int a)
