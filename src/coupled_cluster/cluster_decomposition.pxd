cdef class ClusterDecomposition:
    cdef object recipes_f
    cdef int [:,:] recipes_D
    cdef int [:,:] recipes_T
    cdef int [:,:] recipes_Q
    cdef int [:,:] recipes_5
    cdef int [:,:] recipes_6
    cdef int [:,:] recipes_7 
    cdef int [:,:] recipes_8
    
    cdef bint loaded_D
    cdef bint loaded_T
    cdef bint loaded_Q
    cdef bint loaded_5
    cdef bint loaded_6
    cdef bint loaded_7
    cdef bint loaded_8
    
    cdef int[:,:] select_recipe(self, int rank)
    cdef int n_rules(self, int rank)
    cdef decompose(self, alpha_hp, beta_hp, mode=*)
