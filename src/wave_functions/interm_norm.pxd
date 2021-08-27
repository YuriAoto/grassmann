from wave_functions.general cimport WaveFunction
from wave_functions.general import WaveFunction
cdef enum ExcType:
   ALL, A, B, AA, AB, BB

cdef class IntermNormWaveFunction(WaveFunction):
    cdef public bint use_CISD_norm
    cdef double _norm
    cdef double[:] amplitudes
    cdef int[:] ini_blocks_S
    cdef int[:,:] ini_blocks_D
    cdef int first_bb_pair
    cdef int first_ab_pair
    cdef int _n_ampl
    cdef int _add_block_for_calc_ini_blocks(self,
                                            int i_in_D,
                                            int spirrep_a,
                                            int spirrep_b,
                                            int [:] raveled_ini_blocks_D)
    cdef inline int get_ij_pos_from_i_j(self,
                                        int i,
                                        int j,
                                        int irrep_i,
                                        ExcType exc_type) except -1
    cdef int initialize_amplitudes(self) except -1
    cdef void _calc_ini_blocks(self)
