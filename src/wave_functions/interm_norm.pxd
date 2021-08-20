from wave_functions.general cimport WaveFunction
from wave_functions.general import WaveFunction


cdef class IntermNormWaveFunction(WaveFunction):
    cdef double _norm
    cdef bint use_CISD_norm
    cdef double[:] amplitudes
    cdef int[:] ini_blocks_S
    cdef int[:,:] ini_blocks_D
    cdef int first_bb_pair
    cdef int first_ab_pair
    cdef int _n_ampl
