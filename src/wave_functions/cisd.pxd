from wave_functions.general cimport WaveFunction

cdef class CISD_WaveFunction(WaveFunction):
    cdef object C0
    cdef object Cs
    cdef object Cd
    cdef object Csd
