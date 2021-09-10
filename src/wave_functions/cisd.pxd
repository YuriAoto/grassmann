from wave_functions.general cimport WaveFunction

cdef class CISDWaveFunction(WaveFunction):
    cdef readonly float C0
    cdef readonly object Cs
    cdef readonly object Cd
    cdef readonly object Csd
