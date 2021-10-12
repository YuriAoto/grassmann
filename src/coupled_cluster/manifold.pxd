from orbitals.orbital_space cimport FullOrbitalSpace
from orbitals.orbital_space import FullOrbitalSpace

cdef int update_indep_amplitudes(double [:] amplitudes,
                                 double [:] z,
                                 int n_singles,
                                 int n_ampl,
                                 int n_indep_ampl,
                                 FullOrbitalSpace orbspace) except -1
