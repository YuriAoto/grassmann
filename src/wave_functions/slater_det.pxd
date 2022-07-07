from orbitals.orbital_space cimport FullOrbitalSpace, OrbitalSpace

cdef class SlaterDet():
    cdef public double c
    cdef public int[:] alpha_occ, beta_occ
    cdef OrbitalSpace orbspace(self, FullOrbitalSpace)