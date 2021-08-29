"""


"""
import copy

import numpy as np

from orbitals.orbital_space cimport FullOrbitalSpace

cdef class OccOrbital:
    """Class to run over occupied orbitals

    Usage:
    ------
    # alpha and beta correlated orbitals
    corr_orb = np.array([5,2,2,0, 4,2,2,0], dtype=int_dtype)
    # number of orbitals before that irrep
    orbs_before = np.array([0, 10, 15, 20, 22])

    i = OccOrbital(corr_orb, orbs_before, True)
    i.pos_in_occ  # 0
    i.orb  # 0
    i.orbirp  # 0
    i.spirrep  # 0
    i.alive  # True

    i.next_()

    i.pos_in_occ  # 1
    i.orb  # 1
    i.orbirp  # 1
    i.spirrep  # 0

    for k in range(3):
       i.next_()

    i.pos_in_occ  # 4
    i.orb  # 4
    i.orbirp  # 4
    i.spirrep  # 0

    i.next_()

    i.pos_in_occ  # 5
    i.orb  # 10
    i.orbirp  # 0
    i.spirrep  # 1

    for k in range(4):
       i.next_()

    i.pos_in_occ # 9 > n of occupied orb
    i.alive  # False

    i.rewind()
    i.pos_in_occ  # 0
    i.orb  # 0
    i.orbirp  # 0
    i.spirrep  # 0
    i.alive  # True


    """
    def __init__(self,
                  FullOrbitalSpace orbspace,
                  bint is_alpha):
        cdef int irrep, spirrep
        self._n_irrep = orbspace.n_irrep
        self.is_alpha = is_alpha
        self._corr_orb = np.array(orbspace.corr)
        self._orbs_before = orbspace.orbs_before
        self._n_occ = 0
        self.spirrep = 0 if is_alpha else self._n_irrep
        spirrep = self.spirrep
        self.orb = -1
        for irrep in range(self._n_irrep):
            self._n_occ += self._corr_orb[spirrep]
            if self.orb == -1 and self._corr_orb[spirrep] > 0:
                self.orb = self._orbs_before[irrep]
                self.spirrep = spirrep
            spirrep += 1
        if self._n_occ == 0:
            self.alive = False
            return
        self.alive = True
        self.pos_in_occ = 0
        self.orbirp = 0

    def __str__(self):
        return f'Occupied orbital: {self.orb} of spirrep {self.spirrep}'

    cpdef rewind(self):
        cdef int irrep = 0
        if self._n_occ == 0:
            self.alive = False
            return
        self.alive = True
        self.spirrep = 0 if self.is_alpha else self._n_irrep
        while self._corr_orb[self.spirrep] == 0:
            self.spirrep += 1
            irrep += 1
        self.orb = self._orbs_before[irrep]
        self.orbirp = 0
        self.pos_in_occ = 0

    cpdef next_(self):
        self.pos_in_occ += 1
        if self.pos_in_occ < self._n_occ:
            self.orb += 1
            self.orbirp += 1
            if self.orb < (self._orbs_before[self.spirrep % self._n_irrep]
                           + self._corr_orb[self.spirrep]):
                return
            self.spirrep += 1
            while self._corr_orb[self.spirrep] == 0:
                self.spirrep += 1
            self.orb = self._orbs_before[self.spirrep % self._n_irrep]
            self.orbirp = 0
        else:
            self.alive = False
