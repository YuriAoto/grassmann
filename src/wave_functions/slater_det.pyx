from collections import namedtuple

import numpy as np

from input_output import molpro
from util.variables import int_dtype
from util.other import int_array
from orbitals.orbital_space cimport FullOrbitalSpace, OrbitalSpace
from orbitals.orbital_space import FullOrbitalSpace, OrbitalSpace


class UndefOrbspace(Exception):
    pass


cdef class SlaterDet:
    """A generic Slater determinant
    
    Attributes:
    -----------
    c (float)
        The coefficient associated to this Slater determinant
    
    {alpha,beta}_occ (array-like of int)
        The occupied orbitals for {alpha,beta} orbitals
    """
    def __init__(self, c, alpha_occ, beta_occ):
        self.c = c
        self.alpha_occ = int_array(alpha_occ)
        self.beta_occ = int_array(beta_occ)

    def __eq__(self, SlaterDet other):
        cdef int i
        if abs(self.c - other.c) > 1.0E-8:
            return False
        if (self.alpha_occ.size != other.alpha_occ.size
            or self.beta_occ.size != other.beta_occ.size):
            return False
        for i in range(self.alpha_occ.size):
            if self.alpha_occ[i] != self.alpha_occ[i]:
                return False
        for i in range(self.beta_occ.size):
            if self.beta_occ[i] != other.beta_occ[i]:
                return False
        return True

    @classmethod
    def from_molpro_line(cls,
                         str line,
                         double Ms,
                         OrbitalSpace froz_orb,
                         molpro_output='',
                         line_number=-1,
                         zero_coefficient=False):
        """Constructor from a FCI configuration from Molpro output
        
        Parameters:
        -----------
        line (str)
            The line with a configuration, from the FCI program in Molpro
            to be converted to a Slater Determinant.
        
        Ms (float)
            the Ms
        
        froz_orb (OrbitalSpace)
            The frozen orbitals
        
        molpro_output (str, optional, default='')
            The output file name (only for error message)
        
        line_number (int, optional, default=-1)
            The line number in Molpro output
        
        zero_coefficient (bool, optional, default=False)
            If True, the coefficient is set to zero and the value
            in line is discarded
        
        Return:
        --------
        An instance of SlaterDet containing only active orbitals.
        That is, all frozen electrons are discarded and the first
        non-frozen orbital is indexed by 0.
        All irreps are merged in a single array
        
        Raise:
        ------
        molpro.MolproInputError
        
        Examples:
        ---------
        
        # if n_irrep = 4, orb_dim = (6,2,2,0), froz_orb = (0,0,0,0) then
        
        -0.162676901257  1  2  7  1  2  7
        gives
        c=-0.162676901257; alpha_occ=[0,1,6]; beta_occ=[0,1,6]
        
        -0.049624632911  1  2  4  1  2  6
        gives
        c=-0.049624632911; alpha_occ=[0,1,3]; beta_occ=[0,1,5]
        
        0.000000000000  1  2  9  1  2 10
        gives
        c=-0.000000000000; alpha_occ=[0,1,8]; alpha_occ=[0,1,9]
        
        # but if froz_orb = (1,1,0,0) then the above cases give
            (because frozen electrons are indexed first in Molpro convention
             and they do not enter in the returned SlaterDet)
        
        c=-0.162676901257; alpha_occ=[4]; beta_occ=[4]
        
        c=-0.049624632911; alpha_occ=[1]; beta_occ=[3]
        
        c=-0.000000000000; alpha_occ=[6]; beta_occ=[7]
        """
        cdef int n_total_frozen = len(froz_orb) // 2
        lspl = line.split()
        try:
            coeff = float(lspl[0])
            occ = [int(x) - 1 - n_total_frozen
                   for x in lspl[1:]
                   if int(x) > n_total_frozen]
        except Exception as e:
            raise molpro.MolproInputError(
                "Error when reading FCI configuration. Exception was:\n"
                + str(e),
                line=line,
                line_number=line_number,
                file_name=molpro_output)
        n_alpha = (len(occ) + int(2 * Ms)) // 2
        n_beta = (len(occ) - int(2 * Ms)) // 2
        if n_beta + n_alpha != len(occ):
            raise ValueError('Ms = {} is not consistent with occ = {}'.format(
                Ms, occ))
        return cls(c=0.0 if zero_coefficient else coeff,
                   alpha_occ=int_array(occ[:n_alpha]),
                   beta_occ=int_array(occ[n_alpha:]))

    @classmethod
    def from_excitation(cls,
                        SlaterDet ref_det,
                        double c,
                        alpha_hp,
                        beta_hp):
        """Create a new SlaterDet as an excitation on top of reference
        
        Parameters:
        -----------
        ref_det (SlaterDet)
            The reference Slater determinant
        
        c (float)
            The coefficient of the returned Slater determinant
        
        alpha_hp (2-tuple of np.array)
            The alpha holes and particles of the excitation
        
        beta_hp (2-tuple of np.array)
            The beta holes and particles of the excitation
        
        Return:
        -------
        The new Slater determinant
        
        """
        return cls(c=c,
                   alpha_occ=np.sort(
                       np.concatenate(
                           (np.array([x for x in ref_det.alpha_occ
                                      if x not in alpha_hp[0]],
                                     dtype=int_dtype),
                            alpha_hp[1]))),
                   beta_occ=np.sort(
                       np.concatenate(
                           (np.array([x for x in ref_det.beta_occ
                                      if x not in beta_hp[0]],
                                     dtype=int_dtype),
                            beta_hp[1]))))

    @classmethod
    def from_orbspace(cls, OrbitalSpace orbspace, int [:] orbs_before):
        """Create a Slater determinant whose orbital_space is orbspace
        
        Parameters:
        -----------
        orbspace (OrbitalSpace)
            The orbital space
        
        orbs_before (array of int)
            The number or orbitals before each irrep (see FullOrbitalSpace)
        
        """
        cdef irrep, p
        ref_alpha = []
        ref_beta = []
        for irrep in range(orbspace.n_irrep):
            for p in range(orbspace[irrep]):
                ref_alpha.append(p + orbs_before[irrep])
            for p in range(orbspace[irrep + orbspace.n_irrep]):
                ref_beta.append(p + orbs_before[irrep])
        return SlaterDet(c=0.0,
                         alpha_occ=np.array(ref_alpha, dtype=int_dtype),
                         beta_occ=np.array(ref_beta, dtype=int_dtype))

    cdef OrbitalSpace orbspace(self, FullOrbitalSpace full_orbsp):
        """The orbital space defined by the Slater Determinant
        
        This is the inverse function of from_orbspace.
        
        Return:
        -------
        The orbital space defined by this Slater determinant
        
        Raise:
        ------
        UndefOrbspace if the occupied orbitals of the Slater determinant
        are not the first orbitals of each irrep
        
        """
        cdef int i, p, irrep
        dim = np.zeros(2*full_orbsp.n_irrep, dtype=int_dtype)
        full_orbsp.calc_remaining()
        for i in range(self.alpha_occ.size):
            p, irrep = full_orbsp.get_local_index(self.alpha_occ[i], True)
            if p == 0 or self.alpha_occ[i-1] == self.alpha_occ[i]-1:
                dim[irrep] += 1
            else:
                raise UndefOrbspace(f'Cannot be used as reference: {self}')
        for i in range(self.beta_occ.size):
            p, irrep = full_orbsp.get_local_index(self.beta_occ[i], False)
            if p == 0 or self.beta_occ[i-1] == self.beta_occ[i]-1:
                dim[irrep + full_orbsp.n_irrep] += 1
            else:
                raise UndefOrbspace(f'Cannot be used as reference: {self}')
        return OrbitalSpace(dim=dim[:2*full_orbsp.n_irrep])

    def __str__(self):
        return (f'Slater determinant: c = {self.c:15.12f} ; '
                + '^'.join(map(str, self.alpha_occ))
                + ' ^ '
                + '^'.join(map(str, self.beta_occ)))

    def set_coef(self, double value):
        """Set the coefficient"""
        self.c = value
