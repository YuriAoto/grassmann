from collections import namedtuple

import numpy as np

from util.variables import int_dtype
from util.other import int_array


def get_slater_det_from_fci_line(line, Ms, froz_orb,
                                 molpro_output='', line_number=-1,
                                 zero_coefficient=False):
    """Read a FCI configuration from Molpro output and return a Slater Determinant
    
    Parameters:
    -----------
    line (str)
        The line with a configuration, from the FCI program in Molpro
        to be converted to a Slater Determinant.

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
    lspl = line.split()
    n_total_frozen = len(froz_orb) // 2
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
    return SlaterDet(c=0.0 if zero_coefficient else coeff,
                     alpha_occ=int_array(occ[:n_alpha]),
                     beta_occ=int_array(occ[n_alpha:]))


def get_slater_det_from_excitation(ref_det, c, alpha_hp, beta_hp):
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
    return SlaterDet(c=c,
                     alpha_occ=np.array(sorted(
                         [x for x in ref_det.alpha_occ
                          if x not in alpha_hp[0]] + alpha_hp[1]),
                                        dtype=int_dtype),
                     beta_occ=np.array(sorted(
                         [x for x in ref_det.beta_occ
                          if x not in beta_hp[0]] + beta_hp[1]),
                                       dtype=int_dtype))


class SlaterDet(namedtuple('SlaterDet',
                           ['c',
                            'alpha_occ',
                            'beta_occ'])):
    """A namedtuple for a generic Slater determinant
    
    Attributes:
    -----------
    c (float)
        The coefficient associated to this Slater determinant
    
    {alpha,beta}_occ (1D np.arrays of int)
        The occupied orbitals for {alpha,beta} orbitals
    """
    __slots__ = ()
    
    def __str__(self):
        return ('Slater determinant: c = {0:15.12f} ; '.
                format(self.c)
                + '^'.join(map(str, self.alpha_occ))
                + ' ^ '
                + '^'.join(map(str, self.beta_occ)))
