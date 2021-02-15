from collections import namedtuple

import numpy as np

from util.variables import int_dtype


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
