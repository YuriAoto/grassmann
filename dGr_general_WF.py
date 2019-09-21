"""General electronic wave function

Abstract base class for electronic wave functions.

"""
import logging
from collections import namedtuple
from collections.abc import Sequence, Mapping, Collection
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

class Spirrep_String_Index(Collection):
    """An string index for a single spirrep
    
    Attributes:
    -----------
    
    standard_position_of_string (int)
        The index of this string in the standard order
    
    occ_orb (np.ndarray)
        An numpy array of int8 with the occupied orbitals
    """
    __slots__ = ('I_spirreps',
                 'occ_orb')
    
    def __init__(self, n_elec):
        self.standard_position_of_string = 0
        self.occ_orb = np.zeros(n_elec, dtype=np.int8)
    
    def __contains__(self, x):
        return x in self.occ_orb
    
    def __iter__(self):
        return self.occ_orb
    
    def __len__(self):
        return len(self.occ_orb)
    
    def __int__(self):
        return self.standard_position_of_string

class String_Index(Mapping):
    """A full string index, for all spirreps
    
    Attributes:
    -----------
    
    spirrep_indices (list of Spirrep_String_Index)
        The index of each spirrep
    """
    __slots__ = ('spirrep_indices')
    
    def __init__(self, spirrep_indices=None):
        if spirrep_indices is None:
            self.spirrep_indices = []
        else:
            self.spirrep_indices = spirrep_indices
    
    def __iter__(self):
        return self.spirrep_indices
    
    def __getitem__(self, spirrep):
        return self.spirrep_indices[spirrep]
    
    def __len__(self):
        return len(self.spirrep_indices)


class Wave_Function(ABC, Sequence):
    """An abstract base class for electronic wave functions
    
    Atributes:
    ----------
    
    n_irrep (int)                     the number of irreducible representations
    n_strings (list of int)           the number string in each spirrep
    orb_dim (n_irrep-tuple of int)    dimension of the orbital space of each irrep
    n_elec (int)                      number of electrons
    n_alpha (int)                     number of alpha electrons
    n_beta (int)                      number of beta electrons
    n_frozen (int)                    number of frozen orbitals
                                       (assumed to be equal for alpha and beta)
    ref_occ (n_irrep-tuple of int)    occupation of the reference determinant
    WF_type (str)                     type of wave function
    
    Some rules about Sequence's abstract methods:
    
    __getitem__ should accept an instance of String_Index and return
    the corresponding CI coefficient
    
    __len__  should return the number of distinct determinants
    """
    
    def __init__(self):
        self.n_irrep = 0
        self.n_strings = [0]
        self.orb_dim = []
        self.n_elec = 0
        self.n_alpha = 0
        self.n_beta = 0
        self.n_frozen = 0
        self.ref_occ = []
        self.WF_type = ''
    
    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = []
        x.append('-'*50)
        x.append('n irrep: {}'.format(self.n_irrep))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('n frozen: {}'.format(self.n_frozen))
        x.append('WF type: {}'.format(self.WF_type))
        x.append('-'*50)
        return '\n'.join(x)
    
    def spirrep_blocks(self):
        """Yield the possible spin and irreps, as a single integer."""
        for spirrep in range(2 * n_irrep):
            yield spirrep
    
    @abstractmethod
    def string_indices(self, spirrep=None, coupled_to=None, no_occ_orb=False):
        """A generator that yields all string indices
        
        Behaviour:
        ----------
        
        The indices that this generator yield should be an instance
        of String_Index or of Spirrep_String_Index.
        The wave function should be indexable by the values
        that this function yield, returning the corresponding coefficient.
        That is, the following construction should print all
        CI coefficients:
        for I in wf.string_indices():
             print(wf[I])
        
        Examples:
        ---------
        
        for a FCI wave function this yields all
        possible such subindices.
        
        for a CISD wave function this yields all subindices
        that differ from range(ref_occ(irrep)) for at most
        two inices (that is, a double excitation)
        
        Parameters:
        -----------
        
        spirrep (int, default=None)
            If passed, Spirrep_String_Index of this spirrep are yield
        
        coupled_to (tuple, default=None)
            If passed, it should be a even-length tuple
            with each pair being (spirrep_i, I_i), and the function should
            yield all String_Index that have the subindice I for spirrep,
            or all Spirrep_String_Index of the given spirrep that are
            coupled to spirrep
        
        no_occ_orb (bool, default=False)
            If True, do not waste time filling the attribute occ_orb.
        
        Yield:
        ------
        
        Instances of String_Index or of Spirrep_String_Index (if spirrep
        was given)
        """
        pass
    
    @abstractmethod
    def make_Jac_Hess(self, analytic = True):
        """Construct the Jacobian and the Hessian of the function overlap.
    
        Behaviour:
        ----------
        
        he function is f(x) = <wf(x), det1(x)>
        where x parametrises the orbital rotations and
        det1 is the first determinant in wf.
        
        Parameters:
        -----------
        
        analytic (bool, default = True)
            If True, calculate the Jacobian and the Hessian by the
            analytic expression, if False calculate numerically
        
        Returns:
        --------
        
        The tuple (Jac, Hess), with the Jacobian and the Hessian
        """
        pass

    @abstractmethod
    def calc_wf_from_z(self, z, just_C0 = False):
        """Calculate the wave function in a new orbital basis
        
        Behaviour:
        ----------
        
        Given the wave function in the current orbital basis,
        a new (representation of the) wave function is constructed
        in a orbital basis that has been modified by a step z.
        
        Paramters:
        ----------
        
        z        the update in the orbital basis (given in the space of the
             K_i^a parameters) from the position z=0 (that is, the orbital
             basis used to construct the current representation of the
             wave function
        cur_wf   current representation of the wave function
             (as Molpro_FCI_Wave_Function)
        just_C0  Calculates only the first coefficient (see transform_wf)
        
        Return:
        -------
        
        a tuple (new_wf, Ua, Ub) where new_wf is a Molpro_FCI_Wave_Function
        with the wave function in the new representation, and Ua and Ub
        are the transformations from the previous to the new orbital
        basis (alpha and beta, respectively).
        """
        pass
    
    @abstractmethod
    def change_orb_basis(self, Ua, Ub, just_C0 = False):
        """Transform the wave function as induced by a transformation in the orbital basis
        
        Behaviour:
        ----------
        
        If the coefficients of wf are given in the basis |u_I>:
        
        |wf> = \sum_I c_I |u_I>
        
        it calculates the wave function in the basis |v_I>:
        
        |wf> = \sum_I d_I |v_I>
        
        and Ua and Ub are the matrix transformations of the MO from the basis |v_I>
        to the basis |u_I>:
    
        |MO of (u)> = |MO of (v)> U

        Parameters:
        -----------
        
        wf   the initial wave function as Molpro_FCI_Wave_Function
        Ua   the orbital transformation for alpha orbitals
        Ub   the orbital transformation for beta orbitals
        just_C0   If True, calculates the coefficients of the initial determinant only
              (default False)
        
        Return:
        -------
        
        The transformed wave function
        """
        pass


