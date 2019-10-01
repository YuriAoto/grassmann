"""General electronic wave function

Abstract base class for electronic wave functions.

"""
import logging
from collections import namedtuple
from collections.abc import Sequence, Mapping, Sized, Iterable, Container
from abc import ABC, abstractmethod

import numpy as np

from dGr_exceptions import *

logger = logging.getLogger(__name__)

class Spirrep_String_Index(Sized, Iterable, Container):
    """An string index for a single spirrep
    
    Attributes:
    -----------
    
    standard_position_of_string (int)
        The index of this string in the standard order
    
    occ_orb (np.ndarray)
        An numpy array of int8 with the occupied orbitals
    """
    __slots__ = ('standard_position_of_string',
                 'occ_orb')
    
    def __init__(self, n_elec):
        self.standard_position_of_string = 0
        self.occ_orb = np.arange(n_elec, dtype=np.int8)
    
    def __contains__(self, x):
        return x in self.occ_orb
    
    def __iter__(self):
        return self.occ_orb
    
    def __setitem__(self, key, value):
        self.occ_orb.__setitem__(key, value)
    
    def __len__(self):
        return len(self.occ_orb)
    
    def __int__(self):
        return self.standard_position_of_string
    
    def __repr__(self):
        return ('I = ' + str(self.standard_position_of_string) + ': '
                + str(self.occ_orb))
    
    def __str__(self):
        return (str(self.occ_orb))

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
    
    def __setitem__(self, key, value):
        self.spirrep_indices.__setitem__(key, value)
    
    def __len__(self):
        return len(self.spirrep_indices)
    
    def append(self, value):
        self.spirrep_indices.append(value)
    
    def __repr__(self):
        return '\n'.join(map(str, self.spirrep_indices))
    
    def __str__(self):
        return ' ^ '.join(map(str, self.spirrep_indices))


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
    n_core (int)                      number of core orbitals
                                       (assumed to be equal for alpha and beta)
    ref_occ (n_irrep-tuple of int)    occupation of the reference determinant
    WF_type (str)                     type of wave function
    source (str)                      The source of this wave function
    
    Some rules about Sequence's abstract methods:
    
    __getitem__ should accept an instance of String_Index and return
    the corresponding CI coefficient
    
    __len__  should return the number of distinct determinants
    """
    
    def __init__(self):
        self.restricted = None
        self.point_group = None
        self.n_irrep = None
        self.n_core = None
        self.n_act = None
        self.orb_dim = None
        self.ref_occ = None
        self.n_alpha = None
        self.n_beta = None
        self.WF_type = None
        self.source = None
    
    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = []
        x.append('-'*50)
        x.append('point group: {}'.format(self.point_group))
        x.append('n irrep: {}'.format(self.n_irrep))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('n core: {}'.format(self.n_core))
        x.append('n act: {}'.format(self.n_act))
        x.append('ref occ: {}'.format(self.ref_occ))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('WF type: {}'.format(self.WF_type))
        x.append('source: {}'.format(self.source))
        x.append('-'*50)
        return '\n'.join(x)
    
    def initialize_data(self):
        if self.point_group is None:
            raise dGrValueError('I still do not know the point group!')
        if self.restricted is None:
            raise dGrValueError('I still do not know if it is a restricted'
                                + ' or unrestricted wave function!')
        self.n_alpha = self.n_beta = 0
        self.orb_dim = np.zeros(self.n_irrep,
                                dtype=np.uint8)
        self.ref_occ = np.zeros(self.n_irrep * (1 if self.restricted else 2),
                                dtype=np.uint8)
        self.n_core = np.zeros(self.n_irrep,
                               dtype=np.uint8)
        self.n_act = np.zeros(self.n_irrep,
                              dtype=np.uint8)
        self.n_strings = np.zeros(self.n_irrep * (1 if self.restricted else 2),
                                  dtype=np.uint8)
    
    def spirrep_blocks(self):
        """Yield the possible spin and irreps, as a single integer."""
        for spirrep in range(2 * n_irrep):
            yield spirrep
    
    @property
    def n_elec(self):
        if self.n_alpha is None or self.n_beta is None:
            return None
        else:
            return self.n_alpha + self.n_beta
    
    @property
    def n_corr_alpha(self):
        if (self.n_alpha is None
            or self.n_corr_orb is None):
            return None
        else:
            return self.n_alpha - sum(self.n_core[:self.n_irrep])
    
    @property
    def n_corr_beta(self):
        if (self.n_beta is None
            or self.n_corr_orb is None):
            return None
        else:
            if self.restricted:
                return self.n_corr_alpha
            else:
                return self.n_beta - sum(self.n_core[:self.n_irrep])
    
    @property
    def n_corr_elec(self):
        ncorr_a = self.n_corr_alpha
        if restricted:
            return (None
                    if ncorr_a is None else
                    2 * ncorr_a)
        else:
            ncorr_b = self.n_corr_alpha
            return (None
                    if (ncorr_a is None or ncorr_b is None) else
                    ncorr_a + ncorr_b)
    
    @property
    def n_ext(self):
        if (self.orb_dim is None
            or self.ref_occ is None
            or self.n_act is None):
            return None
        else:
            if self.restricted:
                return self.orb_dim - self.ref_occ
            else:
                return np.concatenate((self.orb_dim,
                                       self.orb_dim)) - self.ref_occ
    
    @property
    def n_corr_orb(self):
        if (self.ref_occ is None
            or self.n_core is None):
            return None
        else:
            if self.restricted:
                return self.ref_occ - self.n_core
            else:
                return self.ref_occ - np.concatenate((self.n_core,
                                                      self.n_core))
    
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


