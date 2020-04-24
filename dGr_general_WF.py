"""Abstract base classs for electronic wave functions and alike.

Classes:
--------

Spirrep_String_Index
String_Index
Orbitals_Sets
Spirrep_Index
Wave_Function
"""
import logging
from collections import namedtuple
from collections.abc import Sequence, Mapping, Sized, Iterable, Container
from abc import ABC, abstractmethod

import numpy as np

from dGr_exceptions import *
from dGr_util import number_of_irreducible_repr

class Spirrep_String_Index(Sized, Iterable, Container):
    """An string index for a single spirrep
    
    Behaviour:
    -----------
    
    This class contains the information of orbital occupation
    (an orbital string) for a single spin/irrep (= spirrep)
    
    Attributes:
    -----------
    occ_orb (1D nd.array)
        An array with the indices of the occupied orbitals, starting by 0,
        in ascending order. For example:
        [0 1 2 3]    The first four orbitals are occupied;
        [1 2 3 4]    The first orbital (0) is empty and the following 4
                     are occupied (ie. an single exception from 0 to 4
                     over the above case);
        [1 2 6 9]    Orbitals 0 and 3 are empty, 6 and 9 are occupied.

    wf (Wave_Function)
        The wave function of this Spirrep_String_Index
    
    spirrep (int)
        The spirrep of this Spirrep_String_Index
    
    Data Model:
    -----------
    in
        Check if orbital is occupied: if orb.occ_orb = [0,2,3,5]:
        4 in orb -> False
        2 in orb -> True
    
    iter
        Iterates over occupied orbitals
    
    []
        Get/set occ_orb
    
    len
        Number of occupied orbitals
    
    int
        The position of this spirrep string in the wave function
        int(self) should be the the value index that one obtains self
        when iterating over Wave_Function.string_indices(spirrep=spirrep).
        That is, the following code should not raise an exception:
        
        for i, I in wf.string_indices(spirrep=spirrep):
            if int(I) != i:
                raise Exception('int of Spirrep_String_Index not consistent!')
    
    += (int)
        increase int of the object by the given value:
    
    """
    __slots__ = ('_std_pos',
                 '_occ_orb',
                 '_clear_std_pos_in_setitem',
                 'wf',
                 'spirrep')
    
    def __init__(self, n_elec):
        self._occ_orb = np.arange(n_elec, dtype=np.int8)
        self._std_pos = None
        self._clear_std_pos_in_setitem = True
        self.wf = None
        self.spirrep = None
    
    @property
    def occ_orb(self):
        return self._occ_orb
    
    def __contains__(self, x):
        return x in self._occ_orb
    
    def __iter__(self):
        return iter(self._occ_orb)
    
    def __setitem__(self, key, value):
        self._occ_orb.__setitem__(key, value)
        if self._clear_std_pos_in_setitem:
            self.clear_std_pos()
    
    def __getitem__(self, key):
        return self._occ_orb.__getitem__(key)
    
    def __len__(self):
        return len(self._occ_orb)
    
    def __int__(self):
        if self.is_std_pos_clear():
            if self.wf is None or self.spirrep is None:
                raise dGrValueError(
                    'Can not calculate std_pos without wave function and spirrep.')
            self._std_pos = self.wf.get_std_pos(occupation=self._occ_orb,
                                                spirrep=self.spirrep)
        return int(self._std_pos)
    
    def __iadd__(self, n):
        if not isinstance(n, int):
            raise dGrValueError('We need an integer.')
        if self.is_std_pos_clear():
            raise dGrValueError('Current index has not been started.')
        self._std_pos += n
        return self
    
    def __repr__(self):
        return ('I = ' + str(self._std_pos) + ': '
                + str(self._occ_orb))
    
    def __str__(self):
        return (str(self._occ_orb))
    
    def do_not_clear_std_pos(self):
        """Do not clear the standard position when occ_orb is changed."""
        self._clear_std_pos_in_setitem = False
    
    def do_clear_std_pos(self):
        """Do clear the standard position when occ_orb is changed."""
        self._clear_std_pos_in_setitem = True
    
    def clear_std_pos(self):
        """Clear the standard position."""
        self._std_pos = None
    
    def is_std_pos_clear(self):
        """Check if standard position is clear."""
        return self._std_pos is None
    
    def set_std_pos(self, new_pos):
        """Set a standard position."""
        self._std_pos = new_pos
    
    def start(self):
        """Set the standard position to 0."""
        self._std_pos = 0
    
    @classmethod
    def make_hole(cls, n_elec, holes):
        """Create a Spirrep_String_Index with hole(s) at holes"""
        if isinstance(holes, (np.integer, int)): 
            holes = (holes,)
        holes = sorted(holes, reverse=True)
        new_I = cls(n_elec)
        i_virt = n_elec
        for i in holes:
            if i >= n_elec:
                continue
            new_I._occ_orb[i:] = np.roll(new_I._occ_orb[i:],-1)
            if len(new_I._occ_orb) > 0:
                new_I._occ_orb[-1] = i_virt
            i_virt += 1
        return new_I

class String_Index(Mapping):
    """The string index for all spirreps
    
    Attributes:
    -----------
    
    spirrep_indices (list of Spirrep_String_Index)
        The string index of each spirrep
    
    Data Model:
    -----------
    
    []
        Get/set the Spirrep_String_Index of a spirrep
    
    len
        The number of Spirrep_String_Index. It is the number of
        irreps for restricted case, or two times the number of
        irreps for the unrestricted case.
    
    iter
        Iterates over all spirreps, giving the corresponding
        Spirrep_String_Index
    
    """
    __slots__ = ('spirrep_indices')
    
    def __init__(self, spirrep_indices=None):
        if spirrep_indices is None:
            self.spirrep_indices = []
        else:
            self.spirrep_indices = spirrep_indices
    
    def __iter__(self):
        return iter(self.spirrep_indices)
    
    def __getitem__(self, spirrep):
        return self.spirrep_indices[spirrep]
    
    def __setitem__(self, key, value):
        self.spirrep_indices.__setitem__(key, value)
    
    def __len__(self):
        return len(self.spirrep_indices)
    
    def __repr__(self):
        return '\n'.join(map(str, self.spirrep_indices))
    
    def __str__(self):
        return ' ^ '.join(map(str, self.spirrep_indices))
    
    def append(self, value):
        """Append value to the spirrep indices."""
        self.spirrep_indices.append(value)


class Orbitals_Sets(Sequence):
    """The number of occupied orbitals per spirrep
    
    Behaviour:
    ----------
    
    Objects of this class represents the number of occupied
    orbitals in each spirrep.
    
    Attributes:
    -----------
    occ_type (str)
        It can be 'R' for restricted,
                  'A' for just alpha orbitals
                  'B' for just beta orbitals
                  'F' for full set of orbitals
    
    Data Model:
    -----------
    []
        Get/set the number of occupied orbitals for the given spirrep
    
    len
        The total number of occupied orbitals.
    
    ==
        Gives True if the number of occupied orbitals is the same for all
        spirreps, even for different occ_type.
    
    +, -, +=
        Return a new instance of Orbitals_Sets (or act on self),
        adding/subtracting the number of occupied orbitals of each spirrep.
        If occ_type is the same, preserves occ_type; otherwise the returned
        object has occ_type = 'F'.
    
    """
    def __init__(self,
                 occ_or_n_irrep,
                 occ_type='F'):
        """Initialises the object
        
        Parameters:
        -----------
        occ_or_n_irrep (int or object that can be converted to
                        an 1D np.ndarray of int)
            The number of irreducible representations or the occupation
            number for each spirrrep.
        """
        self._type = occ_type
        if isinstance(occ_or_n_irrep, (int, np.integer)):
            self._n_irrep = occ_or_n_irrep
            self._occupation = np.zeros((2
                                         if self._type == 'F' else
                                         1) * self._n_irrep,
                                        dtype=np.uint8)
        else:
            self._occupation = np.array(occ_or_n_irrep,
                                        dtype=np.uint8)
            self._n_irrep = (len(self._occupation) // 2
                            if self._type == 'F' else
                            len(self._occupation))

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key < 0 or key >= 2 * self._n_irrep:
                return IndexError('Orbital ' + str(key) + ' is out of range for ' + str(self) + '.')
            if self._type == 'F':
                return self._occupation[key]
            if self._type == 'R':
                return self._occupation[key % self._n_irrep]
            if self._type == 'A':
                return self._occupation[key] if key < self._n_irrep else 0
            if self._type == 'B':
                return self._occupation[key] if key >= self._n_irrep else 0
        return self._occupation[key]

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            if 0 > key < 2 * self._n_irrep:
                return IndexError('Key ' + str(key) + ' is out of range for ' + str(self) + '.')
            if self._type == 'A' and key >= self._n_irrep:
                raise dGrValueError('Cannot set occupation for beta orbital for ' + str(self) + '.')
            if self._type == 'B' and key < self._n_irrep:
                raise dGrValueError('Cannot set occupation for alpha orbital for ' + str(self) + '.')
            if self._type == 'R':
                key = key % self._n_irrep
        self._occupation[key] = value

    def __len__(self):
        return (2
                if self._type == 'R' else
                1) * sum(self._occupation)

    def __str__(self):
        x = []
        
        if self._type != 'B':
            if self._type == 'R':
                x.append('alpha/beta: [')
            else:
                x.append('alpha: [')
            for i in self._occupation[:self._n_irrep]:
                x.append(str(i))
        if self._type != 'A' and self._type != 'R':
            x.append(']; beta: [')
            for i in (self._occupation[self._n_irrep:]
                      if self._type == 'F' else
                      self._occupation[:self._n_irrep]):
                x.append(str(i))
        x.append(']')
        return ' '.join(x)
    
    def __eq__(self, other):
        if self._n_irrep != other._n_irrep:
            raise dGrValueError('Cannot compare Orbitals_Sets for different number of irreps')
        for i in range((1
                        if self._type == 'R' and other._type == 'R' else
                        2) * self._n_irrep):
            if self[i] != other[i]:
                return False
        return True

    def __add__(self, other):
        if not isinstance(other, Orbitals_Sets):
            raise dGrValueError('Orbitals_Sets adds only with another Orbitals_Sets.')
        if self._n_irrep != other._n_irrep:
            raise dGrValueError('Both instances of Orbitals_Sets must have same len.')
        if self._type == other._type:
            new_occupation = self._occupation + other._occupation
            new_occ_type = self._type
        else:
            new_occ_type = 'F'
            new_occupation = np.zeros(self._n_irrep * 2,
                                      dtype=np.int8)
            if self._type != 'B':
                new_occupation[:self._n_irrep] += self._occupation[:self._n_irrep]
            if self._type != 'A':
                new_occupation[self._n_irrep:] += (self._occupation[self._n_irrep:]
                                                  if self._type == 'F' else
                                                  self._occupation)
            if other._type != 'B':
                new_occupation[:self._n_irrep] += other._occupation[:other._n_irrep]
            if other._type != 'A':
                new_occupation[other._n_irrep:] += (other._occupation[other._n_irrep:]
                                                  if other._type == 'F' else
                                                  other._occupation)
        return Orbitals_Sets(new_occupation,
                             new_occ_type)

    def __iadd__(self, other):
        if not isinstance(other, Orbitals_Sets):
            raise dGrValueError('Orbitals_Sets adds only with another Orbitals_Sets.')
        if self._n_irrep != other._n_irrep:
            raise dGrValueError('Both instances of Orbitals_Sets must have same len.')
        if self._type == other._type:
            self_occupation += other._occupation
        else:
            self_occ = self._occupation
            self._occupation = np.zeros(self._n_irrep * 2,
                                        dtype=np.int8)
            if self._type != 'B':
                self._occupation[:self._n_irrep] += self_occ[:self._n_irrep]
            if self._type != 'A':
                self._occupation[self._n_irrep:] += (self_occ[self._n_irrep:]
                                                     if self._type == 'F' else
                                                     self_occupation)
            self._type = 'F'
            if other._type != 'B':
                self._occupation[:self._n_irrep] += other._occupation[:other._n_irrep]
            if other._type != 'A':
                self._occupation[other._n_irrep:] += (other._occupation[other._n_irrep:]
                                                      if other._type == 'F' else
                                                      other._occupation)
        return self

    def __sub__(self, other):
        if not isinstance(other, Orbitals_Sets):
            raise dGrValueError('Orbitals_Sets adds only with another Orbitals_Sets.')
        if self._n_irrep != other._n_irrep:
            raise dGrValueError('Both instances of Orbitals_Sets must have same n_irrep.')
        if self._type == other._type:
            new_occupation = self._occupation - other._occupation
            new_occ_type = self._type
        else:
            new_occ_type = 'F'
            new_occupation = np.zeros(self._n_irrep * 2,
                                      dtype=np.int8)
            if self._type != 'B':
                new_occupation[:self._n_irrep] += self._occupation[:self._n_irrep]
            if self._type != 'A':
                new_occupation[self._n_irrep:] += (self._occupation[self._n_irrep:]
                                                  if self._type == 'F' else
                                                  self._occupation)
            if other._type != 'B':
                new_occupation[:self._n_irrep] -= other._occupation[:other._n_irrep]
            if other._type != 'A':
                new_occupation[other._n_irrep:] -= (other._occupation[other._n_irrep:]
                                                  if other._type == 'F' else
                                                  other._occupation)
        return Orbitals_Sets(new_occupation,
                             new_occ_type)
    
    def restrict_it(self):
        """Transform occ_type to 'R' if possible, or raise dGrValueError."""
        if self._type == 'R':
            return
        if self._type == 'F':
            if (self._occupation[:self._n_irrep] != self._occupation[self._n_irrep:]).any():
                raise dGrValueError('Cannot restrict ' + str(self) + '.')
            self._occupation = self._occupation[:self._n_irrep]
            self._type = 'R'
            return
        raise dGrValueError('Cannot restrict ' + str(self) + '.')
    
    @property
    def occ_type(self):
        return self._type

class Spirrep_Index(namedtuple('Spirrep_Index',
                               ['spirrep',
                                'I'])):
    """A namedtuple for a pair spirrep/Spirrep_String_Index
    
    Attributes:
    -----------
    spirrep (int)
        The spirrep
    I (Spirrep_String_Index)
        The index
    """
    __slots__ = ()


class Wave_Function(ABC, Sequence):
    """An abstract base class for electronic wave functions
    
    Atributes:
    ----------
    
    restricted (bool)
        Restricted (alpha and beta parameters are the same) or
        unrestricted (alpha and beta parameters differ) wave function 
    
    point_group (str)
        The point group
    
    n_irrep (int, property)
        The number of irreducible representations
    
    n_core (Orbitals_Sets)
        Number of core orbitals per spirrep (restricted)
    
    n_act (Orbitals_Sets)
        Number of active orbitals per spirrep (restricted)
    
    orb_dim (Orbitals_Sets)
        Dimension of the orbital space of each irrep
    
    ref_occ (Orbitals_Sets)
        Number of occupied orbitals per spirrep in reference determinant
    
    n_ext, n_corr_orb (Orbitals_Sets)
        n_ext = orb_dim - ref_occ
        n_corr_orb = ref_occ - n_core
    
    n_alpha, n_beta, n_elec, n_corr_alpha, n_corr_beta, n_corr_elec (int)
        Number of alpha, beta, total, correlated alpha, correlated beta,
        and total correlated electrons, respectively

    WF_type (str)
       Type of wave function
    
    source (str)
        The source of this wave function
    
    Data Model:
    -----------
    Some rules about Sequence's abstract methods:
    
    __getitem__ should accept an instance of String_Index and return
    the corresponding CI coefficient
    
    __len__  should return the number of distinct determinants
    """

    def __init__(self):
        self.restricted = None
        self.point_group = None
        self.Ms = None
        self.n_core = None
        self.n_act = None
        self.orb_dim = None
        self.ref_occ = None
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
        x.append('Ms: {}'.format(self.Ms))
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
        self.orb_dim = Orbitals_Sets(self.n_irrep,
                                     occ_type='R')
        self.ref_occ = Orbitals_Sets(self.n_irrep,
                                     occ_type='R' if self.restricted else 'F')
        self.n_core = Orbitals_Sets(self.n_irrep,
                                    occ_type='R')
        self.n_act = Orbitals_Sets(self.n_irrep,
                                   occ_type='A')
    
    def spirrep_blocks(self, restricted=None):
        """Yield the possible spin and irreps, as a single integer."""
        if restricted is None:
            restricted = self.restricted
        for i in range(self.n_irrep * (1
                                       if restricted else
                                       2)):
            yield i
    
    @property
    def n_irrep(self):
        if self.point_group is None:
            return None
        return number_of_irreducible_repr[self.point_group]
    
    @property
    def n_ext(self):
        if (self.orb_dim is None
            or self.ref_occ is None):
            return None
        else:
            return self.orb_dim - self.ref_occ
    
    @property
    def n_corr_orb(self):
        if (self.ref_occ is None
            or self.n_core is None):
            return None
        else:
            return self.ref_occ - self.n_core
    
    @property
    def n_alpha(self):
        if self.ref_occ is None:
            return None
        if self.restricted and self.ref_occ.occ_type == 'R':
            return len(self.ref_occ) // 2
        if self.ref_occ.occ_type == 'F':
            return sum([self.ref_occ[i] for i in range(self.n_irrep)])
        len_n_act = len(self.n_act)
        return len_n_act + (len(self.ref_occ) - len_n_act) // 2
    
    @property
    def n_beta(self):
        if self.ref_occ is None:
            return None
        if self.restricted and self.ref_occ.occ_type == 'R':
            return len(self.ref_occ) // 2
        if self.ref_occ.occ_type == 'F':
            return sum([self.ref_occ[i] for i in range(self.n_irrep, 2 * self.n_irrep)])
        return (len(self.ref_occ) - len(self.n_act)) // 2
    
    @property
    def n_elec(self):
        if self.ref_occ is None:
            return None
        return len(self.ref_occ)
    
    @property
    def n_corr_alpha(self):
        if (self.n_alpha is None
            or self.n_corr_orb is None):
            return None
        return self.n_alpha - len(self.n_core) // 2
    
    @property
    def n_corr_beta(self):
        if (self.n_beta is None
            or self.n_corr_orb is None):
            return None
        return self.n_beta - len(self.n_core) // 2
    
    @property
    def n_corr_elec(self):
        n_corr_orb = self.n_corr_orb
        if n_corr_orb is None:
            return None
        return len(n_corr_orb)
    
    @abstractmethod
    def string_indices(self,
                       spirrep=None,
                       coupled_to=None,
                       no_occ_orb=False,
                       only_ref_occ=False,
                       only_this_occ=None):
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
            If passed, it should be a tuple of Spirrep_Index,
            and the function should yield all String_Index that have the
            .I for .spirrep, or all Spirrep_String_Index of the
            given spirrep that are coupled to spirrep for that wave function
        
        no_occ_orb (bool, default=False)
            If True, do not waste time filling the attribute occ_orb.
        
        only_ref_occ (bool, default=False)
            If True, yield only the string indices that have the same
            occupation of the reference wave function per irrep
        
        only_this_occ (int or tuple of int, default=None)
            If passed, should be an int if parameter spirrep was
            also given, otherwise a tuple where the entries are the
            occupation per irrep. Thus, only indices with such occupation
            are yield. If not given, the occupation of reference is used.
        
        Yield:
        ------
        
        Instances of String_Index or of Spirrep_String_Index (if spirrep
        was given)
        """
        pass
    
    @abstractmethod
    def make_Jac_Hess_overlap(self, analytic = True):
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
        
        z   the update in the orbital basis (given in the space of the
            K_i^a parameters) from the position z=0 (that is, the orbital
            basis used to construct the current representation of the
            wave function
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


