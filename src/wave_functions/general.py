"""Abstract base class for electronic wave functions and alike.


# Covention for orbitals ordering:

Consider a multideterminantal wave function that:
1) is symmetry adapted: It is presented in a basis of symmetry adapted
   (spin-)orbitals, such that tach orbital belongs to one of g possible
   irreducible representations (irrep).
2) is based on excitations on top of a Slater determinant,
   that is the reference determinant.
3) Might have a number of frozen core orbitals.

The convention for the orbital ordering is:

First all frozen orbitals, ordered by symmetry;
Then all orbitals of first irrep, then all orbitals of second irrep, and
so on.;
Inside each irrep, first comes all orbitals that are occupied in the reference,
then all orbitals that are not occupied in the reference.

If the wave function is not of restricted type (either with restricted orbitals
or not), first come all frozen alpha orbitals, then all frozen beta orbitals,
then all alpha (non-frozen) and finally all beta (non-frozen)

Example (restricted type):

orb_dim   [10, 6, 6, 4]   Dimension of orbital space by irrep (with frozen)
froz_orb  [ 2, 1, 1, 1]   Frozen (core) orbitals
ref_orb   [ 5, 3, 3, 1]   Reference occupation (with frozen)
corr_orb  [ 3, 2, 2, 0]   Correlated orbitals
virt_orb  [ 5, 3, 3, 3]   Virtual orbitals

orbs_before = [0, 8, 13, 18, 21]

Orbital order (irrep = irreducible representation):

Frozen (Only formally, not really used)

orb     irrep

 -5      0
 -4   
==============
 -3      1
==============
 -2      2
==============
 -1      3

Non frozen

global     index
orbital    within    block     irrep
index      block

  0        0
  1        1         occ   
  2        2
------------------------
  3        0                     0
  4        1
  5        2         virt
  6        3
  7        4
==========================================
  8        0         occ
  9        1
------------------------         1
  10       0
  11       1         virt
  12       2
==========================================
  13       0         occ
  14       1
------------------------         2
  15       0
  16       1         virt
  17       2
==========================================
------------------------
  18       0 
  19       1         virt        3
  20       2



Example (unrestricted type):

Orbital dimension obviously does not depend on alpha/beta.
For frozen, we will consider only same frozen orbitals
in alpha and beta

orb_dim   [10, 6, 6, 4]                 As above
froz_orb  [ 2, 1, 1, 1]                 As above
ref_orb   [ 5, 3, 3, 1,  3, 3, 3, 1]    Reference occupation
                                        (with frozen, alpha/beta)
corr_orb  [ 3, 2, 2, 0,  1, 2, 2, 0]    Correlated orbitals
                                        (alpha/beta)
virt_orb  [ 5, 3, 3, 3,  7, 3, 3, 3]    Virtual orbitals

Orbital order (irrep = irreducible representation):

Frozen (Only formally, not really used)

orb     irrep

 -5      0
 -4   
==============
 -3      1
==============
 -2      2
==============
 -1      3

Non frozen

global     index
orbital    within    block     irrep
index      block

___ALPHA____

  0        0
  1        1         occ   
  2        2
------------------------
  3        0                     0
  4        1
  5        2         virt
  6        3
  7        4
==========================================
  8        0         occ
  9        1
------------------------         1
  10       0
  11       1         virt
  12       2
==========================================
  13       0         occ
  14       1
------------------------         2
  15       0
  16       1         virt
  17       2
==========================================
------------------------
  18       0 
  19       1         virt        3
  20       2

___BETA____

  0        0         occ
------------------------
  1        0
  2        1
  3        2                     0
  4        3
  5        4         virt
  6        5
  7        6
==========================================
  8        0         occ
  9        1
------------------------         1
  10       0
  11       1         virt
  12       2
==========================================
  13       0         occ
  14       1
------------------------         2
  15       0
  16       1         virt
  17       2
==========================================
------------------------
  18       0 
  19       1         virt        3
  20       2






Classes:
--------

SpirrepStringIndex
StringIndex
OrbitalsSets
SpirrepIndex
WaveFunction
"""
import logging
from collections import namedtuple
from collections.abc import Sequence, Mapping, Sized, Iterable, Container
from abc import ABC, abstractmethod

import numpy as np

from util import number_of_irreducible_repr, int_dtype
import memory

logger = logging.getLogger(__name__)


class SpirrepStringIndex(Sized, Iterable, Container):
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

    wf (WaveFunction)
        The wave function of this SpirrepStringIndex
    
    spirrep (int)
        The spirrep of this SpirrepStringIndex
    
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
        when iterating over WaveFunction.string_indices(spirrep=spirrep).
        That is, the following code should not raise an exception:
        
        for i, I in wf.string_indices(spirrep=spirrep):
            if int(I) != i:
                raise Exception('int of SpirrepStringIndex not consistent!')
    
    += (int)
        increase int of the object by the given value:
    
    """
    __slots__ = ('_std_pos',
                 '_occ_orb',
                 '_clear_std_pos_in_setitem',
                 'wf',
                 'spirrep')
    
    def __init__(self, n_elec):
        self._occ_orb = np.arange(n_elec, dtype=int_dtype)
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
                raise ValueError(
                    'Can not calculate std_pos without'
                    + ' wave function and spirrep.')
            self._std_pos = self.wf.get_std_pos(occupation=self._occ_orb,
                                                spirrep=self.spirrep)
        return int(self._std_pos)
    
    def __iadd__(self, n):
        if not isinstance(n, int):
            raise ValueError('We need an integer.')
        if self.is_std_pos_clear():
            raise ValueError('Current index has not been started.')
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
        """Create a SpirrepStringIndex with hole(s) at holes"""
        if isinstance(holes, (np.integer, int)):
            holes = (holes,)
        holes = sorted(holes, reverse=True)
        new_I = cls(n_elec)
        i_virt = n_elec
        for i in holes:
            if i >= n_elec:
                continue
            new_I._occ_orb[i:] = np.roll(new_I._occ_orb[i:], -1)
            if len(new_I._occ_orb) > 0:
                new_I._occ_orb[-1] = i_virt
            i_virt += 1
        return new_I


class StringIndex(Mapping):
    """The string index for all spirreps
    
    Attributes:
    -----------
    
    spirrep_indices (list of SpirrepStringIndex)
        The string index of each spirrep
    
    Data Model:
    -----------
    
    []
        Get/set the SpirrepStringIndex of a spirrep
    
    len
        The number of SpirrepStringIndex. It is the number of
        irreps for restricted case, or two times the number of
        irreps for the unrestricted case.
    
    iter
        Iterates over all spirreps, giving the corresponding
        SpirrepStringIndex
    
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
    
    @classmethod
    def make_reference(cls, ref_orb, n_irrep, wf=None):
        ref_indices = []
        for spirrep in range(2 * n_irrep):
            ref_indices.append(SpirrepStringIndex(ref_orb[spirrep]))
            ref_indices[-1].spirrep = spirrep
            ref_indices[-1].wf = wf
        return cls(ref_indices)
    
    def set_wave_function(self, wf):
        for Index in self.spirrep_indices:
            Index.wf = wf
    
    def append(self, value):
        """Append value to the spirrep indices."""
        if not isinstance(value, SpirrepStringIndex):
            raise ValueError(
                'Only SpirrepStringIndex can be appended to a StringIndex')
        self.spirrep_indices.append(value)
        self.spirrep_indices[-1].spirrep = len(self) - 1

    def swap_spirreps(self, i, j):
        """Swap the strings associated to spirreps i and j
        
        After this, the attribute spirrep of both SpirrepStringIndex
        is still consistent with the order of self."""
        self[i], self[j] = self[j], self[i]
        self[i].spirrep = i
        self[j].spirrep = j


class OrbitalsSets(Sequence):
    """The number of orbitals per spirrep
    
    Behaviour:
    ----------
    
    Objects of this class represents orbitals in each spirrep.
    
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
        Get/set the number of orbitals for the given spirrep
    
    len
        The total number of orbitals (if 'R', considers both alpha and beta).
    
    ==
        Gives True if the number of orbitals is the same for all
        spirreps, even for different occ_type.
    
    +, -, +=
        Return a new instance of OrbitalsSets (or act on self),
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
                                        dtype=int_dtype)
        else:
            self._occupation = np.array(occ_or_n_irrep,
                                        dtype=int_dtype)
            self._n_irrep = (len(self._occupation) // 2
                             if self._type == 'F' else
                             len(self._occupation))

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key < 0 or key >= 2 * self._n_irrep:
                return IndexError('Orbital ' + str(key)
                                  + ' is out of range for '
                                  + str(self) + '.')
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
                return IndexError('Key ' + str(key)
                                  + ' is out of range for '
                                  + str(self) + '.')
            if self._type == 'A' and key >= self._n_irrep:
                raise ValueError(
                    'Cannot set occupation for beta orbital for '
                    + str(self) + '.')
            if self._type == 'B' and key < self._n_irrep:
                raise ValueError(
                    'Cannot set occupation for alpha orbital for '
                    + str(self) + '.')
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
            raise ValueError(
                'Cannot compare OrbitalsSets for different number of irreps')
        for i in range((1
                        if self._type == 'R' and other._type == 'R' else
                        2) * self._n_irrep):
            if self[i] != other[i]:
                return False
        return True

    def __add__(self, other):
        if not isinstance(other, OrbitalsSets):
            raise ValueError(
                'OrbitalsSets adds only with another OrbitalsSets.')
        if self._n_irrep != other._n_irrep:
            raise ValueError(
                'Both instances of OrbitalsSets must have same len.')
        if self._type == other._type:
            new_occupation = self._occupation + other._occupation
            new_occ_type = self._type
        else:
            new_occ_type = 'F'
            new_occupation = np.zeros(self._n_irrep * 2,
                                      dtype=int_dtype)
            if self._type != 'B':
                new_occupation[:self._n_irrep] += (
                    self._occupation[:self._n_irrep])
            if self._type != 'A':
                new_occupation[self._n_irrep:] += (
                    self._occupation[self._n_irrep:]
                    if self._type == 'F' else
                    self._occupation)
            if other._type != 'B':
                new_occupation[:self._n_irrep] += (
                    other._occupation[:other._n_irrep])
            if other._type != 'A':
                new_occupation[other._n_irrep:] += (
                    other._occupation[other._n_irrep:]
                    if other._type == 'F' else
                    other._occupation)
        return OrbitalsSets(new_occupation,
                            new_occ_type)

    def __iadd__(self, other):
        if not isinstance(other, OrbitalsSets):
            raise ValueError(
                'OrbitalsSets adds only with another OrbitalsSets.')
        if self._n_irrep != other._n_irrep:
            raise ValueError(
                'Both instances of OrbitalsSets must have same len.')
        if self._type == other._type:
            self._occupation += other._occupation
        else:
            self_occ = self._occupation
            self._occupation = np.zeros(self._n_irrep * 2,
                                        dtype=int_dtype)
            if self._type != 'B':
                self._occupation[:self._n_irrep] += self_occ[:self._n_irrep]
            if self._type != 'A':
                self._occupation[self._n_irrep:] += (self_occ[self._n_irrep:]
                                                     if self._type == 'F' else
                                                     self._occupation)
            self._type = 'F'
            if other._type != 'B':
                self._occupation[:self._n_irrep] += (
                    other._occupation[:other._n_irrep])
            if other._type != 'A':
                self._occupation[other._n_irrep:] += (
                    other._occupation[other._n_irrep:]
                    if other._type == 'F' else
                    other._occupation)
        return self

    def __sub__(self, other):
        if not isinstance(other, OrbitalsSets):
            raise ValueError(
                'OrbitalsSets adds only with another OrbitalsSets.')
        if self._n_irrep != other._n_irrep:
            raise ValueError(
                'Both instances of OrbitalsSets must have same n_irrep.')
        if self._type == other._type:
            new_occupation = self._occupation - other._occupation
            new_occ_type = self._type
        else:
            new_occ_type = 'F'
            new_occupation = np.zeros(self._n_irrep * 2,
                                      dtype=int_dtype)
            if self._type != 'B':
                new_occupation[:self._n_irrep] += (
                    self._occupation[:self._n_irrep])
            if self._type != 'A':
                new_occupation[self._n_irrep:] += (
                    self._occupation[self._n_irrep:]
                    if self._type == 'F' else
                    self._occupation)
            if other._type != 'B':
                new_occupation[:self._n_irrep] -= (
                    other._occupation[:other._n_irrep])
            if other._type != 'A':
                new_occupation[other._n_irrep:] -= (
                    other._occupation[other._n_irrep:]
                    if other._type == 'F' else
                    other._occupation)
        return OrbitalsSets(new_occupation,
                            new_occ_type)
    
    def restrict_it(self):
        """Transform occ_type to 'R' if possible, or raise ValueError."""
        if self._type == 'R':
            return
        if self._type == 'F':
            if (self._occupation[:self._n_irrep]
                    != self._occupation[self._n_irrep:]).any():
                raise ValueError('Cannot restrict ' + str(self) + '.')
            self._occupation = self._occupation[:self._n_irrep]
            self._type = 'R'
            return
        raise ValueError('Cannot restrict ' + str(self) + '.')
    
    def as_array(self):
        """Return a copy of the array that represents it in Full"""
        if self.occ_type == 'F':
            return np.array(self._occupation)
        if self.occ_type == 'R':
            return np.concatenate((self._occupation, self._occupation))
        if  self.occ_type == 'A':
            return np.concatenate((self._occupation, np.zeros(self._n_irrep)))
        return np.concatenate((np.zeros(self._n_irrep), self._occupation))
    
    @property
    def occ_type(self):
        return self._type


class SpirrepIndex(namedtuple('SpirrepIndex',
                              ['spirrep',
                               'Index'])):
    """A namedtuple for a pair spirrep/SpirrepStringIndex
    
    Attributes:
    -----------
    spirrep (int)
        The spirrep
    Index (SpirrepStringIndex)
        The index
    """
    __slots__ = ()


class WaveFunction(ABC, Sequence):
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
    
    froz_orb (OrbitalsSets)
        Number of core orbitals per spirrep (restricted)
    
    act_orb (OrbitalsSets)
        Number of active orbitals per spirrep (restricted)
    
    orb_dim (OrbitalsSets)
        Dimension of the orbital space of each irrep
    
    ref_orb (OrbitalsSets)
        Number of occupied orbitals per spirrep in reference determinant
    
    virt_orb, corr_orb (OrbitalsSets)
        virt_orb = orb_dim - ref_orb
        corr_orb = ref_orb - froz_orb
    
    orb_before (np.array of int)
        The number of orbitals (without frozen orbitals) before each irrep.
        orb_before[irrep] = sum(corr_orb[:irrep]) + sum(virt_orb[:irrep])
    
    n_alpha, n_beta, n_elec, n_corr_alpha, n_corr_beta, n_corr_elec (int)
        Number of alpha, beta, total, correlated alpha, correlated beta,
        and total correlated electrons, respectively
    
    n_orb n_orb_nocore (int)
        Number of spatial orbitals (with and without core orbitals

    WF_type (str)
       Type of wave function
    
    source (str)
        The source of this wave function
    
    Data Model:
    -----------
    Some rules about Sequence's abstract methods:
    
    __getitem__ should accept an instance of StringIndex and return
    the corresponding CI coefficient
    
    __len__  should return the number of distinct determinants
    """

    def __init__(self):
        self.restricted = None
        self.point_group = None
        self.Ms = None
        self.orb_dim = None
        self.froz_orb = None
        self.ref_orb = None
        self.act_orb = None
        self._orb_before = None
        self.WF_type = None
        self.source = None
        self.mem = 0.0

    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = [super().__repr__()]
        x.append('-' * 50)
        x.append('point group: {}'.format(self.point_group))
        x.append('n irrep: {}'.format(self.n_irrep))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('n core: {}'.format(self.froz_orb))
        x.append('n act: {}'.format(self.act_orb))
        x.append('ref occ: {}'.format(self.ref_orb))
        x.append('Ms: {}'.format(self.Ms))
        x.append('restricted: {}'.format(self.restricted))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('WF type: {}'.format(self.WF_type))
        x.append('source: {}'.format(self.source))
        x.append('-' * 50)
        return '\n'.join(x)
    
    def __del__(self):
        memory.free(self.mem)
    
    def _set_memory(self, destination=None, calc_args=()):
        """Set the amount of memory used by this wave function
        
        If the new amount of memory does not exceed available
        memory, sets self.mem and changes mem. Otherwise raises
        memory.MemoryExceededError
        
        Parameter
        ---------
        destination (str)
            Where the memory is used. For the message in Exception
        
        calc_args (tuple)
            A tuple with extra parameters for calc_memory
        
        """
        if destination is None:
            destination = "For wave function {} from {}".format(
                self.WF_type, self.source)
        new_mem = self.calc_memory(*calc_args)
        memory.allocate(new_mem, destination)
        self.mem = new_mem
    
    @abstractmethod
    def calc_memory(self):
        """Calculate and return the memory used (or to be used)
        
        This method shall not store the needed or used memory,
        but only calculates it and returns it, as a float in the unit
        used internaly in the module memory. This unit can be accessed
        by memory.unit(). See module memory for more details and auxiliary
        functions;
        
        Optionally, add extra arguments to this method, that should
        be given to _set_memory through calc_args
        """
        pass
    
    def initialize_data(self):
        if self.point_group is None:
            raise ValueError('I still do not know the point group!')
        self.orb_dim = OrbitalsSets(self.n_irrep,
                                    occ_type='R')
        self.ref_orb = OrbitalsSets(self.n_irrep,
                                    occ_type='F')
        self.froz_orb = OrbitalsSets(self.n_irrep,
                                     occ_type='R')
        self.act_orb = OrbitalsSets(self.n_irrep,
                                    occ_type='A')
    
    def spirrep_blocks(self, restricted=None):
        """Yield the possible spin and irreps, as a single integer."""
        if restricted is None:
            restricted = self.restricted
        for i in range(self.n_irrep * (1
                                       if restricted else
                                       2)):
            yield i
    
    def get_orb_spirrep(self, orb):
        """Return the spirrep of orb.
        
        Parameters:
        -----------
        orb (int)
            The global orbital index
        
        Return:
        -------
        The integer of that spirrep
        
        """
        for i_irrep in self.spirrep_blocks(restricted=True):
            if orb < self.orbs_before[i_irrep+1]:
                return i_irrep

            
    @property
    def n_irrep(self):
        if self.point_group is None:
            return None
        return number_of_irreducible_repr[self.point_group]
    
    @property
    def virt_orb(self):
        if (self.orb_dim is None
                or self.ref_orb is None):
            return None
        else:
            return self.orb_dim - self.ref_orb
    
    @property
    def corr_orb(self):
        if (self.ref_orb is None
                or self.froz_orb is None):
            return None
        else:
            return self.ref_orb - self.froz_orb
    
    @property
    def n_alpha(self):
        if self.ref_orb is None:
            return None
        if self.restricted and self.ref_orb.occ_type == 'R':
            return len(self.ref_orb) // 2
        if self.ref_orb.occ_type == 'F':
            return sum([self.ref_orb[i] for i in range(self.n_irrep)])
        len_act_orb = len(self.act_orb)
        return len_act_orb + (len(self.ref_orb) - len_act_orb) // 2
    
    @property
    def n_beta(self):
        if self.ref_orb is None:
            return None
        if self.restricted and self.ref_orb.occ_type == 'R':
            return len(self.ref_orb) // 2
        if self.ref_orb.occ_type == 'F':
            return sum([self.ref_orb[i]
                        for i in range(self.n_irrep, 2 * self.n_irrep)])
        return (len(self.ref_orb) - len(self.act_orb)) // 2
    
    @property
    def n_elec(self):
        if self.ref_orb is None:
            return None
        return len(self.ref_orb)
    
    @property
    def n_orb(self):
        if self.orb_dim is None:
            return None
        return len(self.orb_dim) // 2

    @property
    def n_orb_nocore(self):
        if self.orb_dim is None or self.froz_orb is None:
            return None
        return len(self.orb_dim - self.froz_orb) // 2

    @property
    def n_corr_alpha(self):
        if (self.n_alpha is None
                or self.corr_orb is None):
            return None
        return self.n_alpha - len(self.froz_orb) // 2
    
    @property
    def n_corr_beta(self):
        if (self.n_beta is None
                or self.corr_orb is None):
            return None
        return self.n_beta - len(self.froz_orb) // 2
    
    @property
    def n_corr_elec(self):
        corr_orb = self.corr_orb
        if corr_orb is None:
            return None
        return len(corr_orb)
    
    @property
    def orb_before(self):
        if self._orb_before is None:
            self._orb_before = [0]
            for irrep in self.spirrep_blocks(restricted=True):
                self._orb_before.append(self.corr_orb[irrep]
                                        + self.virt_orb[irrep])
            self._orb_before = np.array(self._orb_before)
        return self._orb_before
    
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
        of StringIndex or of SpirrepStringIndex.
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
        that differ from range(ref_orb(irrep)) for at most
        two inices (that is, a double excitation)
        
        Parameters:
        -----------
        
        spirrep (int, default=None)
            If passed, SpirrepStringIndex of this spirrep are yield
        
        coupled_to (tuple, default=None)
            If passed, it should be a tuple of SpirrepIndex,
            and the function should yield all StringIndex that have the
            .Index for .spirrep, or all SpirrepStringIndex of the
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
        
        Instances of StringIndex or of SpirrepStringIndex (if spirrep
        was given)
        """
        pass
    
    @abstractmethod
    def make_Jac_Hess_overlap(self, analytic=True):
        """Construct the Jacobian and the Hessian of the function overlap.
        
        Behaviour:
        ----------
        
        The function is f(x) = <wf(x), det1(x)>
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
    def calc_wf_from_z(self, z, just_C0=False):
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
        
        a tuple (new_wf, Ua, Ub) where new_wf is a WaveFunction
        with the wave function in the new representation, and Ua and Ub
        are the transformations from the previous to the new orbital
        basis (alpha and beta, respectively).
        """
        pass
    
    @abstractmethod
    def change_orb_basis(self, U, just_C0=False):
        r"""Transform the wave function after a change in the orbital basis
        
        Behaviour:
        ----------
        
        If the coefficients of wf are given in the basis |u_I>:
        
        |wf> = \sum_I c_I |u_I>
        
        it calculates the wave function in the basis |v_I>:
        
        |wf> = \sum_I d_I |v_I>
        
        and Ua and Ub are the matrix transformations of the
        MO from the basis |v_I> to the basis |u_I>:
    
        |MO of (u)> = |MO of (v)> U

        Parameters:
        -----------
        
        wf   the initial wave function as WaveFunction
        U    the orbital transformation
        just_C0   If True, calculates the coefficients of
                  the initial determinant only (default False)
        
        Return:
        -------
        
        The transformed wave function
        """
        pass
