"""Abstract base class for electronic wave functions

All wave functions should inherit from this class.


# Covention for orbitals ordering:

Consider a multideterminantal wave function that:
1) is symmetry adapted: It is presented in a basis of symmetry adapted
   (spin-)orbitals, such that each orbital belongs to one of g possible
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
WaveFunction


"""
import copy
from abc import ABC, abstractmethod

import numpy as np

from util.variables import int_dtype
from util import memory
from molecular_geometry.symmetry import (number_of_irreducible_repr,
                                         irrep_product)
from orbitals.symmetry import OrbitalsSets



class WaveFunction(ABC):
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
    
    irrep (int, property)
        The irreducible representation that this wave function belong
    
    orb_dim (OrbitalsSets)
        Dimension of the orbital space of each irrep ("R" type)
    
    froz_orb (OrbitalsSets)
        Number of frozen orbitals per irrep ("R" type)
    
    act_orb (OrbitalsSets)
        Number of active orbitals per irrep ("A" type)
    
    ref_orb (OrbitalsSets)
        Number of occupied orbitals per spirrep in reference determinant
    
    virt_orb, corr_orb (OrbitalsSets)
        virt_orb = orb_dim - ref_orb
        corr_orb = ref_orb - froz_orb
    
    orbs_before (np.array of int)
        The number of orbitals (without frozen orbitals) before each irrep.
        orbs_before[irrep] = sum(corr_orb[:irrep]) + sum(virt_orb[:irrep])
    
    corr_orbs_before (np.array of int)
        The number of correlated orbitals before each irrep.
        orbs_before[irrep] = sum(corr_orb[:irrep])
    
    n_alpha, n_beta, n_elec, n_corr_alpha, n_corr_beta, n_corr_elec (int)
        Number of alpha, beta, total, correlated alpha, correlated beta,
        and total correlated electrons, respectively
    
    n_orb n_orb_nofrozen (int)
        Number of spatial orbitals (with and without frozen orbitals
    
    wf_type (str)
       Type of wave function
    
    source (str)
        The source of this wave function
    """
    
    def __init__(self):
        self.restricted = None
        self.point_group = None
        self._irrep = None
        self.Ms = None
        self.orb_dim = None
        self.froz_orb = None
        self.ref_orb = None
        self.act_orb = None
        self._orbs_before = None
        self._corr_orbs_before = None
        self.wf_type = None
        self.source = None
        self.mem = 0.0
    
    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = [super().__repr__()]
        x.append('-' * 50)
        x.append('point group: {}'.format(self.point_group))
        x.append('n irrep: {}'.format(self.n_irrep))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('n frozen: {}'.format(self.froz_orb))
        x.append('n act: {}'.format(self.act_orb))
        x.append('ref occ: {}'.format(self.ref_orb))
        x.append('irrep: {}'.format(self.irrep))
        x.append('Ms: {}'.format(self.Ms))
        x.append('restricted: {}'.format(self.restricted))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('WF type: {}'.format(self.wf_type))
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
        
        Usage:
        ------
        Whenever you create a subclass of WaveFunction, you should:
        1) Implement the method calc_memory, see its documentation;
        2) Call this method together with initialisation of large arrays.
           Whenever possible, do this **before** you allocate memory,
           so that the memory handler will detect if memory will be exceeded
           prior its allocation.
        
        """
        if destination is None:
            destination = "For wave function {} from {}".format(
                self.wf_type, self.source)
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
    
    @classmethod
    @abstractmethod
    def similar_to(cls, wf, restricted=None):
        """Constructs a WaveFunction with same basic attributes as wf
        
        Derived classes should call super() and initialize further attributes
        
        Parameters:
        -----------
        See get_parameters_from
        
        """
        new_wf = cls()
        new_wf.get_parameters_from(wf, restricted=restricted)
        new_wf.mem = 0.0
        return new_wf
    
    def get_parameters_from(self, wf, restricted=None):
        """Get parameters from wf
        
        Derived classes may call super() to copy further attributes
        
        Parameters:
        -----------
        wf (WaveFunction)
            The wave function that will be used to construct the new instance
        
        restricted (bool or None, optional, default=None)
            If not None, indicates if the contructed wave_function will be
            of restricted type:
            If None, it will be as wf.
            If False, creates a unrestricted wave function, even if wf is
              restricted. In such case, attributes such as ref_orb are of "F"
              type, to allow unrestricted case, but with alpha and beta
              dimensions
            If True, creates a restricted wave function if possible. If wf
              has different any alpha and beta dimensions, ValueError is raised
        
        """
        self.point_group = wf.point_group
        self.Ms = wf.Ms
        self.orb_dim = wf.orb_dim
        self.froz_orb = wf.froz_orb
        self._orbs_before = wf._orbs_before
        self._corr_orbs_before = None
        if restricted is None:
            self.restricted = wf.restricted
            self.ref_orb = wf.ref_orb
            self.act_orb = wf.act_orb
        elif restricted:
            self.restricted = True
            if np.any(wf.act_orb.as_array()):
                raise ValueError(
                    'act_orb is not empty, cannot be of restricted type!')
            self.act_orb = wf.act_orb
            self.ref_orb = copy.deepcopy(wf.ref_orb)
            self.ref_orb.restrict_it()
        else:
            self.restricted = False
            self.act_orb = wf.act_orb
            self.ref_orb = wf.ref_orb
    
    def initialize_orbitals_sets(self):
        """Initialize all orbitals sets
        
        Initialize orb_dim, froz_orb, ref_orb, and act_orb
        
        Point group must be already set, otherwise the number
        or irreps is not known.
        """
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
        """Yield the possible spin and irreps, as a single integer.
        
        Parameters:
        -----------
        restricted (None or bool, optional, default=None)
        If True, yield only up to n_irrep-1, that is, one for each irrep
        If False, yield up to 2*n_irrep-1, that is, alpha and beta for
            each irrep
        If None, uses self.restricted
        
        Yield:
        ------
        An integer: the spirrep
        
        """
        if restricted is None:
            restricted = self.restricted
        for i in range(self.n_irrep * (1
                                       if restricted else
                                       2)):
            yield i
    
    def get_orb_irrep(self, orb):
        """Return the irrep of orb.
        
        Parameters:
        -----------
        orb (int)
            The global orbital index
        
        Return:
        -------
        The integer of that irrep
        
        """
        for i_irrep in self.spirrep_blocks(restricted=True):
            if orb < self.orbs_before[i_irrep+1]:
                return i_irrep
    
    @property
    def n_irrep(self):
        """Number or irreducible representations"""
        if self.point_group is None:
            return None
        return number_of_irreducible_repr[self.point_group]
    
    @property
    def irrep(self):
        """The irreducible representation that this wave function belong"""
        if self._irrep is None:
            if self.restricted:
                self._irrep = 0
            total_irrep = 0
            for irrep in self.spirrep_blocks(restricted=True):
                n_extra = abs(self.ref_orb[irrep]
                              - self.ref_orb[irrep + self.n_irrep])
                if n_extra % 2 == 1:
                    total_irrep = irrep_product[total_irrep, irrep]
            self._irrep = total_irrep
        return self._irrep
    
    @property
    def virt_orb(self):
        """Virtual orbitals per spirrep"""
        if (self.orb_dim is None
                or self.ref_orb is None):
            return None
        else:
            return self.orb_dim - self.ref_orb
    
    @property
    def corr_orb(self):
        """Correlated orbitals per spirrep"""
        if (self.ref_orb is None
                or self.froz_orb is None):
            return None
        else:
            return self.ref_orb - self.froz_orb
    
    @property
    def n_alpha(self):
        """Number of alpha electrons"""
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
        """Number of beta electrons"""
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
        """Number of electrons"""
        if self.ref_orb is None:
            return None
        return len(self.ref_orb)
    
    @property
    def n_orb(self):
        """Number of (spatial) orbitals"""
        if self.orb_dim is None:
            return None
        return len(self.orb_dim) // 2
    
    @property
    def n_orb_nofrozen(self):
        """Number of (spatial) orbitals without frozen orbitals"""
        if self.orb_dim is None or self.froz_orb is None:
            return None
        return len(self.orb_dim - self.froz_orb) // 2
    
    @property
    def n_corr_alpha(self):
        """Number of correlated alpha electrons"""
        if (self.n_alpha is None
                or self.corr_orb is None):
            return None
        return self.n_alpha - len(self.froz_orb) // 2
    
    @property
    def n_corr_beta(self):
        """Number of correlated beta electrons"""
        if (self.n_beta is None
                or self.corr_orb is None):
            return None
        return self.n_beta - len(self.froz_orb) // 2
    
    @property
    def n_corr_elec(self):
        """Number od correlated electrons"""
        corr_orb = self.corr_orb
        if corr_orb is None:
            return None
        return len(corr_orb)
    
    @property
    def orbs_before(self):
        """Number of orbitals (without frozens) before each irrep"""
        if self._orbs_before is None:
            self._orbs_before = [0]
            for irrep in self.spirrep_blocks(restricted=True):
                self._orbs_before.append(self._orbs_before[irrep]
                                         + self.corr_orb[irrep]
                                         + self.virt_orb[irrep])
            self._orbs_before = np.array(self._orbs_before, dtype=int_dtype)
        return self._orbs_before
    
    @property
    def corr_orbs_before(self):
        """Number of correlated orb. (without frozens) before each spirrep"""
        if self._corr_orbs_before is None:
            _corr_orbs_before = [0]
            for irrep in self.spirrep_blocks():
                if irrep == self.n_irrep - 1:
                    _corr_orbs_before.append(0)
                else:
                    _corr_orbs_before.append(_corr_orbs_before[irrep]
                                             + self.corr_orb[irrep])
            self._corr_orbs_before = np.array(_corr_orbs_before)
        return self._corr_orbs_before
