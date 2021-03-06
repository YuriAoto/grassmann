"""base class for electronic wave functions

All wave functions should inherit from this class.


# Covention for orbitals ordering:

Consider a multideterminantal wave function that:
1) is possibly symmetry adapted: It is presented in a basis of symmetry adapted
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

Example. Consider the following orbital space (see FullOrbitalSpace),
of restricted type:

full  [10, 6, 6, 4]   Dimension of orbital space by irrep (with frozen)
froz  [ 2, 1, 1, 1]   Frozen (core) orbitals
ref   [ 5, 3, 3, 1]   Reference occupation (with frozen)
corr  [ 3, 2, 2, 0]   Correlated orbitals
virt  [ 5, 3, 3, 3]   Virtual orbitals

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

full  [10, 6, 6, 4]                 As above
froz  [ 2, 1, 1, 1]                 As above
ref   [ 5, 3, 3, 1,  3, 3, 3, 1]    Reference occupation
                                        (with frozen, alpha/beta)
corr  [ 3, 2, 2, 0,  1, 2, 2, 0]    Correlated orbitals
                                        (alpha/beta)
virt  [ 5, 3, 3, 3,  7, 3, 3, 3]    Virtual orbitals

corr_orbs_before = [0, 3, 5, 7, 0, 1, 3, 5, 0]


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
#from abc import ABC, abstractmethod

import numpy as np
from libc.stdlib cimport free

from util.variables import int_dtype
from util import memory
from molecular_geometry.symmetry import (number_of_irreducible_repr,
                                         irrep_product)
from orbitals.orbital_space cimport FullOrbitalSpace
#from orbitals.orbital_space import FullOrbitalSpace


cdef class WaveFunction:
    """To be used as base class for electronic wave functions
    
    Atributes (and attributes-like methods):
    ----------------------------------------
    
    In the following you will also read about methods that could
    have been made as @property, but were not to avoid interaction
    with python. These are indicated by "()", and thus you must call
    them as methods.
    
    wf_type (str)
       Type of wave function
    
    source (str)
        The source of this wave function
    
    restricted (bool)
        Restricted (alpha and beta parameters are the same) or
        unrestricted (alpha and beta parameters differ) wave function
    
    point_group (str)
        The point group
    
    n_irrep (int)
        The number of irreducible representations
    
    irrep (int)
        The irreducible representation that this wave function belong
    
    orbspace (FullOrbitalSpace)
        The dimensions of the orbital space and its relevant subspaces
    
    n_alpha, n_beta, n_elec,
    n_corr_alpha, n_corr_beta, n_corr_elec (int)
        Number of alpha, beta, total, correlated alpha, correlated beta,
        and total correlated electrons, respectively
    
    TODO:
    Do we have to deallocate orbspace in __dealloc__??
    
    """
    
    def __cinit__(self):
        self.orbspace = FullOrbitalSpace()
        
    def __init__(self):
        self.point_group = None
        self.source = None
        self.wf_type = None
        self._irrep = -1
        self.mem = 0.0
    
    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = [super().__repr__()]
        x.append('-' * 50)
        x.append('point group: {}'.format(self.point_group))
        x.append('n irrep: {}'.format(self.n_irrep))
        x.append('Orbital space:\n{}'.format(str(self.orbspace)))
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
    
    def __dealloc__(self):
        memory.free(self.mem)
    
    def _set_memory(self, object destination=None, object calc_args=()):
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
        new_mem = self.calc_memory(calc_args)
        memory.allocate(new_mem, destination)
        self.mem = new_mem
    
    def calc_memory(self, calc_args):
        """Calculate and return the memory used (or to be used)
        
        This method shall not store the needed or used memory,
        but only calculates it and returns it, as a float in the unit
        used internaly in the module memory. This unit can be accessed
        by memory.unit(). See module memory for more details and auxiliary
        functions;
        
        Optionally, add extra arguments to this method, that should
        be given to _set_memory through calc_args
        """
        raise NotImplementedError('Implement it for the subclass!')
    
    @classmethod
    def similar_to(cls, wf, restricted=None):
        """Constructs a WaveFunction with same basic attributes as wf
        
        Derived classes should call super() and initialize further attributes
        
        Parameters:
        -----------
        See get_parameters_from
        
        """
        new_wf = cls()
        new_wf.get_attributes_from(wf, restricted)
        new_wf.mem = 0.0
        return new_wf
    
        
    def str_compare_with(self, WaveFunction other):
        """A string that compares both wave functions"""
        space_between = 4
        self_str = str(self).split('\n')
        other_str = str(other).split('\n')
        maxlen = max(map(len, self_str)) + space_between
        x1 = 'Wave function 1:'
        x2 = 'Wave function 2:'
        paste = [x1 + ' '*(maxlen - len(x1)) + x2,
                 '=' * (maxlen * 2 - space_between)]
        
        for i, x in enumerate(self_str):
            paste.append(x + ' '*(maxlen - len(x)) + other_str[i])
        return '\n'.join(paste)

    def get_attributes_from(self, WaveFunction wf, restricted=None):
        """Get parameters from wf
        
        Derived classes may call super() and copy further attributes
        
        Parameters:
        -----------
        wf (WaveFunction)
            The wave function that will be used to construct the new instance
        
        restricted (bool or None, optional, default=None)
            If not None, indicates if the contructed wave_function will be
            of restricted type:
            If None, it will be as wf.
            If False, creates a unrestricted wave function, even if wf is
              restricted. In such case, attributes such as orbspace.ref are of "F"
              type, to allow unrestricted case, but with alpha and beta
              dimensions
            If True, creates a restricted wave function if possible. If wf
              has different any alpha and beta dimensions, ValueError is raised
        
        """
        self.restricted = wf.restricted if restricted is None else restricted
        self.point_group = wf.point_group
        self.Ms = wf.Ms
        self.orbspace.get_attributes_from(wf.orbspace)
        if restricted is not None and restricted:
            if np.any(np.array(wf.orbspace.act)):
                raise ValueError(
                    'act_orb is not empty, cannot be of restricted type!')
    
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
    
    def symmetry_allowed_det(self, det):
        """Return True if the determinant is symmetry allowed in this wave function"""
        total_irrep = 0
        for p in det.alpha_occ:
            total_irrep = irrep_product[total_irrep, self.orbspace.get_orb_irrep(p)]
        for p in det.beta_occ:
            total_irrep = irrep_product[total_irrep, self.orbspace.get_orb_irrep(p)]
        return total_irrep == self.irrep
    
    cpdef bint symmetry_allowed_exc(self, SDExcitation exc):
        """Return True if the excitation is symmetry allowed in this wave function
        
        Parameters:
        -----------
        exc (SDExcitation)
            The excitation
        """
        cdef int i, h_irrep = 0, p_irrep = 0
        for i in range(exc.alpha_rank):
            h_irrep = irrep_product[h_irrep, self.orbspace.get_orb_irrep(exc.alpha_h[i])]
            p_irrep = irrep_product[p_irrep, self.orbspace.get_orb_irrep(exc.alpha_p[i])]
        for i in range(exc.beta_rank):
            h_irrep = irrep_product[h_irrep, self.orbspace.get_orb_irrep(exc.beta_h[i])]
            p_irrep = irrep_product[p_irrep, self.orbspace.get_orb_irrep(exc.beta_p[i])]
        return h_irrep == p_irrep

    @property
    def n_irrep(self):
        """Number or irreducible representations"""
        n_irrep = self.get_n_irrep()
        return n_irrep if n_irrep > 0 else None

    cdef int get_n_irrep(self):
        """Number or irreducible representations"""
        if self.point_group is None:
            return -1
        return number_of_irreducible_repr[self.point_group]

    @property
    def irrep(self):
        """The irreducible representation that this wave function belong"""
        if self._irrep == -1:
            if self.restricted:
                self._irrep = 0
            total_irrep = 0
            for irrep in self.spirrep_blocks(restricted=True):
                n_extra = abs(self.orbspace.ref[irrep]
                              - self.orbspace.ref[irrep + self.n_irrep])
                if n_extra % 2 == 1:
                    total_irrep = irrep_product[total_irrep, irrep]
            self._irrep = total_irrep
        return self._irrep
    
    @property
    def n_alpha(self):
        """Number of alpha electrons"""
        if not self.orbspace.n_irrep:
            return None
        if self.restricted and self.orbspace.ref.orb_type == 'R':
            return len(self.orbspace.ref) // 2
        if self.orbspace.ref.orb_type == 'F':
            return sum([self.orbspace.ref[i] for i in range(self.n_irrep)])
        len_act_orb = len(self.orbspace.act)
        return len_act_orb + (len(self.orbspace.ref) - len_act_orb) // 2
    
    @property
    def n_beta(self):
        """Number of beta electrons"""
        if not self.orbspace.n_irrep:
            return None
        if self.restricted and self.orbspace.ref._type == 'R':
            return len(self.orbspace.ref) // 2
        if self.orbspace.ref._type == 'F':
            return sum([self.orbspace.ref[i]
                        for i in range(self.n_irrep, 2 * self.n_irrep)])
        return (len(self.orbspace.ref) - len(self.orbspace.act)) // 2

    @property
    def n_elec(self):
        """Number of electrons"""
        if not self.orbspace.n_irrep:
            return None
        return len(self.orbspace.ref)
        
    @property
    def n_corr_alpha(self):
        """Number of correlated alpha electrons"""
        if not self.orbspace.n_irrep:
            return None
        return self.n_alpha - len(self.orbspace.froz) // 2
    
    @property
    def n_corr_beta(self):
        """Number of correlated beta electrons"""
        if not self.orbspace.n_irrep:
            return None
        return self.n_beta - len(self.orbspace.froz) // 2
    
    @property
    def n_corr_elec(self):
        """Number of correlated electrons"""
        if not self.orbspace.n_irrep:
            return None
        return len(self.orbspace.corr)
    
