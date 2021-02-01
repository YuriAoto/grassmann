"""

Classes:
--------

OrbitalsSets


"""
from collections.abc import Sequence

import numpy as np

from util.variables import int_dtype


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
            return np.array(self._occupation, dtype=int_dtype)
        if self.occ_type == 'R':
            return np.concatenate((self._occupation, self._occupation))
        if self.occ_type == 'A':
            return np.concatenate((self._occupation, np.zeros(self._n_irrep)))
        return np.concatenate((np.zeros(self._n_irrep), self._occupation))
    
    @property
    def occ_type(self):
        return self._type
