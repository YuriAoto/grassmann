"""

Classes:
--------

SpirrepStringIndex
StringIndex
SpirrepIndex
SD_StringIndex


"""
from collections.abc import Sized, Iterable, Container, Mapping
from collections import namedtuple

import numpy as np

from util.variables import int_dtype

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


class SD_StringIndex(StringIndex):
    """The string index for wave function with single and doubles
    
    Atributes:
    ----------
    exc_type (str, of one character)
        'R' reference determinant
        'S' single excitation
        'D' double excitation
    
    C (float)
        The coefficient
    """
    def __init__(self, spirrep_indices=None):
        super().__init__(spirrep_indices)
        self.exc_type = None
        self.C = None
    
    def __str__(self):
        return str(self.exc_type) + ': ' + super().__str__()
    
    def is_coupled_to(self, coupled_to):
        """Check if self is coupled to the elements of coupled_to
        
        Parameters:
        -----------
        coupled_to (list of gen_wf.SpirrepIndex)
            Return True only if all elements of coupled_to
            are part of self, respecting the spirreps
        """
        if coupled_to is None:
            return True
        for cpl in coupled_to:
            if (len(cpl.Index) != len(self[cpl.spirrep])
                    or int(cpl.Index) != int(self[cpl.spirrep])):
                return False
        return True
