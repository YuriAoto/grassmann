# cython: profile=False
"""The orbital spaces

"""
import numpy as np

from util.variables import int_dtype


cdef class OrbitalSpace():
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
                 int n_irrep=0,
                 dim=None,
                 str orb_type='F'):
        """Initialises the object
        
        Parameters:
        -----------
        n_irrep (int, optional, default=None)
            The number of irreducible representations.
        
        dim (iterable of int, optional, default=None)
            The dimension of each irrep
        
        orb_type (str, optional, default='F')
            Whether the orbital space is:
                'F': for all spirreps (full)
                'R': the same for alpha and beta (restricted)
                'A': only for alpha orbitals 
                'B': only for beta orbitals
       
        """
        cdef int i, x
        self.n_irrep = 0
        self._type = orb_type
        if n_irrep:
            self.n_irrep = n_irrep
        if dim is not None:
            if len(dim) > 16:
                raise ValueError('dim must have at most 16 entries.')
            if not n_irrep:
                self.n_irrep = (len(dim) // 2
                                 if self._type == 'F' else
                                 len(dim))
            if orb_type == 'F':
                for i, x in enumerate(dim):
                    self._dim_per_irrep[i] = x
            if orb_type == 'A' or orb_type == 'R':
                for i, x in enumerate(dim):
                    self._dim_per_irrep[i] = x
            if orb_type == 'R' or orb_type == 'B':
                for i, x in enumerate(dim):
                    self._dim_per_irrep[i + self.n_irrep] = x
        if self.n_irrep and self.n_irrep not in (1, 2, 4, 8):
            raise ValueError('number of irreps must be 1, 2, 4, or 8.')


    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key < 0 or key >= 2 * self.n_irrep:
                raise IndexError('Orbital ' + str(key)
                                 + ' is out of range for '
                                 + str(self) + '.')
            # if self._type == 'F':
            #     return self._dim_per_irrep[key]
            # if self._type == 'R':
            #     return self._dim_per_irrep[key % self.n_irrep]
            # if self._type == 'A':
            #     return self._dim_per_irrep[key] if key < self.n_irrep else 0
            # if self._type == 'B':
            #     return self._dim_per_irrep[key] if key >= self.n_irrep else 0
        return np.array(self._dim_per_irrep)[key]

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            if 0 > key < 2 * self.n_irrep:
                raise IndexError('Key ' + str(key)
                                  + ' is out of range for '
                                  + str(self) + '.')
            if self._type == 'A' and key >= self.n_irrep:
                raise ValueError(
                    'Cannot set occupation for beta orbital for '
                    + str(self) + '.')
            if self._type == 'B' and key < self.n_irrep:
                raise ValueError(
                    'Cannot set occupation for alpha orbital for '
                    + str(self) + '.')
        self._dim_per_irrep[key] = value
        if self._type == 'R':
            if key < self.n_irrep:
                self._dim_per_irrep[key + self.n_irrep] = value
            else:
                self._dim_per_irrep[key % self.n_irrep] = value

    def __len__(OrbitalSpace self):
        cdef int i, s = 0
        for i in range(2*self.n_irrep):
            s += self._dim_per_irrep[i]
        return s

    def __str__(OrbitalSpace self):
        x = []
        if self._type != 'B':
            if self._type == 'R':
                x.append('alpha/beta: [')
            else:
                x.append('alpha: [')
            for i in self._dim_per_irrep[:self.n_irrep]:
                x.append(str(i))
        if self._type != 'A' and self._type != 'R':
            x.append(']; beta: [')
            if self._type == 'F':
                for i in self._dim_per_irrep[self.n_irrep:2*self.n_irrep]:
                    x.append(str(i))
            else:
                for i in self._dim_per_irrep[:self.n_irrep]:
                    x.append(str(i))
        x.append(']')
        return ' '.join(x)

    def __array__(self):
        """Return a copy of the array that represents it in Full"""
        cdef int i
        return np.array([self._dim_per_irrep[i] for i in range(2*self.n_irrep)],
                        dtype=np.intc)

    def __eq__(OrbitalSpace self, OrbitalSpace other):
        cdef int i
        if self.n_irrep != other.n_irrep:
            raise ValueError('Cannot compare OrbitalsSets'
                             ' for different number of irreps')
        for i in range((1
                        if self._type == 'R' and other._type == 'R' else
                        2) * self.n_irrep):
            if self[i] != other[i]:
                return False
        return True

    def __add__(OrbitalSpace self, OrbitalSpace other):
        cdef int i
        cdef OrbitalSpace new_orbsp
        if self.n_irrep != other.n_irrep:
            raise ValueError('Both instances of OrbitalSpace'
                             ' must have same number of irreps.')
        new_orbsp = OrbitalSpace(n_irrep=self.n_irrep,
                                 orb_type=self._type
                                 if self._type == other._type else
                                 'F')
        for i in range(16):
            new_orbsp._dim_per_irrep[i] = self._dim_per_irrep[i] + other._dim_per_irrep[i]
        return new_orbsp

    def __iadd__(OrbitalSpace self, OrbitalSpace other):
        cdef int i
        cdef OrbitalSpace new_orbsp
        if self.n_irrep != other.n_irrep:
            raise ValueError('Both instances of OrbitalSpace'
                             ' must have same number of irreps.')
        self._type = (self._type
                      if self._type == other._type else
                      'F')
        for i in range(16):
            self._dim_per_irrep[i] += other._dim_per_irrep[i]
        return self

    def __sub__(OrbitalSpace self, OrbitalSpace other):
        cdef int i
        cdef OrbitalSpace new_orbsp
        if self.n_irrep != other.n_irrep:
            raise ValueError('Both instances of OrbitalSpace'
                             ' must have same number of irreps.')
        new_orbsp = OrbitalSpace(n_irrep=self.n_irrep,
                                 orb_type=self._type
                                 if self._type == other._type else
                                 'F')
        for i in range(16):
            new_orbsp._dim_per_irrep[i] = self._dim_per_irrep[i] - other._dim_per_irrep[i]
        return new_orbsp

    def __isub__(OrbitalSpace self, OrbitalSpace other):
        cdef int i
        if self.n_irrep != other.n_irrep:
            raise ValueError('Both instances of OrbitalSpace'
                             ' must have same number of irreps.')
        self._type = (self._type
                      if self._type == other._type else
                      'F')
        for i in range(16):
            self._dim_per_irrep[i] -= other._dim_per_irrep[i]
        return self

    def set_(OrbitalSpace self, OrbitalSpace other, bint force=False):
        cdef int i
        if force:
            self.n_irrep = other.n_irrep
            self._type = other._type
        else:
            if self.n_irrep != other.n_irrep:
                raise ValueError('Both instances of OrbitalSpace'
                                 ' must have same number of irreps.')
            self._type = (self._type
                          if self._type == other._type else
                          'F')
        for i in range(16):
            self._dim_per_irrep[i] = other._dim_per_irrep[i]
        return self
    
    def restrict_it(self):
        """Transform occ_type to 'R' if possible, or raise ValueError."""
        cdef int i
        if self._type == 'R':
            return
        if self._type == 'F':
            for i in range(self.n_irrep):
                if self._dim_per_irrep[i] != self._dim_per_irrep[i + self.n_irrep]:
                    raise ValueError('Cannot restrict ' + str(self) + '.')
            self._type = 'R'
            return
        raise ValueError('Cannot restrict ' + str(self) + '.')
    
    @property
    def orb_type(self):
        return self._type



cdef class FullOrbitalSpace:
    """The (dimensions of) orbital spaces
    
    
    The full orbital space is divided into subspaces:
    
    
             alpha                                beta
    ________________________          ________________________  
   |                        |        |                        |  \
   |         virtual        |        |                        |   |
   |                        |        |         virtual        |   |
   +------------+-----------+ ------ |                        |   |
   |            |           | active |                        |   |
   | correlated |           | ------ |------------+-----------+    > full
   |            |           |        | correlated |           |   |
   +------------+ reference |        |------------+           |   |
   |            |           |        |            | reference |   |
   |   frozen   |           |        |   frozen   |           |   |
   |____________|___________|        |____________|___________|  /
    
    
    This division is similar to each irrep. Each of these orbital subspaces
    is represented by an instance of OrbitalSpace.
    
    
    Atributes:
    ----------
    n_irrep (int)
        Number of irreducible representations
    
    full (OrbitalSpace)
        Full dimension of the orbital space of each irrep. It is always
        of "R" type.
    
    froz (OrbitalSpace)
        Number of frozen orbitals per irrep. It is always of "R" type.
    
    act (OrbitalSpace)
        Number of active orbitals per irrep ("A" type). Note that this is
        the difference between alpha and beta correlated orbitals, and thus
        is of "A" type.
    
    ref (OrbitalSpace)
        Number of occupied orbitals per spirrep in reference determinant.
        This includes the frozen orbitals. It is of "F" type
    
    virt, corr (OrbitalSpace)
        virt = dim - ref
        corr = ref - froz
    
    orbs_before (np.array of int)
        The number of orbitals (without frozen orbitals) before each irrep.
        orbs_before[irrep] = sum(corr[:irrep]) + sum(virt[:irrep])
    
    corr_orbs_before (np.array of int)
        The number of correlated orbitals before each irrep.
        corr_orbs_before[irrep] = sum(corr_orb[:irrep])
    
    n_orb n_orb_nofrozen (int)
        Number of spatial orbitals (with and without frozen orbitals
        
    
    """
    def __init__(self, n_irrep=0):
        self.n_irrep = n_irrep
        self.full = OrbitalSpace(n_irrep=n_irrep)
        self.froz = OrbitalSpace(n_irrep=n_irrep)
        self.ref = OrbitalSpace(n_irrep=n_irrep)
        self.virt = OrbitalSpace(n_irrep=n_irrep)
        self.corr = OrbitalSpace(n_irrep=n_irrep)
        self.act = OrbitalSpace(n_irrep=n_irrep)

    def __str__(OrbitalSpace self):
        return (f'full:       {self.full}\n'
                f'--------\n'
                f'virtual:    {self.virt}\n'
                f'correlated: {self.corr}\n\n'
                f'reference:  {self.ref}\n'
                f'active:     {self.act}\n'
                f'========\n'
                f'frozen:     {self.froz}\n')
    
    def __eq__(self, other):
        if self.full != other.full: return False
        if self.froz != other.froz: return False
        if self.ref != other.ref: return False
        if self.corr != other.corr: return False
        if self.virt != other.virt: return False
        if self.act != other.act: return False
        return True

    cpdef set_n_irrep(self, int n):
        self.n_irrep = n
        self.full.n_irrep = n
        self.froz.n_irrep = n
        self.ref.n_irrep = n
        self.virt.n_irrep = n
        self.corr.n_irrep = n
        self.act.n_irrep = n

    cpdef set_full(self, OrbitalSpace other, bint update=True):
        self.full.set_(other)
        if update: self.calc_remaining()

    cpdef add_to_full(self, OrbitalSpace other, bint update=True):
        self.full += other
        if update: self.calc_remaining()

    cpdef set_froz(self, OrbitalSpace other, bint update=True, bint add_to_full=False,
                   bint add_to_ref=False):
        self.froz.set_(other)
        if add_to_full: self.full += other
        if add_to_ref: self.ref += other
        if update: self.calc_remaining()
            
    cpdef add_to_froz(self, OrbitalSpace other, bint update=True, bint add_to_full=False,
                   bint add_to_ref=False):
        self.froz += other
        if add_to_full: self.full += other
        if add_to_ref: self.ref += other
        if update: self.calc_remaining()

    cpdef set_ref(self, OrbitalSpace other, bint update=True, bint add_to_full=False):
        self.ref.set_(other)
        if add_to_full: self.full += other
        if update: self.calc_remaining()
            
    cpdef add_to_ref(self, OrbitalSpace other, bint update=True, bint add_to_full=False):
        self.ref += other
        if add_to_full: self.full += other
        if update: self.calc_remaining()

    cpdef set_act(self, OrbitalSpace other, bint update=True, bint add_to_full=False):
        self.act.set_(other)
        if add_to_full: self.full += other
        if update: self.calc_remaining()

    cpdef add_to_act(self, OrbitalSpace other, bint update=True, bint add_to_full=False):
        self.act.set_(other)
        if add_to_full: self.full += other
        if update: self.calc_remaining()
    
    cdef calc_remaining(self):
        """Calculate the dependent attributes
        
        The following attributes are calculated (the "dependent attributes"):
        
        virt
        corr
        n_orb
        n_orb_nofrozen
        orbs_before
        corr_orbs_before
        
        They depend on the other attributes:
        
        full
        ref
        froz
        
        This is more like an implementation choice. One could for instance calculate
        ref from corr, that is also reasonable.
        
        Note that the dependent attributes are overwriten here!
        
        """
        cdef int irrep
        self.virt = self.full - self.ref
        self.corr = self.ref - self.froz
        self.n_orb = len(self.full) // 2
        self.n_orb_nofrozen = len(self.full - self.froz) // 2
        for irrep in range(self.n_irrep):
            self.orbs_before[irrep + 1] = (self.orbs_before[irrep]
                                           + self.corr[irrep]
                                           + self.virt[irrep])
        for irrep in range(2*self.n_irrep):
            if irrep == self.n_irrep - 1:
                self.corr_orbs_before[irrep + 1] = 0
            else:
                self.corr_orbs_before[irrep + 1] = (self.corr_orbs_before[irrep]
                                                    + self.corr[irrep])

    cpdef get_attributes_from(self, FullOrbitalSpace other):
        """Deepy copy all attributes from other"""
        cdef int i
        self.n_irrep = other.n_irrep
        self.n_orb = other.n_orb
        self.n_orb_nofrozen = other.n_orb_nofrozen
        for i in range(9):
            self.orbs_before[i] = other.orbs_before[i]
        for i in range(17):
            self.corr_orbs_before[i] = other.corr_orbs_before[i]
        self.full.set_(other.full, force=True)
        self.froz.set_(other.froz, force=True)
        self.ref.set_(other.ref, force=True)
        self.virt.set_(other.virt, force=True)
        self.corr.set_(other.corr, force=True)
        self.act.set_(other.act, force=True)
        # needed???
        # if restricted:
        #     if np.any(np.array(other.act)):
        #         raise ValueError(
        #             'act_orb is not empty, cannot be of restricted type!')
        #     self.orbspace.ref.restrict_it()
    
    cdef inline int first_virtual(self, int spirrep):
        """The index of first virtual orbital of spirrep"""
        return self.orbs_before[spirrep % self.n_irrep] + self.corr[spirrep]

    cdef int get_orb_irrep(self, int orb) except -1:
        """Return the irrep of orb.
        
        Parameters:
        -----------
        orb (int)
            The global orbital index (without frozen orbitals)
        
        Return:
        -------
        The integer of that irrep
        
        """
        cdef int irrep
        for irrep in range(self.n_irrep):
            if orb < self.orbs_before[irrep + 1]:
                return irrep

    cdef (int, int) get_local_index(self, int p, bint alpha_orb) except *:
        """Return index of p within its block of corr. or virtual orbitals
        
        Given the global index of an orbital, calculates its index within
        the block of orbitals it belongs
        
        Example:
        --------
        
        Parameters:
        -----------
        
        
        """
        cdef int irrep, spirrep
        irrep = self.get_orb_irrep(p)
        p -= self.orbs_before[irrep]
        spirrep = irrep + (0
                           if alpha_orb else
                           self.n_irrep)
        if p >= self.corr[spirrep]:
            p -= self.corr[spirrep]
        return p, irrep
        
    cdef int get_absolute_index(self,
                                int p,
                                int irrep,
                                bint occupied,
                                bint alpha_orb) except -1:
        """Return the absolute index of p
        
        Given the local index p, within a block of orbitals it belongs,
        its irrep, and whether it is occupied/virtual or alpha/beta,
        return its global index (within alpha or beta)
        
        It is the inverse of get_local_index
        
        Example:
        --------
        
        Parameters:
        -----------
        
        
        """
        p += self.orbs_before[irrep]
        if occupied:
            p += self.corr[
                irrep + (0
                         if alpha_orb or self.restricted else
                         self.n_irrep)]
        return p

    cdef int get_abs_corr_index(self, int p, int irrep, bint alpha_orb) except -1:
        """Return the absolute index of p within correlated orbitals
        
        Given the local index p, within a correlated block of orbitals,
        its irrep, and whether it is alpha/beta,
        return its global index (within alpha or beta correlated orbitals)
        
        It is the inverse of get_local_index
        
        Example:
        --------
        
        Parameters:
        -----------
        
        
        """
        return p + self.corr_orbs_before[irrep + (0
                                                  if alpha_orb else
                                                  self.n_irrep)]
