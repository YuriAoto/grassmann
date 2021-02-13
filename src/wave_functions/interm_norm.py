"""An wave function in the intermediate normalisation

Classes:
--------

IntermNormWaveFunction
"""
import math
import logging

import numpy as np

from util.array_indices import (triangular,
                                get_pos_from_rectangular,
                                get_ij_from_triang,
                                get_n_from_triang)
from util.variables import int_dtype
from util.memory import mem_of_floats
from input_output import molpro
from molecular_geometry.symmetry import irrep_product
from wave_functions.general import WaveFunction
from orbitals.occ_orbitals import OccOrbital
#from wave_functions.singles_doubles import (
#    EXC_TYPE_ALL,
#    EXC_TYPE_A, EXC_TYPE_B,
#    EXC_TYPE_AA, EXC_TYPE_BB, EXC_TYPE_AB)

### temporary: use from singles_doubles afther cython
EXC_TYPE_ALL = 0
EXC_TYPE_A = 1
EXC_TYPE_B = 2
EXC_TYPE_AA = 3
EXC_TYPE_AB = 4
EXC_TYPE_BB = 5



logger = logging.getLogger(__name__)


class IntermNormWaveFunction(WaveFunction):
    """An electronic wave function in intermediate normalisation
    
    This class stores the wave function by its amplitudes,
    that form a cluster operator T, with up to double excitations:
    
    T = \sum_{i,a} t_i^a a_a^\dagger a_i
        + \sum_{i<j, a<b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i
    
    where i, j run over occupied (and correlated) orbitals and a, b
    over virtual orbitals
    
    Atributes:
    ----------
    norm (float)
        The norm of the wave function.
        Recall that, in the intermediate normalisation,
        the wave function is given as:
        |ref> + |ext>
        where |ref> is the normalised reference wave function
        (here, a single Slater determinant), and <ref|ext> = 0
        The external wave function is stored by its amplitudes.
    
    amplitudes (a 1D np.ndarray)
        t_i^a  and t_ij^ab
        
        Amplitudes of single and double excitations.
        
        Order convention:
        
        first all singles (if present) followed by all doubles
        
        For singles, t_i^a:
        
        These are stored in groups of spirrep.
        Recall that irrep(i) = irrep(a) for a nonzero amplitude.
        There are, thus, self.corr_orb[spirrep]*self.virt_orb[spirrep]
        amplitudes in each spirrep block.
        
        If restricted:
        first all amplitudes of {i, a} in irrep = 0,
        then all amplitudes of {i, a} in irrep = 1, etc.
        In each of these blocks, orbital indices run faster:
        
        t_0^0, t_0^1, t_0^2, ..., t_0^{m-1}, t_1^0, ..., t_1^{m-1}, ...
        
        If unrestricted:
        similar to unrestricted, but all alpha first, then all beta.
        
        
        For doubles:
        
        the amplitudes are stored in groups of {i, j} pairs and,
        in each of these pairs, in groups of irrep of a.
        Recall that irrep(i)*irrep(j) = irrep(a)*irrep(b) for a
        nonzero amplitude. Thus, for each pair {i, j}, and each
        irrep of a, the irrep of b is automatically determined.
        
        
        If restricted:
        i and j run over all occupied and correlated
        (spatial) orbitals, from the first to the last irrep,
        with the restriction i <= j for the absolute ordering.
        i runs faster, and thus the order of pairs i,j is:
        
        i,j
        
        0,0
        0,1  1,1
        0,2  1,2  2,2
        0,3  1,3  2,3  3,3
        
        where for example, orbital 0 is the first orbital
        of first irrep, etc. If we change orbital to a pair
        (orbital, irrep) the above ordering is:
        
        (0,0);(0,0)
        (0,0);(1,0)   (1,0);(1,0)
        (0,0);(2,0)   (1,0);(2,0)   (2,0);(2,0)
        (0,0);(0,1)   (1,0);(0,1)   (2,0);(0,1)   (0,1),(0,1)
        
        in the case of 3 occupied orbitals of first irrep and
        one for the second:
        
        absolute    i,irrep
         order
        0           0,0
        1           1,0
        2           2,0
        3           0,1
        
        Within these blocks for {i,j} pairs, there are n_irrep
        subblocks, running over the irrep of a. In each of these
        subblocks the amplitudes for pairs of virtual orbitals,
        {a,b}, are stored. There are
        virt_orb[irrep_a]*virt_orb[irrep_b]
        of such elements, and remember that irrep_b is determined
        by the other irreps. b indices run faster:
        
        t_{ij}^{00}, t_{ij}^{01}, ...,
            t_{ij}^{0{virt_orb[irrep_b]-1}}, t_{ij}^{10}, ...
            t_{ij}^{1{virt_orb[irrep_b]-1}}, ...
                t_{ij}^{{virt_orb[irrep_a]-1}{virt_orb[irrep_b]-1}}
                     
        
        # For unrestricted wave functions:
        i and j run first over alpha-alpha spin-orbitals,
        then over beta-beta spin-orbitals, and thus over alpha-beta
        spin-orbitals.
        For the first two cases, i < j (since i == j means the same
        orbital, and there can be no double excitation).
        For the last case there is no restriction, because i and j
        are surely of different spin-orbitals (i alpha and j beta)
        and all cases (i > j, i == j, and i < j) are possible.
        From the point of view of spin-orbitals, (i,alpha) < (j,beta)
        The order, for 5 occupied alpha-orbitals and 4 occupied
        beta-orbitals, is:
        
        
        aa (alpha-alpha):
        0,1                    0
        0,2  1,2               1   2
        0,3  1,3  2,3          3   4   5
        0,4  1,4  2,4  3,4     6   7   8   9
        
        bb (beta-beta):
        0,1                10
        0,2  1,2           11  12
        0,3  1,3  2,3      13  14  15
        
        ab (alpha-beta):
        0,0  1,0  2,0  3,0  4,0       16  17  18  19  20
        0,1  1,1  2,1  3,1  4,1       21  22  23  24  25
        0,2  1,2  2,2  3,2  4,2       26  27  28  29  30
        0,3  1,3  2,3  3,3  4,3       31  32  33  34  35
        
        
        To help handling the order, the functions N_from_ij and
        ij_from_N are provided.
        
        Then, for each of these pairs, the situation is similar to
        in the restricted case: there are blocks for the irrep of a
        and for the virt_orb[irrep_a]*virt_orb[irrep_b] amplitudes.
        In this case, however, t_{ij}{ab} = -t_{ij}{ba}. We opted
        to store both amplitudes, and thus the last subblock is
        antisymmetric if irrep_a == irrep_b, or it is minus the
        transpose of another subblock.
        
    
    Data Model:
    -----------
    [key]
        key should be a tuple, with length 3, 4 or 8:
        
        if len(key) == 3:
            represents alpha holes and particles of that excitation:
            key = (rank,
                   (alpha_holes, alpha_particles),
                   (beta_holes, beta_particles))
            rank should be 1 or 2, indicating the rank of that
            excitation
            alpha_holes should be a iterable with the holes in
            alpha orbitals
            similar for alpha_particles, beta_holes, and beta_particles
        
        if len(key) == 4:
            represents a single excitation, t_i^a
            key = (i, a, irrep, exc_type)
        
        if len(key) == 8:
            represents a double excitation, t_{ij}^{ab}
            key = (i, j, a, b, i_irrep, j_irrep, a_irrep, exc_type)
       
        for the last two cases, i, j, a, b are indices local to the
        correlated or virtual orbitals; i_irrep, j_irrep, and a_irrep
        are irreducible representation and exc_type the excitation type
        
        Return:
        -------
        The amplitude t_i^a or t_{ij}^{ab}
        

        Examples:
        ---------
        for:
            restricted   True
            orb_dim      [9, 5, 5, 2]
            froz_orb     [1, 0, 0, 0]
            ref_orb      [4, 2, 2, 0]
            corr_orb     [3, 2, 2, 0]
            virt_orb     [5, 3, 3, 2]
        
        rank = 1
        alpha_hp = ([0], [3])
        beta_hp = ([], [])
          => self.singles[0][0, 0]
        
        rank = 1
        alpha_hp = ([], [])
        beta_hp = ([0], [3])
          => self.singles[0][0, 0]
        
        rank = 1
        alpha_hp = ([2], [7])
        beta_hp = ([], [])
          => self.singles[0][2, 4]
        
        rank = 1
        alpha_hp = ([1], [5])
        beta_hp = ([], [])
          => self.singles[0][1, 2]
        
        rank = 1
        alpha_hp = ([8], [10])
        beta_hp = ([], [])
          => self.singles[1][0, 0]
        
        rank = 1
        alpha_hp = ([9], [12])
        beta_hp = ([], [])
          => self.singles[1][1, 2]
        
        
        rank = 2
        alpha_hp = ([], [12])
        beta_hp = ([], [])
          => self.singles[1][1, 2]
        
        for:
            restricted   False
            orb_dim      [9, 5, 5, 2]
            froz_orb     [1, 0, 0, 0]
            ref_orb      [4, 2, 2, 0, 3, 2, 2, 0]
            corr_orb     [3, 2, 2, 0, 2, 2, 2, 0]
            virt_orb     [5, 3, 3, 2, 6, 3, 3, 2]
        
        __ALPHA__           __BETA__
        0                   0
        1                   1
        2                   ----
        ----                2
        3                   3
        4                   4
        5                   5
        6                   6
        7                   7
        ====                ====
        8                   8
        9                   9
        ----                ----
        10                  10
        11                  11
        12                  12
        ====                ====
        8                   8
        9                   9
        ----                ----
        10                  10
        11                  11
        12                  12
        ====                ====
        ----                ----
        13                  13
        14                  14

        rank = 1
        alpha_hp = ([0], [3])
        beta_hp = ([], [])
          => self.singles[0][0, 0]

        rank = 1
        alpha_hp = ([], [])
        beta_hp = ([0], [3])
          => self.singles[4][0, 1]

        rank = 1
        alpha_hp = ([2], [7])
        beta_hp = ([], [])
          => self.singles[0][2, 4]

        rank = 1
        alpha_hp = ([], [])
        beta_hp = ([1], [7])
          => self.singles[4][1, 5]

        rank = 1
        alpha_hp = ([1], [5])
        beta_hp = ([], [])
          => self.singles[0][1, 2]

        rank = 1
        alpha_hp = ([8], [10])
        beta_hp = ([], [])
          => self.singles[1][0, 0]

        rank = 1
        alpha_hp = ([9], [12])
        beta_hp = ([], [])
          => self.singles[1][1, 2]

        rank = 2
        alpha_hp = ([0, 1], [5, 7])
        beta_hp = ([], [])
          => self.doubles[0][0][2, 4]

        rank = 2
        alpha_hp = ()
        beta_hp = ([0, 1], [5, 7])
          => self.doubles[N][0][3, 5]   # N is the first beta-beta pair ij

        rank = 2
        alpha_hp = ([0, 1], [10, 11])
        beta_hp = ()
          => self.doubles[0][1][0, 1]

        rank = 2
        alpha_hp = ([0], [3])
        beta_hp = ([0], [3])
          => self.doubles[0][1][0, 1]

        Parameters:
        -----------
        rank (int)
            The excitation rank

        alpha_hp (2-tuple of np-array)
            the tuple (alpha_holes, alpha_particles),
            without considering frozen orbitals

        beta_hp (2-tuple of np-array)
            the tuple (beta_holes, beta_particles),
            without considering frozen orbitals

        Return:
        -------
        A float, with the amplitude





    
    len
        The number of (initialized) amplitudes
        
    """
    def __init__(self):
        super().__init__()
        self._norm = None
        self.use_CISD_norm = True
        self.amplitudes = None
        self.ini_blocks_S = None
        self.ini_blocks_D = None
        self.first_bb_pair = None
        self.first_ab_pair = None
        self._n_ampl = None
    
    def __repr__(self):
        """Return representation of the wave function."""
        return ('<Wave Function in Intermediate normalisation>\n'
                + super().__repr__())
    
    def __str__(self):
        """Return string version the wave function."""
        return super().__repr__()
    
    def __len__(self):
        """Return the number of amplitudes amplitudes (stored or to be stored)
        
        Return:
        -------
        A integer
        """
        if self.corr_orb is None or self.virt_orb is None:
            return 0
        if self._n_ampl is None:
            self._calc_ini_blocks()
        return self._n_ampl


    def occupied_pairs(self, exc_type='all'):
        """Generator for all pairs i,j
        
        """
        if self.restricted:
            raise NotImplementedError('Have to do for restricted')
            i = OccOrbital(self.corr_orb.as_array(),
                           self.orbs_before.as_array(),
                           True)
            j = OccOrbital(self.corr_orb.as_array(),
                           self.orbs_before.as_array(),
                           True)
            while j.alive:
                yield i, j
                if i.pos_in_occ == j.pos_in_occ:
                    j.next_()
                    i.rewind()
                else:
                    i.next_()
        else:
            if exc_type in ('all', EXC_TYPE_AA):
                i = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               True)
                j = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               True)
                j.next_()
                while j.alive:
                    yield i, j
                    if i.pos_in_occ == j.pos_in_occ - 1:
                        j.next_()
                        i.rewind()
                    else:
                        i.next_()
            if exc_type in ('all', EXC_TYPE_BB):
                i = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               False)
                j = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               False)
                j.next_()
                while j.alive:
                    yield i, j
                    if i.pos_in_occ == j.pos_in_occ - 1:
                        j.next_()
                        i.rewind()
                    else:
                        i.next_()
            if exc_type in ('all', EXC_TYPE_AB):
                i = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               True)
                j = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               False)
                while j.alive: # alpha, beta
                    yield i, j
                    i.next_()
                    if not i.alive:
                        j.next_()
                        i.rewind()

    def _calc_ini_blocks(self):
        """Calculate the arrays with the initial index of amplitude blocks"""
        def add_block(i_in_D, irrep_a, irrep_b):
            n = (self.ini_blocks_D[i_in_D]
                 + self.virt_orb[irrep_a] * self.virt_orb[irrep_b])
            i_in_D += 1
            if i_in_D < self.ini_blocks_D.size:
                self.ini_blocks_D[i_in_D] = n
            else:
                self._n_ampl = n
            return i_in_D

        n_occ_alpha = (self.corr_orbs_before[self.n_irrep - 1]
                       + self.corr_orb[self.n_irrep - 1])
        n_occ_beta = self.corr_orbs_before[-1]
        if self.restricted:
            self.ini_blocks_S = np.zeros(self.n_irrep, dtype=int_dtype)
            self.ini_blocks_D = np.zeros(triangular(n_occ_alpha) 
                                         * self.n_irrep,
                                         dtype=int_dtype)
        else:
            self.ini_blocks_S = np.zeros(2*self.n_irrep, dtype=int_dtype)
            self.ini_blocks_D = np.zeros(
                (triangular(n_occ_alpha - 1)
                 + triangular(n_occ_beta - 1)
                 + n_occ_alpha * n_occ_beta) * self.n_irrep,
                dtype=int_dtype)
        if "SD" in self.wf_type:
            self.ini_blocks_S[0] = 0
            for spirrep in self.spirrep_blocks():
                new_index = (self.ini_blocks_S[spirrep]
                             + self.corr_orb[spirrep] * self.virt_orb[spirrep])
                if spirrep == (self.n_irrep
                               if self.restricted else
                               (2*self.n_irrep)) - 1:
                    self.ini_blocks_D[0] = new_index
                else:
                    self.ini_blocks_S[spirrep + 1] = new_index
        else:
            self.ini_blocks_D[0] = 0
        i_in_D = 0
        if self.restricted:
            for i, j in self.occupied_pairs():
                for a_irrep in range(self.n_irrep - 1):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[
                            a_irrep, irrep_product[i.spirrep,
                                                   j.spirrep]])
        else:
            for i, j in self.occupied_pairs(EXC_TYPE_AA):
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[a_irrep,
                                      irrep_product[i.spirrep,
                                                    j.spirrep]])
            self.first_bb_pair = self.ini_blocks_D[i_in_D] // self.n_irrep
            for i, j in self.occupied_pairs(EXC_TYPE_BB):
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[a_irrep,
                                      irrep_product[i.spirrep - self.n_irrep,
                                                    j.spirrep - self.n_irrep]])
            self.first_ab_pair = self.ini_blocks_D[i_in_D] // self.n_irrep
            for i, j in self.occupied_pairs(EXC_TYPE_AB):
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[a_irrep,
                                      irrep_product[i.spirrep,
                                                    j.spirrep - self.n_irrep]])
        self.ini_blocks_D = np.reshape(self.ini_blocks_D,
                                       (self.ini_blocks_D.size // self.n_irrep,
                                        self.n_irrep))
        if i_in_D != self.ini_blocks_D.size:
            raise Exception(f'BUG: {i_in_D} != {self.ini_blocks_D.size}')
    
    def __setitem__(self, key, value):
        """Set the amplitude associated to that excitation. See class doc."""
        self.amplitudes[self.pos_in_ampl(key)] = value
        
    def __getitem__(self, key):
        """Get the amplitude associated to that excitation. See class doc."""
        return self.amplitudes[self.pos_in_ampl(key)]
    
    def pos_in_ampl(self, key):
        """The position of that key in amplitudes vector"""
        if len(key) == 3:
            if key[0] == 1:
                return self.pos_in_ampl(
                    self.indices_of_singles(key[1], key[2]))
            elif key[0] == 2:
                return self.pos_in_ampl(
                    self.indices_of_doubles(key[1], key[2]))
            else:
                raise IndexError('Rank must be 1 or 2!')
        if len(key) == 4:
            i, a, irrep, exc_type = key
            if exc_type == EXC_TYPE_B:
                irrep += self.n_irrep
            return (self.ini_blocks_S[irrep]
                    + get_pos_from_rectangular(
                        i, a, self.virt_orb[irrep]))
        if len(key) == 8:
            i, j, a, b, i_irrep, j_irrep, a_irrep, exc_type = key
            i = self.get_abs_corr_index(i, i_irrep,
                                        exc_type in (EXC_TYPE_AA, EXC_TYPE_AB))
            j = self.get_abs_corr_index(j, j_irrep,
                                        exc_type == EXC_TYPE_AA)
            
            if self.restricted: ############ ATTENTION! CHANGE GET_N_FROM_TRIANG
                ij = get_n_from_triang(j, i, with_diag=True)
            elif exc_type == EXC_TYPE_AA:
                ij = get_n_from_triang(j, i, with_diag=False)
            elif exc_type == EXC_TYPE_BB:
                ij = self.first_bb_pair + get_n_from_triang(j, i,
                                                            with_diag=False)
            else:  #exc_type == EXC_TYPE_AB
                ij = self.first_ab_pair + get_pos_from_rectangular(
                    j, i, self.corr_orb[i_irrep])
            return (self.ini_blocks_D[ij, a_irrep]
                    + get_pos_from_rectangular(
                        a, b, self.virt_orb[
                            irrep_product[a_irrep,
                                          irrep_product[i_irrep,
                                                        j_irrep]]]))
        raise KeyError('Key must have 3, 4 (S) or 8 (D) entries!')
    
    def indices_of_singles(self, alpha_hp, beta_hp):
        """Return i, a, irrep, exc_type of single excitation"""
        exc_type = (EXC_TYPE_A
                    if alpha_hp[0].size == 1 else
                    EXC_TYPE_B)
        i, a = ((alpha_hp[0][0], alpha_hp[1][0])
                if exc_type == EXC_TYPE_A else
                (beta_hp[0][0], beta_hp[1][0]))
        irrep = self.get_orb_irrep(i)
        i -= self.orbs_before[irrep]
        a -= self.orbs_before[irrep]
        a -= self.corr_orb[
            irrep + (0
                     if (not self.restricted
                         and exc_type == EXC_TYPE_A) else
                     self.n_irrep)]
        return i, a, irrep, exc_type

    def indices_of_doubles(self, alpha_hp, beta_hp):
        if self.restricted:
            raise NotImplementedError('Missing for restricted')
        if alpha_hp[0].size == 2:
            exc_type = EXC_TYPE_AA
            i, i_irrep = self.get_local_index(alpha_hp[0][0], True)
            j, j_irrep = self.get_local_index(alpha_hp[0][1], True)
            a, a_irrep = self.get_local_index(alpha_hp[1][0], True)
            b, b_irrep = self.get_local_index(alpha_hp[1][1], True)
        elif alpha_hp[0].size == 1:
            exc_type = EXC_TYPE_AB
            i, i_irrep = self.get_local_index(alpha_hp[0][0], True)
            j, j_irrep = self.get_local_index(beta_hp[0][0], False)
            a, a_irrep = self.get_local_index(alpha_hp[1][0], True)
            b, b_irrep = self.get_local_index(beta_hp[1][0], False)
        else:
            exc_type = EXC_TYPE_BB
            i, i_irrep = self.get_local_index(beta_hp[0][0], False)
            j, j_irrep = self.get_local_index(beta_hp[0][1], False)
            a, a_irrep = self.get_local_index(beta_hp[1][0], False)
            b, b_irrep = self.get_local_index(beta_hp[1][1], False)
        return i, j, a, b, i_irrep, j_irrep, a_irrep, exc_type

    def calc_memory(self):
        """Calculate memory needed for amplitudes
        
        Parameters:
        -----------
        
        Return:
        -------
        A float, with the memory used to store the wave function amplitudes
        """
        return mem_of_floats(len(self))
    
    def initialize_amplitudes(self):
        """Initialize the list for the amplitudes."""
        self._set_memory('Amplitudes array in IntermNormWaveFunction')
        self.amplitudes = np.zeros(len(self))
    
    def update_amplitudes(self, z):
        """Update the amplitudes by z
        
        Parameters:
        -----------
        z (np.array, same len as self.amplitudes)
            The update for the amplitudes.
        
        """
        if len(z) != len(self):
            raise ValueError(
                'Update vector does not have same length as amplitude.')
        self.amplitudes += z
    
    @classmethod
    def similar_to(cls, wf, wf_type, restricted):
        new_wf = super().similar_to(wf, restricted=restricted)
        new_wf.wf_type = wf_type
        new_wf.initialize_amplitudes()
        return new_wf

    @classmethod
    def from_zero_amplitudes(cls, point_group,
                             ref_orb, orb_dim, froz_orb,
                             level='SD', wf_type='CC'):
        """Construct a new wave function with all amplitudes set to zero
        
        Parameters:
        -----------
        ref_orb (orbitals.symmetry.OrbitalsSets)
            The reference occupation
        
        orb_dim (orbitals.symmetry.OrbitalsSets)
            The dimension of orbital spaces
        
        froz_orb (orbitals.symmetry.OrbitalsSets)
            The frozen orbitals
        
        Limitations:
        ------------
        Only for restricted wave functions. Thus, ref_orb must be of 'R' type
        
        """
        new_wf = cls()
        new_wf.restricted = ref_orb.occ_type == 'R'
        new_wf.wf_type = wf_type + level
        new_wf.point_group = point_group
        new_wf.initialize_orbitals_sets()
        new_wf.ref_orb += ref_orb
        new_wf.orb_dim += orb_dim
        new_wf.froz_orb += froz_orb
        if new_wf.restricted:
            new_wf.Ms = 0.0
        else:
            new_wf.Ms = 0
            for i_irrep in range(self.n_irrep):
                new_wf.Ms += (new_wf.ref_orb[i_irrep]
                              - new_wf.ref_orb[i_irrep + self.n_irrep])
            new_wf.Ms /= 2
        new_wf.initialize_amplitudes()
        return new_wf
    
    @classmethod
    def from_Molpro(cls, molpro_output,
                    start_line_number=1,
                    wf_type=None,
                    point_group=None):
        """Load the wave function from Molpro output.
        
        Parameters:
        -----------
        See fci.get_coeff_from_molpro for a description of the parameters.
        The difference here is that molpro_output must have a CISD/CCSD wave
        function described using intermediate normalisation.
        """
        raise NotImplementedError('Do it!!')
        new_wf = cls()
        new_wf.Ms = 0.0
        new_wf.restricted = None
        check_pos_ij = False
        sgl_found = False
        dbl_found = False
        MP2_step_passed = True
        if isinstance(molpro_output, str):
            f = open(molpro_output, 'r')
            f_name = molpro_output
        else:
            f = molpro_output
            f_name = f.name
            if wf_type is None:
                raise ValueError(
                    'wf_type is mandatory when molpro_output'
                    + ' is a file object')
            if point_group is None:
                raise ValueError(
                    'point_group is mandatory when molpro_output'
                    + ' is a file object')
            new_wf.wf_type = wf_type
            new_wf.point_group = point_group
            new_wf.initialize_data()
        new_wf.source = 'From file ' + f_name
        for line_number, line in enumerate(f, start=start_line_number):
            if new_wf.wf_type is None:
                try:
                    new_wf.point_group = molpro.get_point_group_from_line(
                        line, line_number, f_name)
                except molpro.MolproLineHasNoPointGroup:
                    if line in (molpro.CISD_header,
                                molpro.CCSD_header,
                                molpro.BCCD_header,
                                molpro.UCISD_header,
                                molpro.RCISD_header,
                                molpro.UCCSD_header,
                                molpro.RCCSD_header):
                        new_wf.wf_type = line[11:15]
                        new_wf.initialize_data()
                continue
            if ('Number of closed-shell orbitals' in line
                    or 'Number of frozen orbitals' in line):
                new_orbitals = molpro.get_orb_info(
                    line, line_number,
                    new_wf.n_irrep, 'R')
                new_wf.ref_orb += new_orbitals
                new_wf.orb_dim += new_orbitals
                if 'Number of frozen orbitals' in line:
                    new_wf.froz_orb += new_orbitals
            elif 'Number of active  orbitals' in line:
                new_wf.act_orb = molpro.get_orb_info(
                    line, line_number,
                    new_wf.n_irrep, 'A')
                new_wf.orb_dim += molpro.get_orb_info(
                    line, line_number,
                    new_wf.n_irrep, 'R')
                new_wf.ref_orb += new_wf.act_orb
                new_wf.Ms = len(new_wf.act_orb) / 2
            elif 'Number of external orbitals' in line:
                new_wf.orb_dim += molpro.get_orb_info(
                    line, line_number,
                    new_wf.n_irrep, 'R')
            elif 'Starting RMP2 calculation' in line:
                MP2_step_passed = False
            elif 'RHF-RMP2 energy' in line:
                MP2_step_passed = True
            elif molpro.CC_sgl_str in line and MP2_step_passed:
                sgl_found = True
                new_wf.restricted = not ('Alpha-Alpha' in line
                                         or 'Beta-Beta' in line)
                if new_wf.singles is None:
                    new_wf.initialize_amplitudes()
                exc_type = EXC_TYPE_B if 'Beta-Beta' in line else EXC_TYPE_A
            elif molpro.CC_dbl_str in line and MP2_step_passed:
                dbl_found = True
                prev_Molpros_i = prev_Molpros_j = -1
                pos_ij = i = j = -1
                restricted = not ('Alpha-Alpha' in line
                                  or 'Beta-Beta' in line
                                  or 'Alpha-Beta' in line)
                if (new_wf.restricted is not None
                        and restricted != new_wf.restricted):
                    raise molpro.MolproInputError(
                        'Found restricted/unrestricted inconsistence '
                        + 'between singles and doubles!',
                        line=line,
                        line_number=line_number,
                        file_name=f_name)
                else:
                    new_wf.restricted = restricted
                if new_wf.singles is None and new_wf.doubles is None:
                    if new_wf.wf_type == 'CCSD':
                        new_wf.wf_type = 'CCD'
                    elif new_wf.wf_type == 'CCSD':
                        new_wf.wf_type = 'CID'
                    new_wf.initialize_amplitudes()
                if new_wf.restricted or 'Alpha-Alpha' in line:
                    exc_type = EXC_TYPE_AA
                elif 'Beta-Beta' in line:
                    exc_type = EXC_TYPE_BB
                elif 'Alpha-Beta' in line:
                    exc_type = EXC_TYPE_AB
            elif dbl_found:
                lspl = line.split()
                if len(lspl) == 7:
                    (Molpros_i, Molpros_j,
                     irrep_a, irrep_b,
                     a, b) = map(
                         lambda x: int(x) - 1, lspl[0:-1])
                    C = float(lspl[-1])
                    if exc_type in (EXC_TYPE_AA, EXC_TYPE_AB):
                        a -= new_wf.act_orb[irrep_a]
                    if exc_type == EXC_TYPE_AA:
                        b -= new_wf.act_orb[irrep_b]
                    if a < 0 or b < 0:
                        if abs(C) > zero:
                            raise molpro.MolproInputError(
                                'This coefficient of'
                                + ' singles should be zero!',
                                line=line,
                                line_numbe=line_number,
                                file_name=f_name)
                        continue
                    if (Molpros_i, Molpros_j) != (prev_Molpros_i,
                                                  prev_Molpros_j):
                        (prev_Molpros_i,
                         prev_Molpros_j) = Molpros_i, Molpros_j
                        # In Molpro's output,
                        # both occupied orbitals
                        # (alpha and beta) follow the
                        # same notation.
                        i, i_irrep = new_wf.get_occ_irrep(
                            Molpros_i, True)
                        j, j_irrep = new_wf.get_occ_irrep(
                            Molpros_j, True)
                        pos_ij = new_wf.N_from_ij(i, j,
                                                  i_irrep, j_irrep,
                                                  exc_type)
                    elif check_pos_ij:
                        my_i, my_i_irrep = new_wf.get_occ_irrep(
                            Molpros_i, True)
                        my_j, my_j_irrep = new_wf.get_occ_irrep(
                            Molpros_j, True)
                        my_pos_ij = new_wf.N_from_ij(
                            my_i, my_j,
                            my_i_irrep, my_j_irrep,
                            exc_type)
                        if (i != my_i
                            or j != my_j
                            or i_irrep != my_i_irrep
                            or j_irrep != my_j_irrep
                                or pos_ij != my_pos_ij):
                            logger.warning(
                                'check_pos_ij: differ -->'
                                + ' i=(%s %s) j=(%s %s)'
                                + 'i_irrep=(%s %s)'
                                + ' j_irrep=(%s %s)'
                                + ' pos_ij=(%s %s)',
                                i, my_i, j, my_j,
                                i_irrep, my_i_irrep,
                                j_irrep, my_j_irrep,
                                pos_ij, my_pos_ij)
                    new_wf.doubles[pos_ij][irrep_a][a, b] = C
            elif sgl_found:
                lspl = line.split()
                if len(lspl) == 4:
                    i, irrep, a = map(lambda x: int(x) - 1,
                                      lspl[0:-1])
                    C = float(lspl[-1])
                    spirrep = (irrep
                               if exc_type == EXC_TYPE_A else
                               irrep + new_wf.n_irrep)
                    i -= (sum(new_wf.ref_orb[:irrep])
                          - sum(new_wf.froz_orb[:irrep]))
                    if exc_type == EXC_TYPE_A:
                        a -= new_wf.act_orb[irrep]
                    if (a < 0
                        or i >= (new_wf.ref_orb[spirrep]
                                 - new_wf.froz_orb[irrep])):
                        if abs(C) > zero:
                            raise molpro.MolproInputError(
                                'This coefficient of singles'
                                + ' should be zero!',
                                line=line,
                                line_number=line_number,
                                file_name=f_name)
                        continue
                    if new_wf.wf_type == 'BCCD':
                        new_wf.BCC_orb_gen[spirrep][i, a] = C
                    else:
                        new_wf.singles[spirrep][i, a] = C
            if (MP2_step_passed
                and ('RESULTS' in line
                     or 'Spin contamination' in line)):
                if not dbl_found:
                    raise molpro.MolproInputError(
                        'Double excitations not found!')
                break
        if new_wf.restricted:
            new_wf.ref_orb.restrict_it()
        if isinstance(molpro_output, str):
            f.close()
        return new_wf




        
    @property
    def norm(self):
        if self._norm is None:
            raise NotImplementedError('Do it')
        return self._norm
    
    @property
    def C0(self):
        """The coefficient of reference"""
        return 1.0 / self.norm


    def N_from_ij(self, i, j, i_irrep, j_irrep, exc_type):
        """Get the initial position of block ij
        
        The indices i and j are relative to the correlated
        orbitals only, that is, start at 0 for the first correlated
        orbital (and thus frozen orbitals are excluded)
        
        Parameters:
        -----------
        i, j (int)
            The hole indices i and j
        
        i_irrep, j_irrep (int)
            The irrep of i and j
        
        exc_type (int)
            EXC_TYPE_AA, EXC_TYPE_BB, or EXC_TYPE_AB,
            indicating the exception type
        
        Returns:
        --------
        The global index for hole pairs
        """
        spin_shift = 0
        if exc_type in (EXC_TYPE_AB, EXC_TYPE_BB):
            spin_shift += self.n_corr_alpha * (self.n_corr_alpha - 1) // 2
            if exc_type in (EXC_TYPE_AA, EXC_TYPE_AB):
                spin_shift += self.n_corr_beta * (self.n_corr_beta - 1) // 2
        if self.restricted or exc_type in (EXC_TYPE_AA, EXC_TYPE_AB):
            i += sum(self.corr_orb[:i_irrep])
        else:
            i += sum(self.corr_orb[self.n_irrep:self.n_irrep + i_irrep])
        if self.restricted or exc_type in (EXC_TYPE_AB, EXC_TYPE_BB):
            j += sum(self.corr_orb[:j_irrep])
        else:
            j += sum(self.corr_orb[self.n_irrep:self.n_irrep + j_irrep])
        if self.restricted:
            pos_ij = get_n_from_triang(i, j)
        elif exc_type in (EXC_TYPE_AA, EXC_TYPE_BB):
            pos_ij = get_n_from_triang(i, j, with_diag=False)
        elif exc_type == EXC_TYPE_AB:
            pos_ij = j + i * self.n_corr_beta
        return spin_shift + pos_ij
    
    def test_indices_func(self, N,
                          i, j,
                          i_irrep, j_irrep,
                          exc_type):
        """Check if N is the global index of i, j, ... and log the result."""
        my_N = self.N_from_ij(i, j, i_irrep, j_irrep, exc_type)
        logger.debug(
            '(i=%s, j=%s, i_irrep=%s, j_irrep=%s, exc_type=%s):'
            + '  my_N, N = %s, %s%s',
            i, j, i_irrep, j_irrep, exc_type,
            my_N, N,
            '' if my_N == N else ' <<< my_N != N')
        (my_i, my_j,
         my_i_irrep, my_j_irrep,
         my_exc_type) = self.ij_from_N(N)
        logger.debug(
            'N = %s: i=(%s %s) j=(%s %s)'
            + ' i_irrep=(%s %s) j_irrep=(%s %s) exc_type=(%s %s)%s',
            N, i, my_i, i, my_j,
            i_irrep, my_i_irrep,
            j_irrep, my_j_irrep,
            exc_type, my_exc_type,
            '' if (i == my_i
                   and j == my_j
                   and i_irrep == my_i_irrep
                   and j_irrep == my_j_irrep
                   and exc_type == my_exc_type) else
            ' <<< differ')
