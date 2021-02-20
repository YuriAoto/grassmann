"""An wave function in the intermediate normalisation

Classes:
--------

IntermNormWaveFunction
"""
import copy
import logging

import numpy as np

from util.array_indices import (triangular,
                                n_from_triang_with_diag,
                                n_from_triang,
                                n_from_rect)
from util.variables import int_dtype, zero
from util.memory import mem_of_floats
from input_output import molpro
from molecular_geometry.symmetry import irrep_product
from wave_functions.general import WaveFunction
from orbitals.occ_orbitals import OccOrbital


# from wave_functions.singles_doubles import (
#     EXC_TYPE_A, EXC_TYPE_B,
#     EXC_TYPE_AA, EXC_TYPE_BB, EXC_TYPE_AB)
EXC_TYPE_ALL = 0
EXC_TYPE_A = 1
EXC_TYPE_B = 2
EXC_TYPE_AA = 3
EXC_TYPE_AB = 4
EXC_TYPE_BB = 5


def _transpose(X, nrow, ncol):
    """Return the ravel form of the transpose of the (raveled) block
    
    Given X as a ravelled 2D array, return its transpose, also ravelled,
    such that:
    
    X[n_from_rect(i, j, ncol)] == \
        transpose(X, nrow, ncol)[n_from_rect(j, i, nrow)]
    
    is True for all (0, 0) <= (i, j) < (nrow, ncol)
    
    Parameters:
    -----------
    X (1D np.array)
        The ravelled array to be transposed
    
    nrow, ncol (int)
        The number of rows and columns of the 2D version of X.
        ncol is associated to the index that runs faster
        X.size == nrow * ncol must be True
    
    """
    return np.ravel(np.reshape(X, (nrow, ncol),
                               order='C'),
                    order='F')


logger = logging.getLogger(__name__)


class IntermNormWaveFunction(WaveFunction):
    r"""An electronic wave function in intermediate normalisation
    
    This class stores the wave function by its amplitudes,
    that form a cluster operator T, with up to double excitations:
    
    T = \sum_{i,a} t_i^a a_a^\dagger a_i
        + \sum_{i<j, a<b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i
    
    where i, j run over occupied (and correlated) orbitals and a, b
    over virtual orbitals
    
    Atributes:
    ----------
    
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
        
        
        EXC_TYPE_AA (alpha-alpha):
        0,1                    0
        0,2  1,2               1   2
        0,3  1,3  2,3          3   4   5
        0,4  1,4  2,4  3,4     6   7   8   9
        
        EXC_TYPE_BB (beta-beta):
        0,1                10
        0,2  1,2           11  12
        0,3  1,3  2,3      13  14  15
        
        EXC_TYPE_AB (alpha-beta):
        0,0  1,0  2,0  3,0  4,0       16  17  18  19  20
        0,1  1,1  2,1  3,1  4,1       21  22  23  24  25
        0,2  1,2  2,2  3,2  4,2       26  27  28  29  30
        0,3  1,3  2,3  3,3  4,3       31  32  33  34  35
        
        
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

    def occupied_pairs(self, exc_type=EXC_TYPE_ALL):
        """Generator for all pairs i,j
        
        """
        if self.restricted:
            i = OccOrbital(self.corr_orb.as_array(),
                           self.orbs_before,
                           True)
            j = OccOrbital(self.corr_orb.as_array(),
                           self.orbs_before,
                           True)
            while j.alive:
                yield i, j
                if i.pos_in_occ == j.pos_in_occ:
                    j.next_()
                    i.rewind()
                else:
                    i.next_()
        else:
            if exc_type in (EXC_TYPE_ALL, EXC_TYPE_AA):
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
            if exc_type in (EXC_TYPE_ALL, EXC_TYPE_BB):
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
            if exc_type in (EXC_TYPE_ALL, EXC_TYPE_AB):
                i = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               True)
                j = OccOrbital(self.corr_orb.as_array(),
                               self.orbs_before,
                               False)
                while j.alive:  # alpha, beta
                    yield i, j
                    i.next_()
                    if not i.alive:
                        j.next_()
                        i.rewind()

    def _calc_ini_blocks(self):
        """Calculate the arrays with the initial index of amplitude blocks"""
        def add_block(i_in_D, spirrep_a, spirrep_b):
            n = (self.ini_blocks_D[i_in_D]
                 + self.virt_orb[spirrep_a] * self.virt_orb[spirrep_b])
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
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[
                            a_irrep, irrep_product[i.spirrep,
                                                   j.spirrep]])
        else:
            self.first_bb_pair = 0
            for i, j in self.occupied_pairs(EXC_TYPE_AA):
                self.first_bb_pair += 1
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[a_irrep,
                                      irrep_product[i.spirrep,
                                                    j.spirrep]])
            self.first_ab_pair = self.first_bb_pair
            for i, j in self.occupied_pairs(EXC_TYPE_BB):
                self.first_ab_pair += 1
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep + self.n_irrep,
                        irrep_product[a_irrep,
                                      irrep_product[i.spirrep - self.n_irrep,
                                                    j.spirrep - self.n_irrep]]
                        + self.n_irrep)
            for i, j in self.occupied_pairs(EXC_TYPE_AB):
                for a_irrep in range(self.n_irrep):
                    i_in_D = add_block(
                        i_in_D,
                        a_irrep,
                        irrep_product[a_irrep,
                                      irrep_product[i.spirrep,
                                                    j.spirrep - self.n_irrep]]
                        + self.n_irrep)
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
            if exc_type == EXC_TYPE_B and not self.restricted:
                irrep += self.n_irrep
            return (self.ini_blocks_S[irrep]
                    + n_from_rect(
                        i, a, self.virt_orb[irrep]))
        if len(key) == 8:
            i, j, a, b, i_irrep, j_irrep, a_irrep, exc_type = key
            i = self.get_abs_corr_index(i, i_irrep,
                                        exc_type in (EXC_TYPE_AA, EXC_TYPE_AB))
            j = self.get_abs_corr_index(j, j_irrep,
                                        exc_type == EXC_TYPE_AA)
            ij = self.get_ij_pos_from_i_j(i, j, i_irrep, exc_type)
            return (self.ini_blocks_D[ij, a_irrep]
                    + n_from_rect(
                        a, b, self.virt_orb[
                            irrep_product[a_irrep,
                                          irrep_product[i_irrep,
                                                        j_irrep]]]))
        raise KeyError('Key must have 3, 4 (S) or 8 (D) entries!')
    
    def get_ij_pos_from_i_j(self, i, j, i_irrep, exc_type):
        """Get the position of pair i,j as stored in self.ini_blocks_D
        
        Parameters:
        -----------
        i, j (int)
            The absolute position (within all occupied orbitals of same spin)
            of occupied orbitals i and j.
        
        i_irrep (int)
            The irreducible representation of orbital i
        
        exc_type (int, an excitation type)
            The excitation type
            
        """
        if self.restricted:
            ij = n_from_triang_with_diag(i, j)
        elif exc_type == EXC_TYPE_AA:
            ij = n_from_triang(i, j)
        elif exc_type == EXC_TYPE_BB:
            ij = self.first_bb_pair + n_from_triang(i, j)
        else:  # exc_type == EXC_TYPE_AB
            ij = self.first_ab_pair + n_from_rect(
                j, i,
                self.corr_orbs_before[self.n_irrep - 1]
                + self.corr_orb[self.n_irrep - 1])
        return ij
    
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
        # if self.restricted:
        #     raise NotImplementedError('Missing for restricted')
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
            for i_irrep in range(new_wf.n_irrep):
                new_wf.Ms += (new_wf.ref_orb[i_irrep]
                              - new_wf.ref_orb[i_irrep + new_wf.n_irrep])
            new_wf.Ms /= 2
        new_wf.initialize_amplitudes()
        return new_wf
    
    @classmethod
    def restrict(cls, ur_wf):
        """A contructor that return a restricted version of wf
        
        The constructed wave function should be the same as wf, however,
        of a restricted type. This method will work only if the amplitudes
        associated to alpha and beta excitations are equal, within the
        tolerance dictated by util.variables.zero.
        
        Parameters:
        -----------
        ur_wf (IntermNormWaveFunction)
        The wave function (in general of unrestricted type). If restricted,
        a deepcopy  is returned
        
        Raise:
        ------
        ValueError if the wave function cannot be restricted.
        
        """
        if ur_wf.restricted:
            return copy.deepcopy(ur_wf)
        if "S" in ur_wf.wf_type:
            if not np.allclose(
                    ur_wf.amplitudes[:ur_wf.ini_blocks_S[ur_wf.n_irrep]],
                    ur_wf.amplitudes[ur_wf.ini_blocks_S[ur_wf.n_irrep]:
                                     ur_wf.ini_blocks_D[0, 0]]
            ):
                raise ValueError('alpha and beta singles are not the same!')
        if not np.allclose(
                ur_wf.amplitudes[ur_wf.ini_blocks_D[0, 0]:
                                 ur_wf.ini_blocks_D[ur_wf.first_bb_pair, 0]],
                ur_wf.amplitudes[ur_wf.ini_blocks_D[ur_wf.first_bb_pair, 0]:
                                 ur_wf.ini_blocks_D[ur_wf.first_ab_pair, 0]]
        ):
            raise ValueError(
                'alpha,alpha and beta,beta singles are not the same!')
        n_corr = ur_wf.n_corr_alpha
        for i, j in ur_wf.occupied_pairs(EXC_TYPE_AB):
            pos_ij = (ur_wf.first_ab_pair
                      + n_from_rect(j.pos_in_occ,
                                    i.pos_in_occ,
                                    n_corr))
            if i.pos_in_occ >= j.pos_in_occ:
                pos_ji = (ur_wf.first_ab_pair
                          + n_from_rect(i.pos_in_occ,
                                        j.pos_in_occ,
                                        n_corr))
                for irrep_a in range(ur_wf.n_irrep):
                    irrep_b = irrep_product[
                        irrep_a, irrep_product[i.spirrep,
                                               j.spirrep - ur_wf.n_irrep]]
                    block_len = ur_wf.virt_orb[irrep_a]*ur_wf.virt_orb[irrep_b]
                    ij_ini = ur_wf.ini_blocks_D[pos_ij, irrep_a]
                    ji_ini = ur_wf.ini_blocks_D[pos_ji, irrep_b]
                    if not np.allclose(
                            ur_wf.amplitudes[ij_ini:
                                             ij_ini + block_len],
                            _transpose(ur_wf.amplitudes[ji_ini:
                                                        ji_ini + block_len],
                                       ur_wf.virt_orb[irrep_b],
                                       ur_wf.virt_orb[irrep_a])):
                        a_block = ur_wf.amplitudes[ij_ini:
                                                   ij_ini + block_len]
                        b_block = _transpose(
                            ur_wf.amplitudes[ji_ini:
                                             ji_ini + block_len],
                            ur_wf.virt_orb[irrep_b],
                            ur_wf.virt_orb[irrep_a])
                        to_msg = []
                        for iampl in range(block_len):
                            to_msg.append(
                                f'{a_block[iampl]:10.7f}'
                                + f'  {b_block[iampl]:10.7f}'
                                + f'  {a_block[iampl]-b_block[iampl]}')
                        raise ValueError(
                            'alpha,beta with non equivalent blocks:\n'
                            + '\n'.join(to_msg))
        r_wf = cls.similar_to(ur_wf, ur_wf.wf_type, True)
        if "S" in ur_wf.wf_type:
            r_wf.amplitudes[:r_wf.ini_blocks_D[0, 0]] = \
                ur_wf.amplitudes[:ur_wf.ini_blocks_S[ur_wf.n_irrep]]
        r_pos_ij = 0
        for i, j in ur_wf.occupied_pairs(EXC_TYPE_AB):
            pos_ij = (ur_wf.first_ab_pair
                      + n_from_rect(j.pos_in_occ,
                                    i.pos_in_occ,
                                    n_corr))
            if i.pos_in_occ <= j.pos_in_occ:
                r_wf.amplitudes[
                    r_wf.ini_blocks_D[r_pos_ij, 0]:
                    r_wf.ini_blocks_D[r_pos_ij + 1, 0]
                    if r_pos_ij + 1 < r_wf.ini_blocks_D.shape[0] else
                    len(r_wf)] = ur_wf.amplitudes[
                        ur_wf.ini_blocks_D[pos_ij, 0]:
                        ur_wf.ini_blocks_D[pos_ij + 1, 0]
                        if pos_ij + 1 < ur_wf.ini_blocks_D.shape[0] else
                        len(ur_wf)]
                r_pos_ij += 1
        return r_wf
    
    @classmethod
    def unrestrict(cls, r_wf):
        """A constructor that return a unrestricted version of wf
        
        The constructed wave function should be the same as wf, however,
        of a unrestricted type. Thus, the amplitudes are "duplicated" to
        hold both alpha and beta amplitudes
        
        Parameters:
        -----------
        r_wf (IntermNormWaveFunction)
        The wave function (in general of restricted type). If unrestricted,
        a deepcopy is returned
        """
        if not r_wf.restricted:
            return copy.deepcopy(r_wf)
        ur_wf = cls.similar_to(r_wf, r_wf.wf_type, False)
        if "S" in r_wf.wf_type:
            ur_wf.amplitudes[:r_wf.ini_blocks_D[0, 0]] = \
                r_wf.amplitudes[:r_wf.ini_blocks_D[0, 0]]
            ur_wf.amplitudes[r_wf.ini_blocks_D[0, 0]:
                             ur_wf.ini_blocks_D[0, 0]] = \
                r_wf.amplitudes[:r_wf.ini_blocks_D[0, 0]]
        ij = 0
        ij_diff = 0
        first_bb_ampl = (ur_wf.ini_blocks_D[ur_wf.first_bb_pair, 0]
                         - ur_wf.ini_blocks_D[0, 0])
        n_corr = r_wf.n_corr_alpha
        for i, j in r_wf.occupied_pairs(EXC_TYPE_ALL):
            for irrep_a in range(r_wf.n_irrep):
                irrep_b = irrep_product[
                    irrep_a, irrep_product[i.spirrep,
                                           j.spirrep]]
                block_len = r_wf.virt_orb[irrep_a] * r_wf.virt_orb[irrep_b]
                ini_bl = r_wf.ini_blocks_D[ij, irrep_a]
                this_block = r_wf.amplitudes[ini_bl:
                                             ini_bl + block_len]
                ij_ab_pos = (ur_wf.first_ab_pair
                             + n_from_rect(j.pos_in_occ,
                                           i.pos_in_occ,
                                           n_corr))
                ini_bl = ur_wf.ini_blocks_D[ij_ab_pos, irrep_a]
                ur_wf.amplitudes[ini_bl:
                                 ini_bl + block_len] = this_block
                if i.pos_in_occ != j.pos_in_occ:
                    transp_block = _transpose(this_block,
                                              r_wf.virt_orb[irrep_a],
                                              r_wf.virt_orb[irrep_b])
                    ji_ab_pos = (ur_wf.first_ab_pair
                                 + n_from_rect(i.pos_in_occ,
                                               j.pos_in_occ,
                                               n_corr))
                    ini_bl = ur_wf.ini_blocks_D[ji_ab_pos, irrep_b]
                    ur_wf.amplitudes[ini_bl:
                                     ini_bl + block_len] = transp_block
                    # ----- direct term
                    # alpha,alpha -> alpha,alpha
                    ini_bl = ur_wf.ini_blocks_D[ij_diff, irrep_a]
                    ur_wf.amplitudes[ini_bl:
                                     ini_bl + block_len] += this_block
                    # beta,beta -> beta,beta
                    ini_bl += first_bb_ampl
                    ur_wf.amplitudes[ini_bl:
                                     ini_bl + block_len] += this_block
                    # ----- minus transpose term
                    # alpha,alpha -> alpha,alpha
                    ini_bl = ur_wf.ini_blocks_D[ij_diff, irrep_b]
                    ur_wf.amplitudes[ini_bl:
                                     ini_bl + block_len] -= transp_block
                    # beta,beta -> beta,beta
                    ini_bl += first_bb_ampl
                    ur_wf.amplitudes[ini_bl:
                                     ini_bl + block_len] -= transp_block
            ij += 1
            if i.pos_in_occ != j.pos_in_occ:
                ij_diff += 1
        return ur_wf
    
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
        new_wf = cls()
        new_wf.Ms = 0.0
        new_wf.restricted = None
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
            new_wf.initialize_orbitals_sets()
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
                        new_wf.initialize_orbitals_sets()
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
                if new_wf.amplitudes is None:
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
                if new_wf.amplitudes is None:
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
                    all_indices = map(
                        lambda x: int(x) - 1, lspl[0:-1])
                    if exc_type == EXC_TYPE_AB:
                        (Molpros_i, Molpros_j,
                         irrep_a, irrep_b,
                         a, b) = all_indices
                    else:
                        # We will invert the pairs i,j and a,b
                        # because we use the convention i <= j,
                        # contrary to molpro convention.
                        # For alpha-beta (above) our convention
                        # is like in molpro
                        (Molpros_j, Molpros_i,
                         irrep_b, irrep_a,
                         b, a) = all_indices
                    spirrep_b = irrep_b + (
                        0 if exc_type == EXC_TYPE_AA else new_wf.n_irrep)
                    C = float(lspl[-1])
                    if exc_type in (EXC_TYPE_AA, EXC_TYPE_AB):
                        a -= new_wf.act_orb[irrep_a]
                    if exc_type == EXC_TYPE_AA:
                        b -= new_wf.act_orb[irrep_b]
                    if a < 0 or b < 0:
                        if abs(C) > zero:
                            raise molpro.MolproInputError(
                                'This coefficient of'
                                + ' doubles should be zero!',
                                line=line,
                                line_numbe=line_number,
                                file_name=f_name)
                        continue
                    if (Molpros_i, Molpros_j) != (prev_Molpros_i,
                                                  prev_Molpros_j):
                        (prev_Molpros_i,
                         prev_Molpros_j) = Molpros_i, Molpros_j
                        # In Molpro's output, both occupied orbitals
                        # (alpha and beta) follow the same notation:
                        # Let us remove "active" orbitals from beta,
                        # that are occupied in alpha but virtual in beta
                        i, j = Molpros_i, Molpros_j
                        irrep_i = 0
                        while (irrep_i < new_wf.n_irrep
                               and i >= new_wf.corr_orbs_before[irrep_i + 1]):
                            irrep_i += 1
                        if exc_type in (EXC_TYPE_AB, EXC_TYPE_BB):
                            irrep_j = 0
                            while (irrep_j < new_wf.n_irrep
                                   and
                                   j >= new_wf.corr_orbs_before[irrep_j + 1]):
                                irrep_j += 1
                            if irrep_j > 0:
                                j -= sum(new_wf.act_orb[:irrep_j])
                            if exc_type == EXC_TYPE_BB:
                                if irrep_i > 0:
                                    i -= sum(new_wf.act_orb[:irrep_i])
                        pos_ij = new_wf.get_ij_pos_from_i_j(i, j,
                                                            irrep_i,
                                                            exc_type)
                    new_wf.amplitudes[new_wf.ini_blocks_D[pos_ij, irrep_a]
                                      + n_from_rect(
                                          a, b,
                                          new_wf.virt_orb[spirrep_b])] = C
            elif sgl_found:
                lspl = line.split()
                if len(lspl) == 4:
                    i, irrep, a = map(lambda x: int(x) - 1,
                                      lspl[0:-1])
                    C = float(lspl[-1])
                    spirrep = irrep + (0
                                       if exc_type == EXC_TYPE_A else
                                       new_wf.n_irrep)
                    i -= new_wf.corr_orbs_before[irrep]
                    if exc_type == EXC_TYPE_A:
                        a -= new_wf.act_orb[irrep]
                    if a < 0 or i >= new_wf.corr_orb[spirrep]:
                        if abs(C) > zero:
                            raise molpro.MolproInputError(
                                'This coefficient of singles'
                                + ' should be zero!',
                                line=line,
                                line_number=line_number,
                                file_name=f_name)
                        continue
                    if new_wf.wf_type == 'BCCD':
                        raise NotImplementedError('Not done for BCCD')
                    else:
                        new_wf[i, a, irrep, exc_type] = C
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
