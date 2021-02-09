"""An wave function in the intermediate normalisation

Classes:
--------

IntermNormWaveFunction
"""
import math
import logging

import numpy as np

from util.array_indices import (triangular,
                                get_ij_from_triang,
                                get_n_from_triang)
from util.variables import int_dtype
from util.memory import mem_of_floats
from input_output import molpro
from molecular_geometry.symmetry import irrep_product
from wave_functions.general import WaveFunction

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
        (0,0);(0,1)   (1,0);(0,1)   (2,0);(0,1)   (1,1),(0,1)
        
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
        For the first two cases, i > j (since i == j means the same
        orbital, and there can be no double excitation).
        For the last case there is no restriction, because i and j
        are surely of different spin-orbitals (i alpha and j beta)
        and all cases (i > j, i == j, and i < j) are possible.
        The order, for 4 occupied alpha-orbitals and 5 occupied
        beta-orbitals, is:
        
        aa (alpha-alpha):
        0,1                0
        0,2  1,2           1  2
        0,3  1,3  2,3      3  4  5
        
        bb (beta-beta):
        0,1                    6
        0,2  1,2               7   8
        0,3  1,3  2,3          9   10  11
        0,4  1,4  2,4  3,4     12  13  14  15
        
        ab (alpha-beta):
        0,0  1,0  2,0  3,0  4,0      16  17  18  19  20
        0,1  1,1  2,1  3,1  4,1      21  22  23  24  25
        0,2  1,2  2,2  3,2  4,2      26  27  28  29  30
        0,3  1,3  2,3  3,3  4,3      31  32  33  34  35
        
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
    [int]
        The amplitude at that position
    
    len
        The number of (initialized) amplitudes
        
    """
    def __init__(self):
        super().__init__()
        self._norm = None
        self.use_CISD_norm = True
        self.amplitudes = None
        self.ini_blocks = None
    
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
        if self.amplitudes is not None:
            return self.amplitudes.size
        if self.corr_orb is None or self.virt_orb is None:
            return 0
        if self.ini_blocks is None:
            self._calc_ini_blocks()
        return self.ini_blocks[-1]
    
    def _calc_ini_blocks(self):
        """Calculate the array with the initial index of amplitude blocks"""
        def add_block(n):
            self.ini_blocks.append(self.ini_blocks[-1] + n)
        self.ini_blocks = [0]
        if "SD" in self.wf_type:
            for spirrep in self.spirrep_blocks():
                add_block(self.corr_orb[spirrep]
                          * self.virt_orb[spirrep])
        if self.restricted:
            for j_irrep in range(self.n_irrep):
                for j in range(self.corr_orb[j_irrep]):
                    for i_irrep in range(j_irrep + 1):
                        for i in range(j + 1
                                       if i_irrep == j_irrep else
                                       self.corr_orb[i_irrep]):
                            for a_irrep in range(self.n_irrep):
                                b_irrep = irrep_product[
                                    a_irrep, irrep_product[i_irrep,
                                                           j_irrep]]
                                add_block(self.virt_orb[a_irrep]
                                          * self.virt_orb[b_irrep])
        else:
            for j_irrep in range(self.n_irrep):  # alpha alpha
                for j in range(self.corr_orb[j_irrep]):
                    for i_irrep in range(j_irrep):
                        for i in range(j
                                       if i_irrep == j_irrep else
                                       self.corr_orb[i_irrep]):
                            for a_irrep in range(self.n_irrep):
                                b_irrep = irrep_product[
                                    a_irrep, irrep_product[i_irrep,
                                                           j_irrep]]
                                add_block(self.virt_orb[a_irrep]
                                          * self.virt_orb[b_irrep])
            for j_irrep in range(self.n_irrep):  # beta beta
                for j in range(self.corr_orb[j_irrep + self.n_irrep]):
                    for i_irrep in range(j_irrep):
                        for i in range(j
                                       if i_irrep == j_irrep else
                                       self.corr_orb[i_irrep + self.n_irrep]):
                            for a_irrep in range(self.n_irrep):
                                b_irrep = irrep_product[
                                    a_irrep, irrep_product[i_irrep,
                                                           j_irrep]]
                                add_block(self.virt_orb[a_irrep
                                                        + self.n_irrep]
                                          * self.virt_orb[b_irrep
                                                          + self.n_irrep])
            for j_irrep in range(self.n_irrep): # alpha (i,a) beta (j,b)
                for j in range(self.corr_orb[self.n_irrep + j_irrep]):
                    for i_irrep in range(self.n_irrep):
                        for i in range(self.corr_orb[i_irrep]):
                            for a_irrep in range(self.n_irrep):
                                b_irrep = irrep_product[
                                    a_irrep, irrep_product[i_irrep,
                                                           j_irrep]]
                                add_block(self.virt_orb[a_irrep]
                                          * self.virt_orb[b_irrep
                                                          + self.n_irrep])
        self.ini_blocks = np.array(self.ini_blocks)

    def __getitem__(self, key):
        """Return the amplitude associated to that excitation
        
        Parameters:
        -----------
        key (tuple, with length 4 or 8):
        if len(key) == 4, it is assumed that it is a single excitation,
        with key = (i, a, irrep, exc_type)
        if len(key) == 8, it is assumed that it is a single excitation,
        with key = (i, j, a, b, i_irrep, j_irrep, a_irrep, exc_type)
        
        Return:
        -------
        The amplitude t_i^a or t_{ij}^{ab}
        
        """
        if len(key) == 4:
            i, a, irrep, exc_type = key
            if exc_type == 'b':
                irrep += self.n_irrep
            return self.amplitudes[self.ini_blocks_S[irrep]
                                   + get_pos_from_rectangular(
                                       i, a, self.virt_orb[irrep])]
        if len(key) == 8:
            i, j, a, b, i_irrep, j_irrep, a_irrep, exc_type = key
            i = get_abs_position(i, i_irrep, exc_type[0])
            j = get_abs_position(j, j_irrep, exc_type[1])
            irrep_b = irrep_product[irrep_product[i_irrep, j_irrep],
                                    a_irrep]
            if self.restricted:
                ij = get_n_from_triang(i, j)
            elif exc_type == 'aa':
                ij = get_n_from_triang(i, j)
            elif exc_type == 'bb':
                ij = self.first_bb_pair + get_n_from_triang(i, j)
            else:  #exc_type == 'ab'
                ij = self.first_ab_pair + get_n_from_rectangular(
                    i, j, self.corr_orb[j_irrep + self.n_irrep])
            return self.amplitudes[self.ini_blocks_D[ij, a_irrep]
                                   + get_pos_from_rectangular(
                                       a, b, self.virt_orb[b_irrep])]
        raise IndexError('Key must have 4 (S) or 8 (D) entries!')
        
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
    
    @property
    def norm(self):
        if self._norm is None:
            raise NotImplementedError('Do it')
        return self._norm
    
    @property
    def C0(self):
        """The coefficient of reference"""
        return 1.0 / self.norm
    
    def get_irrep_from_global(self, p, alpha_orb):
        """Get index and irrep of p, given in global ordering
        
        
        """
        irrep = self.get_orb_irrep(p)
        p -= self.orbs_before[irrep]
        spirrep = irrep + (0
                           if alpha_orb or self.restricted else
                           self.n_irrep)
        if p < self.ref_orb[spirrep]:
            p -= self.corr_orbs_before[spirrep]
        else:
            p -= self.corr_orb[spirrep]
        return p, irrep

    def get_occ_irrep(self, i, alpha_orb):
        """Get irrep of occupied index i, that runs all occupied orbitals
        
        Do not confuse with WaveFunction.get_orb_irrep.
        The present function considers the indices of all ocuppied orbitals
        running in sequence
        
        """
        prev_corr_sum = corr_sum = 0
        shift = (0
                 if alpha_orb or self.restricted else
                 self.n_irrep)
        for irrep in range(self.n_irrep):
            corr_sum += self.corr_orb[irrep + shift]
            if i < corr_sum:
                return i - prev_corr_sum, irrep
            prev_corr_sum = corr_sum
        raise Exception('BUG: This point should never be reached!'
                        + ' Data structure is incorrrect!')
    
    def ij_from_N(self, N):
        """Transform global index N to ij
        
        The returned indices i and j are relative to the correlated
        orbitals only, that is, start at 0 for the first correlated
        orbital (and thus frozen orbitals are excluded)
        
        Parameters:
        -----------
        N (int)
            The global index for hole pairs
        
        Returns:
        --------
        i, j, i_irrep, j_irrep, exc_type
        """
        raise NotImplementedError('Not sure if this is needed')
        n_aa = self.n_corr_alpha * (self.n_corr_alpha - 1) // 2
        n_bb = self.n_corr_beta * (self.n_corr_beta - 1) // 2
        if self.restricted or N < n_aa:
            exc_type = 'aa'
            i, j = get_ij_from_triang(N)
        elif N < n_aa + n_bb:
            exc_type = 'bb'
            i, j = get_ij_from_triang(N - n_aa)
        else:
            exc_type = 'ab'
            i = (N - (n_aa + n_bb)) // self.n_corr_beta
            j = (N - (n_aa + n_bb)) % self.n_corr_beta
        if not self.restricted and exc_type[0] == exc_type[1]:
            i += 1
        i, i_irrep = self.get_occ_irrep(i, exc_type[0] == 'a')
        j, j_irrep = self.get_occ_irrep(j, exc_type[1] == 'a')
        return (i, j,
                i_irrep, j_irrep,
                exc_type)
    
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
        
        exc_type (str)
            'aa', 'bb', or 'ab', indicating the exception type
        
        Returns:
        --------
        The global index for hole pairs
        """
        spin_shift = 0
        if exc_type[1] == 'b':
            spin_shift += self.n_corr_alpha * (self.n_corr_alpha - 1) // 2
            if exc_type[0] == 'a':
                spin_shift += self.n_corr_beta * (self.n_corr_beta - 1) // 2
        if self.restricted or exc_type[0] == 'a':
            i += sum(self.corr_orb[:i_irrep])
        else:
            i += sum(self.corr_orb[self.n_irrep:self.n_irrep + i_irrep])
        if self.restricted or exc_type[1] == 'b':
            j += sum(self.corr_orb[:j_irrep])
        else:
            j += sum(self.corr_orb[self.n_irrep:self.n_irrep + j_irrep])
        if self.restricted:
            pos_ij = get_n_from_triang(i, j)
        elif exc_type[0] == exc_type[1]:
            pos_ij = get_n_from_triang(i, j, with_diag=False)
        elif exc_type == 'ab':
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

    def indices_of_singles(self, alpha_hp, beta_hp):
        """Return spirrep, i and a of single excitation"""
        beta_exc = beta_hp[0].size == 2
        i, a = ((beta_hp[0][0], beta_hp[1][0])
                if beta_exc else
                (alpha_hp[0][0], alpha_hp[1][0]))
        irrep = self.get_orb_irrep(i)
        i -= self.orbs_before[irrep]
        a -= self.orbs_before[irrep]
        if not self.restricted and beta_exc:
            irrep += self.n_irrep
        a -= self.corr_orb[irrep]
        return irrep, i, a

    def indices_of_doubles(self, alpha_hp, beta_hp):
        if self.restricted:
            raise NotImplementedError('Missing for restricted')
        if alpha_hp[0].size == 2:
            exc_type = 'aa'
            i, i_irrep = self.get_irrep_from_global(alpha_hp[0][0], True)
            j, j_irrep = self.get_irrep_from_global(alpha_hp[0][1], True)
            a, a_irrep = self.get_irrep_from_global(alpha_hp[1][0], True)
            b, b_irrep = self.get_irrep_from_global(alpha_hp[1][1], True)
        elif alpha_hp[0].size == 1:
            exc_type = 'ab'
            i, i_irrep = self.get_irrep_from_global(alpha_hp[0][0], True)
            j, j_irrep = self.get_irrep_from_global(beta_hp[0][0], False)
            a, a_irrep = self.get_irrep_from_global(alpha_hp[1][0], True)
            b, b_irrep = self.get_irrep_from_global(beta_hp[1][0], False)
        else:
            exc_type = 'bb'
            i, i_irrep = self.get_irrep_from_global(beta_hp[0][0], False)
            j, j_irrep = self.get_irrep_from_global(beta_hp[0][1], False)
            a, a_irrep = self.get_irrep_from_global(beta_hp[1][0], False)
            b, b_irrep = self.get_irrep_from_global(beta_hp[1][1], False)
        N = self.N_from_ij(i, j, i_irrep, j_irrep, exc_type)
        return N, a_irrep, a, b
    
    def set_amplitude(self, t, rank, alpha_hp, beta_hp):
        """Set the amplitude associated to the given excitation
        
        Note that we assume that frozen orbital are not considered
        in the arrays of holes and particles.
        
        The amplitude to be set is obtained from the parameters
        in the same way as get_amplitude. See its documentation for
        examples
        
        Parameters:
        -----------
        t (float)
            The amplitude to be set
        
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
        """
        if rank == 1:
            spirrep, i, a = self.indices_of_singles(alpha_hp, beta_hp)
            self.singles[spirrep][i, a] = t
        elif rank == 2:
            N, a_irrep, a, b = self.indices_of_doubles(alpha_hp, beta_hp)
            self.doubles[N][a_irrep][a, b] = t

    def get_amplitude(self, rank, alpha_hp, beta_hp):
        """Return the amplitude associated to the given excitation
        
        Note that we assume that frozen orbital are not considered
        in the arrays of holes and particles
        
        
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


        """
        if rank == 1:
            spirrep, i, a = self.indices_of_singles(alpha_hp, beta_hp)
            return self.singles[spirrep][i, a]
        elif rank == 2:
            N, a_irrep, a, b = self.indices_of_doubles(alpha_hp, beta_hp)
            return self.doubles[N][a_irrep][a, b]
        return None
    

    def get_std_pos(self, **kargs):
        """Calculate the std_pos of a occupation index


        """
        if 'spirrep' not in kargs:
            raise ValueError('Give the spirrep if occupation was passed.')
        spirrep = kargs['spirrep']
        i = []
        a = []
        if 'occupation' in kargs:
            occ_case = len(kargs['occupation']) - self.ref_orb[spirrep]
            for ii in np.arange(self.ref_orb[spirrep],
                                dtype=int_dtype):
                if ii not in kargs['occupation']:
                    i.append(ii - self.froz_orb[spirrep])
            for aa in np.arange(self.ref_orb[spirrep],
                                self.orb_dim[spirrep],
                                dtype=int_dtype):
                if aa in kargs['occupation']:
                    a.append(aa - self.ref_orb[spirrep])
        else:
            if 'occ_case' not in kargs:
                raise ValueError(
                    'Give occ_case if occupation was not passed.')
            if 'occ_case' not in kargs:
                raise ValueError(
                    'Give occ_case if occupation is not given.')
            occ_case = kargs['occ_case']
            if 'i' in kargs:
                i.extend(list(kargs['i']))
            if 'a' in kargs:
                a.extend(list(kargs['a']))
        if occ_case == -2:
            if len(i) != 2 or len(a) != 0:
                return None
            return get_n_from_triang(max(i), min(i),
                                     with_diag=False)
        elif occ_case == -1:
            if len(i) != 1 or len(a) != 0:
                return None
            return i[0]
        elif occ_case == 0:
            if len(i) != len(a):
                return None
            if len(i) == 0:
                return 0
            if len(i) == 1:
                return 1 + i[0] * self.virt_orb[spirrep] + a[0]
            if len(i) == 2:
                return (1 + self.corr_orb[spirrep] * self.virt_orb[spirrep]
                        + (get_n_from_triang(max(i), min(i), with_diag=False)
                           * triangular(self.virt_orb[spirrep] - 1))
                        + get_n_from_triang(max(a), min(a), with_diag=False))
            return None
        elif occ_case == 1:
            if len(i) != 0 or len(a) != 1:
                return None
            return a[0]
        elif occ_case == 2:
            if len(i) != 0 or len(a) != 2:
                return None
            return get_n_from_triang(max(a), min(a),
                                     with_diag=False)
        return None

    def update_amplitudes(self, z, mode='continuous'):
        """Update the amplitudes by z


        """
        if len(z) != len(self.amplitudes):
            raise ValueError(
                'update vector does not have same length as amplitude.')
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
                exc_type = 'b' if 'Beta-Beta' in line else 'a'
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
                    exc_type = 'aa'
                elif 'Beta-Beta' in line:
                    exc_type = 'bb'
                elif 'Alpha-Beta' in line:
                    exc_type = 'ab'
            elif dbl_found:
                lspl = line.split()
                if len(lspl) == 7:
                    (Molpros_i, Molpros_j,
                     irrep_a, irrep_b,
                     a, b) = map(
                         lambda x: int(x) - 1, lspl[0:-1])
                    C = float(lspl[-1])
                    if exc_type[0] == 'a':
                        a -= new_wf.act_orb[irrep_a]
                    if exc_type[1] == 'a':
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
                               if exc_type == 'a' else
                               irrep + new_wf.n_irrep)
                    i -= (sum(new_wf.ref_orb[:irrep])
                          - sum(new_wf.froz_orb[:irrep]))
                    if exc_type == 'a':
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
