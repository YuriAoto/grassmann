"""An wave function in the intermediate normalisation

Classes:
--------

IntermNormWaveFunction
"""
import math
import logging
from collections import namedtuple

import numpy as np

from util.array_indices import (triangular,
                                ij_from_triang,
                                n_from_triang,
                                ij_from_triang_with_diag,
                                n_from_triang_with_diag)
from util.variables import int_dtype
from util.memory import mem_of_floats
from input_output import molpro
from molecular_geometry.symmetry import irrep_product
from orbitals.symmetry import OrbitalsSets
from wave_functions.general import WaveFunction
from string_indices.string_indices import (SD_StringIndex, SpirrepIndex,
                                           SpirrepStringIndex)

logger = logging.getLogger(__name__)


DoublesTypes = namedtuple('DoublesTypes',
                          'baba abab abba baab aaaa bbbb')


class IntermNormWaveFunction(WaveFunction):
    """An electronic wave function in intermediate normalisation
    
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
    
    singles (list of np.ndarrays)
        t_i^a = singles[spirrep][i, a]
        
        Amplitudes of single excitations.
        
        Each element of this list is associated to an spirrep.
        There are n_irrep entries for a restricted wave function,
        and 2*n_irrep for an unrestricted wave function.
        Each entry is an 2D np.ndarray, with shape:
        
        (self.corr_orb[spirrep], self.virt_orb[spirrep])
    
    doubles (list of list of np.ndarrays)
        t_ij^ab = doubles[N][irrep_a][a, b]
        
        Amplitudes of double excitations
        
        Each element of this list is associated
        to a pair i,j (N) of occupied orbitals.
        
        # For restricted wave functions:
        i and j run over all occupied and correlated
        (spatial) orbitals, from the first to the last irrep,
        with the restriction i >= j for the absolute ordering.
        j runs faster, and thus the order of pairs i,j is:
        
        0,0                   0
        1,0  1,1              1    2
        2,0  2,1  2,2         3    4    5
        3,0  3,1  3,2  3,3    6    7    8    9
        
        where for example, electron 0 is the first orbital
        of first irrep, etc. If we change orbital to a pair
        (orbital, irrep) the above ordering is:
        
        (0,0);(0,0)
        (1,0);(0,0)   (1,0);(1,0)
        (2,0);(0,0)   (2,0);(1,0)    (3,0);(2,0)
        (0,1);(0,0)   (0,1);(1,0)    (0,1);(2,0)  (0,1);(3,0)
        
        in the case of 3 occupied orbitals of first irrep and
        one for the second:
        
        absolute    i,irrep
         order
        0           0,0
        1           1,0
        2           2,0
        3           0,1
        
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
        1,0                0
        2,0  2,1           1  2
        3,0  3,1  3,2      3  4  5
        
        bb (beta-beta):
        1,0                    6
        2,0  2,1               7   8
        3,0  3,1  3,2          9   10  11
        4,0  4,1  4,2  4,3     12  13  14  15
        
        ab (alpha-beta):
        0,0  0,1  0,2  0,3  0,4      16  17  18  19  20
        1,0  1,1  1,2  1,3  1,4      21  22  23  24  25
        2,0  2,1  2,2  2,3  2,4      26  27  28  29  30
        3,0  3,1  3,2  3,3  3,4      31  32  33  34  35
        
        To help handling the order, the functions N_from_ij and
        ij_from_N are provided.
        
        Then, for each of these pairs, there is a list indexed
        by the irrep of the orbital a (associated to the
        occupied orbital i). The irrep of the orbital b is
        fixed by symmetry: it has to be such that
        irrep(i) x irrep(j) = irrep(a) x irrep(b)
        => irrep(b) = irrep(a) x irrep(i) x irrep(j)
        The last equality holds because, for the point groups
        we use:
         gamma x gamma = Totally symmetric irrep
        for all irreps gamma.
        
        Finally, for each pair ij and irrep of a, the amplitudes
        are stored in a 2D np.ndarray of shape:
        
        (self.virt_orb(spirrep of a), self.virt_orb(spirrep of b))
       
    
    Data Model:
    -----------
    [(SD_StringIndex)]
        Only get the CI coefficient (of the normalised version!)
        of that determinant
    
    len
        len(singles) + len(doubles).
        TODO: should be something more significant, such as the number
        determinants
        
    """
    def __init__(self):
        super().__init__()
        self._norm = None
        self.use_CISD_norm = True
        self.singles = None
        self.BCC_orb_gen = None
        self.doubles = None
    
    def __getitem__(self, I):
        """Return the CI coefficient from a SD_StringIndex"""
        return I.C
    
    def __len__(self, I):
        length = 0
        if self.singles is not None:
            length += len(self.singles)
        if self.doubles is not None:
            length += len(self.doubles)
        return length
    
    def __repr__(self):
        """Return representation of the wave function."""
        x = ['']
        if self.singles is not None:
            x.append('Amplitudes of single excitations:')
            x.append('')
            for spirrep, S in enumerate(self.singles):
                x.append('Spirrep {}:'.format(spirrep))
                x.append(str(S))
                x.append('')
            x.append('=' * 50)
        if self.BCC_orb_gen is not None:
            x.append('Orbital rotation parameters for BCC reference:')
            x.append('')
            for spirrep, S in enumerate(self.BCC_orb_gen):
                x.append('Spirrep {}:'.format(spirrep))
                x.append(str(S))
                x.append('')
            x.append('=' * 50)
        if self.doubles is not None:
            x.append('Amplitudes of double excitations:')
            x.append('')
            for N, Dij in enumerate(self.doubles):
                i, j, i_irrep, j_irrep, exc_type = self.ij_from_N(N)
                x.append(('N = {0:} (i={1:}, j={2:}, '
                          + 'i_irrep={3:}, j_irrep={4:}, exc_type={5:}):').
                         format(N, i, j, i_irrep, j_irrep, exc_type))
                x.append('')
                for a_irrep, D in enumerate(Dij):
                    b_irrep = irrep_product[a_irrep,
                                            irrep_product[i_irrep, j_irrep]]
                    x.append('a_irrep, b_irrep = {}, {}'.format(a_irrep,
                                                                b_irrep))
                    x.append(str(D))
                    x.append('')
                x.append('-' * 50)
        return ('<Wave Function in Intermediate normalisation>\n'
                + super().__repr__()
                + '\n'.join(x))
    
    def __str__(self):
        """Return string version the wave function."""
        return super().__repr__()
        S_main = None
        D_main = None
        if self.singles is not None:
            for S in self.singles:
                if S_main is None or abs(S_main.t) < abs(S.t):
                    S_main = S
        if self.doubles is not None:
            for D in self.doubles:
                if D_main is None or abs(D_main.t) < abs(D.t):
                    D_main = D
        return ('|0> + {0:5f} |{1:d} -> {2:d}> + ... + '
                + '{3:5f} |{4:d},{5:d} -> {6:d},{7:d}> + ...').format(
                    S_main.t, S_main.i, S_main.a,
                    D_main.t, D_main.i, D_main.j,
                    D_main.a, D_main.b)

    @property
    def norm(self):
        if self._norm is None:
            if (self.wf_type == 'CISD'
                    or self.wf_type in ('CCSD', 'BCCD', 'CCD')
                    and self.use_CISD_norm):
                self._norm = 1.0
                S = 0.0
                for I in self.string_indices():
                    S += self[I]**2
                self._norm = math.sqrt(S)
                if self.wf_type != 'CISD':
                    logger.warning(
                        'We are calculating the CCSD norm'
                        + ' considering only SD determinants!!')
            elif self.wf_type in ('CCSD', 'BCCD', 'CCD'):
                self._norm = 1.0
                logger.warning('CCSD norm set to 1.0!!')
            else:
                raise ValueError(
                    'We do not know how to calculate the norm for '
                    + self.wf_type + '!')
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
        n_aa = self.n_corr_alpha * (self.n_corr_alpha - 1) // 2
        n_bb = self.n_corr_beta * (self.n_corr_beta - 1) // 2
        if self.restricted:
            exc_type = 'aa'
            j, i = ij_from_triang_with_diag(N)
        elif N < n_aa:
            exc_type = 'aa'
            j, i = ij_from_triang(N)
        elif N < n_aa + n_bb:
            exc_type = 'bb'
            j, i = ij_from_triang(N - n_aa)
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
        """Transform indices to the global index N
        
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
            pos_ij = n_from_triang_with_diag(j, i)
        elif exc_type[0] == exc_type[1]:
            pos_ij = n_from_triang(j, i)
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

    def calc_n_ampl(self, with_singles, with_BCC_orb_gen):
        """Calculate the number of amplitudes amplitudes

        Parameters:
        -----------
        with_singles and with_BCC_orb_gen are booleans,
        similar to the arguments of initialize_SD_lists,
        and if True the memory needed to store singles and the
        BCC orbital generators are also taken into account

        Return:
        -------
        A integer
        """
        n_ampl = 0
        if with_singles:
            for spirrep in self.spirrep_blocks():
                n_ampl += self.corr_orb[spirrep] * self.virt_orb[spirrep]
        if with_BCC_orb_gen:
            for spirrep in self.spirrep_blocks():
                n_ampl += self.corr_orb[spirrep] * self.virt_orb[spirrep]
        for exc_type in (['aa']
                         if self.restricted else
                         ['aa', 'bb', 'ab']):
            for i_irrep in range(self.n_irrep):
                i_spirrep = i_irrep + (self.n_irrep
                                       if exc_type[0] == 'b' else 0)
                for i in range(self.corr_orb[i_spirrep]):
                    for j_irrep in range(self.n_irrep):
                        j_spirrep = j_irrep + (self.n_irrep
                                               if exc_type[1] == 'b' else 0)
                        for j in range(self.corr_orb[j_spirrep]):
                            if self.restricted or exc_type[0] == exc_type[1]:
                                if i_irrep < j_irrep:
                                    continue
                                elif i_irrep == j_irrep:
                                    if i < j:
                                        continue
                                    elif i == j and not self.restricted:
                                        continue
                            for a_spirrep in range(self.n_irrep):
                                b_spirrep = irrep_product[
                                    a_spirrep, irrep_product[i_irrep,
                                                             j_irrep]]
                                if exc_type[0] == 'b':
                                    a_spirrep += self.n_irrep
                                if exc_type[1] == 'b':
                                    b_spirrep += self.n_irrep
                                n_ampl += (self.virt_orb[a_spirrep]
                                           * self.virt_orb[b_spirrep])
        return n_ampl

    def calc_memory(self, with_singles, with_BCC_orb_gen):
        """Calculate memory needed for amplitudes

        Parameters:
        -----------
        with_singles and with_BCC_orb_gen are booleans,
        similar to the arguments of initialize_SD_lists,
        and if True the memory needed to store singles and the
        BCC orbital generators are also taken into account

        Return:
        -------
        A float, with the memory used to store the wave function amplitudes
        """
        return mem_of_floats(self.calc_n_ampl(with_singles,
                                              with_BCC_orb_gen))

    def initialize_SD_lists(self,
                            with_singles=True,
                            with_BCC_orb_gen=False):
        """Initialize the lists for singles and doubles amplitudes."""
        self._set_memory('SD lists of IntermNormWaveFunction',
                         calc_args=(with_singles, with_BCC_orb_gen))
        test_ind_func = False
        if with_singles:
            self.singles = []
            for spirrep in self.spirrep_blocks():
                self.singles.append(np.zeros((self.corr_orb[spirrep],
                                              self.virt_orb[spirrep]),
                                             dtype=np.float64))
        if with_BCC_orb_gen:
            self.BCC_orb_gen = []
            for spirrep in self.spirrep_blocks():
                self.BCC_orb_gen.append(np.zeros((self.corr_orb[spirrep],
                                                  self.virt_orb[spirrep]),
                                                 dtype=np.float64))
        self.doubles = []
        N_iter = 0
        for exc_type in (['aa']
                         if self.restricted else
                         ['aa', 'bb', 'ab']):
            for i_irrep in range(self.n_irrep):
                i_spirrep = i_irrep + (self.n_irrep
                                       if exc_type[0] == 'b' else 0)
                for i in range(self.corr_orb[i_spirrep]):
                    for j_irrep in range(self.n_irrep):
                        j_spirrep = j_irrep + (self.n_irrep
                                               if exc_type[1] == 'b' else 0)
                        for j in range(self.corr_orb[j_spirrep]):
                            if self.restricted or exc_type[0] == exc_type[1]:
                                if i_irrep < j_irrep:
                                    continue
                                elif i_irrep == j_irrep:
                                    if i < j:
                                        continue
                                    elif i == j and not self.restricted:
                                        continue
                            new_dbl_ij = []
                            for a_spirrep in range(self.n_irrep):
                                b_spirrep = irrep_product[
                                    a_spirrep, irrep_product[i_irrep,
                                                             j_irrep]]
                                if exc_type[0] == 'b':
                                    a_spirrep += self.n_irrep
                                if exc_type[1] == 'b':
                                    b_spirrep += self.n_irrep
                                new_dbl_ij.append(
                                    np.zeros((self.virt_orb[a_spirrep],
                                              self.virt_orb[b_spirrep]),
                                             dtype=np.float64))
                            self.doubles.append(new_dbl_ij)
                            if test_ind_func and logger.level <= logging.DEBUG:
                                self.test_indices_func(N_iter,
                                                       i, j,
                                                       i_irrep, j_irrep,
                                                       exc_type)
                                N_iter += 1

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
            return n_from_triang(min(i), max(i))
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
                        + (n_from_triang(min(i), max(i))
                           * triangular(self.virt_orb[spirrep] - 1))
                        + n_from_triang(min(a), max(a)))
            return None
        elif occ_case == 1:
            if len(i) != 0 or len(a) != 1:
                return None
            return a[0]
        elif occ_case == 2:
            if len(i) != 0 or len(a) != 2:
                return None
            return n_from_triang(min(a), max(a))
        return None

    def spirrep_blocks(self, restricted=None):
        """Yield the possible spin and irreps, as a single integer."""
        if restricted is None:
            restricted = self.restricted
        for i in range(self.n_irrep * (1
                                       if restricted else
                                       2)):
            yield i

    def string_indices(self,
                       spirrep=None,
                       coupled_to=None,
                       no_occ_orb=False,
                       only_ref_occ=False,
                       only_this_occ=None,
                       print_info_to_log=False):
        """Yield StringIndex or SD_StringIndex

        Parameters:
        -----------
        no_occ_orb (bool, optional, default)
            does not add orbital occupation in the returned Index
               (not implemented!!)

        Behaviour:
        ----------
        This function defines the "std_pos" of the strings.
        This means: for each spirrep, there is a standard order
        associated to all possible strings of this spirrep
        The attribute standard_position_of_string
        of gen_wf.SpirrepStringIndex is the position of such
        string in this standard order.
        We will describe such ordering here:
        There are, in fact, a standard order for each possible
        occupation number of the spirrep.
        In CISD wave function, this can be -2, -1, 0, 1, or 2,
        relative to the occupation of the reference (self.ref_orb[spirrep]).
        This is the order that
        string_indices(spirrep=spirrep, only_this_occ=occ)
        will yield.
        Hole refer to an empty spin-orbital that is occupied
        in the reference.
        Particle refer to an occupied spin-orbital that is
        empty in the reference
        In the expressions below, we are showing explicitly the removal
        of frozen electrons or occupied electrons.
        That is, the indices are the
        absloute indices of orbitals.

        # Case -2 (that is, occ=ref_orb[spirrep] - 2):
        There are two holes and no particles. These two holes must be in
        different spin-orbitals. If these are i and j, with i > j:

        std_pos = n_from_triang(j - self.froz_orb[irrep],
                                i - self.froz_orb[irrep])

        That is, we start with both holes at the lowest positions and
        go up following a trianglar arrangement
        (see how double excitations are stored).

        # Case -1:
        There is one hole. Note that, it is not possible to
        have two holes and one particle, because this would lead a
        determinant with the wrong irrep or Ms.
        If i is where this hole is:

        std_pos = i - self.froz_orb[irrep]

        # Case 0:
        This is the most complicated, as several possibilities exist.
        We start with the reference:

        std_pos = 0  # for the occupation as in the reference

        Then come the single excitations. There are

        self.corr_orb[spirrep] * self.virt_orb[spirrep]

        possible single excitations within spirrep.
        These are ordered starting from the lowest orbital
        and the index of virtual orbitals running faster.
        Thus, if the hole and particle are at i and a:

        std_pos = (1
                   + (i - self.froz_orb[spirrep]) * self.virt_orb[spirrep]
                   + (a - self.ref_orb[spirrep]))

        There is a plus one, becaus of the reference.

        Then come the double excitations. There are

        triangular(self.virt_orb[spirrep] - 1)
        * triangular(self.corr_orb[spirrep] - 1)

        of them. Again, indices of virtual orbitals run faster.
        If the holes are at i and j (i > j)
        and the particles at a and b (a > b)

        std_pos = (1 + self.corr_orb[spirrep] * self.virt_orb[spirrep]
                   + (n_from_triang(j - self.froz_orb[irrep],
                                    i - self.froz_orb[irrep])
                      * triangular(self.virt_orb[spirrep] - 1))
                   + n_from_triang(a - self.ref_orb[spirrep],
                                   b - self.ref_orb[spirrep]))

        Although the expression looks complicated, you can look at it as:

        std_pos = (n_ref + n_singles
                   + (pos(i,j)
                      * n_virt)
                   + pos(a,b))

        That is: the first terms are the total number of single
        and double excitation;
        pos(i,j) is the position of holes, as in case -2, times the
        total number of pairs of virtuals, n_virt
        (Recall: virtuals run faster);
        pos(a,b) is the position of particles, as in case 2.


        # Case 1:
        There is one particle. Like in case -2, it is not
        possible to have two particles and one hole.
        If a is where this hole is:

        std_pos = i - self.ref_orb[spirrep]

        # Case 2:
        There are two particles and no hole. These two particles must be in
        different spin-orbitals. If these are a and b, with a > b:

        std_pos = n_from_triang(b - self.ref_orb[spirrep],
                                a - self.ref_orb[spirrep])

        TODO:
        -----
        This whole subroutine must be refactored out...It is too complicated,
        with too many inedentation levels
        """
        print_info_to_log = print_info_to_log and logger.level <= logging.DEBUG
        if self.wf_type not in ('CISD', 'CCSD', 'CCD', 'BCCD'):
            raise NotImplementedError('Currently works only for CISD and CCSD')
        if self.wf_type == 'CCSD':
            logger.warning(
                'string_indices yields only singly and doubly'
                + ' excited determinants for CCSD!!')
        if only_ref_occ and only_this_occ is not None:
            raise ValueError(
                'Do not give only_this_occ and set only_ref_occ to True.')
        if only_ref_occ:
            only_this_occ = (self.ref_orb
                             if spirrep is not None else
                             self.ref_orb[spirrep])
        if (spirrep is not None
            and only_this_occ is not None
                and not isinstance(only_this_occ, (int, np.integer))):
            raise ValueError(
                'If spirrep is given, only_this_occ must be an integer.')
        if (spirrep is None
            and only_this_occ is not None
                and not isinstance(only_this_occ, OrbitalsSets)):
            raise ValueError(
                'If spirrep is not given, only_this_occ must be'
                + ' an instance of gen_wf.OrbitalsSets.')
        if coupled_to is not None:
            if not isinstance(coupled_to, tuple):
                raise ValueError('Parameter coupled_to must be a tuple.')
            if not isinstance(coupled_to, SpirrepIndex):
                for cpl in coupled_to:
                    if not isinstance(cpl, SpirrepIndex):
                        raise ValueError(
                            'Parameter coupled_to must be a tuple'
                            + ' of gen_wf.SpirrepIndex.')
            else:
                coupled_to = (coupled_to,)
        logger.debug('only_this_occ:\n%s', only_this_occ)
        if spirrep is None:
            yield from self._string_indices_full_indices(
                coupled_to,
                no_occ_orb,
                only_this_occ,
                print_info_to_log)
        else:
            yield from self._string_indices_spirrep(
                spirrep,
                coupled_to,
                no_occ_orb,
                only_this_occ,
                print_info_to_log)

    def _string_indices_full_indices(self,
                                     coupled_to=None,
                                     no_occ_orb=False,
                                     only_this_occ=None,
                                     print_info_to_log=False):
        if (only_this_occ is None
                or only_this_occ == self.ref_orb):
            Index = SD_StringIndex.make_reference(
                self.ref_orb, self.n_irrep)
            Index.exc_type = 'R'
            Index.C = 1.0 / self.norm
            Index.set_wave_function(self)
            if Index.is_coupled_to(coupled_to):
                yield Index
            for i_spirrep in self.spirrep_blocks():
                yield from self._string_indices_sing_exc(
                    i_spirrep,
                    coupled_to,
                    only_this_occ)
        if self.restricted:
            for N, Dij in enumerate(self.doubles):
                i, j, irrep_i, irrep_j, exc_type = self.ij_from_N(N)
                if print_info_to_log:
                    logger.debug(
                        '\nN=%d; i, irrep_i = %d, %d; j, irrep_j = %d, %d;'
                        + ' exc_type = %s',
                        N, i, irrep_i, j, irrep_j, exc_type)
                for irrep_a, D in enumerate(Dij):
                    irrep_b = irrep_product[irrep_a,
                                            irrep_product[irrep_i,
                                                          irrep_j]]
                    if print_info_to_log:
                        logger.debug('irrep_a, irrep_b = %d, %d',
                                     irrep_a, irrep_b)
                    if irrep_b > irrep_a:
                        continue
                    if (self.virt_orb[irrep_a] == 0
                            or self.virt_orb[irrep_b] == 0):
                        continue
                    if (irrep_i, i) == (irrep_j, j):
                        yield from self._string_indices_D_ii(
                            i, irrep_i, irrep_a,
                            D,
                            coupled_to,
                            only_this_occ,
                            print_info_to_log)
                    else:
                        yield from self._string_indices_D_ij(
                            i, j,
                            irrep_i, irrep_j, irrep_a, irrep_b,
                            D, Dij[irrep_b],
                            coupled_to,
                            only_this_occ,
                            print_info_to_log)
        else:
            raise NotImplementedError(
                'Currently only for restricted wave functions')

    def _string_indices_sing_exc(self, irp,
                                 coupled_to=None,
                                 only_this_occ=None,
                                 print_info_to_log=False):
        if self.singles is None:
            return
        if print_info_to_log:
            to_log = []
        Index = SD_StringIndex.make_reference(self.ref_orb, self.n_irrep)
        Index.exc_type = 'S'
        Index.set_wave_function(self)
        sign = (1 if
                self.corr_orb[irp] % 2 == 0
                else -1)
        for i_occ in range(self.corr_orb[irp]):
            sign = -sign
            if i_occ == 0:
                Index[irp][self.froz_orb[irp]:-1] = np.arange(
                    self.froz_orb[irp] + 1,
                    self.ref_orb[irp],
                    dtype=int_dtype)
            for a_virt in range(self.virt_orb[irp]):
                Index.C = (sign * self.singles[irp][i_occ, a_virt]
                           / self.norm)
                Index[irp][-1] = (self.ref_orb[irp] + a_virt)
                if Index.is_coupled_to(coupled_to):
                    yield Index
                if self.restricted:
                    Index.swap_spirreps(irp, irp + self.n_irrep)
                    if Index.is_coupled_to(coupled_to):
                        yield Index
                    Index.swap_spirreps(irp, irp + self.n_irrep)
            Index[irp][self.froz_orb[irp]
                       + i_occ] = self.froz_orb[irp] + i_occ
        if print_info_to_log:
            logger.debug('\n'.join(to_log))

    def _string_indices_D_ii(self, i, irp_i, irp_a, D,
                             coupled_to=None,
                             only_this_occ=None,
                             print_info_to_log=False):
        """
         Here: irrep_a == irrep_b, by symmetry
             and D[a,b] = D[b,a]

        """
        if print_info_to_log:
            to_log = []
        add_CC_term = self.wf_type == 'CCSD' and irp_a == irp_i
        Index = self._make_occ_indices_for_doubles(
            i + self.froz_orb[irp_i], i + self.froz_orb[irp_i],
            irp_i, irp_i,
            irp_a, irp_a)
        # Maybe this can be made directly:
        if (only_this_occ is not None
            and only_this_occ != OrbitalsSets(
                list(map(len, Index)))):
            return
        Index[irp_a][-1] = self.ref_orb[irp_a]
        Index[self.n_irrep + irp_a][-1] = self.ref_orb[irp_a]
        for a in range(self.virt_orb[irp_a]):
            for b in range(a + 1):
                if print_info_to_log:
                    to_log.append('a b = {} {}'.format(a, b))
                Index.C = D[a, b]
                if add_CC_term:
                    Index.C += (self.singles[irp_i][i, a]
                                * self.singles[irp_i][i, b])
                Index.C /= self.norm
                if Index.is_coupled_to(coupled_to):
                    yield Index
                if a != b:
                    Index.swap_spirreps(irp_a, self.n_irrep + irp_a)
                    if Index.is_coupled_to(coupled_to):
                        yield Index
                    Index.swap_spirreps(irp_a, self.n_irrep + irp_a)
                Index[self.n_irrep + irp_a][-1] += 1
            Index[self.n_irrep + irp_a][-1] = self.ref_orb[irp_a]
            Index[irp_a][-1] += 1
        if print_info_to_log:
            logger.debug('\n'.join(to_log))

    def _string_indices_D_ij(self,
                             i, j,
                             irrep_i, irrep_j, irrep_a, irrep_b,
                             D, D_other,
                             coupled_to=None,
                             only_this_occ=None,
                             print_info_to_log=False):
        """
         irrep_i > irrep_j only
         irrep_a > irrep_b only

        """
        if print_info_to_log:
            to_log = []
        add_CC_term_ia = (self.wf_type == 'CCSD'
                          and irrep_a == irrep_i)
        add_CC_term_ib = (self.wf_type == 'CCSD'
                          and irrep_b == irrep_i)
        # ----- sign determination:
        ij_nab_sign = (1
                       if (i + j
                           + self.ref_orb[irrep_a]
                           + self.ref_orb[irrep_b]) % 2 == 0 else
                       -1)
        if irrep_i == irrep_j:
            abab_baba_sign = abba_baab_sign = ij_nab_sign
            aaaa_bbbb_sign = -ij_nab_sign
        elif irrep_i == irrep_a:
            abab_baba_sign = aaaa_bbbb_sign = ij_nab_sign
            abba_baab_sign = -ij_nab_sign
        else:
            parity = 1 if irrep_j < irrep_b else -1
            if irrep_j < irrep_i:
                parity = -parity
            if irrep_b < irrep_i:
                parity = -parity
            if irrep_i < irrep_a:
                parity = -parity
            if irrep_j < irrep_a:
                parity = -parity
            if irrep_b < irrep_a:
                parity = -parity
            n_between = sum([self.ref_orb[irp]
                             for irp in range(self.n_irrep)
                             if (min(irrep_j,
                                     irrep_b) < irp < max(irrep_j,
                                                          irrep_b)
                                 or min(irrep_i,
                                        irrep_a) < irp < max(irrep_i,
                                                             irrep_a))])
            sum_n_sign = 1 if n_between % 2 == 0 else -1
            aaaa_bbbb_sign = -ij_nab_sign * sum_n_sign * parity
            abab_baba_sign = ij_nab_sign * sum_n_sign
            n_between = sum([self.ref_orb[irp]
                             for irp in range(self.n_irrep)
                             if (min(irrep_i,
                                     irrep_b) < irp < max(irrep_i,
                                                          irrep_b)
                                 or min(irrep_j,
                                        irrep_a) < irp < max(irrep_j,
                                                             irrep_a))])
            sum_n_sign = 1 if n_between % 2 == 0 else -1
            abba_baab_sign = ij_nab_sign * sum_n_sign
        # ----- END sign determination
        indices = self._make_occ_indices_for_doubles(
            i + self.froz_orb[irrep_i], j + self.froz_orb[irrep_j],
            irrep_i, irrep_j,
            irrep_a, irrep_b)
        if only_this_occ is not None:
            indices_occ = DoublesTypes(
                *[OrbitalsSets(occ, occ_type='F')
                  for occ in map(lambda x: list(map(len, x)), indices)])
            if only_this_occ not in indices_occ:
                return
        for a in range(self.virt_orb[irrep_a]):
            for b in range(self.virt_orb[irrep_b]):
                if irrep_a == irrep_b and b > a:
                    continue
                if print_info_to_log:
                    to_log.append('a b = {} {}'.format(a, b))
                a_virt = a + self.ref_orb[irrep_a]
                b_virt = b + self.ref_orb[irrep_b]
                C_ia_jb = D[a, b]
                if add_CC_term_ia:
                    C_ia_jb += (self.singles[irrep_i][i, a]
                                * self.singles[irrep_j][j, b])
                indices.abab.C = indices.baba.C = (
                    abab_baba_sign * C_ia_jb / self.norm)
                if (only_this_occ is None
                        or indices_occ.baba == only_this_occ):
                    indices.baba[irrep_a][-1] = a_virt
                    indices.baba[self.n_irrep + irrep_b][-1] = b_virt
                    if indices.baba.is_coupled_to(coupled_to):
                        yield indices.baba
                if (only_this_occ is None
                        or indices_occ.abab == only_this_occ):
                    indices.abab[irrep_b][-1] = b_virt
                    indices.abab[self.n_irrep + irrep_a][-1] = a_virt
                    if indices.abab.is_coupled_to(coupled_to):
                        yield indices.abab
                if irrep_a != irrep_b or a != b:
                    C_ib_ja = D_other[b, a]
                    if add_CC_term_ib:
                        C_ib_ja += (self.singles[irrep_i][i, b]
                                    * self.singles[irrep_j][j, a])
                    indices.abba.C = indices.baab.C = (
                        abba_baab_sign * C_ib_ja / self.norm)
                    if (only_this_occ is None
                            or indices_occ.abba == only_this_occ):
                        indices.abba[irrep_a][-1] = a_virt
                        indices.abba[self.n_irrep + irrep_b][-1] = b_virt
                        if indices.abba.is_coupled_to(coupled_to):
                            yield indices.abba
                    if (only_this_occ is None
                            or indices_occ.baab == only_this_occ):
                        indices.baab[irrep_b][-1] = b_virt
                        indices.baab[self.n_irrep + irrep_a][-1] = a_virt
                        if indices.baab.is_coupled_to(coupled_to):
                            yield indices.baab
                    indices.aaaa.C = indices.bbbb.C = (
                        aaaa_bbbb_sign * (C_ia_jb - C_ib_ja) / self.norm)
                    if (only_this_occ is None
                            or indices_occ.aaaa == only_this_occ):
                        indices.aaaa[irrep_a][-1] = a_virt
                        if irrep_a != irrep_b:
                            indices.aaaa[irrep_b][-1] = b_virt
                        else:
                            indices.aaaa[irrep_b][-2] = b_virt
                        if indices.aaaa.is_coupled_to(coupled_to):
                            yield indices.aaaa
                    if (only_this_occ is None
                            or indices_occ.bbbb == only_this_occ):
                        indices.bbbb[self.n_irrep + irrep_a][-1] = a_virt
                        if irrep_a != irrep_b:
                            indices.bbbb[self.n_irrep + irrep_b][-1] = b_virt
                        else:
                            indices.bbbb[self.n_irrep + irrep_b][-2] = b_virt
                        if indices.bbbb.is_coupled_to(coupled_to):
                            yield indices.bbbb
        if print_info_to_log:
            logger.debug('\n'.join(to_log))

    def _make_occ_indices_for_doubles(self,
                                      i, j,
                                      irrep_i, irrep_j,
                                      irrep_a, irrep_b):
        r"""Helper function to create indices of double excitations

        Behaviour:
        ----------

        Return indices for the following excitations:

        If i, irrep_i == j, irrep_j,
            (and thus it is assumed that irrep_a == irrep_b):
         j i    b a
               ---\-
               -/---
        -0-0-

        If (i, irrep_i) != (j, irrep_j), a namedtuple with the
        following indices:

          j      i        b     a

                        ---\- -/---      [0] = baba
        ---0-  -0----

                        -/--- ---\-      [1] = abab
        -0---  ----0-

                        ---\- -/---      [2] = abba
        -0---  ----0-

                        -/--- ---\-      [3] = baab
        ---0-  -0----

                        -/--- -/---      [4] = aaaa
        -0---  -0----

                        ---\- ---\-      [5] = bbbb
        ---0-  ----0-

        Where:
        -0---   hole in alpha orbital
        ---0-   hole in beta orbital
        -/---   electron alpha orbital
        ---\-   electron in beta orbital

        Limitations:
        ------------
        Currently only for restricted wave functions

        Parameters:
        -----------
        i,j (int)
            The indices of the occupied orbitals that will
            be excited

        irrep_i, irrep_j (int)
            The irreps of i and j

        irrep_a, irrep_b (int)
            The irreps of the virtual orbitals, where the
            electrons will be excited to

        Return:
        -------
        An instance of, or a namedtuple of, SD_StringIndex

        if (i, irrep_i) == (j, irrep_j), that is,
        the initial orbital is the same for both electrons,
        return a single SD_StringIndex, for a configuration with holes
        i == j, for both alpha and beta.

        if (i, irrep_i) != (j, irrep_j), that is,
        return the namedtuple DoublesTypes for configurations with holes in:
        alpha for i, beta for j, alpha in irrep_a, beta in irrep_b (baba)
        beta for i, alpha for j, beta in irrep_a, alpha in irrep_b (abab)
        beta for i, alpha for j, alpha in irrep_a, beta in irrep_b (abba)
        alpha for i, beta for j, beta in irrep_a, alpha in irrep_b (baab)
        alpha for i and j, alpha for irrep_a and irrep_b (aaaa)
        beta for i and j, beta for irrep_a and irrep_b (bbbb)
        """
        # ---------------------
        # The easy case
        if (irrep_i, i) == (irrep_j, j):
            Index = SD_StringIndex()
            Index.exc_type = 'D'
            for spin in ['alpha', 'beta']:
                for irrep in self.spirrep_blocks():
                    n_electrons = self.ref_orb[irrep]
                    if irrep_a != irrep_i:
                        if irrep == irrep_a:
                            n_electrons += 1
                        if irrep == irrep_i:
                            n_electrons -= 1
                    Index.append(
                        SpirrepStringIndex.make_hole(n_electrons, i)
                        if irrep == irrep_i else
                        SpirrepStringIndex(n_electrons))
            Index.set_wave_function(self)
            return Index
        # ---------------------
        # Now the complex case:
        #
        # Using order "jiba" to name variables: j<i b<a;
        # Example:
        # baba:
        # hole in j beta
        # hole in i alpha
        # virtual in irrep_b beta
        # virtual in irrep_a alpha
        Index = DoublesTypes(SD_StringIndex(),
                             SD_StringIndex(),
                             SD_StringIndex(),
                             SD_StringIndex(),
                             SD_StringIndex(),
                             SD_StringIndex())
        for Ind in Index:
            Ind.exc_type = 'D'
        # ---------------------
        # First, alpha electrons:
        for irrep in self.spirrep_blocks():
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_i:
                n_electrons -= 1
            if irrep == irrep_a:
                n_electrons += 1
            Index.baba.append(
                SpirrepStringIndex.make_hole(n_electrons, i)
                if irrep == irrep_i else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_j:
                n_electrons -= 1
            if irrep == irrep_b:
                n_electrons += 1
            Index.abab.append(
                SpirrepStringIndex.make_hole(n_electrons, j)
                if irrep == irrep_j else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_j:
                n_electrons -= 1
            if irrep == irrep_a:
                n_electrons += 1
            Index.abba.append(
                SpirrepStringIndex.make_hole(n_electrons, j)
                if irrep == irrep_j else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_i:
                n_electrons -= 1
            if irrep == irrep_b:
                n_electrons += 1
            Index.baab.append(
                SpirrepStringIndex.make_hole(n_electrons, i)
                if irrep == irrep_i else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep_a != irrep_i and irrep_a != irrep_j:
                if irrep == irrep_a or irrep == irrep_b:
                    n_electrons += 2 if irrep_i == irrep_j else 1
                if irrep == irrep_i or irrep == irrep_j:
                    n_electrons -= 2 if irrep_i == irrep_j else 1
            if irrep == irrep_i and irrep == irrep_j:
                index = SpirrepStringIndex.make_hole(n_electrons,
                                                     (j, i))
            elif irrep == irrep_i:
                index = SpirrepStringIndex.make_hole(n_electrons, i)
            elif irrep == irrep_j:
                index = SpirrepStringIndex.make_hole(n_electrons, j)
            else:
                index = SpirrepStringIndex(n_electrons)
            Index.aaaa.append(index)
            # ===============
            n_electrons = self.ref_orb[irrep]
            Index.bbbb.append(SpirrepStringIndex(n_electrons))
        # ---------------------
        # Second, the beta electrons
        for irrep in self.spirrep_blocks():
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_j:
                n_electrons -= 1
            if irrep == irrep_b:
                n_electrons += 1
            Index.baba.append(
                SpirrepStringIndex.make_hole(n_electrons, j)
                if irrep == irrep_j else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_i:
                n_electrons -= 1
            if irrep == irrep_a:
                n_electrons += 1
            Index.abab.append(
                SpirrepStringIndex.make_hole(n_electrons, i)
                if irrep == irrep_i else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_i:
                n_electrons -= 1
            if irrep == irrep_b:
                n_electrons += 1
            Index.abba.append(
                SpirrepStringIndex.make_hole(n_electrons, i)
                if irrep == irrep_i else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep == irrep_j:
                n_electrons -= 1
            if irrep == irrep_a:
                n_electrons += 1
            Index.baab.append(
                SpirrepStringIndex.make_hole(n_electrons, j)
                if irrep == irrep_j else
                SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            Index.aaaa.append(SpirrepStringIndex(n_electrons))
            # ===============
            n_electrons = self.ref_orb[irrep]
            if irrep_b != irrep_i and irrep_b != irrep_j:
                if irrep == irrep_a or irrep == irrep_b:
                    n_electrons += 2 if irrep_i == irrep_j else 1
                if irrep == irrep_i or irrep == irrep_j:
                    n_electrons -= 2 if irrep_i == irrep_j else 1
            if irrep == irrep_i and irrep == irrep_j:
                index = SpirrepStringIndex.make_hole(n_electrons,
                                                     (j, i))
            elif irrep == irrep_i:
                index = SpirrepStringIndex.make_hole(n_electrons, i)
            elif irrep == irrep_j:
                index = SpirrepStringIndex.make_hole(n_electrons, j)
            else:
                index = SpirrepStringIndex(n_electrons)
            Index.bbbb.append(index)
        # ---------------------
        for Ind in Index:
            Ind.set_wave_function(self)
        return Index

    def _string_indices_spirrep(self, spirrep,
                                coupled_to=None,
                                no_occ_orb=False,
                                only_this_occ=None,
                                print_info_to_log=False):
        if print_info_to_log:
            to_log = []
        does_yield = True
        if only_this_occ is None:
            only_this_occ = self.ref_orb
        if not isinstance(only_this_occ, (int, np.integer)):
            only_this_occ = only_this_occ[spirrep]
        nel_case = only_this_occ - self.ref_orb[spirrep]
        if only_this_occ <= 0:
            does_yield = False
        if -2 > nel_case > 2:
            does_yield = False
        if coupled_to is not None:
            if len(coupled_to) != 1:
                raise NotImplementedError(
                    'If both coupled_to and spirrep are given,'
                    + ' len(coupled_to) must be 1.'
                    + ' An extension is annoying to implement and will'
                    + ' not be used.')
            cpl_to = coupled_to[0]
            if spirrep == cpl_to.spirrep:
                raise NotImplementedError(
                    'If both coupled_to and spirrep are given,'
                    + ' the spirrep and the spirrep of the'
                    + ' coupled_to should not be the same.'
                    + ' This is annoying to implement and will'
                    + ' not be used.')
            spirrep_cpl_to_other_spin = (
                cpl_to.spirrep + self.n_irrep
                if cpl_to.spirrep < self.n_irrep else
                cpl_to.spirrep - self.n_irrep)
            nel_case_cpl_to = (len(coupled_to[0].Index)
                               - self.ref_orb[cpl_to.spirrep])
            if -2 > nel_case_cpl_to < 2:
                does_yield = False
        else:
            spirrep_cpl_to_other_spin = None
            nel_case_cpl_to = None
        if does_yield:
            if nel_case == -2:
                if self._is_spirrep_coupled_to(
                        -2, spirrep,
                        coupled_to,
                        spirrep_cpl_to_other_spin,
                        nel_case_cpl_to):
                    yield from self._string_indices_case_minus_2(spirrep)
            elif nel_case == -1:
                if (self.n_irrep >= 2
                    and self._is_spirrep_coupled_to(
                        -1, spirrep,
                        coupled_to,
                        spirrep_cpl_to_other_spin,
                        nel_case_cpl_to)):
                    yield from self._string_indices_case_minus_1(spirrep)
            elif nel_case == 0:
                n_electrons = self.ref_orb[spirrep]
                Index = SpirrepStringIndex(n_electrons)
                Index.start()
                yield Index
                if (coupled_to is None
                    or (nel_case_cpl_to == 0
                        and int(cpl_to.Index) < (
                            self.corr_orb[cpl_to.spirrep]
                            * self.virt_orb[cpl_to.spirrep]
                            + 1))):
                    Index = SpirrepStringIndex.make_hole(
                        n_electrons, (self.froz_orb[spirrep],))
                    Index.do_not_clear_std_pos()
                    Index.start()
                    Index += 1
                    for j in np.arange(self.froz_orb[spirrep],
                                       n_electrons,
                                       dtype=int_dtype):
                        for a in np.arange(self.ref_orb[spirrep],
                                           self.orb_dim[spirrep],
                                           dtype=int_dtype):
                            Index[-1] = a
                            yield Index
                            Index += 1
                        if j < n_electrons - 1:
                            Index[j] = j
                    if (coupled_to is None
                        or (nel_case_cpl_to == 0
                            and int(cpl_to.Index) == 0)):
                        last_standard_position = int(Index)
                        Index = SpirrepStringIndex.make_hole(
                            n_electrons,
                            (self.froz_orb[spirrep],
                             self.froz_orb[spirrep] + 1))
                        Index.do_not_clear_std_pos()
                        Index.set_std_pos(last_standard_position)
                        for i in np.arange(self.froz_orb[spirrep] + 1,
                                           n_electrons,
                                           dtype=int_dtype):
                            for j in np.arange(self.froz_orb[spirrep],
                                               i,
                                               dtype=int_dtype):
                                for a in np.arange(self.ref_orb[spirrep],
                                                   self.orb_dim[spirrep],
                                                   dtype=int_dtype):
                                    Index[-1] = a
                                    for b in np.arange(
                                            self.ref_orb[spirrep],
                                            a,
                                            dtype=int_dtype):
                                        Index[-2] = b
                                        yield Index
                                        Index += 1
                                if j == n_electrons - 2:
                                    break
                                if j == i - 1:
                                    Index[self.froz_orb[spirrep]:i] = \
                                        np.arange(self.froz_orb[spirrep] + 1,
                                                  i + 1)
                                else:
                                    Index[j] = j
            elif nel_case == 1:
                if (self.n_irrep > 2
                    and self._is_spirrep_coupled_to(
                        1, spirrep,
                        coupled_to,
                        spirrep_cpl_to_other_spin,
                        nel_case_cpl_to)):
                    yield from self._string_indices_case_plus_1(spirrep)
            elif nel_case == 2:
                if self._is_spirrep_coupled_to(-2, spirrep,
                                               coupled_to,
                                               spirrep_cpl_to_other_spin,
                                               nel_case_cpl_to):
                    yield from self._string_indices_case_plus_2(spirrep)
        if print_info_to_log:
            logger.debug('\n'.join(to_log))

    def _string_indices_case_minus_2(self, spirrep):
        n_electrons = self.ref_orb[spirrep] - 2
        Index = SpirrepStringIndex.make_hole(
            n_electrons,
            (self.froz_orb[spirrep],
             self.froz_orb[spirrep] + 1))
        Index.start()
        for i in np.arange(self.froz_orb[spirrep] + 1,
                           n_electrons + 2,
                           dtype=int_dtype):
            for j in np.arange(self.froz_orb[spirrep], i,
                               dtype=int_dtype):
                yield Index
                Index += 1
                if j == n_electrons:
                    break
                if j == i - 1:
                    Index[:i] = np.arange(1, i + 1)
                else:
                    Index[j] = j

    def _string_indices_case_plus_2(self, spirrep):
        n_electrons = self.ref_orb[spirrep] + 2
        Index = SpirrepStringIndex(n_electrons)
        Index.do_not_clear_std_pos()
        Index.start()
        for a in np.arange(self.ref_orb[spirrep],
                           self.orb_dim[spirrep],
                           dtype=int_dtype):
            Index[-1] = a
            for b in np.arange(self.ref_orb[spirrep],
                               a,
                               dtype=int_dtype):
                Index[-2] = b
                yield Index
                Index += 1

    def _string_indices_case_minus_1(self, spirrep):
        n_electrons = self.ref_orb[spirrep] - 1
        Index = SpirrepStringIndex.make_hole(n_electrons,
                                             (self.froz_orb[spirrep],))
        Index.do_not_clear_std_pos()
        Index.start()
        for j in np.arange(self.froz_orb[spirrep], n_electrons + 1,
                           dtype=int_dtype):
            yield Index
            Index += 1
            if j < n_electrons:
                Index[j] = j

    def _string_indices_case_plus_1(self, spirrep):
        n_electrons = self.ref_orb[spirrep] + 1
        Index = SpirrepStringIndex(n_electrons)
        Index.do_not_clear_std_pos()
        Index.start()
        for a in np.arange(self.ref_orb[spirrep] + 1,
                           self.orb_dim[spirrep] + 1,
                           dtype=int_dtype):
            yield Index
            Index[-1] = a
            Index += 1

    def _is_spirrep_coupled_to(self, this_nel_case, spirrep,
                               coupled_to,
                               spirrep_cpl_to_other_spin,
                               nel_case_cpl_to):
        """Helper to check which strings are coupled to a given one

        This is made for nel cases 1 and 2. It also has some assumptions
        regarding how coupled_to is, based on what is actually used
        by string_indices()
        """
        if coupled_to is None:
            return True
        # Assuming len(coupled_to) == 1
        # and spirrep != coupled_to[0].spirrep
        # See the NotImplementedError that are raised by
        # string_indices
        cpl_to = coupled_to[0]
        if abs(this_nel_case) == 2:
            if abs(nel_case_cpl_to) == 2:
                return (nel_case_cpl_to == -this_nel_case
                        and (spirrep // self.n_irrep
                             == cpl_to.spirrep // self.n_irrep))
            return nel_case_cpl_to == 0 and int(cpl_to.Index) == 0
        # Perhaps we can/have to consider n_irrep and see the possible
        # cases (irrep_i == irep_j, irrep_i == i_rrep_a, etc.)
        if abs(this_nel_case) == 1:
            if abs(nel_case_cpl_to) == 2:
                return False
            if abs(nel_case_cpl_to) == 0:
                return int(cpl_to.Index) == 0
            return True
        return True

    def n_strings(self, spirrep, occupation):
        """The number of strings that string_indices(spirrep=spirrep) yield

        Parameters:
        -----------

        spirrep (int)
            The spirrep

        occupation (int)
            The occupation of that spirrep
        """
        diff_elec_to_ref = occupation - self.ref_orb[spirrep]
        if diff_elec_to_ref == -2:
            return triangular(self.corr_orb[spirrep] - 1)
        if diff_elec_to_ref == -1:
            return self.corr_orb[spirrep]
        if diff_elec_to_ref == 0:
            return ((1 if self.ref_orb[spirrep] > 0 else 0)
                    + (self.corr_orb[spirrep]
                       * self.virt_orb[spirrep])
                    + (triangular(self.virt_orb[spirrep] - 1)
                       * triangular(self.corr_orb[spirrep] - 1)))
        if diff_elec_to_ref == 1:
            return self.virt_orb[spirrep]
        if diff_elec_to_ref == 2:
            return triangular(self.virt_orb[spirrep] - 1)
        return 0

    def update_amplitudes(self, z, mode='continuous'):
        """Update the amplitudes by z


        """
        pos = 0
        if self.singles is not None:
            for spirrep in self.spirrep_blocks():
                self.singles[spirrep] += np.reshape(
                    z[pos:
                      pos + self.corr_orb[spirrep] * self.virt_orb[spirrep]],
                    (self.corr_orb[spirrep], self.virt_orb[spirrep]))
                pos += self.corr_orb[spirrep] * self.virt_orb[spirrep]
        N_ij = 0
        for exc_type in (['aa']
                         if self.restricted else
                         ['aa', 'bb', 'ab']):
            for i_irrep in range(self.n_irrep):
                i_spirrep = i_irrep + (self.n_irrep
                                       if exc_type[0] == 'b' else 0)
                for i in range(self.corr_orb[i_spirrep]):
                    for j_irrep in range(self.n_irrep):
                        j_spirrep = j_irrep + (self.n_irrep
                                               if exc_type[1] == 'b' else 0)
                        for j in range(self.corr_orb[j_spirrep]):
                            if self.restricted or exc_type[0] == exc_type[1]:
                                if i_irrep < j_irrep:
                                    continue
                                elif i_irrep == j_irrep:
                                    if i < j:
                                        continue
                                    elif i == j and not self.restricted:
                                        continue
                            for a_spirrep in range(self.n_irrep):
                                b_spirrep = irrep_product[
                                    a_spirrep, irrep_product[i_irrep,
                                                             j_irrep]]
                                if exc_type[0] == 'b':
                                    a_spirrep += self.n_irrep
                                if exc_type[1] == 'b':
                                    b_spirrep += self.n_irrep
                                if a_spirrep <= b_spirrep:
                                    self.doubles[N_ij][a_spirrep % self.n_irrep] += np.reshape(
                                        z[pos:
                                          pos
                                          + self.virt_orb[a_spirrep]
                                          * self.virt_orb[b_spirrep]],
                                        (self.virt_orb[a_spirrep],
                                         self.virt_orb[b_spirrep]))
                                    if a_spirrep == b_spirrep:
                                        for a in range(self.virt_orb[a_spirrep]):
                                            for b in range(a):
                                                self.doubles[N_ij][a_spirrep
                                                                   % self.n_irrep][a, b] = \
                                                    self.doubles[N_ij][a_spirrep
                                                                       % self.n_irrep][b, a]
                                else:
                                    self.doubles[N_ij][b_spirrep] += np.reshape(
                                        z[pos:
                                          pos
                                          + self.virt_orb[a_spirrep]
                                          * self.virt_orb[b_spirrep]],
                                        (self.virt_orb[a_spirrep],
                                         self.virt_orb[b_spirrep])).T
                                pos += self.virt_orb[a_spirrep] * self.virt_orb[b_spirrep]
                            N_ij += 1

    def calc_wf_from_z(self):
        raise NotImplementedError('Not implemented yet: calc_wf_from_z')

    def change_orb_basis(self):
        raise NotImplementedError('Not implemented yet: change_orb_basis')

    def make_Jac_Hess_overlap(self):
        raise NotImplementedError('Not implemented yet: make_Jac_Hess')

    @classmethod
    def similar_to(cls, wf, wf_type, restricted):
        new_wf = super().similar_to(wf, restricted=restricted)
        new_wf.wf_type = wf_type
        new_wf.initialize_SD_lists(
            with_singles=('SD' in new_wf.wf_type),
            with_BCC_orb_gen=new_wf.wf_type == 'BCCD')
        return new_wf

    @classmethod
    def from_zero_amplitudes(cls, point_group,
                             ref_orb, orb_dim, froz_orb,
                             level='SD', wf_type='CC'):
        """Construct a new wave function with all amplitudes set to zero

        Parameters:
        -----------
        ref_orb (OrbitalsSets)
            The reference occupation

        orb_dim (OrbitalsSets)
            The dimension of orbital spaces

        froz_orb (OrbitalsSets)
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
        new_wf.initialize_SD_lists(
            with_singles='SD' in new_wf.wf_type,
            with_BCC_orb_gen=new_wf.wf_type == 'BCCD')
        return new_wf
    
    @classmethod
    def from_Molpro(cls, molpro_output,
                    start_line_number=1,
                    wf_type=None,
                    point_group=None,
                    zero_tol=1.0E-10):
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
                if new_wf.singles is None:
                    new_wf.initialize_SD_lists(
                        with_singles=new_wf.wf_type in ('CISD',
                                                        'CCSD'),
                        with_BCC_orb_gen=new_wf.wf_type == 'BCCD')
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
                    new_wf.initialize_SD_lists(
                        with_singles=False,
                        with_BCC_orb_gen=new_wf.wf_type == 'BCCD')
                if new_wf.restricted or 'Alpha-Alpha' in line:
                    exc_type = 'aa'
                elif 'Beta-Beta' in line:
                    exc_type = 'bb'
                elif 'Alpha-Beta' in line:
                    exc_type = 'ab'
            elif dbl_found:
                lspl = line.split()
                if len(lspl) == 7:
                    try:
                        (Molpros_i, Molpros_j,
                         irrep_a, irrep_b,
                         a, b) = map(
                             lambda x: int(x) - 1, lspl[0:-1])
                    except ValueError:
                        continue
                    C = float(lspl[-1])
                    if exc_type[0] == 'a':
                        a -= new_wf.act_orb[irrep_a]
                    if exc_type[1] == 'a':
                        b -= new_wf.act_orb[irrep_b]
                    if a < 0 or b < 0:
                        if abs(C) > zero_tol:
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
                        if abs(C) > zero_tol:
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
