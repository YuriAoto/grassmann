# cython: profile=False
"""An wave function in the intermediate normalisation

Classes:
--------

IntermNormWaveFunction
"""
import logging

from libc.math cimport sqrt

import numpy as np
from scipy.spatial import distance

from util.array_indices cimport (triangular,
                                 n_from_triang_with_diag,
                                 n_from_triang,
                                 n_from_rect)
from util.variables import int_dtype, zero
from util.memory import mem_of_floats
from util.other import int_array
from input_output import molpro
from molecular_geometry.symmetry import irrep_product
from wave_functions.general cimport WaveFunction
from wave_functions.general import WaveFunction
from wave_functions.slater_det cimport SlaterDet
from wave_functions.slater_det import SlaterDet
import wave_functions.strings_rev_lexical_order as str_order
from coupled_cluster.cluster_decomposition cimport ClusterDecomposition
from coupled_cluster.manifold cimport update_indep_amplitudes
from orbitals.occ_orbitals cimport OccOrbital
from orbitals.occ_orbitals import OccOrbital
from orbitals.orbital_space cimport FullOrbitalSpace
from orbitals.orbital_space import FullOrbitalSpace
from coupled_cluster.excitation cimport SDExcitation
from coupled_cluster.excitation import SDExcitation

cdef ClusterDecomposition cluster_dec = ClusterDecomposition()

cdef void _translate(int [:] X, int ini, int end):
    """X[..., ini, ini+1, ..., end-1, end, ...] -> X[..., ini+1, ..., end-1, end, end, ...]
    """
    # In numpy:
    if ini <= end:
        X[ini:end] = X[ini+1:end+1]
    else:
        X[end+1:ini+1] = X[end:ini]
    # In Cython:
    # cdef int ii
    # if ini <= end:
    #     for ii in range(ini, end):
    #         X[ii] = X[ii + 1]
    # else:
    #     for ii in range(ini - end):
    #         alpha_occ[ini-ii] = alpha_occ[end - (ii+1)]


cdef (int, int) _set_virtpos_for_proj_aa_bb(int old_virt_pos_a,
                                            int old_virt_pos_b,
                                            OccOrbital i,
                                            OccOrbital j,
                                            int a_spirrep,
                                            int b_spirrep,
                                            int [:] occ,
                                            FullOrbitalSpace orbsp):
    """Helper for from_projected_fci: set position of virtual orbitals
    
    Obtains the position of virtual orbitals in the excited determinant,
    in occ, posibly shifing some positions of the occupied orbitals
    in occ to have the orbitals in ascending order
    
    Parameters:
    -----------
    old_virt_pos_a, old_virt_pos_b
        The position of virtual orbitals previously
    
    i, j
        The holes, namely, occupied orbitals that will be excited
    
    a_spirrep, b_spirrep
        Spirrep of virtual orbitals (where the excitation will end)
    
    occ
        The occupied orbitals
    
    Return:
    -------
    The 2-tuple with the calculated position of where the virtual orbitals
    should be
    
    Side Effect
    -----------
    Some entries of occupied orbitals are shifted to have the orbitals
    in ascending order
    
    """
    cdef int new_virt_pos_a = orbsp.corr_orbs_before[a_spirrep] + orbsp.corr[a_spirrep]
    cdef int new_virt_pos_b = orbsp.corr_orbs_before[b_spirrep] + orbsp.corr[b_spirrep]
    new_virt_pos_b += 1
    if j.spirrep <= a_spirrep:
        new_virt_pos_a -= 2
        new_virt_pos_b -= 2
    elif i.spirrep <= a_spirrep:
        new_virt_pos_a -= 1
        new_virt_pos_b -= 1
        if j.spirrep <= b_spirrep:
            new_virt_pos_b -= 1
    elif i.spirrep <= b_spirrep:
        new_virt_pos_b -= 1 
        if j.spirrep <= b_spirrep:
            new_virt_pos_b -= 1 
    if old_virt_pos_a < new_virt_pos_a:
        _translate(occ, old_virt_pos_b, new_virt_pos_b)
        _translate(occ, old_virt_pos_a, new_virt_pos_a)
    else:
        _translate(occ, old_virt_pos_a, new_virt_pos_a)
        _translate(occ, old_virt_pos_b, new_virt_pos_b)
    return new_virt_pos_a, new_virt_pos_b


def singles_contr_from_clusters_fci(alpha_hp, beta_hp, fci_wf):
    """Return the contribution singles as cluster decomposition of doubles
    
    Parameters:
    -----------
    alpha_hp (list of arrays)
        alpha holes and particles
    
    beta_hp (list of arrays)
        beta holes and particles
    
    fci_wf (FCIWaveFunction)
        The FCI wave function that from which the contribution of singles
        will come.
    
    TODO:
    -----
    This is essentially similar to fci.contribution_from_clusters, perhaps
    we could join them. The problem is that here the contribution comes
    from a fci wave function, whereas there it comes from a IntermNormWaveFunction
    If the access is unified, this can be joined.
    """
    decomposition = cluster_dec.decompose(alpha_hp, beta_hp, mode='SD')
    C = 0.0
    for d in decomposition[1:]:  # skipping first contribution: it is the double.
        new_contribution = d[0]
        add_contr = True
        for cluster_exc in d[1:]:
            if not fci_wf.symmetry_allowed_exc(cluster_exc):
                add_contr = False
                break
            new_contribution *= fci_wf[fci_wf.index(SlaterDet.from_excitation(
                fci_wf.ref_det, 0.0, cluster_exc))]
        if add_contr:
            C += new_contribution
    return C


cdef double [:] _transpose(double [:] X, int nrow, int ncol):
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


cdef class IntermNormWaveFunction(WaveFunction):
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
        
        # For singles, t_i^a:
        
        These are stored in groups of spirrep.
        Recall that irrep(i) = irrep(a) for a nonzero amplitude.
        There are, thus:
        
            self.orbspace.corr[spirrep]*self.orbspace.virt[spirrep]
        
        amplitudes in each spirrep block.
        
        * If restricted:
        first all amplitudes of {i, a} in irrep = 0,
        then all amplitudes of {i, a} in irrep = 1, etc.
        In each of these blocks, virtual orbital indices run faster:
        
        t_0^0, t_0^1, t_0^2, ..., t_0^{m-1}, t_1^0, ..., t_1^{m-1}, ...
        
        * If unrestricted:
        similar to unrestricted, but all alpha first, then all beta.
        
        
        To help finding the amplitude, the attribute ini_blocks_S is provided,
        see its documentation
        
        
        # For doubles:
        
        the amplitudes are stored in groups of {i, j} pairs and,
        in each of these pairs, in groups of irrep of a.
        Recall that irrep(i)*irrep(j) = irrep(a)*irrep(b) for a
        nonzero amplitude. Thus, for each pair {i, j}, and each
        irrep of a, the irrep of b is automatically determined.
        
        
        * If restricted:
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
        virt[irrep_a]*virt[irrep_b]
        of such elements, and remember that irrep_b is determined
        by the other irreps. b indices run faster:
        
        t_{ij}^{00}, t_{ij}^{01}, ...,
            t_{ij}^{0{virt[irrep_b]-1}}, t_{ij}^{10}, ...
            t_{ij}^{1{virt[irrep_b]-1}}, ...
                t_{ij}^{{virt[irrep_a]-1}{virt[irrep_b]-1}}
                     
        
        * If unrestricted:
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
        
        
        ExcType.AA (alpha-alpha):
        0,1                    0
        0,2  1,2               1   2
        0,3  1,3  2,3          3   4   5
        0,4  1,4  2,4  3,4     6   7   8   9
        
        ExcType.BB (beta-beta):
        0,1                10
        0,2  1,2           11  12
        0,3  1,3  2,3      13  14  15
        
        ExcType.AB (alpha-beta):
        0,0  1,0  2,0  3,0  4,0       16  17  18  19  20
        0,1  1,1  2,1  3,1  4,1       21  22  23  24  25
        0,2  1,2  2,2  3,2  4,2       26  27  28  29  30
        0,3  1,3  2,3  3,3  4,3       31  32  33  34  35
        
        
        Then, for each of these pairs, the situation is similar to
        in the restricted case: there are blocks for the irrep of a
        and for the virt[irrep_a]*virt[irrep_b] amplitudes.
        In this case, however, t_{ij}{ab} = -t_{ij}{ba}. We opted
        to store both amplitudes, and thus the last subblock is
        antisymmetric if irrep_a == irrep_b, or it is minus the
        transpose of another subblock.
        
        To help finding the amplitude, the attributes ini_blocks_D,
        first_bb_pair, and first_ab_pair are provided, see its
        documentation.
        
        Reading the code of __str__ is a good way to understand this
        order convention.
    
    ini_blocks_S (1D array of int)
        The position of the first amplitude of singles, for each
        block of spirrep. For example, setting:
        
        inibl = self.ini_blocks_S[irrep]
        n_corr = self.orbspace.corr[irrep]
        n_virt = self.orbspace.virt[irrep]
        
        then
        
        self.amplitudes[inibl:inibl + n_corr*n_virt]
        
        gives the whole block of amplitudes associated to irrep
    
    ini_blocks_D (2D array of int)
        The position of the first amplitude of doubles, for each
        block of spirrep, for each pair of occupied orbitals.
        For example, if the position of the pair of occupied orbitals
        is ij, and the product of their irreps is irrep_ij, then setting:
        
        inibl = self.ini_blocks_D[ij, irrep_a]
        n_virt_a = self.orbspace.virt[irrep_a]
        n_virt_b = self.orbspace.virt[irrep_product[irrep_a, irrep_ij]]
        
        self.amplitudes[inibl:inibl + n_virt_a*n_virt_b]
        
        gives the whole block of amplitudes associated to this pair of
        occupied orbitals, with the virtual belonging to irrep_a
        (and the irrep of the second virtual, b, is automatically defined
         by symmetry)
    
    first_bb_pair, first_ab_pair (int)
        The index of first beta-beta or alpha-alpha pair. In the example
        of ini_blocks_D, if ij is first_bb_pair or first_ab_pair, then
        it represents the first pair of beta-beta or alpha-beta orbitals.
        evidently 0 is the first pair of alpha-alpha orbitals.
        
    
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
            key = (i, j, a, b, irrep_i, irrep_j, irrep_a, exc_type)
       
        for the last two cases, i, j, a, b are indices local to the
        correlated or virtual orbitals; irrep_i, irrep_j, and irrep_a
        are irreducible representation and exc_type the excitation type
        
        Return:
        -------
        The amplitude t_i^a or t_{ij}^{ab}
        

        Examples:
        ---------
        for:
            restricted   True
            full     [9, 5, 5, 2]
            froz     [1, 0, 0, 0]
            ref      [4, 2, 2, 0]
            corr     [3, 2, 2, 0]
            virt     [5, 3, 3, 2]
        
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
            full     [9, 5, 5, 2]
            froz     [1, 0, 0, 0]
            ref      [4, 2, 2, 0, 3, 2, 2, 0]
            corr     [3, 2, 2, 0, 2, 2, 2, 0]
            virt     [5, 3, 3, 2, 6, 3, 3, 2]
        
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
        self._n_indep_ampl = -1
        self._norm = -1.0
        self.use_CISD_norm = True
        self.first_bb_pair = 0
        self.first_ab_pair = 0
        self._n_ampl = 0
        self.set_eq_tol()
    
    def __repr__(self):
        """Return representation of the wave function."""
        return ('<Wave Function in Intermediate normalisation>\n'
                + super().__repr__())
    
    def __str__(self):
        """Return string version the wave function."""
        cdef int irrep, spirrep, ia_bl, ia, ij
        cdef OccOrbital i, j
        def get_all_doubl_ij(int ij, int irrep_ij, int exc_type):
            """The doubles for pair ij, whose irrep product is irrep_ij
            
            """
            cdef int a_beta_shift, irrep_a, irrep_b, n_ampl, ini_bl, ab
            a_beta_shift = self.n_irrep if exc_type == ExcType.BB else 0
            b_beta_shift = 0 if exc_type == ExcType.AA else self.n_irrep
            xx = []
            for irrep_a in range(self.n_irrep):
                irrep_b = irrep_product[irrep_a, irrep_ij]
                n_ampl = (self.orbspace.virt[irrep_a + a_beta_shift]
                          * self.orbspace.virt[irrep_b + b_beta_shift])
                if n_ampl == 0:
                    continue
                ini_bl = self.ini_blocks_D[ij, irrep_a]
                xx.append(f'a irrep = {irrep_a} ( => b irrep = {irrep_b})')
                for ab in range(n_ampl):
                    xx.append(f'{self.amplitudes[ini_bl + ab]}')
            return xx
            
        x = [super().__repr__()]
        if self.has_singles:
            x.append('Singles, t_i^a (a runs faster within each irrep)')
            if not self.restricted:
                x.append('alpha -> alpha')
            for irrep in range(self.n_irrep):
                x.append(f'irrep = {irrep}')
                ia_bl = self.ini_blocks_S[irrep]
                for ia in range(self.orbspace.corr[irrep] * self.orbspace.virt[irrep]):
                    x.append(f'{self.amplitudes[ia_bl + ia]}')
            if not self.restricted:
                x.append('beta -> beta')
                for irrep in range(self.n_irrep):
                    spirrep = irrep + self.n_irrep
                    x.append(f'irrep = {irrep}')
                    ia_bl = self.ini_blocks_S[spirrep]
                    for ia in range(self.orbspace.corr[spirrep]
                                    * self.orbspace.virt[spirrep]):
                        x.append(f'{self.amplitudes[ia_bl + ia]}')
        x.append('Doubles, t_{ij}^{ab} (b runs faster for each pair ij)')
        if self.restricted:
            ij = 0
            for i, j in self.occupied_pairs(ExcType.ALL):
                all_dbl = get_all_doubl_ij(ij,
                                           irrep_product[i.spirrep,
                                                         j.spirrep],
                                           ExcType.AA)
                if all_dbl:
                    x.append(f'Pair ij = {ij}: '
                             f' i=({i.orb}, irrep={i.spirrep})'
                             f' j=({j.orb}, irrep={j.spirrep})')
                    x.extend(all_dbl)
                ij += 1
        else:
            ij = 0
            x.append('alpha, alpha -> alpha, alpha')
            for i, j in self.occupied_pairs(ExcType.AA):
                all_dbl = get_all_doubl_ij(ij,
                                           irrep_product[i.spirrep,
                                                         j.spirrep],
                                           ExcType.AA)
                if all_dbl:
                    x.append(f'Pair ij = {ij}: '
                             f' i=({i.orb}, irrep={i.spirrep})'
                             f' j=({j.orb}, irrep={j.spirrep})')
                    x.extend(all_dbl)
                ij += 1
            x.append('beta, beta -> beta, beta')
            for i, j in self.occupied_pairs(ExcType.BB):
                all_dbl = get_all_doubl_ij(ij,
                                           irrep_product[i.spirrep
                                                         - self.n_irrep,
                                                         j.spirrep
                                                         - self.n_irrep],
                                           ExcType.BB)
                if all_dbl:
                    x.append(f'Pair ij = {ij}: '
                             + f' i=({i.orb}, irrep={i.spirrep-self.n_irrep})'
                             + f' j=({j.orb}, irrep={j.spirrep-self.n_irrep})')
                    x.extend(all_dbl)
                ij += 1
            x.append('alpha, beta -> alpha, beta')
            for i, j in self.occupied_pairs(ExcType.AB):
                all_dbl = get_all_doubl_ij(ij,
                                           irrep_product[i.spirrep,
                                                         j.spirrep
                                                         - self.n_irrep],
                                           ExcType.AB)
                if all_dbl:
                    x.append(f'Pair ij = {ij}: '
                             + f' i=({i.orb}, irrep={i.spirrep})'
                             + f' j=({j.orb}, irrep={j.spirrep-self.n_irrep})')
                    x.extend(all_dbl)
                ij += 1
        return '\n'.join(x)
    
    def __len__(self):
        """Return the number of amplitudes amplitudes (stored or to be stored)
        
        Return:
        -------
        A integer
        """
        ### fix!
        if not self.orbspace.n_irrep:
            return 0
        if not self._n_ampl:
            self._calc_ini_blocks()
        return self._n_ampl
    
    def __array__(self):
        return np.array(self.amplitudes)

    def __imul__(self, factor):
        cdef int i
        for i in range(len(self)):
            self.amplitudes[i] *= factor
        return self

    def __div__(self, factor):
        cdef int i
        for i in range(len(self)):
            self.amplitudes[i] /= factor
        return self

    def dist_to(self, IntermNormWaveFunction other):
        """The distance, in the amplitudes spaces, to another wave function
        
        
        
        Parameters:
        -----------
        other (IntermNormWaveFunction)
            The other wave function to which the distance is to be calculated
        
        Return:
        -------
        A float, the (Euclidean) distance between the amplitudes of self
        and other
        
        Raise:
        ------
        ValueError, if other is not compatible with self to calculate
        the distance.
        
        """
        if (self.orbspace != other.orbspace
            or self.n_alpha != other.n_alpha
                or self.n_beta != other.n_beta):
            raise ValueError('The other wave function is not compatible!')
        if len(self) != len(other):
            raise NotImplementedError(
                'Wave functions have amplitude vectors of different sizes.'
                ' we cannot calculate their distance.')
        return distance.euclidean(self.amplitudes, other.amplitudes)

    def dist_to_ref(self):
        """Distance to the reference Slater determinant
        
        The distance is measured in the metric of the intermediate
        normalisation.
        
        Return:
        -------
        The distance, as a float.
        
        """
        cdef int i
        cdef double S = 0.0, D = 0.0
        if self.wf_type == 'CCSD' or self.wf_type == 'CCSD':
            raise ValueError('Distance to reference for CC wave function must be'
                             'through FCIWaveFunction.')
        if self.restricted:
            raise NotImplementedError("Distance to reference only for"
                                      " unrestricted IntermNormWaveFunction.")
        else:
            for i in range(self.ini_blocks_D[0, 0]):
                S += self.amplitudes[i]**2
            for i in range(self.ini_blocks_D[0, 0],
                           self.ini_blocks_D[self.first_ab_pair, 0]):
                D += self.amplitudes[i]**2
            S += D/2
            for i in range(self.ini_blocks_D[self.first_ab_pair, 0],
                           self._n_ampl):
                S += self.amplitudes[i]**2
        return sqrt(S)
    
    @property
    def n_indep_ampl(self):
        """Number of independent amplitudes
        
        For alpha/alpha and beta/beta, t_ij^ab = -t_ji^ba and thus they are
        not independentcount. This property gives the total number of
        independent amplitudes.
        
        For restricted, this is just the number of entries in amplitudes,
        that equals the len.
        
        For unrestricted it is smaller (or equal) than len.
        
        """
        n_ampl = len(self)
        if self._n_indep_ampl < 0:
            if self.restricted:
                self._n_indep_ampl = n_ampl
                logger.debug('In n_indep_ampl, starting with %d', self._n_indep_ampl)
            else:
                self._n_indep_ampl = self.ini_blocks_D[0, 0]
                logger.debug('In n_indep_ampl, starting with %d', self._n_indep_ampl)
                for exc_type in [ExcType.AA, ExcType.BB]:
                    n_irrep_or_0 = 0 if exc_type == ExcType.A else self.n_irrep
                    for i, j in self.occupied_pairs(exc_type):
                        for a_irrep in range(self.n_irrep):
                            b_irrep = irrep_product[
                                irrep_product[i.spirrep - n_irrep_or_0,
                                              j.spirrep - n_irrep_or_0], a_irrep]
                            a_spirrep = a_irrep + n_irrep_or_0,
                            b_spirrep = b_irrep + n_irrep_or_0
                            if a_irrep < b_irrep:
                                self._n_indep_ampl += (self.orbspace.virt[a_spirrep]
                                                       * self.orbspace.virt[b_spirrep])
                            elif a_irrep == b_irrep:
                                self._n_indep_ampl += triangular(self.orbspace.virt[a_spirrep] - 1)
                self._n_indep_ampl += n_ampl - self.ini_blocks_D[self.first_ab_pair, 0]
        return self._n_indep_ampl
    
    def __eq__(self, IntermNormWaveFunction other):
        """Check if amplitudes are almolst equal to amplitudes of other."""
        return np.allclose(self.amplitudes, other.amplitudes,
                           rtol=self.rtol,
                           atol=self.atol)

    def set_eq_tol(self, atol=1.0E-8, rtol=1.0E-5):
        self.atol = atol
        self.rtol = rtol

    @property
    def has_singles(self):
        return self.ini_blocks_D[0,0] != 0

    def occupied_pairs(self, exc_type=ExcType.ALL):
        """Generator for all pairs i,j
        
        """
        cdef OccOrbital i, j
        if self.restricted:
            i = OccOrbital(self.orbspace, True)
            j = OccOrbital(self.orbspace, True)
            while j.alive:
                yield i, j
                if i.pos_in_occ == j.pos_in_occ:
                    j.next_()
                    i.rewind()
                else:
                    i.next_()
        else:
            if exc_type in (ExcType.ALL, ExcType.AA):
                i = OccOrbital(self.orbspace, True)
                j = OccOrbital(self.orbspace, True)
                j.next_()
                while j.alive:
                    yield i, j
                    if i.pos_in_occ == j.pos_in_occ - 1:
                        j.next_()
                        i.rewind()
                    else:
                        i.next_()
            if exc_type in (ExcType.ALL, ExcType.BB):
                i = OccOrbital(self.orbspace, False)
                j = OccOrbital(self.orbspace, False)
                j.next_()
                while j.alive:
                    yield i, j
                    if i.pos_in_occ == j.pos_in_occ - 1:
                        j.next_()
                        i.rewind()
                    else:
                        i.next_()
            if exc_type in (ExcType.ALL, ExcType.AB):
                i = OccOrbital(self.orbspace, True)
                j = OccOrbital(self.orbspace, False)
                while j.alive:  # alpha, beta
                    yield i, j
                    i.next_()
                    if not i.alive:
                        j.next_()
                        i.rewind()

    cdef int _add_block_for_calc_ini_blocks(self,
                                            int i_in_D,
                                            int spirrep_a,
                                            int spirrep_b,
                                            int [:] raveled_ini_blocks_D):
        cdef int n
        n = (raveled_ini_blocks_D[i_in_D]
             + self.orbspace.virt[spirrep_a] * self.orbspace.virt[spirrep_b])
        i_in_D += 1
        if i_in_D < raveled_ini_blocks_D.size:
            raveled_ini_blocks_D[i_in_D] = n
        else:
            self._n_ampl = n
        return i_in_D

    cdef void _calc_ini_blocks(self):
        """Calculate the arrays with the initial index of amplitude blocks"""
        cdef int spirrep, irrep_a, i_in_D
        cdef int n_irrep = self.get_n_irrep()
        cdef int [:] raveled_ini_blocks_D
        cdef OccOrbital i, j
        n_occ_alpha = (self.orbspace.corr_orbs_before[n_irrep - 1]
                       + self.orbspace.corr[n_irrep - 1])
        n_occ_beta = self.orbspace.corr_orbs_before[2*n_irrep]
        if self.restricted:
            self.ini_blocks_S = np.zeros(n_irrep+1, dtype=int_dtype)
            raveled_ini_blocks_D = np.zeros(triangular(n_occ_alpha)
                                            * n_irrep,
                                            dtype=int_dtype)
        else:
            self.ini_blocks_S = np.zeros(2*n_irrep+1, dtype=int_dtype)
            raveled_ini_blocks_D = np.zeros(
                (triangular(n_occ_alpha - 1)
                 + triangular(n_occ_beta - 1)
                 + n_occ_alpha * n_occ_beta) * n_irrep,
                dtype=int_dtype)
        if "SD" in self.wf_type:
            self.ini_blocks_S[0] = 0
            for spirrep in self.spirrep_blocks():
                new_index = (self.ini_blocks_S[spirrep]
                             + self.orbspace.corr[spirrep] * self.orbspace.virt[spirrep])
                if spirrep == (n_irrep
                               if self.restricted else
                               (2*n_irrep)) - 1:
                    raveled_ini_blocks_D[0] = new_index
                else:
                    self.ini_blocks_S[spirrep + 1] = new_index
            self.ini_blocks_S[n_irrep
                              if self.restricted else
                              (2*n_irrep)] = raveled_ini_blocks_D[0]
        else:
            raveled_ini_blocks_D[0] = 0
        i_in_D = 0
        if self.restricted:
            for i, j in self.occupied_pairs():
                for irrep_a in range(n_irrep):
                    i_in_D = self._add_block_for_calc_ini_blocks(
                        i_in_D,
                        irrep_a,
                        irrep_product[
                            irrep_a, irrep_product[i.spirrep,
                                                   j.spirrep]],
                        raveled_ini_blocks_D)
        else:
            self.first_bb_pair = 0
            for i, j in self.occupied_pairs(ExcType.AA):
                self.first_bb_pair += 1
                for irrep_a in range(n_irrep):
                    i_in_D = self._add_block_for_calc_ini_blocks(
                        i_in_D,
                        irrep_a,
                        irrep_product[irrep_a,
                                      irrep_product[i.spirrep,
                                                    j.spirrep]],
                        raveled_ini_blocks_D)
            self.first_ab_pair = self.first_bb_pair
            for i, j in self.occupied_pairs(ExcType.BB):
                self.first_ab_pair += 1
                for irrep_a in range(n_irrep):
                    i_in_D = self._add_block_for_calc_ini_blocks(
                        i_in_D,
                        irrep_a + n_irrep,
                        irrep_product[irrep_a,
                                      irrep_product[i.spirrep - n_irrep,
                                                    j.spirrep - n_irrep]]
                        + n_irrep,
                        raveled_ini_blocks_D)
            for i, j in self.occupied_pairs(ExcType.AB):
                for irrep_a in range(n_irrep):
                    i_in_D = self._add_block_for_calc_ini_blocks(
                        i_in_D,
                        irrep_a,
                        irrep_product[irrep_a,
                                      irrep_product[i.spirrep,
                                                    j.spirrep - n_irrep]]
                        + n_irrep,
                        raveled_ini_blocks_D)
        self.ini_blocks_D = np.reshape(raveled_ini_blocks_D,
                                       (raveled_ini_blocks_D.size // n_irrep,
                                        n_irrep))
        if i_in_D != self.ini_blocks_D.size:
            raise Exception(f'BUG: {i_in_D} != {self.ini_blocks_D.size}')
    
    def __setitem__(self, key, value):
        """Set the amplitude associated to that excitation. See class doc."""
        if isinstance(key, int):
            self.amplitudes[key] = value
        if isinstance(key, SDExcitation):
            pos = self.pos_in_ampl_for_hp(key)
            if isinstance(pos, tuple):
                if self.restricted:
                    raise KeyError(
                        'Do not use this function for'
                        ' setting AA or BB for restricted')
                self.amplitudes[pos[0]] = value
                self.amplitudes[pos[1]] = -value
            else:
                self.amplitudes[pos] = value
        else:
            self.amplitudes[self.pos_in_ampl(key)] = value
        
    def __getitem__(self, key):
        """Get the amplitude associated to that excitation. See class doc."""
        if isinstance(key, int):
            return self.amplitudes[key]
        if isinstance(key, SDExcitation):
            pos = self.pos_in_ampl_for_hp(key)
            if isinstance(pos, tuple):
                if self.restricted:
                    return self.amplitudes[pos[0]] - self.amplitudes[pos[1]]
                else:
                    return self.amplitudes[pos[0]]
            else:
                return self.amplitudes[pos]
        return self.amplitudes[self.pos_in_ampl(key)]
    
    def pos_in_ampl(self, key):
        """The position of that key in amplitudes vector"""
        if len(key) == 4:
            i, a, irrep, exc_type = key
            if exc_type == ExcType.B and not self.restricted:
                irrep += self.n_irrep
            return (self.ini_blocks_S[irrep]
                    + n_from_rect(
                        i, a, self.orbspace.virt[irrep]))
        if len(key) == 8:
            i, j, a, b, irrep_i, irrep_j, irrep_a, exc_type = key
            i = self.orbspace.get_abs_corr_index(i, irrep_i,
                                                 exc_type in (ExcType.AA,
                                                              ExcType.AB))
            j = self.orbspace.get_abs_corr_index(j, irrep_j,
                                                 exc_type == ExcType.AA)
            ij = self.get_ij_pos_from_i_j(i, j, irrep_i, exc_type)
            return (self.ini_blocks_D[ij, irrep_a]
                    + n_from_rect(
                        a, b, self.orbspace.virt[
                            irrep_product[irrep_a,
                                          irrep_product[irrep_i,
                                                        irrep_j]]]))
        raise KeyError('Key must have 4 (S) or 8 (D) entries!')
    
    cdef inline int get_ij_pos_from_i_j(self,
                                        int i,
                                        int j,
                                        int irrep_i,
                                        ExcType exc_type) except -1:
        """Get the position of pair i,j as stored in self.ini_blocks_D
        
        Parameters:
        -----------
        i, j (int)
            The absolute position (within all occupied orbitals of same spin)
            of occupied orbitals i and j.
        
        irrep_i (int)
            The irreducible representation of orbital i
        
        exc_type (ExcType)
            The excitation type
        
        Return:
        -------
        An integer, with the desired position
            
        """
        if self.restricted:
            return n_from_triang_with_diag(i, j)
        if exc_type == ExcType.AA:
            return n_from_triang(i, j)
        if exc_type == ExcType.BB:
            return self.first_bb_pair + n_from_triang(i, j)
        if exc_type == ExcType.AB:
            return self.first_ab_pair + n_from_rect(
                j, i,
                self.orbspace.corr_orbs_before[self.n_irrep - 1]
                + self.orbspace.corr[self.n_irrep - 1])
        raise ValueError('Excitation type must be AA BB or AB')

    def pos_in_ampl_for_hp(self, SDExcitation exc):
        """Get the position of amplitude(s) associated to the excitation"""
        if exc.rank() == 1:
            return self.pos_in_ampl(self.indices_of_singles(exc))
        elif exc.rank() == 2: # just "else:"?
            (i, j, a, b,
             irrep_i, irrep_j, irrep_a,
             exc_type) = self.indices_of_doubles(exc)
            if exc_type in (ExcType.AA, ExcType.BB):
                return (
                    self.pos_in_ampl((i, j, a, b,
                                      irrep_i, irrep_j,
                                      irrep_a,
                                      exc_type)),
                    self.pos_in_ampl((i, j, b, a,
                                      irrep_i, irrep_j,
                                      irrep_product[irrep_a,
                                                    irrep_product[irrep_i,
                                                                  irrep_j]],
                                      exc_type)))
            else:
                return self.pos_in_ampl(
                    (i, j, a, b,
                     irrep_i, irrep_j,
                     irrep_a,
                     exc_type))
        else:
            raise IndexError('Rank must be 1 or 2!')
    
    def indices_of_singles(self, SDExcitation exc):
        """Return i, a, irrep, exc_type of single excitation"""
        cdef int i, a, irrep
        cdef ExcType exc_type
        exc_type = (ExcType.A
                    if exc.alpha_rank == 1 else
                    ExcType.B)
        i, a = ((exc.alpha_h[0], exc.alpha_p[0])
                if exc_type == ExcType.A else
                (exc.beta_h[0], exc.beta_p[0]))
        irrep = self.orbspace.get_orb_irrep(i)
        i -= self.orbspace.orbs_before[irrep]
        a -= self.orbspace.orbs_before[irrep]
        a -= self.orbspace.corr[
            irrep + (0
                     if (not self.restricted
                         and exc_type == ExcType.A) else
                     self.n_irrep)]
        return i, a, irrep, exc_type
    
    def indices_of_doubles(self, SDExcitation exc):
        cdef int i, j, a, b
        cdef int irrep_i, irrep_j, irrep_a, irrep_b
        cdef ExcType exc_type
        if exc.alpha_rank == 1:
            exc_type = ExcType.AB
            i, irrep_i = self.orbspace.get_local_index(exc.alpha_h[0], True)
            j, irrep_j = self.orbspace.get_local_index(exc.beta_h[0], False)
            a, irrep_a = self.orbspace.get_local_index(exc.alpha_p[0], True)
            b, irrep_b = self.orbspace.get_local_index(exc.beta_p[0], False)
            if self.restricted:
                if irrep_j < irrep_i or (irrep_j == irrep_i and j < i):
                    i, irrep_i, j, irrep_j = j, irrep_j, i, irrep_i
                    a, irrep_a, b, irrep_b = b, irrep_b, a, irrep_a
        elif exc.alpha_rank == 2:
            exc_type = ExcType.AA
            i, irrep_i = self.orbspace.get_local_index(exc.alpha_h[0], True)
            j, irrep_j = self.orbspace.get_local_index(exc.alpha_h[1], True)
            a, irrep_a = self.orbspace.get_local_index(exc.alpha_p[0], True)
            b, irrep_b = self.orbspace.get_local_index(exc.alpha_p[1], True)
        else:
            exc_type = ExcType.BB
            i, irrep_i = self.orbspace.get_local_index(exc.beta_h[0], False)
            j, irrep_j = self.orbspace.get_local_index(exc.beta_h[1], False)
            a, irrep_a = self.orbspace.get_local_index(exc.beta_p[0], False)
            b, irrep_b = self.orbspace.get_local_index(exc.beta_p[1], False)
        return i, j, a, b, irrep_i, irrep_j, irrep_a, exc_type

    def calc_memory(self, calcs_args):
        """Calculate memory needed for amplitudes
        
        Parameters:
        -----------
        
        Return:
        -------
        A float, with the memory used to store the wave function amplitudes
        """
        return mem_of_floats(len(self))
    
    cdef int initialize_amplitudes(self) except -1:
        """Initialize the list for the amplitudes."""
        self._set_memory('Amplitudes array in IntermNormWaveFunction')
        self.amplitudes = np.zeros(len(self))
        return 0
    
    def update_amplitudes(self, double[:] z, Py_ssize_t start=0, mode='direct'):
        """Update the amplitudes by z
        
        Parameters:
        -----------
        z (iterable of (python) floats)
            The update for the amplitudes.
        
        start (int)
            The index to start the update
        
        mode (str, optional, default='direct')
            How amplitudes are updated.
            For 'direct', z should have the same size of z
            and it is just added on the array amplitudes.
            For 'indep ampl', the size of z should be n_indep_ampl,
            and it should have only independent amplitudes (see n_indep_ampl).
            Whenever redundant amplitudes occur, both are updated.
        """
        cdef int i
        if mode == 'direct':
            if <Py_ssize_t> len(z) != len(self):
                raise ValueError('Update vector does not have same length as amplitude.')
            if start + <Py_ssize_t> len(z) > len(self):
                raise ValueError('Update vector is too large.')
            for i in range(len(z)):
                self.amplitudes[start+i] += z[i]
        elif mode == 'indep ampl':
            if len(z) != self.n_indep_ampl:
                raise ValueError('Update vector does not have same length'
                                 ' as the number of independent amplitude.')
            update_indep_amplitudes(self.amplitudes,
                                    z,
                                    self.ini_blocks_D[0, 0],
                                    len(self),
                                    self.n_indep_ampl,
                                    self.orbspace)
        else:
            raise ValueError('Unknown mode!')
    
    @classmethod
    def similar_to(cls, wf, wf_type, restricted):
        cdef IntermNormWaveFunction new_wf
        new_wf = super().similar_to(wf, restricted=restricted)
        new_wf.wf_type = wf_type
        new_wf.initialize_amplitudes()
        return new_wf
    
    @classmethod
    def from_zero_amplitudes(cls, point_group, orbspace,
                             level='SD', wf_type='CC',
                             restricted=True):
        """Construct a new wave function with all amplitudes set to zero
        
        Parameters:
        -----------
        orbspace (orbitals.orbital_space.FullOrbitalSpace)
            The orbital space
        
        Limitations:
        ------------
        Only for restricted wave functions. Thus, orbspace.ref must be of 'R' type
        
        """
        cdef IntermNormWaveFunction new_wf
        cdef int irrep
        if not restricted:
            NotImplementedError('Not implemented for unrestricted (sure?)')
        new_wf = cls()
        new_wf.restricted = restricted
        new_wf.wf_type = wf_type + level
        new_wf.point_group = point_group
        new_wf.orbspace.get_attributes_from(orbspace)
        if new_wf.restricted:
            new_wf.Ms = 0.0
        else:
            new_wf.Ms = 0
            for irrep in range(new_wf.n_irrep):
                new_wf.Ms += (new_wf.orbspace.ref[irrep]
                              - new_wf.orbspace.ref[irrep + new_wf.n_irrep])
            new_wf.Ms /= 2
        new_wf.initialize_amplitudes()
        return new_wf

    @classmethod
    def from_projected_fci(cls, wf, wf_type):
        """Construct a new object with same singles and doubles as in wf
        
        The new wave function is a "vertical" projection on top of the
        CCSD, CCD, CISD or CID manifold (depending on wf_type) of the
        wave function wf. That is:
        If wf_type is CCD, CISD or CID, the amplitudes of the returned
        wave function are the same as the corresponding coefficients in wf.
        If wf_type is CCSD, the singles are the same, but theamplitudes
        of doubles are obtained after a cluster decomposition, such that
        the coefficients of the doubly excited determinants,
            t_ij^ab + t_i^a*t_j^b - t_i^b*t_j^a
        are the coefficients from wf.
        
        Parameters:
        -----------
        wf (FCIWaveFunction)
            The wave function that will be projected
        
        wf_type (str)
            The wave function type (CCSD, CCD, CISD or CID)
        
        Return:
        -------
        An instance of cls, with the new wave function, with restricted=False
        
        Side Effect:
        ------------
        wf is set to maximum coincidence order for orbitals, see
        FCIWaveFunction.set_max_coincidence_orbitals
        
        
        """
        cdef OccOrbital i, j
        cdef IntermNormWaveFunction new_wf
        wf.set_max_coincidence_orbitals()
        pos_ini = np.empty(8, dtype=int_dtype)
        pos = 0
        new_wf = cls.similar_to(wf, wf_type, restricted=False)
        i_alpha_ref, i_beta_ref = wf.i_ref
        alpha_occ = np.empty(wf.n_corr_alpha, dtype=int_dtype)
        beta_occ = np.empty(wf.n_corr_beta, dtype=int_dtype)
        for ii in range(wf.n_corr_alpha):
            alpha_occ[ii] = wf.ref_det.alpha_occ[ii]
        for ii in range(wf.n_corr_beta):
            beta_occ[ii] = wf.ref_det.beta_occ[ii]
        do_decomposition = wf_type == 'CCSD'
        if 'S' in wf_type:
            # --- alpha -> alpha
            for irrep in range(wf.n_irrep):
                spirrep = irrep
                if wf.orbspace.virt[spirrep] == 0 or wf.orbspace.corr[spirrep] == 0:
                    continue
                virt_pos = (wf.orbspace.corr_orbs_before[spirrep]
                            + wf.orbspace.corr[spirrep] - 1)
                for ii in range(wf.orbspace.corr_orbs_before[spirrep], virt_pos):
                    alpha_occ[ii] = alpha_occ[ii + 1]
                for ii in range(wf.orbspace.corr[spirrep]):
                    alpha_occ[virt_pos] = (wf.orbspace.orbs_before[irrep]
                                           + wf.orbspace.corr[spirrep])
                    for a in range(wf.orbspace.virt[spirrep]):
                        new_wf.amplitudes[pos] = wf[
                            str_order.get_index(alpha_occ, wf.alpha_string_graph),
                            i_beta_ref]
                        pos += 1
                        alpha_occ[virt_pos] += 1
                    alpha_occ[wf.orbspace.corr_orbs_before[spirrep] + ii] = \
                        wf.orbspace.orbs_before[irrep] + ii
            # --- beta -> beta
            for irrep in range(wf.n_irrep):
                spirrep = irrep + wf.n_irrep
                if wf.orbspace.virt[spirrep] == 0 or wf.orbspace.corr[spirrep] == 0:
                    continue
                virt_pos = (wf.orbspace.corr_orbs_before[spirrep]
                            + wf.orbspace.corr[spirrep] - 1)
                for ii in range(wf.orbspace.corr_orbs_before[spirrep], virt_pos):
                    beta_occ[ii] = beta_occ[ii + 1]
                for ii in range(wf.orbspace.corr[spirrep]):
                    beta_occ[virt_pos] = (wf.orbspace.orbs_before[irrep]
                                          + wf.orbspace.corr[spirrep])
                    for a in range(wf.orbspace.virt[spirrep]):
                        new_wf.amplitudes[pos] = wf[
                            i_alpha_ref,
                            str_order.get_index(beta_occ, wf.beta_string_graph)]
                        pos += 1
                        beta_occ[virt_pos] += 1
                    beta_occ[wf.orbspace.corr_orbs_before[spirrep] + ii] = \
                        wf.orbspace.orbs_before[irrep] + ii
        # --- alpha, alpha -> alpha, alpha
        i = OccOrbital(wf.orbspace, True)
        j = OccOrbital(wf.orbspace, True)
        j.next_()
        beta_hp = (int_array(), int_array())
        while j.alive:
            alpha_hp = (int_array(i.orb, j.orb),
                        int_array(0, 0))
            for ii in range(len(alpha_occ)):
                alpha_occ[ii] = wf.ref_det.alpha_occ[ii]
            virt_pos_a = i.pos_in_occ
            virt_pos_b = j.pos_in_occ
            for a_irrep in range(wf.n_irrep):
                pos_ini[a_irrep] = pos
                b_irrep = irrep_product[
                    irrep_product[i.spirrep,
                                  j.spirrep], a_irrep]
                a_spirrep = a_irrep
                b_spirrep = b_irrep
                if wf.orbspace.virt[a_spirrep] == 0 or wf.orbspace.virt[b_spirrep] == 0:
                    continue
                if a_irrep <= b_irrep:
                    if a_irrep == b_irrep and wf.orbspace.virt[a_spirrep] < 2:
                        pos += 1
                        continue
                    virt_pos_a, virt_pos_b = _set_virtpos_for_proj_aa_bb(
                        virt_pos_a, virt_pos_b,
                        i, j, a_spirrep, b_spirrep,
                        alpha_occ, wf.orbspace)
                    alpha_occ[virt_pos_a] = (wf.orbspace.orbs_before[a_irrep]
                                             + wf.orbspace.corr[a_spirrep])
                    for a in range(wf.orbspace.virt[a_spirrep]):
                        alpha_hp[1][0] = alpha_occ[virt_pos_a]
                        nvirt_1 = wf.orbspace.virt[a_spirrep] - 1
                        alpha_occ[virt_pos_b] = (wf.orbspace.orbs_before[b_irrep]
                                                 + wf.orbspace.corr[b_spirrep])
                        for b in range(wf.orbspace.virt[b_spirrep]):
                            alpha_hp[1][1] = alpha_occ[virt_pos_b]
                            if a_irrep < b_irrep or a < b:
                                new_wf.amplitudes[pos] = wf[
                                    str_order.get_index(alpha_occ,
                                                        wf.alpha_string_graph),
                                    i_beta_ref]
                                if do_decomposition:
                                    new_wf.amplitudes[pos] += singles_contr_from_clusters_fci(
                                        alpha_hp, beta_hp, wf)
                            elif a > b:
                                new_wf.amplitudes[pos] = -new_wf.amplitudes[
                                    pos - (a-b)*nvirt_1]
                            pos += 1
                            alpha_occ[virt_pos_b] += 1
                        alpha_occ[virt_pos_a] += 1
                else:  # and a_irrep > b_irrep
                    for a in range(wf.orbspace.virt[a_spirrep]):
                        for b in range(wf.orbspace.virt[b_spirrep]):
                            new_wf.amplitudes[pos] = -new_wf.amplitudes[
                                pos_ini[b_irrep] + n_from_rect(
                                    b, a, wf.orbspace.virt[a_spirrep])]
                            pos += 1
            if i.pos_in_occ == j.pos_in_occ - 1:
                j.next_()
                i.rewind()
            else:
                i.next_()
        # --- beta, beta -> beta, beta
        i = OccOrbital(wf.orbspace, False)
        j = OccOrbital(wf.orbspace, False)
        j.next_()
        alpha_hp = (int_array(), int_array())
        while j.alive:
            beta_hp = (int_array(i.orb, j.orb),
                       int_array(0, 0))
            for ii in range(len(beta_occ)):
                beta_occ[ii] = wf.ref_det.beta_occ[ii]
            virt_pos_a = i.pos_in_occ
            virt_pos_b = j.pos_in_occ
            for a_irrep in range(wf.n_irrep):
                pos_ini[a_irrep] = pos
                b_irrep = irrep_product[
                    irrep_product[i.spirrep - wf.n_irrep,
                                  j.spirrep - wf.n_irrep], a_irrep]
                a_spirrep = a_irrep + wf.n_irrep
                b_spirrep = b_irrep + wf.n_irrep
                if wf.orbspace.virt[a_spirrep] == 0 or wf.orbspace.virt[b_spirrep] == 0:
                    continue
                if a_irrep <= b_irrep:
                    if a_irrep == b_irrep and wf.orbspace.virt[a_spirrep] < 2:
                        pos += 1
                        continue
                    virt_pos_a, virt_pos_b = _set_virtpos_for_proj_aa_bb(
                        virt_pos_a, virt_pos_b,
                        i, j, a_spirrep, b_spirrep,
                        beta_occ, wf.orbspace)
                    beta_occ[virt_pos_a] = (wf.orbspace.orbs_before[a_irrep]
                                            + wf.orbspace.corr[a_spirrep])
                    for a in range(wf.orbspace.virt[a_spirrep]):
                        beta_hp[1][0] = beta_occ[virt_pos_a]
                        nvirt_1 = wf.orbspace.virt[a_spirrep] - 1
                        beta_occ[virt_pos_b] = (wf.orbspace.orbs_before[b_irrep]
                                                + wf.orbspace.corr[b_spirrep])
                        for b in range(wf.orbspace.virt[b_spirrep]):
                            beta_hp[1][1] = beta_occ[virt_pos_b]
                            if a_irrep < b_irrep or a < b:
                                new_wf.amplitudes[pos] = wf[
                                    i_alpha_ref,
                                    str_order.get_index(beta_occ,
                                                        wf.beta_string_graph)]
                                if do_decomposition:
                                    new_wf.amplitudes[pos] += singles_contr_from_clusters_fci(
                                        alpha_hp, beta_hp, wf)
                            elif a > b:
                                new_wf.amplitudes[pos] = -new_wf.amplitudes[
                                    pos - (a-b)*nvirt_1]
                            pos += 1
                            beta_occ[virt_pos_b] += 1
                        beta_occ[virt_pos_a] += 1
                else:  # and a_irrep > b_irrep
                    for a in range(wf.orbspace.virt[a_spirrep]):
                        for b in range(wf.orbspace.virt[b_spirrep]):
                            new_wf.amplitudes[pos] = -new_wf.amplitudes[
                                pos_ini[b_irrep] + n_from_rect(
                                    b, a, wf.orbspace.virt[a_spirrep])]
                            pos += 1
            if i.pos_in_occ == j.pos_in_occ - 1:
                j.next_()
                i.rewind()
            else:
                i.next_()
        # --- alpha, beta -> alpha, beta
        i = OccOrbital(wf.orbspace, True)
        j = OccOrbital(wf.orbspace, False)
        while j.alive:
            alpha_hp = (int_array(i.orb), int_array(0))
            beta_hp = (int_array(j.orb), int_array(0))
            for ii in range(len(alpha_occ)):
                alpha_occ[ii] = wf.ref_det.alpha_occ[ii]
            virt_pos_a = i.pos_in_occ
            for ii in range(len(beta_occ)):
                beta_occ[ii] = wf.ref_det.beta_occ[ii]
            virt_pos_b = j.pos_in_occ
            for a_irrep in range(wf.n_irrep):
                b_irrep = irrep_product[
                    irrep_product[i.spirrep, j.spirrep - wf.n_irrep], a_irrep]
                a_spirrep = a_irrep
                b_spirrep = b_irrep + wf.n_irrep
                if wf.orbspace.virt[a_spirrep] == 0 or wf.orbspace.virt[b_spirrep] == 0:
                    continue
                alpha_occ[virt_pos_a] = (wf.orbspace.orbs_before[a_irrep]
                                         + wf.orbspace.corr[a_spirrep])
                new_virt_pos_a = (wf.orbspace.corr_orbs_before[a_spirrep]
                                  + wf.orbspace.corr[a_spirrep])
                new_virt_pos_b = (wf.orbspace.corr_orbs_before[b_spirrep]
                                  + wf.orbspace.corr[b_spirrep])
                if (i.spirrep <= a_spirrep):
                    new_virt_pos_a -= 1
                if (j.spirrep <= b_spirrep):
                    new_virt_pos_b -= 1
                if new_virt_pos_a > virt_pos_a:
                    for ii in range(virt_pos_a, new_virt_pos_a):
                        alpha_occ[ii] = alpha_occ[ii + 1]
                else:
                    for ii in range(virt_pos_a - new_virt_pos_a):
                        alpha_occ[virt_pos_a - ii] = alpha_occ[virt_pos_a
                                                               - (ii+1)]
                if new_virt_pos_b > virt_pos_b:
                    for ii in range(virt_pos_b, new_virt_pos_b):
                        beta_occ[ii] = beta_occ[ii + 1]
                else:
                    for ii in range(virt_pos_b - new_virt_pos_b):
                        beta_occ[virt_pos_b - ii] = beta_occ[virt_pos_b
                                                             - (ii+1)]
                virt_pos_a = new_virt_pos_a
                virt_pos_b = new_virt_pos_b
                alpha_occ[virt_pos_a] = (wf.orbspace.orbs_before[a_irrep]
                                         + wf.orbspace.corr[a_spirrep])
                for a in range(wf.orbspace.virt[a_spirrep]):
                    alpha_hp[1][0] = alpha_occ[virt_pos_a]
                    beta_occ[virt_pos_b] = (wf.orbspace.orbs_before[b_irrep]
                                            + wf.orbspace.corr[b_spirrep])
                    for b in range(wf.orbspace.virt[b_spirrep]):
                        beta_hp[1][0] = beta_occ[virt_pos_b]
                        new_wf.amplitudes[pos] = wf[
                            str_order.get_index(alpha_occ, wf.alpha_string_graph),
                            str_order.get_index(beta_occ, wf.beta_string_graph)]
                        if do_decomposition:
                            new_wf.amplitudes[pos] += singles_contr_from_clusters_fci(
                                alpha_hp, beta_hp, wf)
                        pos += 1
                        beta_occ[virt_pos_b] += 1
                    alpha_occ[virt_pos_a] += 1
            i.next_()
            if not i.alive:
                j.next_()
                i.rewind()
        return new_wf
    
    @classmethod
    def restrict(cls, IntermNormWaveFunction ur_wf):
        """A contructor that return a restricted version of wf
        
        The constructed wave function should be the same as wf, however,
        of a restricted type. This method will work only if the amplitudes
        associated to alpha and beta excitations are equal, within the
        tolerance dictated by util.variables.zero.
        
        Parameters:
        -----------
        ur_wf (IntermNormWaveFunction)
        The wave function (in general of unrestricted type). If restricted,
        a copy  is returned
        
        Raise:
        ------
        ValueError if the wave function cannot be restricted.
        
        """
        cdef IntermNormWaveFunction r_wf
        cdef int iampl, n_corr, r_pos_ij, pos_ij, pos_ji, ij_ini, ji_ini
        cdef int irrep_a, irrep_b, block_len
        cdef double[:] a_block
        cdef double[:] b_block
        cdef OccOrbital i, j
        if ur_wf.restricted:
            r_wf = cls.similar_to(ur_wf, ur_wf.wf_type, True)
            r_wf.amplitudes = np.array(ur_wf.amplitudes)
            return r_wf
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
        for i, j in ur_wf.occupied_pairs(ExcType.AB):
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
                    block_len = ur_wf.orbspace.virt[irrep_a]*ur_wf.orbspace.virt[irrep_b]
                    ij_ini = ur_wf.ini_blocks_D[pos_ij, irrep_a]
                    ji_ini = ur_wf.ini_blocks_D[pos_ji, irrep_b]
                    if not np.allclose(
                            ur_wf.amplitudes[ij_ini:
                                             ij_ini + block_len],
                            _transpose(ur_wf.amplitudes[ji_ini:
                                                        ji_ini + block_len],
                                       ur_wf.orbspace.virt[irrep_b],
                                       ur_wf.orbspace.virt[irrep_a])):
                        a_block = ur_wf.amplitudes[ij_ini:
                                                   ij_ini + block_len]
                        b_block = _transpose(
                            ur_wf.amplitudes[ji_ini:
                                             ji_ini + block_len],
                            ur_wf.orbspace.virt[irrep_b],
                            ur_wf.orbspace.virt[irrep_a])
                        to_msg = []
                        for iampl in range(block_len):
                            to_msg.append(
                                f'{a_block[iampl]:10.7f}'
                                f'  {b_block[iampl]:10.7f}'
                                f'  {a_block[iampl]-b_block[iampl]}')
                        raise ValueError(
                            'alpha,beta with non equivalent blocks:\n'
                            + '\n'.join(to_msg))
        r_wf = cls.similar_to(ur_wf, ur_wf.wf_type, True)
        if "S" in ur_wf.wf_type:
            r_wf.amplitudes[:r_wf.ini_blocks_D[0, 0]] = \
                ur_wf.amplitudes[:ur_wf.ini_blocks_S[ur_wf.n_irrep]]
        r_pos_ij = 0
        for i, j in ur_wf.occupied_pairs(ExcType.AB):
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
    def unrestrict(cls, IntermNormWaveFunction r_wf):
        """A constructor that return a unrestricted version of wf
        
        The constructed wave function should be the same as wf, however,
        of a unrestricted type. Thus, the amplitudes are "duplicated" to
        hold both alpha and beta amplitudes
        
        Parameters:
        -----------
        r_wf (IntermNormWaveFunction)
        The wave function (in general of restricted type). If unrestricted,
        a copy is returned
        """
        cdef IntermNormWaveFunction ur_wf
        cdef int iampl, ij, ij_diff, ji_ab_pos, first_bb_ampl, n_corr
        cdef int irrep_a, irrep_b, block_len, ini_bl
        cdef double[:] this_block, transp_block 
        cdef OccOrbital i, j
        if not r_wf.restricted:
            ur_wf = cls.similar_to(r_wf, r_wf.wf_type, False)
            ur_wf.amplitudes = np.array(r_wf.amplitudes)
            return ur_wf
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
        for i, j in r_wf.occupied_pairs(ExcType.ALL):
            for irrep_a in range(r_wf.n_irrep):
                irrep_b = irrep_product[
                    irrep_a, irrep_product[i.spirrep,
                                           j.spirrep]]
                block_len = r_wf.orbspace.virt[irrep_a] * r_wf.orbspace.virt[irrep_b]
                ini_bl = r_wf.ini_blocks_D[ij, irrep_a]
                this_block = r_wf.amplitudes[ini_bl:
                                             ini_bl + block_len]
                ij_ab_pos = (ur_wf.first_ab_pair
                             + n_from_rect(j.pos_in_occ,
                                           i.pos_in_occ,
                                           n_corr))
                ini_bl = ur_wf.ini_blocks_D[ij_ab_pos, irrep_a]
                for iampl in range(block_len):
                    ur_wf.amplitudes[ini_bl + iampl] = this_block[iampl]
                if i.pos_in_occ != j.pos_in_occ:
                    transp_block = _transpose(this_block,
                                              r_wf.orbspace.virt[irrep_a],
                                              r_wf.orbspace.virt[irrep_b])
                    ji_ab_pos = (ur_wf.first_ab_pair
                                 + n_from_rect(i.pos_in_occ,
                                               j.pos_in_occ,
                                               n_corr))
                    ini_bl = ur_wf.ini_blocks_D[ji_ab_pos, irrep_b]
                    for iampl in range(block_len):
                        ur_wf.amplitudes[ini_bl + iampl] = transp_block[iampl]
                    # ----- direct term
                    # alpha,alpha -> alpha,alpha
                    ini_bl = ur_wf.ini_blocks_D[ij_diff, irrep_a]
                    for iampl in range(block_len):
                        ur_wf.amplitudes[ini_bl + iampl] += this_block[iampl]
                    # beta,beta -> beta,beta
                    ini_bl += first_bb_ampl
                    for iampl in range(block_len):
                        ur_wf.amplitudes[ini_bl + iampl] += this_block[iampl]
                    # ----- minus transpose term
                    # alpha,alpha -> alpha,alpha
                    ini_bl = ur_wf.ini_blocks_D[ij_diff, irrep_b]
                    for iampl in range(block_len):
                        ur_wf.amplitudes[ini_bl + iampl] -= transp_block[iampl]
                    # beta,beta -> beta,beta
                    ini_bl += first_bb_ampl
                    for iampl in range(block_len):
                        ur_wf.amplitudes[ini_bl + iampl] -= transp_block[iampl]
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
        cdef IntermNormWaveFunction new_wf
        cdef ExcType exc_type = ExcType.ALL
        cdef int pos_ij, i, j, prev_Molpros_i, prev_Molpros_j
        prev_Molpros_i = prev_Molpros_j = -1
        pos_ij = i = j = -1
        new_wf = cls()
        new_wf.Ms = 0.0
        sgl_found = False
        dbl_found = False
        MP2_step_passed = True
        if molpro_output is None:
            raise ValueError('Parameter molpro_output cannot be None')
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
            new_wf.orbspace.set_n_irrep(new_wf.n_irrep)
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
                else:
                    new_wf.orbspace.set_n_irrep(new_wf.n_irrep)
                continue
            if ('Number of closed-shell orbitals' in line
                    or 'Number of core orbitals' in line):
                new_orbitals = molpro.get_orb_info(
                    line, line_number,
                    new_wf.n_irrep, 'R')
                new_wf.orbspace.add_to_ref(new_orbitals, update=True, add_to_full=True)
                if 'Number of core orbitals' in line:
                    new_wf.orbspace.froz += new_orbitals
            elif 'Number of active  orbitals' in line:
                new_wf.orbspace.set_act(
                    molpro.get_orb_info(line,
                                        line_number,
                                        new_wf.n_irrep, 'A'))
                new_wf.orbspace.add_to_full(
                    molpro.get_orb_info(
                        line, line_number,
                        new_wf.n_irrep, 'R'))
                new_wf.orbspace.add_to_ref(new_wf.orbspace.act, update=True, add_to_full=False)
                new_wf.Ms = len(new_wf.orbspace.act) / 2
            elif 'Number of external orbitals' in line:
                new_wf.orbspace.add_to_full(
                    molpro.get_orb_info(
                        line, line_number,
                        new_wf.n_irrep, 'R'))
            elif 'Starting RMP2 calculation' in line:
                MP2_step_passed = False
            elif 'RHF-RMP2 energy' in line:
                MP2_step_passed = True
            elif molpro.CC_sgl_str in line and MP2_step_passed:
                sgl_found = True
                new_wf.restricted = not ('Alpha-Alpha' in line
                                         or 'Beta-Beta' in line)
                if not new_wf._n_ampl:
                    new_wf.initialize_amplitudes()
                exc_type = ExcType.B if 'Beta-Beta' in line else ExcType.A
            elif molpro.CC_dbl_str in line and MP2_step_passed:
                dbl_found = True
                prev_Molpros_i = prev_Molpros_j = -1
                pos_ij = i = j = -1
                restricted = not ('Alpha-Alpha' in line
                                  or 'Beta-Beta' in line
                                  or 'Alpha-Beta' in line)
                if (sgl_found
                        and restricted != new_wf.restricted):
                    raise molpro.MolproInputError(
                        'Found restricted/unrestricted inconsistence '
                        + 'between singles and doubles!',
                        line=line,
                        line_number=line_number,
                        file_name=f_name)
                else:
                    new_wf.restricted = restricted
                if not new_wf._n_ampl:
                    if new_wf.wf_type == 'CCSD':
                        new_wf.wf_type = 'CCD'
                    elif new_wf.wf_type == 'CCSD':
                        new_wf.wf_type = 'CID'
                    new_wf.initialize_amplitudes()
                if new_wf.restricted or 'Alpha-Alpha' in line:
                    exc_type = ExcType.AA
                elif 'Beta-Beta' in line:
                    exc_type = ExcType.BB
                elif 'Alpha-Beta' in line:
                    exc_type = ExcType.AB
            elif dbl_found:
                lspl = line.split()
                if len(lspl) == 7:
                    all_indices = map(
                        lambda x: int(x) - 1, lspl[0:-1])
                    if exc_type == ExcType.AB:
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
                        0 if exc_type == ExcType.AA else new_wf.n_irrep)
                    C = float(lspl[-1])
                    if exc_type in (ExcType.AA, ExcType.AB):
                        a -= new_wf.orbspace.act[irrep_a]
                    if exc_type == ExcType.AA:
                        b -= new_wf.orbspace.act[irrep_b]
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
                               and i >= new_wf.orbspace.corr_orbs_before[irrep_i + 1]):
                            irrep_i += 1
                        if exc_type in (ExcType.AB, ExcType.BB):
                            irrep_j = 0
                            while (irrep_j < new_wf.n_irrep
                                   and
                                   j >= new_wf.orbspace.corr_orbs_before[irrep_j + 1]):
                                irrep_j += 1
                            if irrep_j > 0:
                                j -= sum(new_wf.orbspace.act[:irrep_j])
                            if exc_type == ExcType.BB:
                                if irrep_i > 0:
                                    i -= sum(new_wf.orbspace.act[:irrep_i])
                        pos_ij = new_wf.get_ij_pos_from_i_j(i, j,
                                                            irrep_i,
                                                            exc_type)
                    new_wf.amplitudes[new_wf.ini_blocks_D[pos_ij, irrep_a]
                                      + n_from_rect(
                                          a, b,
                                          new_wf.orbspace.virt[spirrep_b])] = C
            elif sgl_found:
                lspl = line.split()
                if len(lspl) == 4:
                    i, irrep, a = map(lambda x: int(x) - 1,
                                      lspl[0:-1])
                    C = float(lspl[-1])
                    spirrep = irrep + (0
                                       if exc_type == ExcType.A else
                                       new_wf.n_irrep)
                    i -= new_wf.orbspace.corr_orbs_before[irrep]
                    if exc_type == ExcType.A:
                        a -= new_wf.orbspace.act[irrep]
                    if a < 0 or i >= new_wf.orbspace.corr[spirrep]:
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
                        'Double excitations not found!',
                        line=line,
                        line_number=line_number,
                        file_name=f_name)
                break
        if new_wf.restricted:
            new_wf.orbspace.ref.restrict_it()
        if isinstance(molpro_output, str):
            f.close()
        return new_wf

    @property
    def norm(self):
        if self._norm < 0:
            logger.debug('IntermNormWaveFunction before normalisation:\n%s', self)
            self._norm = sqrt(1.0
                              + self.S_contrib_to_norm()
                              + self.D_contrib_to_norm())
            logger.debug('Calculated norm: %s', self._norm)
        return self._norm

    cdef double S_contrib_to_norm(self) except -1.0:
        """Contribution of singly excited determinants to norm"""
        cdef double S = 0.0
        cdef int pos
        for pos in range(self.ini_blocks_D[0,0]):
            logger.debug('contribution from single to the norm: %.16f',
                         self.amplitudes[pos]**2)
            S += self.amplitudes[pos]**2
        if self.restricted:
            S *= 2
        return S

    cdef double D_contrib_to_norm(self) except -1.0:
        """Contribution of doubly excited determinants to norm"""
        cdef OccOrbital i, j
        cdef int pos, pos_compl, a_irrep, b_irrep, ij
        cdef int nva, nvb
        cdef double D, C, C2, t_ia_t_jb, t_ja_t_ib
        D = 0.0
        ij = 0
        pos = self.ini_blocks_D[0,0]
        if self.restricted:
            for i, j in self.occupied_pairs(ExcType.ALL):
                ij_differ = i.orb != j.orb
                logger.debug('i=%s, j=%s', i, j)
                for a_irrep in range(self.n_irrep):
                    b_irrep = irrep_product[a_irrep,
                                            irrep_product[i.spirrep,
                                                          j.spirrep]]
                    if b_irrep > a_irrep:
                        pos = self.ini_blocks_D[ij, a_irrep+1]
                        continue
                    nva = self.orbspace.virt[a_irrep]
                    inibla = self.ini_blocks_S[a_irrep]
                    nvb = self.orbspace.virt[b_irrep]
                    iniblb = self.ini_blocks_S[b_irrep]
                    ab_same_irrep = a_irrep == b_irrep
                    ccsd_ia_same_irrep = i.spirrep == a_irrep and self.wf_type == 'CCSD'
                    ccsd_ja_same_irrep = j.spirrep == a_irrep and self.wf_type == 'CCSD'
                    t_ia_t_jb = 0
                    t_ja_t_ib = 0
                    for a in range(self.orbspace.virt[a_irrep]):
                        for b in range(self.orbspace.virt[b_irrep]):
                            if ccsd_ia_same_irrep:
                                t_ia_t_jb = (self.amplitudes[inibla
                                                             + n_from_rect(i.orbirp, a, nva)]
                                             * self.amplitudes[iniblb
                                                               + n_from_rect(j.orbirp, b, nvb)])
                            if ccsd_ja_same_irrep:
                                t_ja_t_ib = (self.amplitudes[inibla
                                                             + n_from_rect(j.orbirp, a, nva)]
                                             * self.amplitudes[iniblb
                                                               + n_from_rect(i.orbirp, b, nvb)])
                            if ab_same_irrep and a == b:
                                C = self.amplitudes[pos]
                                if ccsd_ia_same_irrep: C += t_ia_t_jb
                                C = C**2
                                if ij_differ:
                                    C2 = self.amplitudes[pos]
                                    if ccsd_ja_same_irrep: C2 += t_ia_t_jb
                                    C = C + C2**2
                                logger.debug('Contribution from double to the norm'
                                             ' (a=b=%s irrep=%s): %.16f',
                                             a, a_irrep, C)
                                D += C
                            elif (ab_same_irrep and b < a) or not ab_same_irrep:
                                pos_compl = (self.ini_blocks_D[ij, b_irrep]
                                             + n_from_rect(b, a, nva))
                                if ij_differ:
                                    C = self.amplitudes[pos] - self.amplitudes[pos_compl]
                                    if ccsd_ia_same_irrep: C += t_ia_t_jb
                                    if ccsd_ja_same_irrep: C -= t_ja_t_ib
                                    C = 2 * C**2
                                    C2 = self.amplitudes[pos]
                                    if ccsd_ia_same_irrep: C2 += t_ia_t_jb
                                    C += 2 * C2**2
                                    C2 = self.amplitudes[pos_compl]
                                    if ccsd_ja_same_irrep: C2 += t_ja_t_ib
                                    C += 2 * C2**2
                                else:
                                    C = self.amplitudes[pos] + self.amplitudes[pos_compl]
                                    if ccsd_ia_same_irrep: C += 2 * t_ia_t_jb
                                    C = 0.5* C**2
                                logger.debug('Contribution from double to the norm'
                                             ' (a=%s irrep=%s, b=%s irrep=%s): %.16f',
                                             a, a_irrep, b, b_irrep, C)
                                D += C
                            pos += 1
                ij += 1
        else:
            raise NotImplementedError(
                'We still dont have norm for unrestricted wave function')
        return D
        
    @property
    def C0(self):
        """The coefficient of reference"""
        return 1.0 / self.norm
