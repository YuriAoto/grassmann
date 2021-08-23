"""A FCI-like wave function, normalised to unity

The class defined here stores the wave function coefficients as a matrix
of alpha and beta strings, as described in Helgaker's book.

History:
    Dez 2020 - Start

Yuri
"""
import logging
import numpy as np
from numpy import linalg
from scipy.linalg import lu, norm
from scipy.special import comb
import copy
import math

from input_output import molpro

from util.array_indices import n_from_rect
from util.variables import int_dtype
from util.memory import mem_of_floats
from util.other import int_array
from molecular_geometry.symmetry import irrep_product
from wave_functions.general cimport WaveFunction
from wave_functions.general import WaveFunction
from wave_functions.interm_norm cimport IntermNormWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from wave_functions.slater_det import SlaterDet, get_slater_det_from_fci_line
from coupled_cluster.cluster_decomposition import cluster_decompose, str_dec
import wave_functions.strings_rev_lexical_order as str_order
from orbitals.orbitals import calc_U_from_z
from orbitals.orbital_space import OrbitalSpace

logger = logging.getLogger(__name__)


def get_occ_from_FCI_line(l, orb_dim, froz_orb, n_irrep, Ms,
                          molpro_output='', line_number=-1):
    """Read the configuration of a FCI line Molpro output
    
    Parameters:
    -----------
    l (str)
        The line with a configuration, from the FCI program in Molpro
        to be converted to a Slater Determinant.
    
    orb_dim (OrbitalSpace)
        Dimension of orbital space
    
    froz_orb (OrbitalSpace)
        Dimension of frozen orbitals space
    
    n_irrep (int)
        Number of irreps
    
    Ms (float)
        Ms of total wave function (n_alpha - n_beta)/2
    
    molpro_output (str, optional, default='')
        The output file name (only for error message)
    
    line_number (int, optional, default=-1)
        The line number in Molpro output
    
    Returns:
    --------
    A list of arrays, see the examples
    
    Raises:
    molpro.MolproInputError
    
    Examples:
    ---------
    
    # if n_irrep = 4, orb_dim = (6,2,2,0), froz_orb = (0,0,0,0) then
    
    -0.162676901257  1  2  7  1  2  7
    gives
    [(0,1) (0) () () (0,1) (0) () ()]
    
    -0.049624632911  1  2  4  1  2  6
    gives
    [(0,1,3) () () () (0,1,5) () () ()]
    
    0.000000000000  1  2  9  1  2 10
    gives
    [(0,1) () (0) () (0,1) () (1) ()]
    
    # but if froz_orb = (1,1,0,0) then the above cases give
         (because frozen electrons are indexed first in Molpro convention)
    
    [(0,5) (0) () () (0,5) (0) () ()]
    
    [(0,2) (0) () () (0,4) (0) () ()]
    
    [(0) (0) (0) () (0) (0) (1) ()]


    """
    lspl = l.split()
    final_occ = [list(range(froz_orb[irp])) for irp in range(2 * n_irrep)]
    n_tot_frozen = sum(map(len, final_occ)) // 2
    try:
        coeff = float(lspl[0])
        occ = [int(x) - 1 for x in lspl[1:] if int(x) > n_tot_frozen]
    except Exception as e:
        raise molpro.MolproInputError(
            "Error when reading FCI configuration. Exception was:\n"
            + str(e),
            line=l,
            line_number=line_number,
            file_name=molpro_output)
    if len(occ) + 2 * n_tot_frozen + 1 != len(lspl):
        raise molpro.MolproInputError(
            "Inconsistency in number of frozen orbitals for FCI. froz_orb:\n"
            + str(froz_orb),
            line=l,
            line_number=line_number,
            file_name=molpro_output)
    total_orbs = [sum(froz_orb[irp] for irp in range(n_irrep))]
    for i in range(n_irrep):
        total_orbs.append(total_orbs[-1]
                          + orb_dim[i] - froz_orb[i])
    irrep = irrep_shift = 0
    ini_beta = (len(occ) + int(2 * Ms)) // 2
    for i, orb in enumerate(occ):
        if i == ini_beta:
            irrep_shift = n_irrep
            irrep = 0
        while True:
            if irrep == n_irrep:
                raise molpro.MolproInputError(
                    'Configuration is not consistent with orb_dim = '
                    + str(orb_dim),
                    line=l,
                    line_number=line_number,
                    file_name=molpro_output)
            if total_orbs[irrep] <= orb < total_orbs[irrep + 1]:
                final_occ[irrep + irrep_shift].append(orb - total_orbs[irrep]
                                                      + froz_orb[irrep])
                break
            else:
                irrep += 1
    for i, o in enumerate(final_occ):
        final_occ[i] = np.array(o, dtype=int_dtype)
    return final_occ




def _compare_strings(ref, exc):
    """Compare strings ref and exc and return information about excitation
    
    This function look at exc as being an excitation over ref, and tell
    the rank of the excitation, the particles and the holes.
    
    Parameters:
    -----------
    ref (np.array of int)
        The "reference" string
    
    exc (np.array of int)
        The "excited" string
    
    Return:
    -------
    The tuple (rank, holes, particles), where:
    
    rank is an integer, with the excitation rank
    (0 for no exception, 1 for singly excited, 2 for doubly ...);
    
    holes are np.arrays with the indices that are in ref,
    but not in exc (thus, associated to orbitals where the excitation
    come from);
    
    particles are np.arrays with the indices that are in exc,
    but not in ref (thus, associated to orbitals where the excitation
    go to).
    
    Raise:
    ------
    ValueError if ref and exc have different lengths
    
    """
    holes = []
    particles = []
    for i in ref:
        if i not in exc:
            holes.append(i)
    for i in exc:
        if i not in ref:
            particles.append(i)
    return len(holes), holes, particles


cdef class FCIWaveFunction(WaveFunction):
    """A FCI-like wave function, based on alpha and beta strings
    
    The wave function is stored in a 2D np.array, whose rows and columns
    refer to alpha and beta strings
    
    Data Model:
    -----------
    []
    Should receive an object that can be interpreted as the index
    of a 2D np.array (that is the coefficients matrix for alpha and beta
    strings
    
    """
    rtol = 1E-5
    atol = 1E-8
    
    def __init__(self):
        """Initialize the wave function"""
        super().__init__()
        self.n_alpha_str_init = False
        self.n_beta_str_init = False
    
    def __len__(self):
        try:
            return self._coefficients.size
        except AttributeError:
            return 0
    
    def __getitem__(self, i):
        return self._coefficients[i[0], i[1]]
    
    def __iter__(self):
        """Generator for determinants"""
        aocc = np.arange(self.n_corr_alpha, dtype=int_dtype)
        for ia in range(self._coefficients.shape[0]):
            bocc = np.arange(self.n_corr_beta, dtype=int_dtype)
            for ib in range(self._coefficients.shape[1]):
                yield SlaterDet(c=self._coefficients[ia, ib],
                                alpha_occ=aocc, beta_occ=bocc)
                str_order.next_str(bocc)
            str_order.next_str(aocc)
        
    def __eq__(self, other):
        """Checks if both wave functions are the same within
        TODO: compare all attributes
        """
        global rtol
        global atol
        return np.allclose(self._coefficients, other._coefficients,
                           rtol=rtol, atol=atol)
    
    def __str__(self):
        """Return a string version of the wave function."""
        x = []
        for det in self:
            excinfo = self.get_exc_info(det)
            x.append(
                f'{det!s} >> RANK: {excinfo[0]}'
                + f' ({excinfo[1][0]!s}->{excinfo[1][1]!s})'
                + f' ({excinfo[2][0]!s}->{excinfo[2][1]!s})')
        x.append('-' * 50)
        return '\n'.join(x)

    def get_coefficients(self):
        return np.array(self._coefficients)

    @property
    def alpha_string_graph(self):
        return np.array(self._alpha_string_graph)

    @property
    def beta_string_graph(self):
        return np.array(self._beta_string_graph)

    def enumerate(self):
        """Generator for determinants that also yield the index
        
        Has the same philosophy as __iter__(self), but
        give the indices for alpha and beta strings. Usage:
        
        for ia, ib, det in self.enumerate:
            print(ia, ib, det)
            print((ia, ib) == wf.index(det)) # Should be always True
        """
        aocc = np.arange(self.n_corr_alpha, dtype=int_dtype)
        for ia in range(self._coefficients.shape[0]):
            bocc = np.arange(self.n_corr_beta, dtype=int_dtype)
            for ib in range(self._coefficients.shape[1]):
                yield ia, ib, SlaterDet(c=self._coefficients[ia, ib],
                                        alpha_occ=aocc, beta_occ=bocc)
                str_order.next_str(bocc)
            str_order.next_str(aocc)
    
    @property
    def n_alpha_str(self):
        if not self.n_alpha_str_init:
            self._n_alpha_str = comb(self.orbspace.n_orb_nofrozen,
                                     self.n_corr_alpha,
                                     exact=True)
        return self._n_alpha_str
    
    @property
    def n_beta_str(self):
        if not self.n_beta_str_init:
            self._n_beta_str = comb(self.orbspace.n_orb_nofrozen,
                                    self.n_corr_beta,
                                    exact=True)
        return self._n_beta_str
    
    def calc_memory(self, calc_args):
        """Calculate memory of current determinants in wave function"""
        return mem_of_floats(self.n_alpha_str * self.n_beta_str)
    
    def initialize_coeff_matrix(self):
        """Initialize the matrix with the coefficients"""
        self._alpha_string_graph = str_order.generate_graph(
            self.n_corr_alpha, self.orbspace.n_orb_nofrozen)
        self._beta_string_graph = str_order.generate_graph(
            self.n_corr_beta, self.orbspace.n_orb_nofrozen)
        self._set_memory()
        self._coefficients = np.zeros((self.n_alpha_str, self.n_beta_str))

    def set_slater_det(self, det):
        """Set a the Slater Determinant coefficient to the wave function
        
        Parameter:
        ----------
        det (Slater_Det)
            The Slater determinant
        
        Return:
        -------
        The index of this determinant
        """
        ii = self.index(det)
        self._coefficients[ii[0], ii[1]] = det.c
        return ii
    
    def index(self, det):
        """Return the index of det in the coefficients matrix
        
        Parameters:
        -----------
        det (Slater_Det)
            the Slater_Det whose index must be found
        
        Return:
        -------
        a 2-tuple, the index of element as alpha and beta string indices
        
        Raise:
        ------
        ValueError if element is not in self
        
        """
        return (str_order.get_index(det.alpha_occ, self._alpha_string_graph),
                str_order.get_index(det.beta_occ, self._beta_string_graph))
            
    @property
    def i_ref(self):
        return self.index(self.ref_det)
    
    def get_i_max_coef(self, set_ref_det=False):
        """Return the index of determinant with largest coefficient
        
        If set_ref_det is True, also sets ref_det to the corresponding
        determinant (default=False)
        """
        i_max = np.unravel_index(np.argmax(self._coefficients),
                                 self._coefficients.shape)
        if set_ref_det:
            self.ref_det = SlaterDet(
                c=self[i_max],
                alpha_occ=self.alpha_orb_occupation(i_max[0]),
                beta_occ=self.beta_orb_occupation(i_max[1]))
        return i_max
    
    @property
    def C0(self):
        return self.ref_det.c
    
    def alpha_orb_occupation(self, i):
        """Return the alpha occupied orbitals of index i
        
        Parameters:
        -----------
        i (int)
            The index of the slater determinant in the
            alpha string graph matrix
        
        Return:
        -------
        A 1D np.array with the occupied alpha orbitals (without considering
        frozen orbitals)
        """
        return str_order.occ_from_pos(i, self._alpha_string_graph)

    def beta_orb_occupation(self, i):
        """Return the beta occupied orbitals of index i
        
        Parameters:
        -----------
        i (int)
            The index of the slater determinant in the
            beta string graph matrix
        
        Return:
        -------
        A 1D np.array with the occupied beta orbitals (without considering
        frozen orbitals)
        """
        return str_order.occ_from_pos(i, self._beta_string_graph)
    
    def get_slater_det(self, ii):
        """Return the slater determinant associated to index i
        
        Parameters:
        -----------
        ii (2-tuple of int)
            The index of the slater determinant in the coefficients matrix
        
        Return:
        -------
        An instance of SlaterDet with the coefficient associated to ii
        and the alpha and beta occupation, without considering frozen orbitals
        """
        return SlaterDet(c=self[ii],
                         alpha_occ=self.alpha_orb_occupation(ii[0]),
                         beta_occ=self.beta_orb_occupation(ii[1]))
    
    def get_exc_info(self, det,
                     ref=None,
                     only_rank=False):
        """Return some info about the excitation that lead from ref to det
        
        This will return the holes, the particles, and the rank of det,
        viewed as a excitation over self.ref_orb.
        Particles and holes are returned as lists of Orbital_Info.
        These lists have len = rank.
        If consider_frozen == False, frozen orbitals are not considered,
        and the first correlated orbitals is 0; Otherwise frozen orbitals
        are taken into account and the first correlated orbital is
        froz_orb[spirrep]
        The order that the holes/particles are put in the lists is the
        canonical: follows the spirrep, and the normal order inside each
        spirrep.
        Particles are numbered starting at zero, that is, the index at det
        minus self.ref_orb[spirrep].
        
        Parameters:
        -----------
        det (SlaterDet)
            The Slater determinant to be verified.
        
        ref (None or a SlaterDet)
            The Slater determinant as reference.
            If None, use self.ref_det
        
        only_rank (bool, optional, default=False
            If True, calculates and return only the rank
        
        TODO:
        -----
        consider_frozen (bool, optional, default=True)
            Not implemented yet! Currently it does not consider frozen orbitals
            Whether or not frozen orbitals are considered when
            assigning holes
        
        Returns:
        --------
        rank, (alpha holes, alpha particles), (beta holes, beta particles)
        """
        if ref is None:
            ref = self.ref_det
        rank_a, holes_a, particles_a = _compare_strings(ref.alpha_occ,
                                                        det.alpha_occ)
        rank_b, holes_b, particles_b = _compare_strings(ref.beta_occ,
                                                        det.beta_occ)
        return (rank_a + rank_b,
                (np.array(holes_a, dtype=int_dtype),
                 np.array(particles_a, dtype=int_dtype)),
                (np.array(holes_b, dtype=int_dtype),
                 np.array(particles_b, dtype=int_dtype)))
    
    def normalise(self, mode='unit'):
        """Normalise the wave function
        
        Parameters:
        -----------
        mode (str, optional, default='unit')
            How the wave function is normalised.
            'unit' normalises to unit.
            'intermediate' put in the intermediate normalisation
                (with respect to .ref_det)
        """
        cdef int i, j
        if mode == 'unit':
            S = norm(self._coefficients)
        elif mode == 'intermediate':
            S = self.C0
        for i in range(self._n_alpha_str):
            for j in range(self._n_beta_str):
                self._coefficients[i, j] /= S
        self.set_coeff_ref_det()
    
    def set_ref_det_from_corr_orb(self):
        ref_alpha = []
        ref_beta = []
        for irrep in self.spirrep_blocks(restricted=True):
            for i in range(self.orbspace.corr[irrep]):
                ref_alpha.append(i + self.orbspace.orbs_before[irrep])
            for i in range(self.orbspace.corr[irrep + self.n_irrep]):
                ref_beta.append(i + self.orbspace.orbs_before[irrep])
        self.ref_det = SlaterDet(
            c=0.0,
            alpha_occ=np.array(ref_alpha, dtype=int_dtype),
            beta_occ=np.array(ref_beta, dtype=int_dtype))
        self.set_coeff_ref_det()
    
    def set_coeff_ref_det(self):
        ia, ib = self.index(self.ref_det)
        self.ref_det = SlaterDet(
            c=self[ia, ib],
            alpha_occ=self.ref_det.alpha_occ,
            beta_occ=self.ref_det.beta_occ)
    
    @classmethod
    def similar_to(cls, wf, restricted=None):
        """Construct a FCIWaveFunction with same basic attributes as wf"""
        new_wf = super().similar_to(wf, restricted=restricted)
        new_wf.initialize_coeff_matrix()
        new_wf.set_ref_det_from_corr_orb()
        return new_wf

    @classmethod
    def from_interm_norm(cls, wf, restricted=None):
        """Construct the wave function from wf in intermediate normalisation
        
        This constructor returns an instance of cls that represents
        the same wave function as wf, but as a FCI-like wave function
        
        Parameters:
        -----------
        wf (IntermNormWaveFunction)
            The input wave function
        
        restricted (bool or None, optional, default=None)
            See WaveFunction.get_parameters_from
        
        """
        new_wf = cls.similar_to(wf, restricted=restricted)
        new_wf.wf_type = wf.wf_type + ' as FCI'
        new_wf.source = wf.source
        new_wf.get_coefficients_from_interm_norm_wf(wf)
        return new_wf
    
    @classmethod
    def from_Molpro_FCI(cls, molpro_output,
                        start_line_number=1,
                        point_group=None,
                        state='1.1',
                        zero_coefficients=False):
        """Construct a FCI wave function from an Molpro output
        
        This is a class constructor wrapper for get_coefficients.
        See its documentation for the details.
        """
        new_wf = cls()
        new_wf.get_coeff_from_molpro(molpro_output,
                                     start_line_number=start_line_number,
                                     point_group=point_group,
                                     state=state,
                                     zero_coefficients=zero_coefficients)
        return new_wf
    
    def get_coefficients_from_interm_norm_wf(self, IntermNormWaveFunction wf):
        """Get coefficients from a wave function in the int. normalisation
        
        Parameters:
        -----------
        wf (IntermNormWaveFunction)
            wave function in intermediate normalisation
        
        TODO:
        -----
        If self.restricted, the code can be improved to exploit the
        fact that _coefficients is symmetric (right??)
        """
        _to_print = (-1, -1)
        level = 'SD' if 'SD' in wf.wf_type else 'D'
        wf_type = 'CC' if 'CC' in wf.wf_type else 'CI'
        for ia, ib, det in self.enumerate():
            if not self.symmetry_allowed(det):
                continue
            rank, alpha_hp, beta_hp = self.get_exc_info(det)
            if rank == 0:
                self._coefficients[ia, ib] = 1.0
            elif ((rank == 1 and level == 'SD')
                   or (rank == 2 and (level == 'D' or wf_type == 'CI'))):
                self._coefficients[ia, ib] = wf[rank, alpha_hp, beta_hp]
            elif wf_type == 'CC' and (level == 'SD' or rank % 2 == 0):
                decomposition = cluster_decompose(
                    alpha_hp, beta_hp, self.ref_det,
                    mode=level, recipes_f=None)
                if (ia,ib) == _to_print:
                    print(str_dec(decomposition))
                self._coefficients[ia, ib] = 0.0
                for d in decomposition:
                    if (ia,ib) == _to_print:
                        print('------ new!!')
                    new_contribution = d[0]
                    add_contr = True
                    for cluster_det in d[1:]:
                        if not self.symmetry_allowed(cluster_det):
                            if (ia,ib) == _to_print:
                                print('------ not all')
                            add_contr = False
                            break
                        rank, alpha_hp, beta_hp = self.get_exc_info(
                            cluster_det)
                        new_contribution *= wf[rank, alpha_hp, beta_hp]
                        if (ia,ib) == _to_print:
                            print(alpha_hp, beta_hp, wf[rank, alpha_hp, beta_hp])
                    if add_contr:
                        self._coefficients[ia, ib] -= new_contribution
        self.set_coeff_ref_det()
    
    def get_coeff_from_molpro(self, molpro_output,
                              start_line_number=1,
                              point_group=None,
                              state='1.1',
                              zero_coefficients=False):
        """Read coefficients from molpro_output but keep the structure.
        
        Parameters:
        -----------
        molpro_output (str or file object)
            the molpro output with the FCI wave function.
            If it is a string, opens and closes the file.
            If it is a file object, it is assumed that it is in
            the beginning of a FCI program of Molpro.
            It iterates over the lines,
            stops when the wave function is completely loaded,
            and do not closes the file.
        
        start_line_number (int, optional, default=1)
            line number where file starts to be read.
        
        point_group (str, optional, default=None)
            the point group.
            It is mandatory in case a file object has been
            passed in molpro_output.
        
        state (str, optional, default = '1.1')
            state of interest
        
        zero_coefficients (bool, optional, default=False)
            If False, set coefficients of new determinants to zero
                
        TODO:
        -----
        If use_structure, should we compare old attributes to check
        compatibility?
        """
        cdef int i, j
        FCI_coefficients_found = False
        uhf_alpha_was_read = False
        found_orbital_source = False
        self.wf_type = 'FCI'
        if isinstance(molpro_output, str):
            f = open(molpro_output, 'r')
            f_name = molpro_output
            FCI_prog_found = False
        else:
            f = molpro_output
            f_name = f.name
            FCI_prog_found = True
            if point_group is None:
                raise ValueError(
                    'point_group is mandatory when molpro_output'
                    + ' is a file object')
            self.point_group = point_group
            self.orbspace.set_n_irrep(self.n_irrep)
        self.source = 'From file ' + f_name
        S = 0.0
        first_determinant = True
        unknown_ref_det = True
        for line_number, line in enumerate(f, start=start_line_number):
            if not FCI_prog_found:
                try:
                    self.point_group = molpro.get_point_group_from_line(
                        line, line_number, f_name)
                    self.orbspace.set_n_irrep(self.n_irrep)
                except molpro.MolproLineHasNoPointGroup:
                    if molpro.FCI_header == line:
                        FCI_prog_found = True
                continue
            if FCI_coefficients_found:
                if 'CI Vector' in line:
                    continue
                if 'EOF' in line:
                    break
                det = get_slater_det_from_fci_line(
                    line, self.Ms, self.orbspace.froz,
                    molpro_output=molpro_output,
                    line_number=line_number)
                if first_determinant:
                    first_determinant = False
                    # Improve!! do not use old version
                    # This definition of ref_orb seem to be
                    # quite bad, because the first slater determinant
                    # might not have the same ref_orb (per spirrep) as
                    # the True reference determinant... I think that this
                    # is not a real problem
                    self.orbspace.set_ref(OrbitalSpace(
                        dim=list(map(len, get_occ_from_FCI_line(
                            line, self.orbspace.full, self.orbspace.froz,
                            self.n_irrep, self.Ms)))))
                    self.initialize_coeff_matrix()
                S += det.c**2
                self.set_slater_det(det)
                if unknown_ref_det or abs(det.c) > abs(self.ref_det.c):
                    unknown_ref_det = False
                    self.ref_det = det
                    line_ref = line
            else:
                if ('FCI STATE  ' + state + ' Energy' in line
                        and 'Energy' in line):
                    FCI_coefficients_found = True
                elif 'Frozen orbitals:' in line:
                    self.orbspace.set_froz(molpro.get_orb_info(line, line_number,
                                                               self.n_irrep,
                                                               'R'))
                elif 'Active orbitals:' in line:
                    self.orbspace.set_full(self.orbspace.froz
                                           + molpro.get_orb_info(
                                               line,
                                               line_number,
                                               self.n_irrep,
                                               'R'))
                elif 'Active electrons:' in line:
                    active_el_in_out = int(line.split()[2])
                elif 'Spin quantum number:' in line:
                    self.Ms = float(line.split()[3])
                elif 'Molecular orbitals read from record' in line:
                    if 'RHF' in line:
                        self.restricted = True
                        found_orbital_source = True
                    elif 'UHF/ALPHA' in line:
                        self.restricted = False
                        uhf_alpha_was_read = True
                    elif 'UHF/BETA' in line:
                        if not uhf_alpha_was_read:
                            raise molpro.MolproInputError(
                                "Not sure what to do...\n"
                                + "UHF/BETA orbitals but no UHF/ORBITALS!!",
                                line=line,
                                line_number=line_number,
                                file_name=molpro_output)
                        found_orbital_source = True
                    else:
                        raise molpro.MolproInputError(
                            "Not sure how to treat a wf with these orbitals!\n"
                            + "Neither RHF nor UHF!",
                            line=line,
                            line_number=line_number,
                            file_name=molpro_output)
        if isinstance(molpro_output, str):
            f.close()
        self.orbspace.set_ref(OrbitalSpace(
            dim=list(map(len, get_occ_from_FCI_line(
                line_ref, self.orbspace.full, self.orbspace.froz,
                self.n_irrep, self.Ms)))))
        if self.ref_det.c < 0:
            for i in range(self.n_alpha_str):
                for j in range(self.n_beta_str):
                    self._coefficients[i,j] *= -1
            self.ref_det = SlaterDet(c=-self.ref_det.c,
                                     alpha_occ=self.ref_det.alpha_occ,
                                     beta_occ=self.ref_det.beta_occ)
        if not found_orbital_source:
            raise molpro.MolproInputError(
                'I didnt find the source of molecular orbitals!')
        if abs(self.Ms) > 0.001:
            self.restricted = False
        logger.info('norm of FCI wave function: %f', math.sqrt(S))
        self.orbspace.set_act(OrbitalSpace(n_irrep=self.n_irrep,
                                           orb_type='A'))
        if active_el_in_out + len(self.orbspace.froz) != self.n_elec:
            raise ValueError('Inconsistency in number of electrons:\n'
                             + 'n frozen el = ' + str(self.orbspace.froz)
                             + '; n act el (Molpro output) = '
                             + str(active_el_in_out)
                             + '; n elec = ' + str(self.n_elec))
    
    def get_trans_max_coef(self):
        """
        Return U that the determinant with largest coefficient as the ref
        """
        raise NotImplementedError('do it')
        det_max_coef = None
        for det in self:
            if det_max_coef is None or abs(det.c) > abs(det_max_coef.c):
                det_max_coef = det
        U = []
        for spirrep in self.spirrep_blocks():
            U.append(np.identity(self.orbspace.full[spirrep]))
            extra_in_det = []
            miss_in_det = []
            for orb in det_max_coef.occupation[spirrep]:
                if orb not in range(self.orbspace.ref[spirrep]):
                    extra_in_det.append(orb)
            for orb in range(self.orbspace.ref[spirrep]):
                if orb not in det_max_coef.occupation[spirrep]:
                    miss_in_det.append(orb)
            for p, q in zip(extra_in_det, miss_in_det):
                U[spirrep][p, p] = 0.0
                U[spirrep][q, q] = 0.0
                U[spirrep][p, q] = 1.0
                U[spirrep][q, p] = 1.0
        logger.debug('U:\n%r', U)
        return U

    def make_Jac_Hess_overlap(self, restricted=None, analytic=True, eps=0.001):
        r"""Construct the Jacobian and the Hessian of the function overlap.
        
        The function overlap is f(x) = <self|0> = C0 where |0> is the
        reference determinant of |self>, assumed to be the first.
        Thus, it is the overlap with a determinant |0>.
        
        The Slater determinants are parametrised by the orbital rotations:
        
        |0'> = exp(-K) |0>
        
        where K = \sum_{irrep} K(irrep).
        
        For restricted calculation:
        
        K(irrep) = sum_{i,a \in irrep} K(irrep)_i^a (E_{ia} - E_{ai})
        
        and for unrestricted calculation:
        
        K(irrep) = sum_{i,a \in irrep, alpha}
                        K(irrep,alpha)_i^a (a_i^a - a_a^i)
                   + sum_{i,a \in irrep, beta}
                         K(irrep,beta)_i^a (b_i^a - b_a^i)
        
        where a_p^q and b_p^q are excitation operators of alpha and beta
        orbitals, respectivelly, and E_{pq} = a_p^q + b_p^q are the singlet
        excitation operators.
        
        In each irrep, the elements K(irrep) (or, in each spirrep, the
        elements of K(irrep,sigma)) are:
        
        K_1^{n+1}    K_1^{n+2}   ...  K_1^orb_dim
        K_2^{n+1}    K_2^{n+2}   ...  K_2^orb_dim
          ...
          ...        K_i^a
          ...
        K_n^{n+1}    K_n^{n+2}   ...  K_n^orb_dim
        
        The Jacobian and Hessian are packed using the C order (row-major):
        
        K_1^{n+1}       -> Jac[0]
        K_1^{n+2}       -> Jac[1]
        ...
        K_1^{orb_dim}   -> Jac[virt_orb - 1 = orb_dim-ref_orb-1]
        K_2^{n+1}       -> Jac[virt_orb + 0]
        K_2^{n+2}       -> Jac[virt_orb + 1]
        ...
        K_2^{orb_dim}   -> Jac[2*virt_orb-1]
        ...
        K_i^a           -> Jac[(i-1) * virt_orb + (a-1-virt_orb)]
        ...
        K_n^{n+1}       -> Jac[(n-1) * virt_orb + 0]
        K_n^{n+2}       -> Jac[(n-1) * virt_orb + 1]
        ...
        K_n^{orb_dim}   -> Jac[(n-1) * virt_orb + virt_orb - 1]
        
        The elements are ordered as above inside each irrep (spirrep) block.
        The irreps are then ordered sequentially. For unrestricted first come
        all alpha and then all beta.
        
        Parameters:
        -----------
        restricted (bool, optional, default=same as self
            If True, carries out a restricted calculation.
            In general, the code decides what to do, based on self.
            However, one can enforce restricted=False for a restricted
            wave function (mainly for testing).
            One cannot enforce a restricted optimisation for a unrestricted
            wave function.
        
        analytic (bool, optional, default=True)
            if True, calculates the Jacobian and the Hessian by the
            analytic expression;
            if False calculates numerically.
        
        eps (float, optional, default=0.001)
            The step size for finite differences
            calculation of the derivatives.
            It has effect for analytic=False only
        Returns:
        --------
        the tuple (Jac, Hess), with the Jacobian and the Hessian.
        See above how the elements are packed.
        
        TODO:
        -----
        Consider only correlated orbitals! (it is possible!)
        Implement the unrestricted version
        
        """
        raise NotImplementedError('do it')
        if restricted is None:
            restricted = self.restricted
        if restricted and not self.restricted:
            raise ValueError(
                'Restricted calculation needs a restricted wave function')
        logger.info('Building Jacobian and Hessian: %s procedure',
                    'Analytic' if analytic else 'Numeric')
        n_param = 0
        spirrep_start = [0]
        for spirrep in self.spirrep_blocks(restricted=restricted):
            nK = self.orbspace.ref[spirrep] * self.orbspace.virt[spirrep]
            spirrep_start.append(spirrep_start[-1] + nK)
            n_param += nK
        spirrep_start.pop()
        if restricted:
            spirrep_start *= 2
        Jac = np.zeros(n_param)
        Hess = np.zeros((n_param, n_param))
        # --- Numeric
        if not analytic:
            coef_0 = self.C0
            coef_p = np.zeros(n_param)
            coef_m = np.zeros(n_param)
            coef_pp = np.zeros((n_param, n_param))
            coef_mm = np.zeros((n_param, n_param))
            z = np.zeros(n_param)
            for i in range(n_param):
                z[i] = eps
                coef_p[i] = self.calc_wf_from_z(z, just_C0=True)[0].C0
                for j in range(n_param):
                    z[j] += eps
                    coef_pp[i][j] = self.calc_wf_from_z(z, just_C0=True)[0].C0
                    z[j] = eps if j == i else 0.0
                z[i] = -eps
                coef_m[i] = self.calc_wf_from_z(z, just_C0=True)[0].C0
                for j in range(n_param):
                    z[j] -= eps
                    coef_mm[i][j] = self.calc_wf_from_z(z, just_C0=True)[0].C0
                    z[j] = -eps if j == i else 0.0
                z[i] = 0.0
            for i in range(n_param):
                Jac[i] = (coef_p[i] - coef_m[i]) / (2 * eps)
                for j in range(n_param):
                    Hess[i, j] = (2 * coef_0
                                  + coef_pp[i, j]
                                  - coef_p[i] - coef_p[j]
                                  + coef_mm[i, j]
                                  - coef_m[i] - coef_m[j]) / (2 * eps * eps)
            return Jac, Hess
        # --- Analytic
        for det in self:
            holes, particles, rank = self.get_exc_info(det,
                                                       consider_frozen=True)
            logger.debug('new det: %s', det)
            if rank > 2:
                continue
            if rank == 0:
                for i in range(n_param):
                    Hess[i][i] -= det.c
                logger.debug('Setting diagonal elements of Hess to %f',
                             -det.c)
            elif rank == 1:
                if (restricted
                        and holes[0].spirrep >= self.n_irrep):  # beta
                    continue
                pos = spirrep_start[holes[0].spirrep]
                pos += n_from_rect(
                    holes[0].orb, particles[0].orb,
                    self.orbspace.virt[particles[0].spirrep])
                Jac[pos] += (det.c
                             if (holes[0].orb
                                 + self.orbspace.ref[holes[0].spirrep]) % 2 == 0
                             else -det.c)
                logger.debug('Adding to Jac[%d] = %f', pos, Jac[pos])
            elif rank == 2:
                if (holes[0].spirrep != particles[0].spirrep  # occ != ref_orb
                        and holes[1].spirrep != particles[1].spirrep):
                    continue
                if restricted:
                    if holes[0].spirrep >= self.n_irrep:  # beta beta
                        continue
                    if holes[1].spirrep >= self.n_irrep:  # alpha beta
                        if holes[0].spirrep + self.n_irrep > holes[1].spirrep:
                            continue
                        if holes[0].spirrep + self.n_irrep == holes[1].spirrep:
                            if holes[0].orb > holes[1].orb:
                                continue
                            if (holes[0].orb == holes[1].orb
                                    and particles[0].orb > particles[1].orb):
                                continue
                pos = spirrep_start[holes[0].spirrep]
                pos += n_from_rect(
                    holes[0].orb, particles[0].orb,
                    self.orbspace.virt[particles[0].spirrep])
                pos1 = spirrep_start[holes[1].spirrep]
                pos1 += n_from_rect(
                    holes[1].orb, particles[1].orb,
                    self.orbspace.virt[particles[1].spirrep])
                if holes[0].spirrep == holes[1].spirrep:
                    negative = (holes[0].orb
                                + holes[1].orb) % 2 == 0
                else:
                    negative = (holes[0].orb
                                + holes[1].orb
                                + self.orbspace.ref[holes[0].spirrep]
                                + self.orbspace.ref[holes[1].spirrep]) % 2 == 1
                Hess[pos, pos1] += -det.c if negative else det.c
                if pos != pos1:
                    Hess[pos1, pos] += -det.c if negative else det.c
                logger.debug('Adding to Hess[%d,%d] = %f',
                             pos, pos1, -det.c if negative else det.c)
                if holes[0].spirrep == holes[1].spirrep:
                    pos = spirrep_start[holes[0].spirrep]
                    pos += n_from_rect(
                        holes[0].orb, particles[1].orb,
                        self.orbspace.virt[particles[1].spirrep])
                    pos1 = spirrep_start[holes[1].spirrep]
                    pos1 += n_from_rect(
                        holes[1].orb, particles[0].orb,
                        self.orbspace.virt[particles[0].spirrep])
                    negative = not negative
                    Hess[pos, pos1] += -det.c if negative else det.c
                    if pos1 != pos:
                        Hess[pos1, pos] += -det.c if negative else det.c
                    logger.debug('Adding to Hess[%d,%d] = %f',
                                 pos, pos1, Hess[pos, pos1])
        if restricted:
            Jac *= 2
            Hess *= 2
        return Jac, Hess
    
    def calc_wf_from_z(self, z, just_C0=False):
        """Calculate the wave function in a new orbital basis
        
        A new (representation of the) wave function self is constructed
        in a orbital basis that has been modified by a step z.
        The transformation of the orbital basis is obtained using the
        exponential parametrisation.
        
        
        Paramters:
        ----------
        
        z (1D np array)
            The update in the orbital basis (given in the space of the K_i^a
            parameters) from the position z=0 (that is, the orbital basis used
            to construct the current representation of the wave function).
            The parameters for all spin and irreps are stored in the same
            array,packed as the Jacobian (see make_Jac_Hess_overlap).
            For a Newton step, this is Hess^-1 @ Jac (note that the absence of
            minus, that will be considered here)
        
        just_C0 (bool, optional, default=False)
            Calculates only the first coefficient (see change_orb_basis)
        
        Returns:
        --------
        The tuple (new_wf, U) where new_wf the wave function in the new basis
        and U is the transformation from the previous to the new orbital basis.
        """
        U = calc_U_from_z(z, self)
        logger.info('Matrices U have been calculated.'
                    + ' Calculating transformed wf now.')
        return self.change_orb_basis(U, just_C0=just_C0), U
    
    def change_orb_basis(self, U, just_C0=False,
                         method='traditional'):
        r"""Transform the representation of self to a new orbital basis
        
        Parameters:
        -----------
        U (list of np arrays)
            The orbital transformation, stored per spirrep
            If len(U) == self.n_irrep, assumes a restricted transformation,
            and uses same transformation matrix for both alpha and beta.
            Otherwise len(U) must be 2*self.n_irrep, and alpha and beta
            transformations are independent.
        
        just_C0 (bool, optional, default=False)
            If True, calculates the coefficients of the initial determinant
            only
        
        method (str, optional, default='traditional')
            Selects the method to carry out the transformation.
            'traditional' uses the standart brute force way, based on
            calculating the minors IJ for the transformation matrix
            'Malmqvist' uses the procedure by Malmqvist, at
            Int. J. Quantum Chem. XXX, 479 (1986). Also described in
            "the Bible", end of the chapter about CI.
        
        Behaviour:
        ----------
        
        If the coefficients of self are given in the basis |u_I>:
        
        |self> = \sum_I c_I |u_I>
        
        it calculates the wave function in the basis |v_I>:
        
        |self> = \sum_I d_I |v_I>
        
        where the arrays at U are the matrix transformations of the MO
        from the basis |v_I> to the basis |u_I>:
        
        |MO of (u)> = |MO of (v)> U
        
        Return:
        -------
        The transformed wave function, as instace of NormCI_WaveFunction.
        Optionally, only C0 is calculated, and the final wave function has
        only one determinant (associated to C0)
        """
        if method == 'traditional':
            return self._change_orb_basis_traditional(U, just_C0=just_C0)
        elif method == 'Malmqvist':
            return self._change_orb_basis_Malmqvist(U, just_C0=just_C0)
        else:
            raise ValueError('Unknown method for change_orb_basis: '
                             + method)
    
    def _change_orb_basis_traditional(self, U, just_C0=False):
        raise NotImplementedError('do it')
        new_wf = FCIWaveFunction()
        new_wf.restricted = self.restricted
        new_wf.point_group = self.point_group
        new_wf.Ms = self.Ms
        new_wf.orbspace.get_attributes_from(self)
        new_wf.wf_type = self.wf_type
        new_wf.source = (self.source.replace(' (another basis)', '')
                         + ' (another basis)')
        n_calcs = 0
        if len(U) == self.n_irrep:
            restricted = True
        elif len(U) == 2 * self.n_irrep:
            restricted = False
        else:
            raise ValueError('len(U) = ' + str(len(U))
                             + ' is not compatible to n_irrep = '
                             + str(self.n_irrep))
        for det_J in self:
            new_occ = det_J.occupation
            new_c = 0.0
            logger.debug('====== Starting det %s', new_occ)
            U_J = []
            for spirrep, U_spirrep in enumerate(2 * U
                                                if restricted else
                                                U):
                U_J.append(U_spirrep[:, det_J.occupation[spirrep]])
            len_det_J = list(map(len, det_J.occupation))
            for det_I in self:
                if len_det_J != list(map(len, det_I.occupation)):
                    continue
                if abs(det_I.c) > 1.0E-11:
                    n_calcs += 1
                    C_times_det_minor_IJ = det_I.c
                    for spirrep in self.spirrep_blocks(restricted=False):
                        C_times_det_minor_IJ *= linalg.det(
                            U_J[spirrep][det_I.occupation[spirrep], :])
                    new_c += C_times_det_minor_IJ
                    logger.debug(
                        'New contribution = %f\n det_J:\n%s\n det_I:\n%s',
                        C_times_det_minor_IJ, det_J, det_I)
            new_wf._all_determinants.append(SlaterDet(c=new_c,
                                                      occupation=new_occ))
            if just_C0:
                break
        new_wf._set_memory()
        logger.info('Number of det calculations: %d', n_calcs)
        return new_wf

    def _change_orb_basis_Malmqvist(self, U, just_C0=False):
        raise NotImplementedError('Not yet done...')
        new_wf = copy.copy(self)
        n_calcs = 0
        if logger.level <= logging.DEBUG:
            logger.debug('WF:\n%s', str(self))
        tUa, tLa = lu(Ua, permute_l=True) ### CHECK
        tUb, tLb = lu(Ub, permute_l=True) ### CHECK
        tUa = linalg.inv(tUa)
        tUb = linalg.inv(tUb)
        tLa = np.identity(len(tUa)) - tLa
        tLb = np.identity(len(tUa)) - tLb
        ta = tUa + tLa
        tb = tUb + tLb
        for k_tot in range(2*(self.orbspace.full - self.n_frozen)):
            coeff_delta = []
            if k_tot < self.orbspace.full - self.n_frozen:
                k = k_tot
                spin_shift = 0
                t = ta
                logger.debug('k in alpha')
            else:
                k = k_tot -(self.orbspace.full - self.n_frozen)
                spin_shift = new_wf.n_alpha
                t = tb
                logger.debug('k in beta')
                logger.debug('k = %d; spin_shift = %d', k, spin_shift)
            for det_J in new_wf:
                c_delta = 0.0
                for det_I in new_wf:
                    if abs(det_I.c)>1.0E-11:
                        n_diff = 0
                        p = None
                        if not (k in det_I.occupation):#....[1+spin_shift:1+spin_shift+new_wf.n_alpha]):
                            continue
                        for i_ind,i in enumerate(det_J.occupation):
                            if i_ind < new_wf.n_alpha:
                                if not i in det_I.occupation:#...[1:new_wf.n_alpha+1]:
                                    logger.debug('detI = ', str(det_I))
                                    logger.debug('detJ = ', str(det_J))
                                    n_diff += 1
                                    if not spin_shift:
                                        p = i-1
                                        logger('Current p (alpha): %d', p)
                            else:
                                if not i in det_I.occupation:#...[new_wf.n_alpha+1:]:
                                    n_diff += 1
                                    if spin_shift:
                                        p = i - (self.orbspace.full - self.n_frozen) -1
                                        logger('Current p (beta): %d', p)
                        if n_diff > 1:
                            continue
                        if n_diff == 0:
                            c_delta += det_I.c*(t[k][k]-1)
                        else:
                            if k in det_J.occupation:#...[1+spin_shift:1+spin_shift+new_wf.n_alpha]:
                                continue
    #                        print(p)
    #                        try:
                            c_delta += det_I.c * t[p][k]
    #                        except:
    #                            print(p,k)
    #                            raise
                coeff_delta.append(c_delta)
            for det_J, c_delta in zip(new_wf, coeff_delta):
                det_J.c += c_delta
        logger.info('Number of det calculations: %d', n_calcs)
        return new_wf

    def string_indices(self,
                       spirrep=None,
                       coupled_to=None,
                       no_occ_orb=False,
                       only_ref_orb=False,
                       only_this_occ=None):
        raise NotImplementedError('undone')
    
    def _extract_S_from_D(self, recipes_f=None):
        """Extract the contribution of singles from the doubles
        
        Changes the coefficients of the double excitations
        by its decomposition into singles:
        
        new_c_{ij}^{ab} = c_{ij}^{ab}
                          - (c_{i}^{a}c_{j}^{b} + c_{i}^{b}c_{j}^{a})
        
        Side Effect:
        ------------
        The coefficients of all double excitations are changed!
        
        TODO:
        -----
        Implement an option that creates another coefficients matrix
        instead of changing this matrix.
        (or implement a referse function)
        
        """
        for ia, ib, det in self.enumerate():
            rank, alpha_hp, beta_hp = self.get_exc_info(det)
            if rank == 2:
                decomposition = cluster_decompose(
                    alpha_hp, beta_hp, self.ref_det,
                    mode='SD', recipes_f=recipes_f)
                for d in decomposition[1:]:
                    singles_contr = d[0]
                    for cluster_det in d[1:]:
                        singles_contr *= self[self.index(cluster_det)]
                    self._coefficients[ia, ib] += singles_contr
                    
    def _restore_S_to_D(self, recipes_f=None):
        """The reverse of _extract_S_from_D"""
        raise NotImplementedError('Do it')
                    
    def _coeff_as_order_relative_to_ref(self):
        """Change the coefficients as if the orbitals were relative to ref_det
        
        Because of symmetry, the reference Slater determinant might not be
        that with all occupied orbitals first. This causes some sign problems
        in the cluster decomposition of the wave function, and thus this
        subroutine changes the sign of the determinants as if the orbitals
        were reordered with all occupied orbitals in the reference coming
        first. However, the order of the orbitals (and thus of the alpha
        and beta strings) are not really changed!
        If the wave function is to be used again after decompose, this method
        has to be called again to change back the signs!
        
        See strings_rev_lexical_order.sign_relative_to_ref
        
        """
        for ia, ib, det in self.enumerate():
            rank, alpha_hp, beta_hp = self.get_exc_info(det)
            self._coefficients[ia, ib] *= (
                str_order.sign_relative_to_ref(alpha_hp[0],
                                               alpha_hp[1],
                                               self.ref_det.alpha_occ)
                * str_order.sign_relative_to_ref(beta_hp[0],
                                                 beta_hp[1],
                                                 self.ref_det.beta_occ))
    
    def dist_to(self, other,
                metric='IN',
                normalise=True):
        """Distance between self and other
        
        Parameters:
        -----------
        other (FCIWaveFunction)
            The wave function to which the distance will be calculated
        
        metric (str, optional, default='IN')
            metric to calculate the distance.
            Possible values are:
            'IN' - Intermediate normalisation
            'overlap' - The overlap between self and other
                        (that is not exactly the distance, but related to)
        
        normalise (bool, optional, default=False)
            If True, normalise the wave function in a compatible way
            with the metric option
        
        Return:
        -------
        A float, with the distance
        
        Side Effect:
        ------------
        If normalise is True, the wave function is changed!
        
        """
        if normalise:
            if metric == 'IN':
                self.normalise(mode='intermediate')
                other.normalise(mode='intermediate')
            else:
                self.normalise(mode='unit')
                other.normalise(mode='unit')
        if metric == 'IN':
            return str_order.eucl_distance(self._coefficients,
                                           other._coefficients)
    
    def symmetry_allowed(self, det):
        """Return True if this determinant is symmetry allowed"""
        total_irrep = 0
        for p in det.alpha_occ:
            total_irrep = irrep_product[total_irrep, self.get_orb_irrep(p)]
        for p in det.beta_occ:
            total_irrep = irrep_product[total_irrep, self.get_orb_irrep(p)]
        return total_irrep == self.irrep
    
