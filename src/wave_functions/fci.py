"""A CI like wave function, normalised to unity.

The class defined here stores all determinants explicitly
and contains the methods for the dGr optimisation that uses
the algorithm based on orbital rotations

History:
    Aug 2018 - Start
    Mar 2019 - Add CISD wave function
               Add to git
Yuri
"""
import logging
import numpy as np
from numpy import linalg
from scipy.linalg import lu
import copy
import math
from collections import namedtuple

from util import get_pos_from_rectangular
from wave_functions import general
import orbitals
import molpro_util
import memory

logger = logging.getLogger(__name__)


class Slater_Det(namedtuple('Slater_Det',
                            ['c',
                             'occupation'])):
    """A namedtuple for a generic Slater_Det of the wave function
    
    Attributes:
    -----------
    c (float)
        The coefficient of the Slater determinant in the wave function
    
    occupation (list of np.arrays of int)
        The occupied orbitals for each spirrep
    """
    __slots__ = ()
    
    def __str__(self):
        spirreps = []
        for occ in self.occupation:
            spirreps.append(str(occ))
        return ('Slater_Det: c = {0:15.12f} ; '.
                format(self.c) + '^'.join(spirreps))


Orbital_Info = namedtuple('Orbital_Info', ['orb', 'spirrep'])


def _get_Slater_Det_from_String_Index(Index, n_irrep,
                                      zero_coefficients=False):
    """return the Slater_Det corresponding to String_Index Index"""
    coeff = Index.C
    final_occ = [np.array([], dtype=np.int8) for i in range(2 * n_irrep)]
    for spirrep, spirrep_I in enumerate(Index):
        if spirrep != spirrep_I.spirrep:
            raise Exception(
                'spirrep of spirrep_I is not consistent with position in list')
        final_occ[spirrep_I.spirrep] = np.array(spirrep_I.occ_orb)
    return Slater_Det(c=0.0 if zero_coefficients else coeff,
                      occupation=final_occ)


def _get_Slater_Det_from_FCI_line(l, orb_dim, n_core, n_irrep, Ms,
                                  molpro_output='', line_number=-1,
                                  zero_coefficients=False):
    """Read a FCI configuration from Molpro output and return a Slater Determinant
    
    Parameters:
    -----------
    l (str)
        The line with a configuration, from the FCI program in Molpro
        to be converted to a Slater Determinant.
    
    orb_dim (Orbitals_Sets)
        Dimension of orbital space
    
    n_core (Orbitals_Sets)
        Dimension of frozen orbitals space
    
    n_irrep (int)
        Number of irreps
    
    Ms (float)
        Ms of total wave function (n_alpha - n_beta)/2
    
    molpro_output (str, optional, default='')
        The output file name (only for error message)
    
    line_number (int, optional, default=-1)
        The line number in Molpro output
    
    zero_coefficients (bool, optional, default=False)
        If True, the coefficient is set to zero and the value in l is discarded
    
    Returns:
    --------
    A Slater_Det
    
    Raises:
    molpro_util.MolproInputError
    
    Examples:
    ---------
    
    # if n_irrep = 4, orb_dim = (6,2,2,0), n_core = (0,0,0,0) then
    
    -0.162676901257  1  2  7  1  2  7
    gives
    c=-0.162676901257; occupation=[(0,1) (0) () () (0,1) (0) () ()]
    
    -0.049624632911  1  2  4  1  2  6
    gives
    c=-0.049624632911; occupation=[(0,1,3) () () () (0,1,5) () () ()]
    
    0.000000000000  1  2  9  1  2 10
    gives
    c=-0.000000000000; occupation=[(0,1) () (0) () (0,1) () (1) ()]
    
    # but if n_core = (1,1,0,0) then the above cases give
         (because frozen electrons are indexed first in Molpro convention)
    
    c=-0.162676901257; occupation=[(0,5) (0) () () (0,5) (0) () ()]
    
    c=-0.049624632911; occupation=[(0,2) (0) () () (0,4) (0) () ()]
    
    c=-0.000000000000; occupation=[(0) (0) (0) () (0) (0) (1) ()]


    """
    lspl = l.split()
    final_occ = [list(range(n_core[irp])) for irp in range(2 * n_irrep)]
    n_tot_core = sum(map(len, final_occ)) // 2
    try:
        coeff = float(lspl[0])
        occ = [int(x) - 1 for x in lspl[1:] if int(x) > n_tot_core]
    except Exception as e:
        raise molpro_util.MolproInputError(
            "Error when reading FCI configuration. Exception was:\n"
            + str(e),
            line=l,
            line_number=line_number,
            file_name=molpro_output)
    if len(occ) + 2 * n_tot_core + 1 != len(lspl):
        raise molpro_util.MolproInputError(
            "Inconsistency in number of core orbitals for FCI. n_core:\n"
            + str(n_core),
            line=l,
            line_number=line_number,
            file_name=molpro_output)
    total_orbs = [sum(n_core[irp] for irp in range(n_irrep))]
    for i in range(n_irrep):
        total_orbs.append(total_orbs[-1]
                          + orb_dim[i] - n_core[i])
    irrep = irrep_shift = 0
    ini_beta = (len(occ) + int(2 * Ms)) // 2
    for i, orb in enumerate(occ):
        if i == ini_beta:
            irrep_shift = n_irrep
            irrep = 0
        while True:
            if irrep == n_irrep:
                raise molpro_util.MolproInputError(
                    'Configuration is not consistent with orb_dim = '
                    + str(orb_dim),
                    line=l,
                    line_number=line_number,
                    file_name=molpro_output)
            if total_orbs[irrep] <= orb < total_orbs[irrep + 1]:
                final_occ[irrep + irrep_shift].append(orb - total_orbs[irrep]
                                                      + n_core[irrep])
                break
            else:
                irrep += 1
    for i, o in enumerate(final_occ):
        final_occ[i] = np.array(o, dtype=int)
    return Slater_Det(c=0.0 if zero_coefficients else coeff,
                      occupation=final_occ)


class Wave_Function_Norm_CI(general.Wave_Function):
    """A normalised CI-like wave function, with explicit determinants
    
    The wave function is stored explicitly, with all Slater determinants
    (as instances of Slater_Det).
    
    Atributes:
    ----------
    has_FCI_structure (bool)
        If True, it contains all possible Slater determinants
        (of that Ms and irrep), even those that have zero coefficient
        in the wave function.
    
    Data Model:
    -----------
    len
        The number of determinants
    
    [int]
        Get corresponding determinant
    
    iterable
        Iterates over determinants
    
    """
    tol_eq = 1E-8
    
    def __init__(self):
        """Initialise the wave function"""
        super().__init__()
        self._all_determinants = []
        self.has_FCI_structure = False
        self._i_ref = None
    
    def __len__(self):
        return len(self._all_determinants)
    
    def __getitem__(self, i):
        return self._all_determinants[i]
    
    def __iter__(self):
        """Generator for determinants"""
        for det in self._all_determinants:
            yield det
        
    def __str__(self):
        """Return a string version of the wave function."""
        return ("Wave function: "
                + str(self._all_determinants[0].c)
                + ' + |' + str(self.ref_occ) + '> ...')
    
    def __eq__(self, other):
        """Checks if both wave functions are the same within tol
        TODO: compare all attributes
        """
        for i, det in enumerate(self):
            if (det.occupation != other[i].occupation
                    or abs(det.c - other[i].c) > self.tol_eq):
                return False
        return True
        
    def __repr__(self):
        x = []
        for det in self:
            x.append(str(det)
                     + ' >> RANK: {0:d}'.format(
                         self.get_exc_info(det, only_rank=True)))
        x.append('-' * 50)
        return ('<CI-like Wave Function normalised to unity>\n'
                + super().__repr__() + '\n'
                + '\n'.join(x))
    
    def calc_memory(self):
        """Calculate memory of current determinants in wave function"""
        return memory.convert((self.n_elec + 8) * len(self),
                              'B', memory.unit())
    
    @classmethod
    def from_int_norm(cls, wf_intN):
        """Construct the wave function from wf_intN
        
        wf_intN is an wave function in the intermediate normalisation
        
        TODO: renaming: this and related function can be used for any
              wf_intN that have sting_indices properly implemented.
        """
        new_wf = cls()
        new_wf.get_coeff_from_int_norm_WF(wf_intN,
                                          change_structure=True,
                                          use_structure=False)
        return new_wf
    
    @classmethod
    def from_Molpro_FCI(cls, molpro_output,
                        start_line_number=1,
                        point_group=None,
                        state='1.1',
                        zero_coefficients=False,
                        change_structure=True,
                        use_structure=False):
        """Construct a FCI wave function from an Molpro output
        
        This is a class constructor wrapper for get_coefficients.
        See its documentation for the details.
        """
        new_wf = cls()
        new_wf.get_coeff_from_molpro(molpro_output,
                                     start_line_number=start_line_number,
                                     point_group=point_group,
                                     state=state,
                                     zero_coefficients=zero_coefficients,
                                     change_structure=change_structure,
                                     use_structure=use_structure)
        return new_wf
    
    def index(self, element, check_coeff=False):
        """Return index of element
        
        Parameters:
        -----------
        element (Slater_Det)
            the Slater_Det whose index must be found
        
        check_coeff (bool, optional, default=False)
            If True, return the index only if the coefficient
            is the same
        
        Return:
        -------
        the index of element, if in self
        
        Raises
        ------
        ValueError if element is not in self
        
        """
        for idet, det in enumerate(self):
            found_det = True
            for occ1, occ2 in zip(det.occupation,
                                  element.occupation):
                if (len(occ1) != len(occ2)
                        or not np.all(occ1 == occ2)):
                    found_det = False
                    break
            if found_det:
                if (not check_coeff
                        or abs(self[idet].c - element.c) < self.tol_eq):
                    return idet
                raise ValueError(
                    'Found occupation, but coefficient is different')
        raise ValueError('Occupation not found.')
    
    def get_coeff_from_int_norm_WF(self, intN_wf,
                                   change_structure=True,
                                   use_structure=False):
        """Get coefficients from a wave function with int. normalisation
        
        Parameters:
        -----------
        intN_wf (Wave_Function_Int_Norm)
            wave function in intermediate normalisation
        
        change_structure (bool, optional, default=True)
            If True, new determinants coefficients are allowed to be added
        
        use_structure (bool, optional, default=False)
            If True, uses present structure in self.
            Otherwise the eventual Slater determinants already in self
            are discarded.
        
        TODO:
        cross check attributes. The way it is all is overridden,
        what is DANGEROUS
        
        """
        self.restricted = intN_wf.restricted
        self.point_group = intN_wf.point_group
        self.Ms = intN_wf.Ms
        self.n_core = intN_wf.n_core
        self.n_act = intN_wf.n_act
        self.orb_dim = intN_wf.orb_dim
        self.ref_occ = intN_wf.ref_occ
        self.WF_type = intN_wf.WF_type
        self.source = intN_wf.source
        self._i_ref = None
        if not use_structure:
            self._all_determinants = []
        for Index in intN_wf.string_indices():
            new_Slater_Det = _get_Slater_Det_from_String_Index(Index,
                                                               self.n_irrep)
            if use_structure:
                try:
                    idet = self.index(new_Slater_Det)
                except ValueError:
                    if change_structure:
                        self._all_determinants.append(new_Slater_Det)
                else:
                    self._all_determinants[idet] = new_Slater_Det
            elif change_structure:
                self._all_determinants.append(new_Slater_Det)
        self._set_memory()

    def get_coeff_from_molpro(self, molpro_output,
                              start_line_number=1,
                              point_group=None,
                              state='1.1',
                              zero_coefficients=False,
                              change_structure=True,
                              use_structure=False):
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
        
        change_structure (bool, optional, default=True)
            If True, new determinants coefficients are allowed to be added
        
        use_structure (bool, optional, default=False)
            If True, uses present structure in self.
            Otherwise the eventual Slater determinants already in self
            are discarded.
        
        TODO:
        -----
        If use_structure, should we compare old attributes to check
        compatibility?
        """
        sgn_invert = False
        FCI_coefficients_found = False
        uhf_alpha_was_read = False
        found_orbital_source = False
        self.has_FCI_structure = True
        self.WF_type = 'FCI'
        if not use_structure:
            self._all_determinants = []
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
        self.source = 'From file ' + f_name
        S = 0.0
        for line_number, line in enumerate(f, start=start_line_number):
            if not FCI_prog_found:
                try:
                    self.point_group = molpro_util.get_point_group_from_line(
                        line, line_number, f_name)
                except molpro_util.MolproLineHasNoPointGroup:
                    if molpro_util.FCI_header == line:
                        FCI_prog_found = True
                continue
            if FCI_coefficients_found:
                if 'CI Vector' in line:
                    continue
                if 'EOF' in line:
                    break
                new_Slater_Det = _get_Slater_Det_from_FCI_line(
                    line, self.orb_dim, self.n_core, self.n_irrep, self.Ms,
                    molpro_output=molpro_output,
                    line_number=line_number,
                    zero_coefficients=zero_coefficients)
                S += new_Slater_Det.c**2
                if not zero_coefficients:
                    if len(self) == 0:
                        sgn_invert = new_Slater_Det.c < 0.0
                    if sgn_invert:
                        new_Slater_Det = Slater_Det(
                            c=-new_Slater_Det.c,
                            occupation=new_Slater_Det.occupation)
                if use_structure:
                    try:
                        idet = self.index(new_Slater_Det)
                    except ValueError:
                        if change_structure:
                            self._all_determinants.append(new_Slater_Det)
                    else:
                        self._all_determinants[idet] = new_Slater_Det
                elif change_structure:
                    self._all_determinants.append(new_Slater_Det)
            else:
                if ('FCI STATE  ' + state + ' Energy' in line
                        and 'Energy' in line):
                    FCI_coefficients_found = True
                elif 'Frozen orbitals:' in line:
                    self.n_core = molpro_util.get_orb_info(line, line_number,
                                                           self.n_irrep,
                                                           'R')
                elif 'Active orbitals:' in line:
                    self.orb_dim = (self.n_core
                                    + molpro_util.get_orb_info(
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
                            raise molpro_util.MolproInputError(
                                "Not sure what to do...\n"
                                + "UHF/BETA orbitals but no UHF/ORBITALS!!",
                                line=line,
                                line_number=line_number,
                                file_name=molpro_output)
                        found_orbital_source = True
                    else:
                        raise molpro_util.MolproInputError(
                            "Not sure how to treat a wf with these orbitals!\n"
                            + "Neither RHF nor UHF!",
                            line=line,
                            line_number=line_number,
                            file_name=molpro_output)
        if isinstance(molpro_output, str):
            f.close()
        self.get_i_max_coef(set_i_ref=True)
        self.ref_occ = general.Orbitals_Sets(
            list(map(len, self[self.i_ref].occupation)))
        if not found_orbital_source:
            raise molpro_util.MolproInputError(
                'I didnt find the source of molecular orbitals!')
        if abs(self.Ms) > 0.001:
            self.restricted = False
        logger.info('norm of FCI wave function: %f', math.sqrt(S))
        self.n_act = general.Orbitals_Sets(np.zeros(self.n_irrep),
                                           occ_type='A')
        if active_el_in_out + len(self.n_core) != self.n_elec:
            raise ValueError('Inconsistency in number of electrons:\n'
                             + 'n core el = ' + str(self.n_core)
                             + '; n act el (Molpro output) = '
                             + str(active_el_in_out)
                             + '; n elec = ' + str(self.n_elec))
        self._set_memory()
        
    @property
    def i_ref(self):
        if self._i_ref is None:
            for i, det in enumerate(self):
                if self.get_exc_info(det, only_rank=True) == 0:
                    self._i_ref = i
                    break
            if self._i_ref is None:
                raise Exception('Did not find reference for wave function!')
        return self._i_ref

    def get_i_max_coef(self, set_i_ref=False):
        """Return index of determinant with largest coefficient
        
        If set_i_ref == True, also sets i_ref to it (default=False)
        """
        max_coef = 0.0
        i_max_coef = -1
        for i, det in enumerate(self):
            if abs(det.c) > max_coef:
                max_coef = abs(det.c)
                i_max_coef = i
        self._i_ref = i_max_coef
        return i_max_coef

    @property
    def C0(self):
        return self[self.i_ref].c
    
    def get_exc_info(self, det, only_rank=False, consider_core=True):
        """Return some info about the excitation that lead from ref to det
        
        This will return the holes, the particles, and the rank of det,
        viewed as a excitation over self.ref_occ.
        Particles and holes are returned as lists of Orbital_Info.
        These lists have len = rank.
        If consider_core == False, core orbitals are not considered,
        and the first correlated orbitals is 0; Otherwise core orbitals
        are taken into account and the first correlated orbital is
        n_core[spirrep]
        The order that the holes/particles are put in the lists is the
        canonical: follows the spirrep, and the normal order inside each
        spirrep.
        Particles are numbered starting at zero, that is, the index at det
        minus self.ref_occ[spirrep].
        
        Parameters:
        -----------
        det (Slater_Det)
            The Slater determinant to be verified.
        
        only_rank (bool, optional, default=False
            If True, calculates and return only the rank
        
        consider_core (bool, optional, default=True)
            Whether or not core orbitals are considered when
            assigning holes
        
        Returns:
        --------
        The tuple (holes, particles, rank), or just the rank.
        """
        rank = 0
        holes = []
        particles = []
        for spirrep in self.spirrep_blocks(restricted=False):
            ncore = (0
                     if consider_core else
                     self.n_core[spirrep])
            if not only_rank:
                for orb in range(self.ref_occ[spirrep]):
                    if orb not in det.occupation[spirrep]:
                        holes.append(Orbital_Info(
                            orb=orb - ncore,
                            spirrep=spirrep))
            for orb in det.occupation[spirrep]:
                if orb >= self.ref_occ[spirrep]:
                    rank += 1
                    if not only_rank:
                        particles.append(
                            Orbital_Info(orb=orb
                                         - self.ref_occ[spirrep],
                                         spirrep=spirrep))
        if only_rank:
            return rank
        else:
            return holes, particles, rank

    def normalise(self):
        """Normalise the wave function to unity"""
        S = 0.0
        for det in self:
            S += det.c**2
        S = math.sqrt(S)
        for det in self:
            det.c /= S

    def get_trans_max_coef(self):
        """
        Return U that the determinant with larges coefficient as the ref
        """
        det_max_coef = None
        for det in self:
            if det_max_coef is None or abs(det.c) > abs(det_max_coef.c):
                det_max_coef = det
        U = []
        for spirrep in self.spirrep_blocks():
            U.append(np.identity(self.orb_dim[spirrep]))
            extra_in_det = []
            miss_in_det = []
            for orb in det_max_coef.occupation[spirrep]:
                if orb not in range(self.ref_occ[spirrep]):
                    extra_in_det.append(orb)
            for orb in range(self.ref_occ[spirrep]):
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
        K_1^{orb_dim}   -> Jac[n_ext - 1 = orb_dim-ref_occ-1]
        K_2^{n+1}       -> Jac[n_ext + 0]
        K_2^{n+2}       -> Jac[n_ext + 1]
        ...
        K_2^{orb_dim}   -> Jac[2*n_ext-1]
        ...
        K_i^a           -> Jac[(i-1) * n_ext + (a-1-n_ext)]
        ...
        K_n^{n+1}       -> Jac[(n-1) * n_ext + 0]
        K_n^{n+2}       -> Jac[(n-1) * n_ext + 1]
        ...
        K_n^{orb_dim}   -> Jac[(n-1) * n_ext + n_ext - 1]
        
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
            nK = self.ref_occ[spirrep] * self.n_ext[spirrep]
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
            holes, particles, rank = self.get_exc_info(det, consider_core=True)
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
                pos += get_pos_from_rectangular(
                    holes[0].orb, particles[0].orb,
                    self.n_ext[particles[0].spirrep])
                Jac[pos] += (det.c
                             if (holes[0].orb
                                 + self.ref_occ[holes[0].spirrep]) % 2 == 0
                             else -det.c)
                logger.debug('Adding to Jac[%d] = %f', pos, Jac[pos])
            elif rank == 2:
                if (holes[0].spirrep != particles[0].spirrep  # occ != ref_occ
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
                pos += get_pos_from_rectangular(
                    holes[0].orb, particles[0].orb,
                    self.n_ext[particles[0].spirrep])
                pos1 = spirrep_start[holes[1].spirrep]
                pos1 += get_pos_from_rectangular(
                    holes[1].orb, particles[1].orb,
                    self.n_ext[particles[1].spirrep])
                if holes[0].spirrep == holes[1].spirrep:
                    negative = (holes[0].orb
                                + holes[1].orb) % 2 == 0
                else:
                    negative = (holes[0].orb
                                + holes[1].orb
                                + self.ref_occ[holes[0].spirrep]
                                + self.ref_occ[holes[1].spirrep]) % 2 == 1
                Hess[pos, pos1] += -det.c if negative else det.c
                if pos != pos1:
                    Hess[pos1, pos] += -det.c if negative else det.c
                logger.debug('Adding to Hess[%d,%d] = %f',
                             pos, pos1, -det.c if negative else det.c)
                if holes[0].spirrep == holes[1].spirrep:
                    pos = spirrep_start[holes[0].spirrep]
                    pos += get_pos_from_rectangular(
                        holes[0].orb, particles[1].orb,
                        self.n_ext[particles[1].spirrep])
                    pos1 = spirrep_start[holes[1].spirrep]
                    pos1 += get_pos_from_rectangular(
                        holes[1].orb, particles[0].orb,
                        self.n_ext[particles[0].spirrep])
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
        U = orbitals.calc_U_from_z(z, self)
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
        The transformed wave function, as instace of Wave_Function_Norm_CI.
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
        new_wf = Wave_Function_Norm_CI()
        new_wf.restricted = self.restricted
        new_wf.point_group = self.point_group
        new_wf.Ms = self.Ms
        new_wf.n_core = self.n_core
        new_wf.n_act = self.n_act
        new_wf.orb_dim = self.orb_dim
        new_wf.ref_occ = self.ref_occ
        new_wf.WF_type = self.WF_type
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
            new_wf._all_determinants.append(Slater_Det(c=new_c,
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
        for k_tot in range(2*(self.orb_dim - self.n_frozen)):
            coeff_delta = []
            if k_tot < self.orb_dim - self.n_frozen:
                k = k_tot
                spin_shift = 0
                t = ta
                logger.debug('k in alpha')
            else:
                k = k_tot -(self.orb_dim - self.n_frozen)
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
                                        p = i - (self.orb_dim - self.n_frozen) -1
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
                       only_ref_occ=False,
                       only_this_occ=None):
        raise NotImplementedError('undone')
