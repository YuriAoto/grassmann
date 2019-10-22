"""A module for molecular orbitals

Orbitals are generally saved in np.ndarrays, with the coefficients of
the orbitals in a basis set given in the columns. See get_orbitals for
a description.

Classes:
--------
Molecular_Orbitals

"""
import logging
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import inv

from dGr_WF_int_norm import number_of_irreducible_repr
from dGr_exceptions import *

logger = logging.getLogger(__name__)

class Molecular_Orbitals():
    """A set of molecular orbitals
    
    Behaviour:
    ----------
    The orbitals are stored in one or more np.ndarray
    with the coefficients of the i-th orbital in the i-th column:
    
    1st MO      2nd MO    ...    nth MO
    
    C[0][0]    C[1][1]    ...    C[1][n-1]   -> coef of 1st basis function
    C[1][0]    C[2][1]    ...    C[2][n-1]   -> coef of 2nd basis function
    ...        ...        ...    ...
    C[n-1][0]  C[n-1][1]  ...    C[n][n-1]   -> coef of nth basis function
    
    There are n_irrep (restricted=True) or 2*n_irrep (restricted=False)
    such np.ndarray, each of shape (basis lenght, n orb of irrep).
    For unrestricted orbitals, all alpha orbitals for all irreps come first,
    followed by the beta orbitals:
    
    Restricted:
    [np.ndarray for irrep 1, ..., np.ndarray for irrep (n_irrep-1)]
    
    Unrestricted:
    [np.ndarray for alpha/irrep 1, ..., np.ndarray for alpha/irrep (n_irrep-1),
     np.ndarray for beta/irrep 1, ..., np.ndarray for beta/irrep (n_irrep-1)]
    
    Attributes:
    -----------
    n_irrep (int)
        Number of ireducible representations
    
    restricted (bool)
        True if both alpha and beta orbitals are the same
    
    Data model:
    -----------
    [spirrep] 
        Access the coefficients of spirrep
    
    len
        The size of the basis set
    
    iter
        Iterates over spirreps

    """
    def __init__(self):
        self._basis_len = 0
        self.n_irrep = None
        self.restricted = True
        self._coefficients = None

    def __getitem__(self, key):
        return self._coefficients[key]

    def __iter__(self):
        return iter(self._coefficients)

    def __len__(self):
        return self._basis_len

    def __str__(self):
        x = []
        for spirrep, orb_spirrep in enumerate(self._coefficients):
            irrep = spirrep % self.n_irrep
            if irrep == 0:
                if self.restricted:
                    x.append('Restricted orbitals:')
                elif irrep == spirrep:
                    x.append('Alpha orbitals:')
                else:
                    x.append('Beta orbitals:')
            x.append('irrep = {}'.format(irrep))
            x.append(str(orb_spirrep))
            x.append('')
        return '\n'.join(x)

    @classmethod
    def from_file(cls, file_name):
        """Load orbitals from file file_name.
        
        Parameters:
        -----------
        file_name (str)
            The file with orbitals. xml file only.
        
        Returns:
        --------
        An instance of Molecular_Orbitals.
        """
        if file_name[-4:] == '.xml':
            return cls._get_orbitals_from_Molpro_xml(file_name)
        raise dGrValueError('We can read orbitals from xml files only.')

    @classmethod
    def _get_orbitals_from_Molpro_xml(cls, xml_file):
        """Load orbitals from Molpro xml file.
        
        Parameters:
        -----------
        xml_file (str)
            A xml file generated by Molpro's put command
        
        Returns:
        --------
        An instance of Molecular_Orbitals.
        """
        def get_spin_shift(orb_type, method, n_irrep):
            if orb_type == 'BETA':
                return n_irrep
            elif orb_type == 'ALPHA':
                return 0
            elif orb_type == 'NATURAL' and method == 'UHF':
                raise dGrValueError('We do not use these orbitals.')
            elif orb_type == 'NATURAL' or orb_type == 'CANONICAL':
                return 0
            else:
                raise dGrParseException('Unknown type of orbital: ' + orb_type)
        new_orbitals = cls()
        tree = ET.parse(xml_file)
        molpro = tree.getroot()
        ns = {'molpro': 'http://www.molpro.net/schema/molpro-output',
              'xsd':'http://www.w3.org/1999/XMLSchema',
              'cml':'http://www.xml-cml.org/schema',
              'stm':'http://www.xml-cml.org/schema',
              'xhtml':'http://www.w3.org/1999/xhtm'}
        molecule = molpro[0]
        point_group = molecule.find('cml:molecule',ns).find('cml:symmetry',ns).attrib['pointGroup']
        new_orbitals.n_irrep = number_of_irreducible_repr[point_group]
        new_orbitals._basis_len = int(molecule.find('molpro:basisSet',ns).attrib['length'])
        if molecule.attrib['method'] == 'UHF':
            new_orbitals.restricted = False
            n_orb_per_spirrep = np.zeros(2 * new_orbitals.n_irrep, dtype=np.uint)
            cur_orb = np.zeros(2 * new_orbitals.n_irrep, dtype=np.uint)
        else:
            new_orbitals.restricted = True
            n_orb_per_spirrep = np.zeros(new_orbitals.n_irrep, dtype=np.uint)
            cur_orb = np.zeros(new_orbitals.n_irrep, dtype=np.uint)
        for orb_set in molecule.findall('molpro:orbitals', ns):
            try:
                spin_shift = get_spin_shift(orb_set.attrib['type'],
                                            orb_set.attrib['method'],
                                            new_orbitals.n_irrep)
            except dGrValueError:
                continue
            for orb in orb_set:
                n_orb_per_spirrep[spin_shift
                                  + int(orb.attrib['symmetryID']) - 1] += 1
        new_orbitals._coefficients = []
        for n in n_orb_per_spirrep:
            new_orbitals._coefficients.append(np.zeros((new_orbitals._basis_len, n)))
        for orb_set in molecule.findall('molpro:orbitals', ns):
            try:
                spin_shift = get_spin_shift(orb_set.attrib['type'],
                                            orb_set.attrib['method'],
                                            new_orbitals.n_irrep)
            except dGrValueError:
                continue
            for orb in orb_set:
                spirrep = spin_shift + int(orb.attrib['symmetryID']) - 1
                new_orbitals._coefficients[spirrep][:,cur_orb[spirrep]] = np.array(
                    list(map(float,orb.text.split())))
                cur_orb[spirrep] += 1
        return new_orbitals

    def in_the_basis_of(self, other):
        """Return the coefficients of self on the basis of other.
        
        Parameters:
        -----------
        other (Molecular_Orbitals)
            The molecular orbitals that will be used as basis for self.
            These orbitals have to be complete, that is, the number of
            orbitals (in each spin for unrestricted) must be the length
            of the basis set.
        
        Behaviour:
        ----------
        
        If the orbitals in self are in a square matrix C_self
        and the orbitals in other are in a square matrix C_other,
        this function returns the transformation matrix U such that:
        
        C_self = C_other @ U
        
        or:
        
        U = C_other^{-1} @ C_self
        
        The matrix U comes divided in spirrep blocks, whose number depend
        on the n_irrep and the restricted of self and other
        
        Returns:
        --------
        U as a list of np.ndarrays, each entry for a spirrep.
        
        TODO:
        -----
        Maybe return an instance of Molecular_Orbitals and use this
        throughout the code for U?
        Note that in our current implementation U[i].shape[0] is the
        orbital dimension of spirrep i, whereas in Molecular_Orbitals
        (as comes from Molpro) store coefficients of non-symmetry adapted
        basis, and thus implementation U[i].shape[0] is the length of the
        basis, irrespective of the irrep.
        """
        logger.debug("MO (other):\n%s", other)
        logger.debug("MO (self):\n%s", self)
        if len(self) != len(other):
            raise dGrValueError('Orbitals do not have the same basis length!')
        if other.n_irrep == self.n_irrep:
            n_irrep = self.n_irrep
        else:
            n_irrep = 1
        C_inv = np.zeros((len(other), len(other)))
        if not other.restricted:
            C_inv_beta = np.zeros((len(other), len(other)))
        i_C = 0
        for spirrep, C_spirrep in enumerate(other):
            for i in range(C_spirrep.shape[1]):
                if spirrep < other.n_irrep:
                    C_inv[:,i_C] = C_spirrep[:,i]
                else:
                    C_inv_beta[:,i_C % len(other)] = C_spirrep[:,i]
                i_C += 1
        C_inv = inv(C_inv)
        logger.debug('C_inv:\n%s', C_inv)
        if not other.restricted:
            C_inv_beta = inv(C_inv_beta)
            logger.debug('C_inv_beta:\n%s', C_inv_beta)
        Ua = np.zeros((len(other), len(other)))
        lim_irrep = [0]
        for n in range(self.n_irrep):
            lim_irrep.append(lim_irrep[-1] + self[n].shape[1])
        for n in range(self.n_irrep):
            Ua[:,lim_irrep[n]:lim_irrep[n + 1]] = C_inv @ self[n]
        logger.debug('Ua:\n%s',Ua)
        if not other.restricted or not self.restricted:
            Ub = np.zeros((len(other), len(other)))
            for n in range(self.n_irrep):
                Ub[:,lim_irrep[n]:lim_irrep[n + 1]] = (
                    (C_inv
                     if other.restricted else
                     C_inv_beta)
                    @ (self[n]
                       if self.restricted else
                       self[n + self.n_irrep]))
        U = []
        for n in range(n_irrep):
            i_inf, i_sup = lim_irrep[n], lim_irrep[n + 1]
            if not np.allclose(Ua[:i_inf, i_inf:i_sup],
                               np.zeros((i_inf,i_sup - i_inf))):
                logger.warning('Not all zero!!:\n%s', Ua[:i_inf, i_inf:i_sup])
            if not np.allclose(Ua[lim_irrep[n+1]:, i_inf:i_sup],
                               np.zeros((len(other) - i_sup, i_sup - i_inf))):
                logger.warning('Not all zero:\n%s',
                               Ua[i_sup:, i_inf:i_sup])
            U.append(Ua[i_inf:i_sup, i_inf:i_sup])
        if not other.restricted or not self.restricted:
            for n in range(n_irrep):
                i_inf, i_sup = lim_irrep[n], lim_irrep[n + 1]
                if not np.allclose(Ub[:i_inf, i_inf:i_sup],
                                   np.zeros((i_inf,i_sup - i_inf))):
                    logger.warning('Not all zero!!:\n%s', Ub[:i_inf, i_inf:i_sup])
                if not np.allclose(Ub[lim_irrep[n+1]:, i_inf:i_sup],
                                   np.zeros((len(other) - i_sup, i_sup - i_inf))):
                    logger.warning('Not all zero:\n%s',
                                   Ub[i_sup:, i_inf:i_sup])
                U.append(Ub[i_inf:i_sup,
                            i_inf:i_sup])
        return U
