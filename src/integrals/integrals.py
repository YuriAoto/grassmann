"""Basic classes to store molecular integrals

"""
import math
from subprocess import check_output, CalledProcessError
from tempfile import mkdtemp
from shutil import rmtree
import os
import re
from sys import version_info
import logging
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from collections import namedtuple

import numpy as np
from scipy import linalg

from input_output.parser import ParseError
import integrals.integrals_cy
from molecular_geometry.periodic_table import ATOMS_NAME, ATOMS

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


BasisInfo = namedtuple('BasisInfo', ['name', 'atom', 'basis'])

_l_of = 'spdfghijkl'

class BasisSetError(Exception):
    
    def __init__(self, msg, basis):
        super().__init__(msg)
        self.msg = msg
        self.basis = basis
    
    def __str__(self):
        return (super().__str__() + '\n' + 'Basis set: ' + self.basis)

    
def _basis_filename(basis, wmme_dir):
    """The actual file name for the basis set
    
    The following conversions are made, for a decent file name
    
    + -> pl
    * -> st
    ( -> lbr
    ) -> lrbr
    
    """
    replaced_name = (basis.
                     replace('+', 'pl').
                     replace('*','st').
                     replace('(', 'lbr').
                     replace(')', 'rbr'))
    return os.path.join(wmme_dir, f'bases/emsl_{replaced_name}.libmol')


def _get_atomic_number_of_line(line):
    """Get the atomic number from a line '! <ELEMENT_NAME> (xxx) -> [XXX]' """
    rematch = re.match('^!\s*([a-zA-Z]+)\s*\(.+\)\s*->\s*\[.+\]', line)
    if rematch:
        rematch = ATOMS_NAME.index(rematch.group(1).lower())
    return rematch


def _get_n_func(shells):
    """Helper for _from_json_to_wmme: number of pritives and contractions"""
    n_prim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    n_contr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for sh in shells:
        if len (sh['angular_momentum']) == 1:
            ang_mom = sh['angular_momentum'][0]
            n_prim[ang_mom] += len(sh['exponents'])
            n_contr[ang_mom] += len(sh['coefficients'])
        else:
            for i, ang_mom in enumerate(sh['angular_momentum']):
                n_prim[ang_mom] += len(sh['exponents'])
                n_contr[ang_mom] += len(sh['coefficients'][i])
    return [x for x in n_prim if x], [x for x in n_contr if x]


def _from_json_to_wmme(basis):
    """Transform from JSON format to the wmme format
    
    The JSON for the basis is as coming from the basissetexchange webpage
    and the wmme is, as far as I can see, the internal format of Molpro.
    
    Parameters:
    -----------
    basis (str)
        A string with the basis in the JSON format.
    
    Return:
    -------
    A string with the basis in the IR-WMME/Molpro format
    
    """
    raise NotImplementedError('Conversion from json not ready')
    b = json.loads(basis)
    newb = []
    basis_name = b['names'][0]
    elements = b['elements']
    ctr_fmt = '{0}  {1}  {2} :  {3}   {4}'
    for at, at_basis in b['elements'].items():
        n_prim, n_contr = _get_n_func(at_basis['electron_shells'])
        fmt_nprim = ','.join([n + _l_of[i]
                              for i, n in enumerate(map(str, n_prim))])
        fmt_ncontr = ','.join([n + _l_of[i]
                               for i, n in enumerate(map(str, n_contr))])
        newb.append(
            f'! {ATOMS_NAME[int(at)].upper()}      ({fmt_nprim}) -> [{fmt_ncontr}]')
        for ang_mom, n in enumerate(zip(n_prim, n_contr)):
            np , nc = n
            contr_info = ctr_fmt.format(ATOMS[int(at)].upper(),
                                        _l_of[ang_mom].upper(),
                                        basis_name,
                                        np, nc)
            contr_info += '    THE CONTRACTION \n'
            newb.append(contr_info)
            all_exp = []
            all_coef = []
            for sh in at_basis['electron_shells']:
                if ang_mom in sh['angular_momentum']:
                    all_exp.extend(sh['exponents'])
                    for c in sh['coefficients']:
                        all_coef.extend(c)
            for i, x in enumerate(all_exp + all_coef):
                pt_pos = x.index('.')
                if i % 5 == 0:
                    newb.append(' '*(8-pt_pos) + x + ' '*(20-len(x)+pt_pos))
                else:
                    newb[-1] += ' '*(8-pt_pos) + x + ' '*(20-len(x)+pt_pos)

    return '\n'.join(newb)


def _from_molpro_to_wmme(basis):
    """Transform from (input's) molpro format to the wmme format
    
    The format coming from basissetexchange is what we put in the molpro input,
    that is different (essentially a transposision) than the internal format of
    Molpro, used by IR-WMME.
    
    Parameters:
    -----------
    basis (str)
        A string with the basis in the Molpro format.
    
    Return:
    -------
    A string with the basis in the IR-WMME/Molpro format
    
    """
    b = basis.split('\n')
    newb = []
    basisname = ''
    all_coef = []
    all_contractions = []
    for line in b:
        rematch = re.match('^!\s*Basis set:\s*(.+)$', line)
        if rematch:
            basisname = rematch.group(1)
        if _get_atomic_number_of_line(line):
            newb.append(line)
            continue
        try:
            comment_pos = line.index('!')
        except ValueError:
            pass
        else:
            line = line[:comment_pos]
        lspl = list(map(lambda x: x.strip(), line.split(',')))
        if not line or 'basis={' == line:
            continue
        if lspl[0] == 'c':
            all_coef.extend(lspl[2:])
            all_contractions.append(lspl[1])
        else:
            if all_contractions:
                basis_descr += f'   {len(all_exp)}   {len(all_contractions)}   '
                basis_descr += '   '.join(all_contractions) + '\n'
                newb.append(basis_descr)
                for i, x in enumerate(all_exp + all_coef):
                    pt_pos = x.index('.')
                    if i % 5 == 0:
                        newb.append(' '*(8-pt_pos) + x + ' '*(20-len(x)+pt_pos))
                    else:
                        newb[-1] += ' '*(8-pt_pos) + x + ' '*(20-len(x)+pt_pos)
            if line == '}':
                break
            basis_descr = f'{lspl[1]}  {lspl[0].upper()}  {basisname} :'
            all_exp = lspl[2:]
            all_contractions = []
            all_coef = []    
    return '\n'.join(newb) + '\n\n'


def _check_basis(file_name, atoms):
    """Check if basis is in there
    
    Parameters:
    -----------
    basis (str)
        The basis set name. Same as in the basis set exchange
    
    atoms (list of int)
        The atomic numbers of the atoms, for which basis sets will be fetched
    
    Side Effect:
    ------------
    The list atoms are updated, leaving only the elements for which basis set has
    not been found
    
    """
    try:
        with open(file_name, 'r') as f:
            for line in f:
                if not atoms:
                    break
                atom_cand = _get_atomic_number_of_line(line)
                if atom_cand is not None and atom_cand in atoms:
                    atoms.remove(atom_cand)
    except OSError:
        return


def _write_basis(info, wmme_dir):
    """Write basis to files
    
    Parameters:
    -----------
    info (list of tuples)
        The basis information, as returned from _fetch_from_basis_set_exchange
    
    wmme_dir (str)
        A string with the directory for the Knizia's ir-wmme program
    
    Side Effect:
    ------------
    The information from info is written to the directory wmme_dir,
    perhaps updating files if the basis file is already there
    
    """
    for x in info:
        try:
            with open(_basis_filename(x.name, wmme_dir), 'r') as f:
                file_content = f.readlines()
        except OSError:
            file_content = []
        pos_new_at = len(file_content)
        for i, line in enumerate(file_content):
            cur_atom = _get_atomic_number_of_line(line)
            if cur_atom is not None and cur_atom > x.atom:
                pos_new_at = i
                break
        file_content.insert(pos_new_at, x.basis)
        with open(_basis_filename(x.name, wmme_dir), 'w') as f:
            f.write(''.join(file_content))


def _fetch_from_basis_set_exchange(basis, atoms):
    """Get the basis for the listed atoms from https://www.basissetexchange.org/
    
    Parameters:
    -----------
    basis (str)
        The basis set name. Same as in the basis set exchange
    
    atoms (list of int)
        The atomic numbers of the atoms, for which basis sets will be fetched
    
    Return:
    -------
    A list of tuples:
    Each element of this list is the following tuple:
    (basis, atomic number, basis set information)
    
    Raise:
    ------
    BasisSetError if the basis set could not be obtained from the webpage
    
    """
    url_fmt = ('https://www.basissetexchange.org/api/basis/{0}/format/molpro/'
               +'?version={1}&elements={2}')
    basis_info = []
    version = 0
    for at in atoms:
        url = url_fmt.format(basis, version, at)
        try:
            page = urlopen(url)
        except HTTPError:
            raise BasisSetError('HTTP error from https://www.basissetexchange.org:\n'
                                + f'Perhaps they do not have {basis} for {ATOMS[at]}',
                                basis)
        except URLError:
            raise BasisSetError('URL error:\n'
                                + f'Perhaps you have no connection to the internet',
                                basis)
        basis_info.append(BasisInfo(name=basis,
                                    atom=at,
                                    basis=page.read().decode("utf-8")))
    return basis_info


def basis_file(basis, mol_geom, wmme_dir, try_getting_it=True):
    """Return the filename with the basis set.
    
    Parameters:
    -----------
    basis (str)
        The basis set name. Same as in the basis set exchange
    
    mol_geom (MolecularGeometry)
        The molecular geometry for which a basis is required
    
    wmme_dir (str)
        A string with the directory for the Knizia's ir-wmme program
    
    try_getting_it (bool, optional, default=True)
        If True, try to fetch the basis set from the basis set exchange
        webpage. In this case it stores it for further usages.
    
    Return:
    -------
    A string with the filename of the basis set
    
    Raise:
    ------
    BasiSetError, if the basis set for the needed atoms is not found
    nor fetched.
    
    """
    atoms = []
    for at in mol_geom:
        if at.atomic_number not in atoms:
            atoms.append(at.atomic_number)
    if basis == 'univ-JKFIT':
        return os.path.join(wmme_dir, 'bases/def2-nzvpp-jkfit.libmol')
    file_name = _basis_filename(basis, wmme_dir)
    _check_basis(file_name, atoms)
    if atoms:
        if try_getting_it:
            b = _fetch_from_basis_set_exchange(basis, atoms)
        else:
            raise BasisSetError(
                f'We do not have the basis {basis} for these atoms:\n'
                + ' '.join(map(lambda x: ATOMS[x], atoms)),
                basis)
        newb = []
        for at in b:
            newb.append(BasisInfo(name=at[0],
                                  atom=at[1],
                                  basis=_from_molpro_to_wmme(at[2])))
        _write_basis(newb, wmme_dir)
    return file_name

class Integrals():
    """All molecular integrals
    
    Parameters:
    -----------
    
    basis_set (str)
        Name of the basis set to be used

    mol_geo (MolecularGeometry)
<<<<<<< HEAD
        A MolecularGeometry object containing all molecular data
=======
        A MolecularGeometry object conatining all molecular data
>>>>>>> 13792f443e9ff0b612aec97b65cf6cb3aea771bd

    n_func (int)
        Number of contracted functions in the basis set ((??check))

    S (np.ndarray ((??check)))
        The overlap matrix
    
    h (2D np.ndarray)
        one-electron integral matrix

    g (3D np.ndarray)
        two-electron integral matrix

    X (??)
        ??

    """
    def __init__(self, mol_geo, basis_set,
                 basis_fit='univ-JKFIT',
                 method='ir-wmme',
                 orth_method='canonical'):
        self.basis_set = basis_set
        self.basis_fit = basis_fit
        self.mol_geo = mol_geo
        self.n_func = 0
        self.S = None
        self.h = None
        self.g = None
        self.X = None
        if method == 'ir-wmme':
            self.set_wmme_integrals()
        if orth_method != None:
            self.orthogonalise_S(method=orth_method)
        logger.debug('h:\n%r', self.h)

    @classmethod
    def from_atomic_to_molecular(cls, old_int, molecular, cy=True):
        new_int = cls(old_int.mol_geo, old_int.basis_set, method = None, orth_method = None)
        new_int.n_func = old_int.n_func
        irr = 0
        if cy:
            new_int.h = np.zeros((old_int.n_func,old_int.n_func))
            integrals.integrals_cy.from_1e_atomic_to_molecular_cy(new_int.h,molecular[irr],old_int.h,old_int.n_func)
            new_int.S = np.zeros((old_int.n_func,old_int.n_func))
            integrals.integrals_cy.from_1e_atomic_to_molecular_cy(new_int.S,molecular[irr],old_int.S,old_int.n_func)
        else:
            new_int.h = _from_1e_atomic_to_molecular(old_int.h, molecular[irr], old_int.n_func)
            new_int.S = _from_1e_atomic_to_molecular(old_int.S, molecular[irr], old_int.n_func)
        new_int.g = Two_Elec_Int.from_2e_atomic_to_molecular(old_int, molecular, cy=cy) 
        return new_int

    def set_wmme_integrals(self):
        """Set integrals from Knizia's ir-wmme program"""
        try:
            wmme_dir = os.environ['GR_IR_WMME_DIR']
        except KeyError:
            raise ParseError(
                'Please set the environment variable GR_IR_WMME_DIR;'
                + " It should have the directory of Knizia's ir-wmme program")
        wmmeBasePath = mkdtemp(prefix="bf-int.", dir=None)
        wmme_xyz_file = os.path.join(wmmeBasePath, 'coord.xyz')
        wmme_overlap_file = os.path.join(wmmeBasePath, 'overlap.int')
        wmme_coreh_file = os.path.join(wmmeBasePath, 'coreh.int')
        wmme_fint2e_file = os.path.join(wmmeBasePath, 'fint.int')
        cmd = [os.path.join(wmme_dir + 'wmme')]
        cmd.extend(['--basis-fit', self.basis_fit])
        cmd.extend(['--basis-orb', self.basis_set])
        cmd.extend(['--basis-lib', basis_file(self.basis_set,
                                              self.mol_geo,
                                              wmme_dir,
                                              try_getting_it=True)])
        cmd.extend(['--basis-lib', basis_file(self.basis_fit,
                                              self.mol_geo,
                                              wmme_dir,
                                              try_getting_it=True)])
        cmd.extend(['--atoms-au', wmme_xyz_file])
        cmd.extend(['--save-coreh', wmme_coreh_file])
        cmd.extend(['--save-overlap', wmme_overlap_file])
        cmd.extend(['--save-fint2e', wmme_fint2e_file])
        cmd.extend(['--matrix-format', 'npy'])
        endl = '\n'
        f = open(wmme_xyz_file, 'w')
        f.write(str(len(self.mol_geo)) + endl + 'Title:XYZ for WMME' + endl)
        for at in self.mol_geo:
            f.write('{0:5s} {1:15.8f} {2:15.8f} {3:15.8f}\n'.
                    format(at.element, at.coord[0], at.coord[1], at.coord[2]))
        f.close()
        try:
            Output = check_output(cmd, shell=False)
            if (version_info) >= (3, 0):
                Output = Output.decode("utf-8")
        except CalledProcessError as e:
            raise Exception('Integral calculation failed.'
                            + ' Output was:\n{}\nException was: {}'.
                            format(e.output, str(e)))
        self.h = np.load(wmme_coreh_file)
        self.n_func = self.h.shape[0]
        self.S = np.load(wmme_overlap_file)
        self.orthogonalise_S()
        self.g = Two_Elec_Int.from_wmme_fint2e(wmme_fint2e_file, self.n_func)
        rmtree(wmmeBasePath)
    
    def orthogonalise_S(self, method='canonical'):
        """Calculate self.X, that orthogonalise the atomic basis functions."""
        if self.S is None:
            raise Exception('Cannot orthogonalise S: ' +
                            'Overlap integrals not calculated yet.')
        s, self.X = linalg.eigh(self.S)
        for j in range(len(s)):
            if abs(s[j]) < 0.000001:
                raise Exception('LD problems in basis functions, s=' + str(s))
            for i in range(len(s)):
                self.X[i][j] = self.X[i][j]/math.sqrt(s[j])
        if loglevel <= logging.DEBUG:
            logger.debug('X:\n%r', self.X)
            logger.debug('XSX:\n%r', self.X.T @ self.S @ self.X)


class Two_Elec_Int():
    """Two electron integrals and related functions

    Parameters:
    -----------
   
    _format (str)
        Indicates how the integrals are organized (F2e or ijkl)
    
    _integrals (ndarrays)
        contains the integrals values

    n_func (int)
        number of basis functions

    """
    def __init__(self):
        self._format = None
        self._integrals = None
        self.n_func = 0
    
    def __getitem__(self, key):
        """
        If key == F2e
        g_ijkl = np.einsum('Fij,Fkl->ijkl',
                            self.intgrls.g._integrals,
                            self.intgrls.g._integrals)
        """
        i, j, k, l = key
        if self._format == 'F2e':
            return np.dot(self._integrals[:, i, j], self._integrals[:, k, l])
        if self._format == 'ijkl':
            ij = j + i * (i + 1) // 2 if i >= j else i + j * (j + 1) // 2
            kl = l + k * (k + 1) // 2 if k >= l else k + l * (l + 1) // 2
            ijkl = (kl + ij * (ij + 1) // 2
                    if ij >= kl else
                    ij + kl * (kl + 1) // 2)
            return self._integrals[ijkl]

    @classmethod
    def from_wmme_fint2e(cls, wmme_fint2e_file, n_func):
        """Load integrals as intF2e"""
        new_2e_int = cls()
        new_2e_int.n_func = n_func
        new_2e_int._format = 'F2e'
        new_2e_int._integrals = np.load(wmme_fint2e_file)
        new_2e_int._integrals = new_2e_int._integrals.reshape(
            (new_2e_int._integrals.shape[0], n_func, n_func))
        return new_2e_int

    @classmethod
    def from_2e_atomic_to_molecular(cls, atomic, molecular, cy=True):
        """Change a Two_Elec_Int in atomic basis to molecular basis
           TODO: Set atomic int as a parameter of molecular orbitals.
                 Only the moleular will be the input.
                 That way we don't need to check for size compatibility
        """
        new_2e_int = cls()
        new_2e_int.n_func  = atomic.n_func
        new_2e_int._format = 'ijkl'
        if atomic.g._format == 'F2e':
            atomic.g.transform_to_ijkl()
        n_g = atomic.n_func * (atomic.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        mo_integrals = np.zeros(n_g)
        irr = 0 #TODO Generalize the symmetry
        if cy:
            new_2e_int._integrals = integrals.integrals_cy.from_2e_atomic_to_molecular_cy(mo_integrals,molecular[irr],atomic.g._integrals,atomic.n_func)
        else:
            new_2e_int._integrals = _old_from_2e_atomic_to_molecular(mo_integrals, atomic, molecular, irr=irr)
        return new_2e_int


    def transform_to_ijkl(self):
        """Transform integrals to ijkl format"""
        n_g = self.n_func * (self.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        g_in_new_format = np.zeros(n_g)
        ij = -1
        for i in range(self.n_func):
            for j in range(self.n_func):
                if i < j:
                    continue
                ij += 1
                kl = -1
                for k in range(self.n_func):
                    for l in range(self.n_func):
                        if k < l:
                            continue
                        kl += 1
                        if ij < kl:
                            continue
                        ijkl = (kl + ij * (ij + 1) // 2
                                if ij >= kl else
                                ij + kl * (kl + 1) // 2)
                        for Ffit in self._integrals:
                            g_in_new_format[ijkl] += Ffit[i][j] * Ffit[k][l]
        self._format = 'ijkl'
        self._integrals = g_in_new_format

        


#####OLD#####
def _from_1e_atomic_to_molecular(atomic_matrix, molecular, n_func):
    mol_int = np.zeros(len(atomic_matrix))
    for i in range(n_func):
        for j in range(n_func):
        #    if i < j:
        #        continue
        #    ij = i+j*(j+1)//2 ##Check if it's right
            for p in range(n_func):
                for q in range(n_func):
        #            if p < q:
        #                continue
        #            pq = p+q*(q+1)//2
                    #print(i,j,ij,p,q,pq)
                    mol_int[i,j]+= molecular[p,i]*molecular[q,j]*atomic_matrix[p,q]
        #            mol_int[ij]+= molecular[p,i]*molecular[q,j]*atomic_matrix[pq]
        #            if p != q:
        #                mol_int[ij]+= molecular[q,i]*molecular[p,j]*atomic_matrix[pq]
                    #print(mol_int[ij])
    return mol_int




def _old_from_2e_atomic_to_molecular(mo_integrals, atomic, molecular, irr = 0):
    ij=-1
    for i in range(atomic.n_func):                      
        for j in range(atomic.n_func):
            if i < j:
                continue
            ij += 1
            kl = -1
            for k in range(atomic.n_func):
                for l in range(atomic.n_func):
                    if k < l:
                        continue
                    kl += 1
                    if ij < kl:
                        continue
                    ijkl = (kl + ij * (ij + 1) // 2
                            if ij >= kl else
                            ij + kl * (kl + 1) // 2)
                    pq=-1
                    for p in range(atomic.n_func):                      
                        for q in range(atomic.n_func):
                            if p < q:
                                continue
                            pq += 1
                            rs = -1
                            for r in range(atomic.n_func):
                                for s in range(atomic.n_func):
                                    if r < s:
                                        continue
                                    rs += 1
                                    if pq < rs:
                                        continue
                                    pqrs = (rs + pq * (pq + 1) // 2
                                            if pq >= rs else
                                            pq + rs * (rs + 1) // 2)
                                    mo_integrals[ijkl]+= molecular[irr][p,i]*molecular[irr][q,j]*molecular[irr][r,k]*molecular[irr][s,l]*atomic.g._integrals[pqrs]
                                    if p != q and r != s and ( q != s or p != r ):
                                        mo_integrals[ijkl]+= molecular[irr][q,i]*molecular[irr][p,j]*molecular[irr][r,k]*molecular[irr][s,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][p,i]*molecular[irr][q,j]*molecular[irr][s,k]*molecular[irr][r,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][q,i]*molecular[irr][p,j]*molecular[irr][s,k]*molecular[irr][r,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][r,i]*molecular[irr][s,j]*molecular[irr][p,k]*molecular[irr][q,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][r,i]*molecular[irr][s,j]*molecular[irr][q,k]*molecular[irr][p,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][s,i]*molecular[irr][r,j]*molecular[irr][p,k]*molecular[irr][q,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][s,i]*molecular[irr][r,j]*molecular[irr][q,k]*molecular[irr][p,l]*atomic.g._integrals[pqrs]
                                     #    print(8)
                                    elif p != q and r != s:
                                        mo_integrals[ijkl]+= molecular[irr][q,i]*molecular[irr][p,j]*molecular[irr][r,k]*molecular[irr][s,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][p,i]*molecular[irr][q,j]*molecular[irr][s,k]*molecular[irr][r,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][q,i]*molecular[irr][p,j]*molecular[irr][s,k]*molecular[irr][r,l]*atomic.g._integrals[pqrs]
                                     #    print(4)
 
                                    elif p != q:
                                        mo_integrals[ijkl]+= molecular[irr][q,i]*molecular[irr][p,j]*molecular[irr][r,k]*molecular[irr][s,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][r,i]*molecular[irr][s,j]*molecular[irr][p,k]*molecular[irr][q,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][r,i]*molecular[irr][s,j]*molecular[irr][q,k]*molecular[irr][p,l]*atomic.g._integrals[pqrs]
                                     #    print(4)
                                    elif r != s:
                                        mo_integrals[ijkl]+= molecular[irr][p,i]*molecular[irr][q,j]*molecular[irr][s,k]*molecular[irr][r,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][r,i]*molecular[irr][s,j]*molecular[irr][p,k]*molecular[irr][q,l]*atomic.g._integrals[pqrs]
                                        mo_integrals[ijkl]+= molecular[irr][s,i]*molecular[irr][r,j]*molecular[irr][p,k]*molecular[irr][q,l]*atomic.g._integrals[pqrs]
                                     #    print(4)
                                    elif p != r or q != s:
                                        mo_integrals[ijkl]+= molecular[irr][r,i]*molecular[irr][s,j]*molecular[irr][p,k]*molecular[irr][q,l]*atomic.g._integrals[pqrs]
                                     #    print(2)
                                    else:
                                       pass
                                     #    print(1)
    return mo_integrals
