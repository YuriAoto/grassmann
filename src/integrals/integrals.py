"""Basic classes to store molecular integrals

"""
import math
from subprocess import check_output, CalledProcessError
from tempfile import mkdtemp
from shutil import rmtree
import os
from sys import version_info
import logging

import numpy as np
from scipy import linalg

from input_output.parser import ParseError
import integrals.integrals_cy

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


class Integrals():
    """All molecular integrals
    
    Parameters:
    -----------
    
    basis_set (str)
        Name of the basis set to be used

    mol_geo (MolecularGeometry)
        A MolecularGeometry object containing all molecular data

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
                 method='ir-wmme',
                 orth_method='canonical'):
        self.basis_set = basis_set
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
            wmmeDir = os.environ['GR_IR_WMME_DIR']
        except KeyError:
            raise ParseError(
                'Please set the environment variable GR_IR_WMME_DIR')
        wmmeBasePath = mkdtemp(prefix="bf-int.", dir=None)
        wmme_xyz_file = os.path.join(wmmeBasePath, 'coord.xyz')
        wmme_overlap_file = os.path.join(wmmeBasePath, 'overlap.int')
        wmme_coreh_file = os.path.join(wmmeBasePath, 'coreh.int')
        wmme_fint2e_file = os.path.join(wmmeBasePath, 'fint.int')
        basis_files = {'sto-3g': os.path.join(wmmeDir
                                              + '/bases/emsl_sto3g.libmol'),
                       'sto-6g': os.path.join(wmmeDir
                                              + '/bases/emsl_sto6g.libmol'),
                       'cc-pVDZ': os.path.join(wmmeDir
                                               + '/bases/emsl_cc-pVDZ.libmol'),
                       'cc-pVTZ': os.path.join(wmmeDir
                                               + '/bases/emsl_cc-pVTZ.libmol')}
        cmd = [os.path.join(wmmeDir + 'wmme')]
        cmd.extend(['--basis-fit', "univ-JKFIT"])
        cmd.extend(['--basis-orb', self.basis_set])
        cmd.extend(['--basis-lib', basis_files[self.basis_set]])
        cmd.extend(['--basis-lib',
                    os.path.join(wmmeDir + '/bases/def2-nzvpp-jkfit.libmol')])
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
