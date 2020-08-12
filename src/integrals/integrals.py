"""Basic classes to store molecular integrals

"""
import math
from subprocess import check_output, CalledProcessError
from tempfile import mkdtemp
from shutil import rmtree
from os import path
from sys import version_info
import logging

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

wmmeDir = '/home/yuriaoto/Documents/Codes/ir-wmme.20141030/ir-wmme.20141030/'


class Integrals():
    """All molecular integrals
    
    
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
        self.orthogonalise_S(method=orth_method)
        logger.debug('h:\n%r', self.h)

    def set_wmme_integrals(self):
        """Set integrals from Knizia's ir-wmme program"""
        wmmeBasePath = mkdtemp(prefix="bf-int.", dir=None)
        wmme_xyz_file = path.join(wmmeBasePath, 'coord.xyz')
        wmme_overlap_file = path.join(wmmeBasePath, 'overlap.int')
        wmme_coreh_file = path.join(wmmeBasePath, 'coreh.int')
        wmme_fint2e_file = path.join(wmmeBasePath, 'fint.int')
        basis_files = {'sto-3g': path.join(wmmeDir
                                           + '/bases/emsl_sto3g.libmol'),
                       'sto-6g': path.join(wmmeDir
                                           + '/bases/emsl_sto6g.libmol'),
                       'cc-pVDZ': path.join(wmmeDir
                                            + '/bases/emsl_cc-pVDZ.libmol'),
                       'cc-pVTZ': path.join(wmmeDir
                                            + '/bases/emsl_cc-pVTZ.libmol')}
        cmd = [path.join(wmmeDir + 'wmme')]
        cmd.extend(['--basis-fit', "univ-JKFIT"])
        cmd.extend(['--basis-orb', self.basis_set])
        cmd.extend(['--basis-lib', basis_files[self.basis_set]])
        cmd.extend(['--basis-lib',
                    path.join(wmmeDir + '/bases/def2-nzvpp-jkfit.libmol')])
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
        new_int = cls()
        new_int.n_func = n_func
        new_int._format = 'F2e'
        new_int._integrals = np.load(wmme_fint2e_file)
        new_int._integrals = new_int._integrals.reshape(
            (new_int._integrals.shape[0], n_func, n_func))
        return new_int

    def transform_to_ijkl(self):
        """Transform integrals in tho ijkl format"""
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
        self._integrals = g_in_new_format