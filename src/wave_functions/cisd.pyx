"""The CISD wave function

Classes:
--------

CISD_WaveFunction
"""
import logging

import numpy as np

from util.array_indices cimport (triangular, n_from_triang, ij_from_triang,
                                 n_from_rect)
from wave_functions.general cimport WaveFunction
from wave_functions.general import WaveFunction
from wave_functions.interm_norm cimport IntermNormWaveFunction
from wave_functions.interm_norm import IntermNormWaveFunction
from orbitals.occ_orbitals cimport OccOrbital
from orbitals.occ_orbitals import OccOrbital
from wave_functions.interm_norm cimport ExcType
from util.memory import mem_of_floats

logger = logging.getLogger(__name__)


cdef class CISDWaveFunction(WaveFunction):
    """The CISD wave function
    
    For the moment, only restricted and including only
    the coefficients as they contribute to the distance to the
    Grassmannian of the reference (that is, have the same occupation
    of the reference in each spirrep block).
    
    Atributes:
    ----------
    C0 (float)
        The coefficient of the reference determinant
    
    Cs (list of 2D np.ndarrays)
        C_i^a = Cs[irrep][i, a]
        
        Coefficients of single excitations.
        Each element of this list is associated to an irrep.
        There are n_irrep entries (only restricted wave functions).
        Each entry is a 2D np.ndarray with shape
        
        (self.orbspace.corr[irrep], self.orbspace.virt[irrep])
        
        storing the coefficients of the excitations from orbital i to a
    
    Cd (list of 2D np.ndarrays)
        C_ij^ab = Cd[irrep][ij, ab]
        
        Coefficients of double excitations within a single spirrep.
        These are "pure" double excitations a single spin and irrep.
        Each element of this list is associated to an irrep.
        There are n_irrep entries (only restricted wave functions).
        Each entry is a 2D np.ndarray with shape
        
        (triangular(self.orbspace.corr[irrep] - 1),
         triangular(self.orbspace.virt[irrep] - 1))
        
        and store the coefficients for the excitation i,j to a,b.
        Indices for the pairs i,j and a,b are stored in triangular order:
        ij = n_from_triang(j, i)
        ab = n_from_triang(b, a)
        with i>j and a>b.
    
    Csd (list of lists of 4D np.ndarrays)
        C(_i^a)(_j^b) = Csd[irrep][irrep2][i,a,j,b]
        
        These are the coefficients of double excitations that are products
        of a single excitation in each spirrep.
        If irrep == irrep2, it is the coefficient of an alpha/beta excitation
        If irrep != irrep2, it is the sum of alpha/alpha plus beta/beta
        excitation.
        
        Each entry is a 4D np.ndarray with shape
        
        (self.orbspace.corr[irrep],  self.orbspace.virt[irrep],
         self.orbspace.corr[irrep2], self.orbspace.virt[irrep2])
        
        storing the coefficients of the excitations from orbital i to a
        (that is within irrep) and from j to b (that is within irrep2).
        We store them in a "triangular" list of lists,
        with irrep >= irrep2 always:
        
        Csd[0][0]
        Csd[1][0]  Csd[1][1]
        Csd[2][0]  Csd[2][1]  Csd[2][2]
        ....
        
        Note that the order convention here is different that what is
        usually made in Grassmann. This should be corrected in the future
    
    
    Data Model:
    -----------
    [(StringIndexfor_SD)]
        TODO: Only get the CI coefficient (of the normalised version!)
        of that determinant
    
    len
        TODO: should be something more significant, such as the number
        determinants
        
    """
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, Index):
        raise NotImplementedError(
            '[] is not implemented for CISD_WaveFunction!')

    def __len__(self):
        raise NotImplementedError(
            'len() is not implemented for CISD_WaveFunction!')

    def __repr__(self):
        x = ['C0 = {}'.format(self.C0)]
        for irrep in self.spirrep_blocks(restricted=True):
            if 0 not in self.Cs[irrep].shape:
                x.append('Cs[{}]:\n {}'.
                         format(irrep, repr(self.Cs[irrep])))
        for irrep in self.spirrep_blocks(restricted=True):
            if 0 not in self.Cd[irrep].shape:
                x.append('Cd[{}]:\n {}'.
                         format(irrep, repr(self.Cd[irrep])))
        for irrep in self.spirrep_blocks(restricted=True):
            for irrep2 in range(irrep + 1):
                if 0 not in self.Csd[irrep][irrep2].shape:
                    x.append('Csd[{}][{}]:\n {}'.
                             format(irrep, irrep2,
                                    repr(self.Csd[irrep][irrep2])))
        return ('<(Partial) CISD Wave Function>\n'
                + super().__repr__() + '\n'
                + '\n'.join(x))

    def calc_memory(self, calc_args):
        """Calculate memory used by the wave function"""
        n_floats = 0.0
        for irrep in self.spirrep_blocks(restricted=True):
            n_floats += self.orbspace.corr[irrep] * self.orbspace.virt[irrep]
            n_floats += (triangular(self.orbspace.corr[irrep] - 1)
                         * triangular(self.orbspace.virt[irrep] - 1))
            for irrep2 in range(irrep + 1):
                n_floats += (self.orbspace.corr[irrep]
                             * self.orbspace.virt[irrep]
                             * self.orbspace.corr[irrep2]
                             * self.orbspace.virt[irrep2])
        return mem_of_floats(n_floats)

    def initialize_SD_lists(self):
        """Initialize lists of coefficients for singles and doubles with zeros.
        """
        self._set_memory()
        self.C0 = 0.0
        self.Cs = []
        self.Cd = []
        self.Csd = []
        for irrep in self.spirrep_blocks(restricted=True):
            self.Cs.append(np.zeros((self.orbspace.corr[irrep],
                                     self.orbspace.virt[irrep])))
            self.Cd.append(np.zeros((triangular(self.orbspace.corr[irrep] - 1),
                                     triangular(self.orbspace.virt[irrep] - 1))))
            self.Csd.append([])
            for irrep2 in range(irrep + 1):
                self.Csd[irrep].append(np.zeros((self.orbspace.corr[irrep],
                                                 self.orbspace.virt[irrep],
                                                 self.orbspace.corr[irrep2],
                                                 self.orbspace.virt[irrep2])))
    
    def make_Jac_Hess_overlap(self, restricted=None):
        """Construct the Jacobian and the Hessian of the function overlap.
        
        See fci.make_Jac_Hess_overlap for the detailed documentation.
        In the present case, however, we have only analytic=True
        
        And for restricted wave function (restrict=True).
        """
        if restricted is None:
            restricted = self.restricted
        if restricted and not self.restricted:
            raise ValueError(
                'Restricted calculation needs a restricted wave function')
        if not restricted:
            raise NotImplementedError('Currently onçy for restricted case!')
        slices_HJ = []
        nK = []
        for irp in self.spirrep_blocks(restricted=restricted):
            nK.append(self.orbspace.corr[irp] * self.orbspace.virt[irp])
            slice_start = 0 if irp == 0 else slices_HJ[-1].stop
            slices_HJ.append(slice(slice_start, slice_start + nK[-1]))
        Jac = np.zeros(sum(nK))
        Hess = -self.C0 * np.identity(len(Jac))
        for irp in self.spirrep_blocks(restricted=restricted):
            Jac[slices_HJ[irp]] = -np.ravel(self.Cs[irp],
                                            order='C')
            for ij in range(self.Cd[irp].shape[0]):
                j, i = ij_from_triang(ij)
                for ab in range(self.Cd[irp].shape[1]):
                    b, a = ij_from_triang(ab)
                    ia = (slices_HJ[irp].start
                          + n_from_rect(i, a, self.orbspace.virt[irp]))
                    jb = (slices_HJ[irp].start
                          + n_from_rect(j, b, self.orbspace.virt[irp]))
                    ib = (slices_HJ[irp].start
                          + n_from_rect(i, b, self.orbspace.virt[irp]))
                    ja = (slices_HJ[irp].start
                          + n_from_rect(j, a, self.orbspace.virt[irp]))
                    Hess[ia, jb] -= self.Cd[irp][ij, ab]
                    Hess[jb, ia] -= self.Cd[irp][ij, ab]
                    Hess[ib, ja] += self.Cd[irp][ij, ab]
                    Hess[ja, ib] += self.Cd[irp][ij, ab]
            for irp2 in range(irp + 1):
                Hess[slices_HJ[irp],
                     slices_HJ[irp2]] += np.reshape(self.Csd[irp][irp2],
                                                    (nK[irp], nK[irp2]))
                for i in range(self.orbspace.corr[irp]):
                    slice_i = slice(
                        slices_HJ[irp].start + i * self.orbspace.virt[irp],
                        slices_HJ[irp].start + (i + 1) * self.orbspace.virt[irp])
                    if irp == irp2 and (i + self.orbspace.corr[irp]) % 2 == 0:
                        Jac[slice_i] *= -1
                    for j in range(self.orbspace.corr[irp2]):
                        if (i + self.orbspace.corr[irp]
                                + j + self.orbspace.corr[irp2]) % 2 == 0:
                            continue
                        slice_j = slice(
                            slices_HJ[irp2].start + j * self.orbspace.virt[irp2],
                            slices_HJ[irp2].start + (j + 1) * self.orbspace.virt[irp2])
                        Hess[slice_i, slice_j] *= -1
                if irp2 < irp:
                    Hess[slices_HJ[irp2],
                         slices_HJ[irp]] = Hess[slices_HJ[irp],
                                                slices_HJ[irp2]].T
        if restricted:
            Jac *= 2
            Hess *= 2
        return Jac, Hess
    
    @classmethod
    def similar_to(cls, wf, wf_type):
        """Construct a WaveFunctionFCI with same basic attributes as wf"""
        new_wf = super().similar_to(wf)
        new_wf.wf_type = wf_type
        new_wf.initialize_SD_lists()
        return new_wf
    
    @classmethod
    def from_interm_norm(cls, IntermNormWaveFunction wf):
        """Load the wave function from a IntermNormWaveFunction."""
        cdef int irrep, irp, ii, inibl_D, inibl_i, inibl_j, a, b, ij
        cdef int n_virt, n_virt_i, n_virt_j
        cdef OccOrbital i, j
        cdef CISDWaveFunction new_wf
        new_wf = cls()
        new_wf.source = 'From IntermNormWaveFunction: ' + wf.source
        new_wf.restricted = wf.restricted
        if not new_wf.restricted:
            raise NotImplementedError(
                'Currently for restricted wave functions only!')
        new_wf.point_group = wf.point_group
        new_wf.orbspace.get_attributes_from(wf.orbspace)
        if wf.wf_type == 'CISD':
            new_wf.wf_type = wf.wf_type
        elif wf.wf_type == 'CCSD':
            new_wf.wf_type = (
                'CISD (with C_ij^ab = t_ij^ab + t_i^a t_j^b from CCSD)')
        elif wf.wf_type in ('CCD', 'BCCD'):
            new_wf.wf_type = (
                'CISD (with C_ij^ab = t_ij^ab from ' + wf.wf_type + ')')
        else:
            raise ValueError(
                'This is to be used for CISD, CCSD, CCD, and BCCD'
                + ' wave functions only!')
        if wf.wf_type in ('CCSD', 'CCD', 'BCCD'):
            logger.warning(
                'This is actually a CISD wave function using coefficients'
                + ' from coupled-cluster amplitudes!!')
        new_wf.initialize_SD_lists()
        new_wf.C0 = wf.C0
        if wf.has_singles:
            for irrep in range(new_wf.n_irrep):
                new_wf.Cs[irrep] += np.reshape(wf.amplitudes[wf.ini_blocks_S[irrep]:
                                                             wf.ini_blocks_S[irrep+1]],
                                               (wf.orbspace.corr[irrep],
                                                wf.orbspace.virt[irrep]))
                for ii in range(new_wf.orbspace.corr[irrep]):
                    if (new_wf.orbspace.corr[irrep] + ii) % 2 == 0:
                        new_wf.Cs[irrep][ii, :] *= -1
        ij = 0
        #
        # Order convention here is different that what is usually made
        # in Grassmann. This should be corrected in the future
        #
        for j, i in wf.occupied_pairs(ExcType.ALL):
            if i.spirrep != j.spirrep:
                n_virt_i = wf.orbspace.virt[i.spirrep]
                n_virt_j = wf.orbspace.virt[j.spirrep]
                inibl_D = wf.ini_blocks_D[ij, j.spirrep]
                new_wf.Csd[i.spirrep][j.spirrep][i.orbirp, :, j.orbirp, :] = \
                    2 * np.reshape(np.array(wf.amplitudes[inibl_D:
                                                          inibl_D + n_virt_i*n_virt_j]),
                                   (n_virt_j,n_virt_i)).T
                inibl_D = wf.ini_blocks_D[ij, i.spirrep]
                new_wf.Csd[i.spirrep][j.spirrep][i.orbirp, :, j.orbirp, :] -= \
                    np.reshape(np.array(wf.amplitudes[inibl_D:
                                                      inibl_D + n_virt_i*n_virt_j]),
                               (n_virt_i,n_virt_j))
                if wf.wf_type == 'CCSD':
                    inibl_i = wf.ini_blocks_S[i.spirrep] + i.orbirp*n_virt_i
                    inibl_j = wf.ini_blocks_S[j.spirrep] + j.orbirp*n_virt_j
                    new_wf.Csd[i.spirrep][j.spirrep][i.orbirp, :, j.orbirp, :] += \
                        2 * np.outer(wf.amplitudes[inibl_i:inibl_i+n_virt_i],
                                     wf.amplitudes[inibl_j:inibl_j+n_virt_j])
                if (i.orbirp + new_wf.orbspace.ref[i.spirrep]
                      + j.orbirp + new_wf.orbspace.ref[j.spirrep]) % 2 == 1:
                    new_wf.Csd[i.spirrep][j.spirrep][i.orbirp, :, j.orbirp, :] *= -1
            else:
                # i_irrep == j_irrep
                # and a_irrep == b_irrep, for this have only contrib from
                # determinants with same occupation as the reference.
                irp = i.spirrep
                inibl_D = wf.ini_blocks_D[ij, irp]
                n_virt = new_wf.orbspace.virt[irp]
                inibl_i = wf.ini_blocks_S[i.spirrep] + i.orbirp*n_virt
                inibl_j = wf.ini_blocks_S[j.spirrep] + j.orbirp*n_virt
                new_wf.Csd[irp][irp][i.orbirp, :, j.orbirp, :] += np.reshape(
                    np.array(wf.amplitudes[inibl_D:inibl_D + n_virt*n_virt]),
                    (n_virt,n_virt)).T
                new_wf.Csd[irp][irp][j.orbirp, :, i.orbirp, :] += np.reshape(
                    np.array(wf.amplitudes[inibl_D:inibl_D + n_virt*n_virt]),
                    (n_virt,n_virt))
                if wf.wf_type == 'CCSD':
                    new_wf.Csd[irp][irp][i.orbirp, :, j.orbirp, :] += np.outer(
                        wf.amplitudes[inibl_j:inibl_j+n_virt],
                        wf.amplitudes[inibl_i:inibl_i+n_virt])
                    new_wf.Csd[irp][irp][j.orbirp, :, i.orbirp, :] += np.outer(
                        wf.amplitudes[inibl_i:inibl_i+n_virt],
                        wf.amplitudes[inibl_j:inibl_j+n_virt])
                if i.orbirp == j.orbirp:
                    new_wf.Csd[irp][irp][i.orbirp, :, i.orbirp, :] /= 2
                if i.orbirp != j.orbirp:
                    ij_Cd = n_from_triang(j.pos_in_occ, i.pos_in_occ)
                    for a in range(n_virt):
                        for b in range(a):
                            # Increment ab instead?? Check the order
                            ab = n_from_triang(b, a)
                            new_wf.Cd[irp][ij_Cd, ab] = \
                                wf.amplitudes[inibl_D + n_from_rect(a, b, n_virt)] \
                                - wf.amplitudes[inibl_D + n_from_rect(b, a, n_virt)]
                            if wf.wf_type == 'CCSD':
                                new_wf.Cd[irp][ij_Cd, ab] += \
                                    wf.amplitudes[inibl_i + a] \
                                    * wf.amplitudes[inibl_j + b] \
                                    - wf.amplitudes[inibl_i + b] \
                                    * wf.amplitudes[inibl_j + a]
                    if (i.orbirp + j.orbirp) % 2 == 1:
                        new_wf.Csd[irp][irp][i.orbirp, :, j.orbirp, :] *= -1
                        new_wf.Csd[irp][irp][j.orbirp, :, i.orbirp, :] *= -1
                        new_wf.Cd[irp][ij_Cd, :] *= -1
            ij += 1
        for irrep in new_wf.spirrep_blocks(restricted=True):
            new_wf.Cs[irrep] /= wf.norm
            new_wf.Cd[irrep] /= wf.norm
            for irrep2 in range(irrep + 1):
                new_wf.Csd[irrep][irrep2] /= wf.norm
        return new_wf
