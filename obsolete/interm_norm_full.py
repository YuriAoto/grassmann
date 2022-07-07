"""An wave function in the intermediate normalisation (V3)

Classes:
--------

IntermNormWaveFunctionFull
- Same as IntermNormWaveFunction, but the T matix is fully storage (probably obsolete)
- Probably this could be an option in the other IntermNormWaveFunction
"""
import math
import logging

import numpy as np

from util.memory import mem_of_floats
from wave_functions.general import WaveFunction

class IntermNormWaveFunctionFull(WaveFunction):
    def __init__(self):
        super().__init__()
        self.amplitudes = None

    def calc_memory(self):
        """Calculate memory needed for amplitudes
        
        Parameters:
        -----------
        
        Return:
        -------
        A float, with the memory used to store the wave function amplitudes
        """
        return mem_of_floats((self.corr_orb[0]*self.virt_orb[0])**2)


    def initialize_amplitudes(self):
        """Initialize the list for the amplitudes."""
        self._set_memory('Amplitudes array in IntermNormWaveFunctionFull')
        self.amplitudes = np.zeros((self.corr_orb[0],
                                    self.corr_orb[0],
                                    self.virt_orb[0],
                                    self.virt_orb[0]))

    @classmethod
    def from_V2(cls,wf):
        new_wf = cls.similar_to(wf, wf.wf_type, wf.restricted)
        i=0
        j=0
        for ini_ij in range(0,len(wf),new_wf.virt_orb[0]**2):
            new_wf.amplitudes[i,j,:,:]=np.reshape(
                    wf.amplitudes[ini_ij:ini_ij+new_wf.virt_orb[0]**2],
                    (new_wf.virt_orb[0],new_wf.virt_orb[0]))
            if i != j:
                new_wf.amplitudes[j,i,:,:]=new_wf.amplitudes[i,j,:,:].T
            if i < j:
                i+=1
            else:
                i=0
                j+=1
        return new_wf



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

    def __getitem__(self,key):
        return self.amplitudes[key]
