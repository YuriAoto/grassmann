"""The CISD wave function

Classes:
--------

Wave_Function_CISD
"""

import numpy as np

from util import triangular, get_n_from_triang
from wave_functions import general


class Wave_Function_CISD(general.Wave_Function):
    """The CISD wave function
    
    For the moment, only restricted and including only
    the coefficients that contribute to the distance to the
    Grassmannian of the reference (that is, have the same occupation
    of the reference in each spirrep block.
    
    Atributes:
    ----------
    C0 (float)
        The coefficient of the reference determinant
    
    Cs (list of 2D np.ndarrays)
        C_i^a = Cs[irrep][i,a]
        
        Each element of this list is associated to an irrep.
        There are n_irrep entries (only restricted wave functions).
        Each entry is a 2D np.ndarray with shape
        
        (self.n_corr_orb[irrep], self.n_ext[irrep])
        
        storing the coefficients of the excitations from orbital i to a
    
    Cd (list of 2D np.ndarrays)
        C_ij^ab = Cd[irrep][ij,ab]
        
        These are the coefficients of double excitations within
        irrep that are pure double excitations in for a single spin
        Each element of this list is associated to an irrep.
        There are n_irrep entries (only restricted wave functions).
        Each entry is a 2D np.ndarray with shape
        
        (triangular(self.n_corr_orb[irrep] - 1),
         triangular(self.n_ext[irrep] - 1))
        
        and store the coefficients for the excitation i,j to a,b.
        Indices for the pairs i,j and a,b are stored in triangular order:
        ij = get_n_from_triang(i, j, with_diag=False)
        ab = get_n_from_triang(a, b, with_diag=False)
        with i>j and a>b.
    
    Csd (list of lists of 4D np.ndarrays)
        C(_i^a)(_j^b) = Cd[irrep][irrep2][i,a,j,b]
        
        These are the coefficients of double excitations that are products
        of a single excitation in each spirrep.
        Each entry is a 4D np.ndarray with shape
        
        (self.n_corr_orb[irrep], self.n_ext[irrep],
         self.n_corr_orb[irrep2], self.n_ext[irrep2])
        
        storing the coefficients of the excitations from orbital i to a
        (that is within irrep) and from j to b (that is within irrep2).
        We store them in a "triangular" list of lists,
        with irrep >= irrep2 always:
        
        Csd[0][0]
        Csd[1][0]  Csd[1][1]
        Csd[2][0]  Csd[2][1]  Csd[2][2]
        ....
    
    Data Model:
    -----------
    [(String_Index_for_SD)]
        Only get the CI coefficient (of the normalised version!)
        of that determinant
    
    len
        len(singles) + len(doubles).
        TODO: should be something more significant, such as the number
        determinants
        
    """
    def __init__(self):
        super().__init__()
        self.C0 = None
        self.Cs = None
        self.Cd = None
        self.Csd = None
    
    def __getitem__(self, I):
        raise NotImplementedError('[] is not implemented for CISD_WF!')

    def __len__(self):
        raise NotImplementedError('len() is not implemented for CISD_WF!')

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
                             format(irrep, irrep2, repr(self.Csd[irrep][irrep2])))
        return ('<(Partial) CISD Wave Function>\n'
                + super().__repr__() + '\n'
                + '\n'.join(x))

    def initialize_SD_lists(self):
        """
        Initialise lists of coefficients for singles and doubles with zeros.
        """
        self.C0 = 0.0
        self.Cs = []
        self.Cd = []
        self.Csd = []
        for irrep in self.spirrep_blocks(restricted=True):
            self.Cs.append(np.zeros((self.n_corr_orb[irrep],
                                     self.n_ext[irrep])))
            self.Cd.append(np.zeros((triangular(self.n_corr_orb[irrep] - 1),
                                     triangular(self.n_ext[irrep] - 1))))
            self.Csd.append([])
            for irrep2 in range(irrep + 1):
                self.Csd[irrep].append(np.zeros((self.n_corr_orb[irrep],
                                                 self.n_ext[irrep],
                                                 self.n_corr_orb[irrep2],
                                                 self.n_ext[irrep2])))
    
    def string_indices(self,
                       spirrep=None,
                       coupled_to=None,
                       no_occ_orb=False,
                       only_ref_occ=False,
                       only_this_occ=None):
        raise NotImplementedError(
            'string_indices not implemented for CISD_WF!')
    
    def make_Jac_Hess_overlap(self, analytic=True):
        raise NotImplementedError(
            'make_Jac_Hess not implemented for CISD_WF!')
    
    def calc_wf_from_z(self, z, just_C0=False):
        raise NotImplementedError(
            'calc_wf_from_z not implemented for CISD_WF!')
    
    def change_orb_basis(self, U, just_C0=False):
        raise NotImplementedError(
            'change_orb_basis not implemented for CISD_WF!')
        
    @classmethod
    def from_intNorm(cls, intN_wf):
        """Load the wave function from a Wave_Function_Int_Norm."""
        new_wf = cls()
        new_wf.source = 'From int. norm. WF> ' + intN_wf.source
        new_wf.restricted = intN_wf.restricted
        new_wf.point_group = intN_wf.point_group
        new_wf.n_core = intN_wf.n_core
        new_wf.n_act = intN_wf.n_act
        new_wf.orb_dim = intN_wf.orb_dim
        new_wf.ref_occ = intN_wf.ref_occ
        new_wf.WF_type = intN_wf.WF_type
        if new_wf.WF_type != 'CISD':
            raise ValueError(
                'This is to be used for CISD wave functions only!')
        if not new_wf.restricted:
            raise NotImplementedError(
                'Currently for restricted wave functions only!')
        new_wf.initialize_SD_lists()
        new_wf.C0 = 1.0
        for irrep in new_wf.spirrep_blocks(restricted=True):
            new_wf.Cs[irrep] += intN_wf.singles[irrep]
            for i in range(new_wf.n_corr_orb[irrep]):
                if (new_wf.n_corr_orb[irrep] + i) % 2 == 0:
                    new_wf.Cs[irrep][i, :] *= -1
        for N, doubles in enumerate(intN_wf.doubles):
            i, j, i_irrep, j_irrep, exc_type = intN_wf.ij_from_N(N)
            if i_irrep != j_irrep:
                # This is how it was before!!
                # Note that this way it is wrong, but it gives the same result
                # as in algorithm 1. Thus: string_indices in int_N_WF is wrong
                # (with analogous mistake) INVESTIGATE!!!
                # for a in range(new_wf.n_ext[i_irrep]):
                #     for b in range(new_wf.n_ext[j_irrep]):
                #         new_wf.Csd[i_irrep][j_irrep][i, a, j, b] = \
                #             doubles[j_irrep][b, a]
                new_wf.Csd[i_irrep][j_irrep][i, :, j, :] = \
                    2 * doubles[i_irrep][:, :] - doubles[j_irrep][:, :].T
                if (i + new_wf.ref_occ[i_irrep]
                      + j + new_wf.ref_occ[j_irrep]) % 2 == 1:
                    new_wf.Csd[i_irrep][j_irrep][i, :, j, :] *= -1
            else:
                if i != j:
                    ij = get_n_from_triang(i, j, with_diag=False)
                for a in range(new_wf.n_ext[i_irrep]):
                    for b in range(new_wf.n_ext[j_irrep]):
                        if a == b and i == j:
                            new_wf.Csd[i_irrep][j_irrep][i, a, j, b] +=\
                                doubles[i_irrep][a, b]
                        elif a == b:  # i != j
                            new_wf.Csd[i_irrep][j_irrep][i, a, j, b] +=\
                                doubles[i_irrep][a, b]
                            new_wf.Csd[i_irrep][j_irrep][j, a, i, b] +=\
                                doubles[i_irrep][a, b]
                        elif i == j:  # a != b
                            new_wf.Csd[i_irrep][j_irrep][i, a, j, b] +=\
                                (doubles[i_irrep][a, b]
                                 + doubles[i_irrep][b, a]) / 2
                        else:  # a != b and i != j
                            if a > b:
                                new_wf.Csd[i_irrep][j_irrep][i, a, j, b] +=\
                                    doubles[i_irrep][a, b]
                                new_wf.Csd[i_irrep][j_irrep][j, a, i, b] +=\
                                    doubles[i_irrep][b, a]
                            else:  # b > a
                                new_wf.Csd[i_irrep][j_irrep][j, a, i, b] +=\
                                    doubles[i_irrep][b, a]
                                new_wf.Csd[i_irrep][j_irrep][i, a, j, b] +=\
                                    doubles[i_irrep][a, b]
                if i != j:
                    for a in range(new_wf.n_ext[i_irrep]):
                        for b in range(a):
                            # Increment ab instead?? Check the order
                            ab = get_n_from_triang(a, b, with_diag=False)
                            new_wf.Cd[i_irrep][ij, ab] =\
                                (doubles[i_irrep][b, a]
                                 - doubles[i_irrep][a, b])
                if (i + j) % 2 == 1:
                    new_wf.Csd[i_irrep][j_irrep][i, :, j, :] *= -1
                    new_wf.Csd[i_irrep][j_irrep][j, :, i, :] *= -1
                    new_wf.Cd[i_irrep][ij, :] *= -1
        if intN_wf.norm is None:
            intN_wf.calc_norm()
        new_wf.C0 /= intN_wf.norm
        for irrep in new_wf.spirrep_blocks(restricted=True):
            new_wf.Cs[irrep] /= intN_wf.norm
            new_wf.Cd[irrep] /= intN_wf.norm
            for irrep2 in range(irrep + 1):
                new_wf.Csd[irrep][irrep2] /= intN_wf.norm
        return new_wf
