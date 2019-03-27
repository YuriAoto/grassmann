"""f(x) = <0|FCI>, for |0> Slater determinant and |FCI> from Molpro

# Parametrisation of |0>:

The slater determinants are parametrised by the orbital rotations:

|0'> = exp(-K) |0>

where K = K(alpha) + K(beta)
 K(sigma) = sum_{i,a} K_i^a (a_i^a - a_a^i)

 K_1^{n+1}    K_1^{n+2}   ...  K_1^orb_dim
 K_2^{n+1}    K_2^{n+2}   ...  K_2^orb_dim
          ...
              K_i^a
          ...
 K_n^{n+1}    K_n^{n+2}   ...  K_n^orb_dim

  --> packed as:

   Jac[0]   = K_1^{n+1}
   Jac[1]   = K_2^{n+1}
      ...
   Jac[n-1] = K_n^{n+1}
   Jac[0   + n] = K_1^{n+2}
   Jac[1   + n] = K_2^{n+2}
      ...
   Jac[n-1 + n] = K_n^{n+2}
   Jac[0   + 2n] = K_1^{n+3}
      ...
   Jac[i-1 + (a - n - 1)*n] = K_i^a
     ...
   Jac[n-1 + (orb_dim - n - 1)*n = n*(orb_dim - n) - 1] = K_n^orb_dim    <<< last alpha
   Jac[n*(orb_dim - n) + 0] = K_1^2     <<< first beta = Molpro_FCI_Wave_Function.beta_shift
     ...

   -> beta are shifted by  b_shift = n*(orb_dim - n)

# |FCI>

The wave function |FCI> is written as a FCI wave function (that is, with
all possible Slater determinants), but it can be the true FCI wave function
or a CID, or CISD wave function (with explicit zeros for other determinants)

TODO: CCD, CCSD, MP2.

History:
    Aug 2018 - Start
    Mar 2019 - Add CISD wave function
               Add to git
Yuri
"""
import numpy as np
from numpy import linalg
from scipy.linalg import expm, lu
import copy
import math
import re

molpro_FCI_header = ' PROGRAM * FCI (Full CI)     Author: P.J. Knowles, 1984\n'
molpro_MRCI_header = ' PROGRAM * CI (Multireference internally contracted CI)     '\
                     + 'Authors: H.-J. Werner, P.J. Knowles, 1987\n'
molpro_CISD_header = ' PROGRAM * CISD (Closed-shell CI(SD))     '\
                     + 'Authors: C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n'
molpro_CCSD_header = ' PROGRAM * CCSD (Closed-shell coupled cluster)'\
                     + '     Authors: C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n'

MRCI_ref_str = ' Reference coefficients greater than 0.0000000\n'
MRCI_sgl_str = ' Coefficients of singly external configurations greater than 0.0000000\n'
MRCI_dbl_str = ' Coefficients of doubly external configurations greater than 0.0000000\n'

CC_sgl_str = ' Singles amplitudes (print threshold =  0.000E+00):\n'
CC_dbl_str = ' Doubles amplitudes (print threshold =  0.000E+00):\n'


class Molpro_FCI_Wave_Function():
    """A wave function with the structure of Molpro's FCI wave function

    Currently, only for Ms=0, that is, equal number of alpha and beta orbitals.

    Atributes:
    determinants (list) : a list of determinants, each determinant being a list
                          similar to Molpro's FCI output:
                          [Ci, a1, a2, ..., an_a, b1, b2, ..., bn_a]
                          where Ci is the (float) coefficient of the determinant 
                          in the normalised wave function, a_i are the occupied
                          alpha orbitals and b_i the occupied beta orbitals.
                          Ex: [0.9, 1, 2, 1, 2] is the determinant with the first
                          two alpha and beta orbitals occupied, with a coefficient
                          or 0.9.
    orb_dim (int) : dimension of the (spatial) orbital space
    n_frozen (int) : number of frozen orbitals (assumed to be equal for alpha and beta)
    n_elec (int) : number of electrons
    n_alpha (int) : number of alpha electrons
    n_beta (int) : number of beta electrons (always equals n_alpha in the current
                   implementation)
    beta_shift (int) : it is equal to n_alpha*(orb_dim - n_alpha) and is the
                       first position of the parameters in beta space (see the
                       documentation of the module)
    WF_type (str) : type of wave function. Currently it can be FCI, CID and CISD.
    """

    def __init__(self,
                 file_name = None,
                 state='1.1',
                 FCI_file_name = None):
        """Initialize a FCI wave function

        Parameters:
        file_name - the molpro output with the FCI wave function
        state - state of interest
        FCI_file_name - if not None, file_name is assumed to be of a non-FCI
                        wave function, and FCI_file_name has a FCI wave function
                        to be used as template. In the end, all
                        coefficients will be zero, except those from the
                        wave function of file_name.
                        if None, file_name is assumed to be a FCI wave function
        """
        self.determinants = []
        self.orb_dim = None
        self.n_frozen = None
        self.n_elec = None
        self.n_alpha = None
        self.n_beta = None
        self.beta_shift = None
        self.WF_type = None
        sgn_invert = False
        S = 0.0 #TMP
        if FCI_file_name is None:
            self.WF_type = 'FCI'
            FCI_file_name = file_name
        if file_name is not None:
            FCI_found = False
            FCI_prog_found = False
            # Load the structure of a FCI wave function
            with open (FCI_file_name, 'r') as f:
                for l in f:
                    if 'FCI STATE  '+state+' Energy' in l and 'Energy' in l:
                        FCI_found = True
                        continue
                    if molpro_FCI_header == l:
                        FCI_prog_found = True
                        continue
                    if FCI_found:
                        if 'CI Vector' in l:
                            continue
                        if 'EOF' in l:
                            break
                        lspl = l.split()
                        if self.WF_type is None:
                            coef = 0.0
                            S += float(lspl[0])**2 # TMP
                        else:
                            coef = float(lspl[0])
                            if len(self.determinants) == 0:
                                sgn_invert = coef < 0.0
                            if sgn_invert:
                                coef = -coef
                        det_descr = lspl[pos_a_ini:pos_a_fin] + lspl[pos_b_ini:pos_b_fin]
                        det_descr = list(map(lambda x: int(x) - self.n_frozen, det_descr))
                        self.determinants.append( [coef] + det_descr)
                    elif FCI_prog_found:
                        if 'Frozen orbitals:' in l:
                            self.n_frozen = int(l.split()[2])
                        if 'Active orbitals:' in l:
                            self.orb_dim = int(l.split()[2])
                        if 'Active electrons:' in l:
                            self.n_elec = int(l.split()[2])
                            if self.n_elec%2 != 0:
                                raise Exception('Only even number of electrons!')
                            self.n_alpha = self.n_elec//2
                            self.n_beta = self.n_alpha
                            self.beta_shift = self.n_alpha*(self.orb_dim - self.n_alpha)
                            pos_a_ini = 1 + self.n_frozen
                            pos_a_fin = pos_a_ini + self.n_alpha
                            pos_b_ini = 1 + 2*self.n_frozen + self.n_alpha
                            pos_b_fin = pos_b_ini + self.n_beta

#            print('norm of FCI: ', math.sqrt(S))
            if not FCI_prog_found or not FCI_found:
                raise Exception('FCI wave function not found!')

            # Load the required external wave function
            if self.WF_type is None:
                ref_found = False
                sgl_found = False
                dbl_found = False
                ref = 0.0
                singles = []
                doubles = []
                with open (file_name, 'r') as f:
                    for l in f:
                        if self.WF_type is None:
                            if molpro_MRCI_header == l:
                                self.WF_type = 'MRCI'
                            if molpro_CISD_header == l or molpro_CCSD_header == l:
                                self.WF_type = 'CISD' if molpro_CISD_header == l else 'CCSD'

                        elif self.WF_type == 'MRCI':
                            if MRCI_ref_str == l:
                                ref_found = True
                            elif MRCI_sgl_str == l:
                                sgl_found = True
                            elif MRCI_dbl_str == l:
                                dbl_found = True
                            elif dbl_found:
                                lspl = l.split()
                                if len(lspl) == 9:
                                    doubles.append([lspl[-1],
                                                    int(lspl[1].split('.')[0]),
                                                    int(lspl[2].split('.')[0]),
                                                    int(lspl[3].split('.')[0]),
                                                    int(lspl[4].split('.')[0]),
                                                    int(lspl[5])])
                            elif sgl_found:
                                lspl = l.split()
                                if len(lspl) == 3:
                                    singles.append([lspl[-1],
                                                    lspl[0],
                                                    int(lspl[1].split('.')[0])])
                            elif ref_found:
                                lspl = l.split()
                                if len(lspl) == 2:
                                    ref = lspl[1]
                            if 'RESULTS' in l:
                                if not ref_found:
                                    raise Exception('Reference coefficients not found!')
                                if not dbl_found:
                                    raise Exception('Double excitations not found!')
                                break

                        elif self.WF_type == 'CCSD' or self.WF_type == 'CISD':

                            if CC_sgl_str == l:
                                sgl_found = True
                            elif CC_dbl_str == l:
                                dbl_found = True
                            elif dbl_found:
                                lspl = l.split()
                                if len(lspl) == 7:
                                    doubles.append([lspl[-1],
                                                    int(lspl[0]),
                                                    int(lspl[1]),
                                                    int(lspl[4])+self.n_alpha,
                                                    int(lspl[5])+self.n_alpha])
                            elif sgl_found:
                                lspl = l.split()
                                if len(lspl) == 4:
                                    singles.append([lspl[-1],
                                                    int(lspl[0]),
                                                    int(lspl[2])+self.n_alpha])
                            if 'RESULTS' in l:
                                if not dbl_found:
                                    raise Exception('Double excitations not found!')
                                break

            if self.WF_type is None:
                raise Exception('We found no wave function in ' + file_name)

            elif self.WF_type == 'MRCI':
                self.load_MRCI_WF(ref, singles, doubles)

            elif self.WF_type == 'CISD':
                self.load_CISD_WF(singles, doubles)

            elif self.WF_type == 'CCSD':
                self.load_CCSD_WF(singles, doubles)
                                    

    def load_CISD_WF(self, Csgl, Cdbl):
        """Load the CISD wave function, written as CCSD ansatz"""
        # print('Amplitudes of single excitations:')
        # for i in Csgl:
        #     print(i)
        # print('-------------------')
        # print()
        # print('Amplitudes of double excitations:')
        # for i in Cdbl:
        #     print(i)
        # print('-------------------')
        # print()
        for det in self.determinants:
            rk = rank_of_exc(det)
            if rk == 0:
                det[0] = 1.0
            if rk == 1 or rk == 2:
                exc_descr = [] # occ1[, occ2], virt1[, virt2]
                exc_type = ''
                for i in range(self.n_alpha):
                    if (i+1) not in det[1:self.n_alpha+1]:
                        exc_descr.append(i+1)
                        exc_type += 'a'
                for i in range(self.n_alpha):
                    if (i+1) not in det[self.n_alpha+1:]:
                        exc_descr.append(i+1)
                        exc_type += 'b'
                for i in det[1:]:
                    if i > self.n_alpha:
                        exc_descr.append(i)
#               print('This det: ', det, ' exc_descr = ', exc_descr, 'exc_type = ', exc_type)
                if rk == 1:
                    if len(exc_descr) != 2:
                        raise Exception('Length of exc_descr is not 2 for single excitation.')
                    for s in Csgl:
                        if s[1:] == exc_descr:
                            det[0] = float(s[0]) if (self.n_alpha+s[1])%2 == 0 else -float(s[0])
                            break
                if rk == 2:
                    if len(exc_descr) != 4:
                        raise Exception('Length of exc_descr is not 4 for double excitation.')
                    for d in Cdbl:
                        if (set(d[1:3]) == set(exc_descr[0:2]) and
                            set(d[3:5]) == set(exc_descr[2:4])):
                            if d[1] == d[2] and d[3] == d[4]:
                                det[0] += float(d[0])
                            elif d[1] == d[2]:
                                det[0] += float(d[0])/2
                            elif d[3] == d[4]:
                                det[0] += float(d[0]) if (d[1] + d[2])%2 == 0 else -float(d[0])
                            else:
                                if exc_type in ['aa', 'bb']:
                                    if d[3] < d[4]:
                                        det[0] += float(d[0]) if (d[1] + d[2])%2 == 0 else -float(d[0])
                                    else:
                                        det[0] += float(d[0]) if (d[1] + d[2])%2 == 1 else -float(d[0])
                                elif exc_type == 'ab' and d[3] == exc_descr[2]:
                                    det[0] += float(d[0]) if (d[1] + d[2])%2 == 0 else -float(d[0])
        self.normalise()

    def load_CCSD_WF(self, Tsgl, Tdbl):
        """Load the CCSD wave function."""
        print('Amplitudes of single excitations:')
        for i in Tsgl:
            print(i)
        print('-------------------')
        print()
        print('Amplitudes of double excitations:')
        for i in Tdbl:
            print(i)
        print('-------------------')
        print()
        raise Exception('load_CCSD_WF: Not implemented yet!')

    def load_MRCI_WF(self, C0, Csgl, Cdbl):
        """Load the MRCISD wave function."""
        sqrt_2 = math.sqrt(2.0)
        print('ref coef: ', C0)
        print()
        print('Singly exc:')
        for i in Csgl:
            print(i)
        print('-------------------')
        print()
        print('Doubly exc:')
        for i in Cdbl:
            print(i)
        print('-------------------')
        print()
        raise Exception ('load_MRCI_WF is not implemented correctly!')

        for det in self.determinants:
            if rank_of_exc(det) == 0:
                det[0] = float(C0)
            if rank_of_exc(det) == 1:
                occ_of_occ = [0] * self.n_alpha
                ext_occ = None
                for i in range(self.n_alpha):
                    if (i+1) in det[1:self.n_alpha+1]:
                        occ_of_occ[i] += 1
                    if (i+1) in det[self.n_alpha+1:]:
                        occ_of_occ[i] += 1
                orb_ini = occ_of_occ.index(1)+1
                occ_of_occ = ''.join(map(str, occ_of_occ)).replace('1','\\')
                if det[self.n_alpha] > self.n_alpha:
                    ext_occ = det[self.n_alpha]
                if det[-1] > self.n_alpha:
                    if ext_occ is not None:
                        raise Exception(
                            'Found more than one externally occ orb in single excitation!')
                    else:
                        ext_occ = det[-1]

                for s in Csgl:
                    if s[1] == occ_of_occ and s[2] == ext_occ:
                        det[0] = (float(s[0])/sqrt_2 if (self.n_alpha+orb_ini)%2==0 else
                                  -float(s[0])/sqrt_2)
                        print('Replacing coef of singles: ', det, s)
                        break

            if rank_of_exc(det) == 2:
                exc_descr = [] # occ1, occ2, virt1, virt2, NP (-1 => excitation from same spin)
                exc_type = ''
                for i in range(self.n_alpha):
                    if (i+1) not in det[1:self.n_alpha+1]:
                        exc_descr.append(i+1)
                        exc_type += 'a'
                for i in range(self.n_alpha):
                    if (i+1) not in det[self.n_alpha+1:]:
                        exc_descr.append(i+1)
                        exc_type += 'b'
                for i in det[1:]:
                    if i > self.n_alpha:
                        exc_descr.append(i)
                if len(exc_descr) != 4:
                    raise Exception('Len of exc_descr is not 4 for double excitation.')
                exc_descr.append(exc_type)
                
#               print('This det: ', det, ' exc_descr = ', exc_descr)
                for d in Cdbl:
                    if (set(d[1:3]) == set(exc_descr[0:2]) and
                        set(d[3:5]) == set(exc_descr[2:4])):
                        print('Replacing coef of doubles: ', det, ':', exc_descr,'=', d)
                        if d[1] == d[2] and d[3] == d[4]:
                            det[0] += float(d[0])
                        elif d[1] == d[2]:
                            det[0] += float(d[0])
                        elif d[3] == d[4]:
                            det[0] += (float(d[0])/sqrt_2 if (d[1] + d[2])%2 == 0
                                       else -float(d[0])/sqrt_2)
                        else:
                            if exc_descr[4] in ['aa', 'bb']:
                                if d[5] == -1:
                                    det[0] += (float(d[0])/2 if (d[1] + d[2])%2 == 1
                                               else -float(d[0])/2)
                            elif exc_descr[4] == 'ab':
                                if d[5] == 1:
                                    det[0] += (float(d[0])/2 if (d[1] + d[2])%2 == 0
                                               else -float(d[0])/2)
                                else:
                                    det[0] += (-float(d[0])/2 if (d[1] + d[2])%2 == 0
                                               else float(d[0])/2)

    def __str__(self):
        """Return a string version of the wave function."""
        x = '='*50 + '\n'
        x += 'n frozen: ' + str(self.n_frozen) + '\n'
        x += 'n electrons: ' + str(self.n_elec) + '\n'
        x += 'n alpha: ' + str(self.n_alpha) + '\n'
        x += 'n beta: ' + str(self.n_beta) + '\n'
        x += 'orb dim: ' + str(self.orb_dim) + '\n'
        x += 'beta shift: ' + str(self.beta_shift) + '\n'
        for det in self.determinants:
            x += '{0:10.12f} '.format(det[0])
            for i_occ in det[1:]:
                x += '{0:4d} '.format(i_occ)
            x += ' >> RANK: {0:d}\n'.format(rank_of_exc(det))
        x += '='*50 + '\n'
        return x

    def normalise(self):
        """Normalise the wave function to unity"""
        S = 0.0
        for det in self.determinants:
            S += det[0]**2
        S = math.sqrt(S)
        for det in self.determinants:
            det[0] /= S


def rank_of_exc(determinant):
    """Return the excitation rank of the determinant.

    It works correctly only for closed shell case, with n_alpha = n_beta.
    """
    rank = 0
    N = (len(determinant)-1)//2 # number of occupied spatial orbitals
    for i in determinant[1:]:
        if i > N:
            rank += 1
    return rank

def print_matrix(X, f_str):
    """Print the matrix X to f_str."""
    for i in X:
        for j in i:
            f_str.write(' {0:10.6f} '.format(j)\
                        if abs(j) > 1.0E-7 else
                        (' ' + '-'*10 + ' '))
        f_str.write('\n')
    f_str.write('\n')


def get_trans_max_coef(wf):
    """
    Return Ua and Ub that transforms the orbitals
    such that the largest coefficient of wf
    is in the first determinant: 1,2,3,..., 1,2,3,...
    """
    det_max_coef = None
    for det in wf.determinants:
        if det_max_coef is None or abs(det[0]) > abs(det_max_coef[0]):
            det_max_coef = det
    Ua = np.identity(wf.orb_dim)
    Ub = np.identity(wf.orb_dim)
    # alpha orbitals
    extra_in_det = []
    miss_in_det = []
    for i in det_max_coef[1:wf.n_alpha+1]:
        if not (i-1 in range(wf.n_alpha)):
            extra_in_det.append(i-1)
    for i in range(wf.n_alpha):
        if not (i+1 in det_max_coef[1:wf.n_alpha+1]):
            miss_in_det.append(i)
    for i,j in zip(extra_in_det, miss_in_det):
        Ua[i][i] = 0.0
        Ua[j][j] = 0.0
        Ua[j][i] = 1.0
        Ua[i][j] = 1.0
    # beta orbitals
    extra_in_det = []
    miss_in_det = []
    for i in det_max_coef[wf.n_alpha+1:]:
        if not (i-1 in range(wf.n_beta)):
            extra_in_det.append(i-1)
    for i in range(wf.n_beta):
        if not (i+1 in det_max_coef[wf.n_alpha+1:]):
            miss_in_det.append(i)
    for i,j in zip(extra_in_det, miss_in_det):
        Ub[i][i] = 0.0
        Ub[j][j] = 0.0
        Ub[j][i] = 1.0
        Ub[i][j] = 1.0
#    print(Ua)
#    print()
#    print(Ub)
#    print()
    return Ua, Ub

def transform_wf(wf, Ua, Ub, just_C0 = False):
    """Transform the wave function as induced by a transformation in the orbital basis

    Parameters

    wf   the initial wave function as Molpro_FCI_Wave_Function
    Ua   the orbital transformation for alpha orbitals
    Ub   the orbital transformation for beta orbitals
    just_C0   If True, calculates the coefficients of the initial determinant only 
              (default False)
    """
    new_wf = Molpro_FCI_Wave_Function()
    new_wf.n_frozen = wf.n_frozen
    new_wf.orb_dim = wf.orb_dim
    new_wf.n_elec = wf.n_elec
    new_wf.n_alpha = wf.n_alpha
    new_wf.n_beta = wf.n_beta
    new_wf.beta_shift = wf.beta_shift
    n_calcs = 0
    for det_J in wf.determinants:
        new_det = copy.copy(det_J)
        new_det[0] = 0.0
        for det_I in wf.determinants:
            if abs(det_I[0])>1.0E-11:
                n_calcs += 1
                U_minor = np.zeros((2*wf.n_alpha,2*wf.n_beta))
                for i,orb_I in enumerate(det_I[1:]):
                    for j,orb_J in enumerate(det_J[1:]):
                        if i < wf.n_alpha and j < wf.n_alpha:
                            U_minor[i][j] = Ua[orb_I-1][orb_J-1]
                        elif i >= wf.n_alpha and j >= wf.n_alpha:
                            U_minor[i][j] = Ub[orb_I-1][orb_J-1]
                U_det_minor_IJ = linalg.det(U_minor)
                new_det[0] += det_I[0] * U_det_minor_IJ
                #print('='*20)
                #print('det_J: ', det_J)
                #print('det_I: ', det_I)
                #print(U_minor)
                #print('='*20)
        new_wf.determinants.append(new_det)
        if just_C0:
            break
    #print('Number of det calculations:',  n_calcs)
    return new_wf

def transform_wf_2(wf, Ua, Ub, just_C0 = False):
    """Similar to transform_wf, but didn't work (it should be faster)."""
    new_wf = copy.copy(wf)
    n_calcs = 0
#    print(wf)
    tUa, tLa = lu(Ua, permute_l=True) ### CHECK
    tUb, tLb = lu(Ub, permute_l=True) ### CHECK
    tUa = linalg.inv(tUa)
    tUb = linalg.inv(tUb)
    tLa = np.identity(len(tUa)) - tLa
    tLb = np.identity(len(tUa)) - tLb
    ta = tUa + tLa
    tb = tUb + tLb
    for k_tot in range(2*(wf.orb_dim - wf.n_frozen)):
        coeff_delta = []
        if k_tot < wf.orb_dim - wf.n_frozen:
            k = k_tot
            spin_shift = 0
            t = ta
#            print('k in alpha')
        else:
            k = k_tot -(wf.orb_dim - wf.n_frozen)
            spin_shift = new_wf.n_alpha
            t = tb
#            print('k in beta')
#        print('k = ', k, '; spin_shift = ', spin_shift)
        for det_J in new_wf.determinants:
            c_delta = 0.0
            for det_I in new_wf.determinants:
                if abs(det_I[0])>1.0E-11:
                    n_diff = 0
                    p = None
                    if not (k in det_I[1+spin_shift:1+spin_shift+new_wf.n_alpha]):
                        continue
                    for i_ind,i in enumerate(det_J[1:]):
                        if i_ind < new_wf.n_alpha:
                            if not i in det_I[1:new_wf.n_alpha+1]:
#                                print('detI = ', det_I)
#                                print('detJ = ', det_J)
                                n_diff += 1
                                if not spin_shift:
                                    p = i-1
#                                    print('current p (alpha):', p)
                        else:
                            if not i in det_I[new_wf.n_alpha+1:]:
                                n_diff += 1
                                if spin_shift:
                                    p = i - (wf.orb_dim - wf.n_frozen) -1
#                                    print('current p (beta):', p)
                    if n_diff > 1:
                        continue
                    if n_diff == 0:
                        c_delta += det_I[0]*(t[k][k]-1)
                    else:
                        if k in det_J[1+spin_shift:1+spin_shift+new_wf.n_alpha]:
                            continue
#                        print(p)
#                        try:
                        c_delta += det_I[0]*t[p][k]
#                        except:
#                            print(p,k)
#                            raise
            coeff_delta.append(c_delta)
        for det_J, c_delta in zip(new_wf.determinants, coeff_delta):
            det_J[0] += c_delta
    #print('Number of det calculations:',  n_calcs)
    return new_wf


def get_position_in_jac(i, a, n, spin_shift):
    """Pack Jacobian: return the position.
    """
    return i-1 + (a - n-1)*n + spin_shift

def get_i_a_from_pos_in_jac(pos, n, b_shift):
    """Unpack Jacobian: returns (i,a).
    """
    s = 0 if pos < b_shift else b_shift
    return (1 + (pos - s)%n, # i
            1 + (pos - s)//n + n) # a

def construct_Jac_Hess(wf, f_str, analytic = True, verb = 0):
    """Construct the Jacobian and the Hessian of the function overlap.

    the function is f(x) = <wf(x), det1(x)>
    where x parametrises the orbital rotations and
    det1 is the first determinant in wf.


    Parameters
    wf        the external (FCI_ wave function, as Molpro_FCI_Wave_Function
    f_str     the output stream
    analytic  if True, calculates the Jacobian and the Hessian by the 
              analytic expression, if False calculate numerically
              (default = True)
    verb      the print level (default = 0)
              
    

    Returns the tuple (Jac, Hess), with the Jacobian and the 
    Hessian
    """
    if verb > 50:
        f_str.write('='*50 + '\n')
        f_str.write('Building Jacobian and Hessian')
        f_str.write(('Analytic' if analytic else 'Numerical') + ' procudure')
    n_param = wf.n_alpha*(wf.orb_dim - wf.n_alpha)*2
    Jac = np.zeros(n_param)
    Hess = np.zeros((n_param, n_param))
    if analytic:
        for det in wf.determinants:
            exc_rank = rank_of_exc(det)

            if exc_rank == 0:
                for i in range(n_param):
                    Hess[i][i] = -det[0]
                if verb > 50:
                    f_str.write('Setting diagonal of Hess to {0:10.6f} '.\
                                format(-det[0]))

            elif exc_rank == 1:
                exc_to = 0
                for i, exc_to in enumerate(det[1:]):
                    if exc_to > wf.n_alpha:
                        spin_shift = 0 if (i < wf.n_alpha) else wf.n_alpha
                        break
                for j in range(wf.n_alpha):
                    if j+1 not in det[1 + spin_shift: 1 + spin_shift + wf.n_alpha]:
                        exc_from = j+1
                        break
                spin_shift = wf.beta_shift if spin_shift else 0
                pos = get_position_in_jac(exc_from, exc_to, wf.n_alpha, spin_shift)
                if (wf.n_alpha - exc_from)%2 == 0:
                    Jac[pos] = det[0]
                else:
                    Jac[pos] = -det[0]

                if verb > 50:
                    f_str.write('# Single exc: ' + str(det) + '\n')
                    f_str.write('  K_{0:d}^{1:d}  spin: {2:s} ; pos = {3:d} '.\
                                format(exc_from, exc_to,
                                       'b' if spin_shift else 'a',
                                       pos))

            elif exc_rank == 2:
                double_exc_to = [] # [(a1, shift1), (a2, shift2)]
                double_exc_from = []
                for i, exc_to in enumerate(det[1:]):
                    if exc_to > wf.n_alpha:
                        double_exc_to.append((exc_to, 0 if (i<wf.n_alpha) else wf.beta_shift))
                for j in range(wf.n_alpha):
                    if j+1 not in det[1: 1 + wf.n_alpha]:
                        double_exc_from.append((j+1, 0))
                for j in range(wf.n_alpha):
                    if j+1 not in det[1+wf.n_alpha: 1 + 2*wf.n_alpha]:
                        double_exc_from.append((j+1, wf.beta_shift))
                if len(double_exc_to) != 2:
                    raise Exception ('len(double_exc_to) != 2' + str(double_exc_to))
                if len(double_exc_from) != 2:
                    raise Exception ('len(double_exc_from) != 2: ' + str(double_exc_from))

                exc_from1, exc_from2 = tuple(double_exc_from)
                exc_to1  , exc_to2   = tuple(double_exc_to)
                pos1 = get_position_in_jac(exc_from1[0], exc_to1[0], wf.n_alpha, exc_from1[1])
                pos2 = get_position_in_jac(exc_from2[0], exc_to2[0], wf.n_alpha, exc_from2[1])
                if exc_from1[1] == exc_from2[1]: # same spin
                    n_sign = exc_from1[0] - exc_from2[0]
                    if (exc_from1[0] < exc_from2[0] and exc_to1[0] < exc_to2[0]) or\
                       (exc_from1[0] > exc_from2[0] and exc_to1[0] > exc_to2[0]):
                        n_sign += 1
                else:
                    n_sign = wf.n_alpha + wf.n_beta + exc_from1[0] - exc_from2[0]
                if n_sign%2 == 0:
                    Hess[pos1][pos2] += det[0]
                    Hess[pos2][pos1] += det[0]
                else:
                    Hess[pos1][pos2] -= det[0]
                    Hess[pos2][pos1] -= det[0]

                if verb > 50:
                    f_str.write('# Double exc: ' + str(det) + '\n')
                    f_str.write('double_exc_from: ' + str(double_exc_from))
                    f_str.write('double_exc_to:   ' + str(double_exc_to))
                    f_str.write(' K_{{{0:d},{1:d}}}^{{{2:d},{3:d}}} spin: {4:s},{5:s} sign={6:d}\n'.\
                                format(double_exc_from[0][0],
                                       double_exc_from[1][0],
                                       double_exc_to[0][0],
                                       double_exc_to[1][0],
                                       'b' if double_exc_from[0][1] else 'a',
                                       'b' if double_exc_from[1][1] else 'a',
                                       '-' if n_sign%2 else '+'))
                    f_str.write('Added in Hess[{0:d}][{1:d}]\n'.\
                                format(pos1,pos2))

                if exc_from1[1] == exc_from2[1]: # same spin
                    exc_to2  , exc_to1 = exc_to1, exc_to2
                    pos1 = get_position_in_jac(exc_from1[0], exc_to1[0], wf.n_alpha, exc_from1[1])
                    pos2 = get_position_in_jac(exc_from2[0], exc_to2[0], wf.n_alpha, exc_from2[1])
                    if n_sign%2 == 0:
                        Hess[pos1][pos2] -= det[0]
                        Hess[pos2][pos1] -= det[0]
                    else:
                        Hess[pos1][pos2] += det[0]
                        Hess[pos2][pos1] += det[0]
                    if verb > 50:
                        f_str.write('Added in Hess[{0:d}][{1:d}] with (-)\n'.\
                                    format(pos1,pos2))

    else:
        eps = 0.001
        coef_0 = wf.determinants[0][0]
        coef_p = np.zeros(n_param)
        coef_m = np.zeros(n_param)
        coef_pp = np.zeros((n_param,n_param))
        coef_mm = np.zeros((n_param,n_param))
        z = np.zeros(n_param)
        for i in range(n_param):
            z[i] = eps
            wf_tmp, cA, cB = calc_wf_from_z(z, wf, f_str, verb = False)
            coef_p[i] = wf_tmp.determinants[0][0]
            for j in range(n_param):
                z[j] += eps
                wf_tmp, cA, cB = calc_wf_from_z(z, wf, f_str, verb = False)
                coef_pp[i][j] = wf_tmp.determinants[0][0]
                z[j] = eps if j==i else 0.0
            z[i] = -eps
            wf_tmp, cA, cB = calc_wf_from_z(z, wf, f_str, verb = False)
            coef_m[i] = wf_tmp.determinants[0][0]
            for j in range(n_param):
                z[j] -= eps
                wf_tmp, cA, cB = calc_wf_from_z(z, wf, f_str, verb = False)
                coef_mm[i][j] = wf_tmp.determinants[0][0]
                z[j] = -eps if j==i else 0.0
            z[i] = 0.0
        for i in range(n_param):
            Jac[i] = (coef_p[i] - coef_m[i])/(2*eps)
            for j in range(n_param):
                Hess[i][j] = (2*coef_0 \
                              + coef_pp[i][j] - coef_p[i] - coef_p[j] \
                              + coef_mm[i][j] - coef_m[i] - coef_m[j] )/(2*eps*eps)

    return (Jac, Hess)


def print_Jac_Hess(J, H, wf, f_str):
    """Print the Jacobian (J) and the Hessian (H) of wave function wf to f_str."""
    f_str.write('='*50 + '\n')
    f_str.write('Jacobian:\n')
    for i, x in enumerate(J):
        exc_from, exc_to = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        f_str.write('[{0:3d}] = {1:.6f}; spin = {2:s}; K_{3:d}^{4:d}\n'.\
                    format(i, x,
                           'a' if i<wf.beta_shift else 'b',
                           exc_from, exc_to))
    f_str.write('-'*50 + '\n')
    f_str.write('Hessian:\n')
    for i,II in enumerate(H):
        exc_from_i, exc_to_i = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        for j,x in enumerate(II): 
            exc_from_j, exc_to_j = get_i_a_from_pos_in_jac(j, wf.n_alpha, wf.beta_shift)
            f_str_write(' [{0:3d},{1:3d}] = {2:.6f}; K_{3:d}^{4:d} [{5:s}]; K_{6:d}^{7:d} [{8:s}]\n'.\
                        format(i, j,  x,
                               exc_from_i, exc_to_i, 'a' if i<wf.beta_shift else 'b',
                               exc_from_j, exc_to_j, 'a' if j<wf.beta_shift else 'b'))
    f_str.write('='*50 + '\n')


def calc_wf_from_z(z, cur_wf, f_str, verb = 0, just_C0 = False):
    """Calculate the wave function in a new orbital basis

    Given the wave function in the current orbital basis, 
    a new (representation of the) wave function is constructed
    in a orbital basis that has been modified by a step z.

    Paramters:
    z        the update in the orbital basis (given in the space of the 
             K_i^a parameters) from the position z=0 (that is, the orbital
             basis used to construct the current representation of the
             wave function
    cur_wf   current representation of the wave function
             (as Molpro_FCI_Wave_Function)
    f_str    stream to output
    verb     verbose level (default = 0)
             Levels:
          ==  0    nothing is printed
          >=  5    info about steps
             30    prints intermediate matrices
    just_C0  Calculates only the first coefficient (see transform_wf)

    Returns:
    a tuple (new_wf, Ua, Ub) where new_wf is a Molpro_FCI_Wave_Function
    with the wave function in the new representation, and Ua and Ub
    are the transformations from the previous to the new orbital
    basis (alpha and beta, respectively).
    """
    if verb >= 30:
        f_str.write('='*50 + '\n')
        f_str.write('Current z vector\n')
        f_str.write(str(z) + '\n')
        f_str.write('='*30 + '\n')

    # transformation of alpha orbitals
    K = np.zeros((cur_wf.orb_dim, cur_wf.orb_dim))
    for i in range(cur_wf.n_alpha):
        for a in range(cur_wf.n_alpha, cur_wf.orb_dim):
            K[i][a] = -z[get_position_in_jac(i+1, a+1, cur_wf.n_alpha, 0)]
            K[a][i] = -K[i][a]
    Ua = expm(K)
    if verb >= 30:
        f_str.write('='*30 + '\n')
        f_str.write('Current K matrices:\n')
        f_str.write('alpha:\n')
        print_matrix(K, f_str)
    # transformation of beta orbitals
    K = np.zeros((cur_wf.orb_dim, cur_wf.orb_dim))
    for i in range(cur_wf.n_alpha):
        for a in range(cur_wf.n_alpha, cur_wf.orb_dim):
            K[i][a] = -z[get_position_in_jac(i+1, a+1, cur_wf.n_alpha, cur_wf.beta_shift)]
            K[a][i] = -K[i][a]
    Ub = expm(K)
    if verb >= 30:
        f_str.write('beta:\n')
        print_matrix(K, f_str)
        f_str.write('-'*30 + '\n')
        f_str.write('Current U = exp(-K):\n')
        f_str.write('alpha:\n')
        print_matrix(Ua, f_str)
        f_str.write('beta:\n')
        print_matrix(Ub, f_str)
        f_str.write('-'*30 + '\n')
        
        if False:
            f_str.write('U^T * U (should be the identity):\n')
            Id = np.matmul(Ua.transpose(), Ua)
            f_str.write('alpha:\n')
            print_matrix(Id, f_str)
            f_str.write('beta:\n')
            Id = np.matmul(Ub.transpose(), Ub)
            print_matrix(Id, f_str)
            f_str.write('-'*30 + '\n')
        f_str.write('='*50 + '\n')
    if verb >= 5:
        f_str.write('# Matrices Ua and Ub are done. Going to calculate transformed wf now...\n')
    return (transform_wf(cur_wf, Ua, Ub, just_C0 = just_C0), Ua, Ub)


def get_transf_RHF_to_UHF(RHF_output, UHF_output):
    """Return the matrices that transform RHF to UHF orbitals.

    returns:
    (Ua, Ub)

    such that
    
    orb_UHFa = orb_RHF*Ua
    orb_UHFb = orb_RHF*Ub
    """
    coeff_RHF = get_orbitals(RHF_output)
    print(coeff_RHF)
    coeff_RHF_inv = linalg.inv(coeff_RHF)
    coeff_UHF_a, coeff_UHF_b = get_orbitals(UHF_output)

    return np.matmul(coeff_RHF_inv, coeff_UHF_a), np.matmul(coeff_RHF_inv, coeff_UHF_b)


def get_orbitals(output_name):
    """Load (last) orbitals from Molpro output.

    Parameters:
    output_name   Molpro output

    Returns:
    One (for restricted) or two (for unrestricted) numpy matrices with the
    orbital coefficients 
    """
    reading_orbitals = False
    is_RHF = None
    n_orb = None
    coef_a = None
    coef_b = None
    with open(output_name, 'r') as f: 
        for l in f:
            if 'NUMBER OF CONTRACTIONS' in l:
                n_orb = int(l.split()[3])
                if n_orb > 10:
                    raise Exception ('Probably not valid for more than 10 orbitals')
            if 'MOLECULAR ORBITALS' in l:
                is_RHF = True
                coef_a = np.zeros((n_orb,n_orb))
                reading_orbitals = True
                cur_coef = coef_a
            if 'ELECTRON ORBITALS FOR POSITIVE SPIN' in l:
                is_RHF = False
                coef_a = np.zeros((n_orb,n_orb))
                reading_orbitals = True
                cur_coef = coef_a
            if 'ELECTRON ORBITALS FOR NEGATIVE SPIN' in l:
                is_RHF = False
                coef_b = np.zeros((n_orb,n_orb))
                reading_orbitals = True
                cur_coef = coef_b
            if 'HOMO' in l:
                reading_orbitals = False
            if reading_orbitals:
                lspl = l.split()
                if lspl:
                    is_line_orb = re.match('(\d+)\.\d+$', lspl[0])
                    if is_line_orb is not None:
                        for i, c in enumerate(lspl[3:]):
                            cur_coef[i][int(is_line_orb.group(1))-1] = c

    if is_RHF:
        return coef_a
    else:
        return coef_a, coef_b
