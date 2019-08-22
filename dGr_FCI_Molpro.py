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

TODO:
Inherit from dGr_general_WF.Wave_Function

Perhaps storing the orbital indices of the determinants starting at 0,
like Python, is better than starting at 1, as Molprop does.

History:
    Aug 2018 - Start
    Mar 2019 - Add CISD wave function
               Add to git
Yuri
"""
import logging
import numpy as np
from numpy import linalg
from scipy.linalg import expm, lu, orth
import copy
import math
import re

from dGr_util import str_matrix
import dGr_Absil

logger = logging.getLogger(__name__)

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
                          of 0.9.
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
        FCI_file_name - if None, file_name is assumed to be of FCI
                        wave function, that will be the external wave function.
                        if not None, FCI_file_name has a FCI wave function
                        to be used as template. In the end, all
                        coefficients will be zero, except those from the
                        wave function of file_name.
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
        if FCI_file_name is not None:
            FCI_found = False
            FCI_prog_found = False
            # Load the structure of a FCI wave function
            # and eventually the wave function itself
            with open (FCI_file_name, 'r') as f:
                S = 0.0
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
                            S += float(lspl[0])**2
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
                logger.info('norm of FCI (template): %f', math.sqrt(S))
        if file_name is not None:
            # Load the required external wave function
            if self.WF_type is None:
                ref_found = False
                sgl_found = False
                dbl_found = False
                FCI_found = False
                FCI_prog_found = False
                i_FCI_templ = 0
                is_first_det = True
                ref = 0.0
                singles = []
                doubles = []
                with open (file_name, 'r') as f:
                    for l in f:
                        if self.WF_type is None:
                            if l == molpro_FCI_header:
                                self.WF_type = 'FCI'
                            elif l == molpro_MRCI_header:
                                self.WF_type = 'MRCI'
                            elif l == molpro_CISD_header or l == molpro_CCSD_header:
                                self.WF_type = 'CISD' if molpro_CISD_header == l else 'CCSD'
                        elif self.WF_type == 'FCI':
                            if 'FCI STATE  '+state+' Energy' in l and 'Energy' in l:
                                FCI_found = True
                                continue
                            if FCI_found:
                                if 'CI Vector' in l:
                                    continue
                                if 'EOF' in l:
                                    break
                                lspl = l.split()
                                coef = float(lspl[0])
                                if is_first_det:
                                    is_first_det = False
                                    sgn_invert = coef < 0.0
                                if sgn_invert:
                                    coef = -coef
                                det_descr = lspl[pos_a_ini:pos_a_fin] + lspl[pos_b_ini:pos_b_fin]
                                det_descr = list(map(lambda x: int(x) - self.n_frozen, det_descr))
                                if FCI_file_name is not None:
                                    while self.determinants[i_FCI_templ][1:] != det_descr:
                                        i_FCI_templ = (i_FCI_templ + 1
                                                       if i_FCI_templ < len(self.determinants) else
                                                       0)
                                    self.determinants[i_FCI_templ][0] = coef
                                else:
                                    self.determinants.append([coef] + det_descr)
                            else:
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
        if logger.level <= logging.DEBUG:
            logmsg = ['Amplitudes of single excitations:']
            for i in Csgl:
                logmsg.append(str(i))
            logmsg.append('-------------------\n')
            logmsg.append('Amplitudes of double excitations:')
            for i in Cdbl:
                logmsg.append(str(i))
                logmsg.append('-------------------\n')
            logger.debug('\n'.join(logmsg))
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
                logger.debug('This det: %s; exc_descr = %s ; exc_type = %s',
                             str(det), str(exc_descr), exc_type)
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
        if logger.level <= logging.DEBUG:
            logmsg = []
            logmsg.append('Amplitudes of single excitations:')
            for i in Tsgl:
                logmsg.append(str(i))
            logmsg.append('-------------------\n')
            logmsg.append('Amplitudes of double excitations:')
            for i in Tdbl:
                logmsg.append(str(i))
            logmsg.append('-------------------\n')
            logger.debug('\n'.join(logmsg))
        raise Exception('load_CCSD_WF: Not implemented yet!')

    def load_MRCI_WF(self, C0, Csgl, Cdbl):
        """Load the MRCISD wave function."""
        sqrt_2 = math.sqrt(2.0)
        logger.debug('ref coef: %f', C0)
        if logger.level <= logging.DEBUG:
            logmsg = []
            logmsg.append('Singly exc:')
            for i in Csgl:
                logmsg.append(i)
            logmsg.append('-------------------\n')
            logmsg.append('Doubly exc:')
            for i in Cdbl:
                logmsg.append(i)
            logmsg.append('-------------------\n')
            logger.debug('\n'.join(logmsg))
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
                        logger.debug('Replacing coef of singles: %s %s', str(det), str(s))
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

                logger.debug('This det: %s; exc_descr = %s',
                             str(det), str(exc_descr))
                for d in Cdbl:
                    if (set(d[1:3]) == set(exc_descr[0:2]) and
                        set(d[3:5]) == set(exc_descr[2:4])):
                        logger.debug('Replacing coef of doubles: %s: %s = %f',
                                     str(det), exc_descr, d)
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
        x = []
        x.append('-'*50)
        x.append('n frozen: {}'.format(self.n_frozen))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('beta shift: {}'.format(self.beta_shift))
        x.append('-----')
        for det in self.determinants:
            d = []
            d.append('{0:15.12f} '.format(det[0]))
            for i_occ in det[1:]:
                d.append('{0:4d} '.format(i_occ))
            d.append(' >> RANK: {0:d}'.format(rank_of_exc(det)))
            x.append(''.join(d))
        x.append('-'*50)
        return '\n'.join(x)

    def normalise(self):
        """Normalise the wave function to unity"""
        S = 0.0
        for det in self.determinants:
            S += det[0]**2
        S = math.sqrt(S)
        print('norm, inside FCI:', S)
        for det in self.determinants:
            det[0] /= S

    def distance_to_det(self, U, thresh_cI=1E-10, assume_orth = False):
        """Calculates the distance to the determinant U
        
        Behaviour:
        
        Parameters:
        
        U (2D numpy.array or 2-tuple of 2D numpy.array)
            A set of (self.n_alpha + self.n_beta) orbitals that
            form a Slater determinant.
            More specifically, it is the matrix that shows how these orbitals
            are written in the orbital used in the representation of this
            wave function.
            (Thus it is part of the transformation basis)
            If it is a 2D numpy.array, it corresponds to the same set for
            both alpha and beta orbitals (restricted). Its shape should be
            (self.orb_dim, self.n_alpha).
            If it is a 2-tuple,2D numpy.array, the entries are for
            (alpha, beta), with shapes (self.orb_dim, self.n_alpha) and
            (self.orb_dim, self.n_alpha).
        
        thresh_cI (float, optional, default=1E-10)
                      The function consider only terms with |C_I| >= thresh_cI
        
        Return:
        
        A float, the distance between self and U.
        """
        if isinstance(U, tuple):
            Ua, Ub = U
        else:
            Ua = Ub = U
        f = 0.0
        for det in self.determinants:
            if abs(det[0]) < thresh_cI:
                continue
            f += (det[0]
                  * dGr_Absil.calc_fI(Ua, [x-1 for x in det[1:self.n_alpha+1]])
                  * dGr_Absil.calc_fI(Ub, [x-1 for x in det[self.n_alpha+1:]]))
        if not assume_orth:
            Da = linalg.det(np.matmul(Ua.T, Ua))
            Db = linalg.det(np.matmul(Ub.T, Ub))
            f /= math.sqrt(Da * Db)
        return f

    def get_ABC_matrices(self, U, thresh_cI=1E-10):
        """Calculates the arrays A,B,C needed for Absil's algorithm
        
        Behaviour:
        
        Limitations:
        
        This needs n_alpha == n_beta
        
        Parameters:
        
        U (2D numpy.array or 2-tuple of 2D numpy.array)
            See distance_to_det.
        
        thresh_cI (float, optional, default=1E-10)
                      The function consider only terms with |C_I| >= thresh_cI
        
        Return:
        
        The three matrices: A, B, C.
        
        Observations regarding implementation:
        The column j==l in the matrix Hkl is zero.
        We no not calculate these zeros, but store them.
        One could have ndarray for Hkl with one column less,
        but have to carefully handle the other columns to be in the
        right position later in A, what seems too complicated to
        compensate the memory saving.
        
        Some matrices are allocated with the same size for both and
        beta counterparts (see limitation). In case of extending to the
        general case, whatch out the allocation.
        """
        if isinstance(U, tuple):
            Ua, Ub = U
            restricted = False
        else:
            Ua = Ub = U
            restricted = True
        K, na = Ua.shape
        nb = Ub.shape[1]
        if na != nb:
            raise NotImplementedError('We need both Ua and Ub with same shape!')
        if K != self.orb_dim:
            raise ValueError('First dimension of U must match orb_dim!')
        if na != self.n_alpha:
            raise ValueError('Second dimension of Ua must match the n_alpha!')
        if nb != self.n_beta:
            raise ValueError('Second dimension of Ua must match the n_beta!')
        ABshape = Ua.shape*2
        Aa_a = np.zeros(ABshape)
        Aa_b = np.zeros(ABshape)
        Ba = np.zeros(ABshape)
        Ca = np.zeros(Ua.shape)
        if not restricted:
            Ab_a = np.zeros(ABshape)
            Ab_b = np.zeros(ABshape)
            Bb = np.zeros(ABshape)
            Cb = np.zeros(Ub.shape)
        for det in self.determinants:
            if abs(det[0]) < thresh_cI:
                continue
            Ia = [x-1 for x in det[1:self.n_alpha+1]]
            Ib = [x-1 for x in det[self.n_alpha+1:]]
            Fa = dGr_Absil.calc_fI(Ua, Ia)
            Fb = dGr_Absil.calc_fI(Ub, Ib)
            Proj_a = np.identity(K)
            Ga = np.zeros(Ua.shape)
            if not restricted:
                Gb = np.zeros(Ub.shape)
                Proj_b = np.identity(K)
            for k in range(K):
                for l in range(na):
                    Hkl_a = np.zeros(Ua.shape)
                    if not restricted:
                        Hkl_b = np.zeros(Ub.shape)
                    for i in range(K):
                        Proj_a[i,k] -= np.dot(Ua[i,:], Ua[k,:])
                        if not restricted:
                            Proj_b[i,k] -= np.dot(Ub[i,:], Ub[k,:])
                        for j in range(na):
                            if j != l:
                                Hkl_a[i,j] = dGr_Absil.calc_H(Ua, Ia, i, j, k, l)
                                if not restricted:
                                    Hkl_b[i,j] = dGr_Absil.calc_H(Ub, Ib, i, j, k, l)
                            else:
                                Ba[i,j,k,l] += det[0] * Fa * Fb * Proj_a[i,k]
                                if not restricted:
                                    Bb[i,j,k,l] += det[0] * Fa * Fb * Proj_b[i,k]
                    Ga[k,l] = dGr_Absil.calc_G(Ua, Ia, k, l)
                    Gb[k,l] = dGr_Absil.calc_G(Ub, Ib, k, l)
#                    Ga[k,l] += np.dot(Hkl_a[:,na-1], Ua[:,na-1])
#                    if restricted:
#                        Gb[k,l] = dGr_Absil.calc_G(Ub, Ib, k, l)
#                    else:
#                        Gb[k,l] += np.dot(Hkl_b[i,nb-1], Ub[i,nb-1])
                    Aa_a[k,l,:,:] += det[0] * Fb * np.matmul(Proj_a, Hkl_a)
                    if not restricted:
                        Ab_b[k,l,:,:] += det[0] * Fa * np.matmul(Proj_b, Hkl_b)
            det_G_FU = det[0] * (Ga - Fa * Ua)
            Ca += Fb * det_G_FU
            Aa_b += np.multiply.outer(det_G_FU, Gb)
            if not restricted:
                det_G_FU = det[0] * (Gb - Fb * Ub)
                Cb += Fa * det_G_FU
                Ab_a += np.multiply.outer(det_G_FU, Ga)
        if restricted:
            return (Aa_a, Aa_b), Ba, Ca
        else:
            return ((Aa_a, Aa_b),
                    (Ab_a, Ab_b)), (Ba, Bb), (Ca, Cb)


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
    logger.debug('Ua:\n%s', str(Ua))
    logger.debug('Ub:\n%s', str(Ub))
    return Ua, Ub

def transform_wf(wf, Ua, Ub, just_C0 = False):
    """Transform the wave function as induced by a transformation in the orbital basis
    
    Arguments:
    
    wf   the initial wave function as Molpro_FCI_Wave_Function
    Ua   the orbital transformation for alpha orbitals
    Ub   the orbital transformation for beta orbitals
    just_C0   If True, calculates the coefficients of the initial determinant only 
              (default False)
    
    Behaviour:
    
    If the coefficients of wf are given in the basis |u_I>:
    
    |wf> = \sum_I c_I |u_I>
    
    it calculates the wave function in the basis |v_I>:
    
    |wf> = \sum_I d_I |v_I>
    
    and Ua and Ub are the matrix transformations of the MO from the basis |v_I>
    to the basis |u_I>:
    
    |MO of (u)> = |MO of (v)> U
    
    Return:
    new_wf   the transformed wave function
    
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
        logger.debug('====== Starting det %s',
                     str(new_det))
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
                if logger.level <= logging.DEBUG:
                    logger.debug('det_J = %s', str(det_J))
                    logger.debug('det_I = %s', str(det_I))
                    logger.debug('U_minor:\n%s', str(U_minor))
                    logger.debug('I = %s; c_I = %f; det_minor = %f; c_I*det_minor = %f',
                                 str(det_I[1:]), det_I[0], U_det_minor_IJ, det_I[0]*U_det_minor_IJ)
                    logger.debug('current C_J (%s) = %f',
                                 str(new_det[1:]), new_det[0])
        new_wf.determinants.append(new_det)
        if just_C0:
            break
    logger.info('Number of det calculations: %d',  n_calcs)
    return new_wf

def transform_wf_2(wf, Ua, Ub, just_C0 = False):
    """Similar to transform_wf, but didn't work (it should be faster)."""
    new_wf = copy.copy(wf)
    n_calcs = 0
    if logger.level <= logging.DEBUG:
        logger.debug('WF:\n%s', str(wf))
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
            logger.debug('k in alpha')
        else:
            k = k_tot -(wf.orb_dim - wf.n_frozen)
            spin_shift = new_wf.n_alpha
            t = tb
            logger.debug('k in beta')
            logger.debug('k = %d; spin_shift = %d', k, spin_shift)
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
                                logger.debug('detI = ', str(det_I))
                                logger.debug('detJ = ', str(det_J))
                                n_diff += 1
                                if not spin_shift:
                                    p = i-1
                                    logger('Current p (alpha): %d', p)
                        else:
                            if not i in det_I[new_wf.n_alpha+1:]:
                                n_diff += 1
                                if spin_shift:
                                    p = i - (wf.orb_dim - wf.n_frozen) -1
                                    logger('Current p (beta): %d', p)
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
    logger.info('Number of det calculations: %d',  n_calcs)
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

def construct_Jac_Hess(wf, analytic = True):
    """Construct the Jacobian and the Hessian of the function overlap.
    
    the function is f(x) = <wf(x), det1(x)>
    where x parametrises the orbital rotations and
    det1 is the first determinant in wf.
    
    Parameters
    wf        the external (FCI_ wave function, as Molpro_FCI_Wave_Function
    analytic  if True, calculates the Jacobian and the Hessian by the 
              analytic expression, if False calculate numerically
              (default = True)
    
    Returns the tuple (Jac, Hess), with the Jacobian and the 
    Hessian
    """
    logger.info('Building Jacobian and Hessian: %s procedure',
                'Analytic' if analytic else 'Numerical')
    n_param = wf.n_alpha*(wf.orb_dim - wf.n_alpha)*2
    Jac = np.zeros(n_param)
    Hess = np.zeros((n_param, n_param))
    if analytic:
        for det in wf.determinants:
            exc_rank = rank_of_exc(det)

            if exc_rank == 0:
                for i in range(n_param):
                    Hess[i][i] = -det[0]
                logger.debug('Setting diagonal of Hess to {0:10.6f}'.\
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

                logger.debug('Single exc: ' + str(det) + '\n'
                             + '  K_{0:d}^{1:d}  spin: {2:s} ; pos = {3:d} '.\
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

                if logger.level <= logging.DEBUG:
                    logmsg = []
                    logmsg.append('Double exc: ' + str(det))
                    logmsg.append('double_exc_from: ' + str(double_exc_from))
                    logmsg.append('double_exc_to:   ' + str(double_exc_to))
                    logmsg.append(' K_{{{0:d},{1:d}}}^{{{2:d},{3:d}}} spin: {4:s},{5:s} sign={6:s}\n'.\
                                  format(double_exc_from[0][0],
                                         double_exc_from[1][0],
                                         double_exc_to[0][0],
                                         double_exc_to[1][0],
                                         'b' if double_exc_from[0][1] else 'a',
                                         'b' if double_exc_from[1][1] else 'a',
                                         '-' if n_sign%2 else '+'))
                    logmsg.append('Added in Hess[{0:d}][{1:d}]'.\
                                  format(pos1,pos2))
                    logger.debug('\n'.join(logmsg))
                    
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
                    if logger.level <= logging.DEBUG:
                        logmsg.append('Added in Hess[{0:d}][{1:d}] with (-)'.\
                                      format(pos1,pos2))
                if logger.level <= logging.DEBUG:
                    logger.debug('\n'.join(logmsg))

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
            wf_tmp, cA, cB = calc_wf_from_z(z, wf)
            coef_p[i] = wf_tmp.determinants[0][0]
            for j in range(n_param):
                z[j] += eps
                wf_tmp, cA, cB = calc_wf_from_z(z, wf)
                coef_pp[i][j] = wf_tmp.determinants[0][0]
                z[j] = eps if j==i else 0.0
            z[i] = -eps
            wf_tmp, cA, cB = calc_wf_from_z(z, wf)
            coef_m[i] = wf_tmp.determinants[0][0]
            for j in range(n_param):
                z[j] -= eps
                wf_tmp, cA, cB = calc_wf_from_z(z, wf)
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

def str_Jac_Hess(J, H, wf):
    """Return str of Jacobian (J) and the Hessian (H) of wave function wf."""
    JHstr = []
    JHstr.append('Jacobian:')
    for i, x in enumerate(J):
        exc_from, exc_to = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        JHstr.append(('[{0:3d}] = {1:.6f}; spin = {2:s}; K_{3:d}^{4:d}'.\
                      format(i, x,
                             'a' if i<wf.beta_shift else 'b',
                             exc_from, exc_to)))
    JHstr.append('-'*50)
    JHstr.append('Hessian:')
    for i,II in enumerate(H):
        exc_from_i, exc_to_i = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        for j,x in enumerate(II): 
            exc_from_j, exc_to_j = get_i_a_from_pos_in_jac(j, wf.n_alpha, wf.beta_shift)
            JHstr.append(' [{0:3d},{1:3d}] = {2:.6f}; K_{3:d}^{4:d} [{5:s}]; K_{6:d}^{7:d} [{8:s}]'.\
                         format(i, j,  x,
                                exc_from_i, exc_to_i, 'a' if i<wf.beta_shift else 'b',
                                exc_from_j, exc_to_j, 'a' if j<wf.beta_shift else 'b'))
    return('\n'.join(JHstr))


def calc_wf_from_z(z, cur_wf, just_C0 = False):
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
    just_C0  Calculates only the first coefficient (see transform_wf)

    Returns:
    a tuple (new_wf, Ua, Ub) where new_wf is a Molpro_FCI_Wave_Function
    with the wave function in the new representation, and Ua and Ub
    are the transformations from the previous to the new orbital
    basis (alpha and beta, respectively).
    """
    logger.info('Current z vector:\n%s', str(z))
    # transformation of alpha orbitals
    K = np.zeros((cur_wf.orb_dim, cur_wf.orb_dim))
    for i in range(cur_wf.n_alpha):
        for a in range(cur_wf.n_alpha, cur_wf.orb_dim):
            K[i][a] = -z[get_position_in_jac(i+1, a+1, cur_wf.n_alpha, 0)]
            K[a][i] = -K[i][a]
    Ua = expm(K)
    logger.info('Current K (alpha) matrix:\n%s', str_matrix(K))
    # transformation of beta orbitals
    K = np.zeros((cur_wf.orb_dim, cur_wf.orb_dim))
    for i in range(cur_wf.n_alpha):
        for a in range(cur_wf.n_alpha, cur_wf.orb_dim):
            K[i][a] = -z[get_position_in_jac(i+1, a+1, cur_wf.n_alpha, cur_wf.beta_shift)]
            K[a][i] = -K[i][a]
    Ub = expm(K)
    if logger.level <= logging.INFO:
        logger.info('Current K (beta) matrix:\n%s', str_matrix(K))
        logger.info('Current U = exp(-K), alpha:\n%s', str_matrix(Ua))
        logger.info('Current U = exp(-K), beta:\n%s', str_matrix(Ub))
    if logger.level <= logging.DEBUG:
        Id = np.matmul(Ua.transpose(), Ua)
        logger.debug('U^T * U (should be the identity, alpha):\n%s', str_matrix(Id))
        Id = np.matmul(Ub.transpose(), Ub)
        logger.debug('U^T * U (should be the identity, beta):\n%s', str_matrix(Id))
    logger.info('Matrices Ua and Ub are done. Going to calculate transformed wf now.')
    return (transform_wf(cur_wf, Ua, Ub, just_C0 = just_C0), Ua, Ub)

def transf_orb_from_to(orb_i, orb_f):
    """Return the matrices that transform the orbitals from orb1 to orb2.
    
    Arguments:
    orb_i    file with initial orbitals
    orb_f    file with final orbitals
    
    Behaviour:
    
    If the orbitals in orb_x are Cx_a and Cx_b
    (x being i or f, a is alpha and b is beta),
    the transformation matrices are Ua, Ub such that:
    
    Cf_a = Ci_a*Ua
    Cf_b = Ci_b*Ub
    
    Returns:
    (Ua, Ub)
    """
    Ci = get_orbitals(orb_i)
    if isinstance(Ci, tuple):
        Ci_inv_a, Ci_inv_b = map(linalg.inv, Ci)
    else:
        Ci_inv_a = linalg.inv(Ci)
        Ci_inv_b = Ci_inv_a
    Cf = get_orbitals(orb_f)
    if isinstance(Cf, tuple):
        Cf_a, Cf_b = Cf
    else:
        Cf_a, Cf_b = Cf, Cf
    Ua = np.matmul(Ci_inv_a, Cf_a)
    Ub = np.matmul(Ci_inv_b, Cf_b)
    logger.debug('Ua:\n' + str(Ua))
    logger.debug('Ub:\n' + str(Ub))
    return Ua, Ub


def get_orbitals(output_name):
    """Load (last) orbitals from Molpro output.
    
    Arguments:
    output_name    Molpro output
    
    Behaviour:
    This function loads always the last RHF or UHF orbitals.
    The returnece matrix (matrices) have the coefficients of
    the i-th orbital in the ith column:
    
    1st MO      2nd MO    ...    nth MO
    
    C[0][0]    C[1][1]    ...    C[1][n-1]   -> coef of 1st basis function
    C[1][0]    C[2][1]    ...    C[2][n-1]   -> coef of 2nd basis function
    ...        ...        ...    ...
    C[n-1][0]  C[n-1][1]  ...    C[n][n-1]   -> coef of nth basis function
    
    Returns:
    One (for restricted) or two (for unrestricted) numpy matrices with the
    orbital coefficients 
    """
    reading_orbitals = False
    is_RHF = None
    n_orb = None
    coef_a = None
    coef_b = None
    logger.info('File: %s', output_name)
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
                        for i in range(n_orb):
                            cur_coef[i][int(is_line_orb.group(1))-1] = float(l[24+i*10:24+(i+1)*10])
    logger.info('coeff_a:\n' + str(coef_a))
    if not is_RHF:
        logger.info('coeff_b:\n' + str(coef_b))
    if is_RHF:
        return coef_a
    else:
        return coef_a, coef_b
