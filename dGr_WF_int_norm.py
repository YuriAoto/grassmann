"""f(x) = <0|ext>, for |0> Slater determinant and |ext> a CC or CI wave function

# Parametrisation of |0>:  See dGr_FCI_Molpro.py


"""
import math
import logging

import numpy as np
from numpy import linalg

from dGr_util import get_I
import dGr_general_WF as genWF
import dGr_Absil as Absil

logger = logging.getLogger(__name__)


molpro_CISD_header = ' PROGRAM * CISD (Closed-shell CI(SD))     '\
                     + 'Authors: C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n'
molpro_CCSD_header = ' PROGRAM * CCSD (Closed-shell coupled cluster)     '\
                     + 'Authors: C. Hampel, H.-J. Werner, 1991, M. Deegan, P.J. Knowles, 1992\n'

CC_sgl_str = ' Singles amplitudes (print threshold =  0.000E+00):\n'
CC_dbl_str = ' Doubles amplitudes (print threshold =  0.000E+00):\n'


class Wave_Function_Int_Norm(genWF.Wave_Function):
    """An electronic wave function in intermediate normalisation
    
    
    """    
    def __init__(self):
        super().__init__()
        self.norm = None
        self.singles = None
        self.doubles = None
       
    @classmethod
    def from_Molpro(cls, molpro_output):
        """Load the wave function from Molpro output."""
        new_wf = cls()
        sgl_found = False
        dbl_found = False
        with open (molpro_output, 'r') as f:
            for l in f:
                if new_wf.WF_type is None:
                    if l == molpro_CISD_header:
                        new_wf.WF_type = 'CISD'
                    elif l == molpro_CISD_header:
                        new_wf.WF_type = 'CCSD'
                else:
                    if new_wf.WF_type == 'CCSD' or new_wf.WF_type == 'CISD':
                        if 'Number of closed-shell orbitals' in l:
                            new_wf.n_alpha = int(l.split()[4])
                            new_wf.n_beta = new_wf.n_alpha
                            new_wf.n_elec = new_wf.n_alpha + new_wf.n_beta
                        if 'Number of external orbitals' in l:
                            new_wf.orb_dim = new_wf.n_alpha + int(l.split()[4])
                        if CC_sgl_str == l:
                            sgl_found = True
                            new_wf.singles = []
                        elif CC_dbl_str == l:
                            dbl_found = True
                            new_wf.doubles = []
                        elif dbl_found:
                            lspl = l.split()
                            if len(lspl) == 7:
                                new_wf.doubles.append(
                                    genWF.Double_Amplitude(
                                        t = float(lspl[-1]),
                                        i = int(lspl[0]) - 1,
                                        j = int(lspl[1]) - 1,
                                        a = int(lspl[4]) - 1 + new_wf.n_alpha,
                                        b = int(lspl[5]) - 1 + new_wf.n_alpha))
                        elif sgl_found:
                            lspl = l.split()
                            if len(lspl) == 4:
                                new_wf.singles.append(
                                    genWF.Single_Amplitude(
                                        t = float(lspl[-1]),
                                        i = int(lspl[0]) - 1,
                                        a = int(lspl[2]) - 1 + new_wf.n_alpha))
                        if 'RESULTS' in l:
                            if not dbl_found:
                                raise Exception('Double excitations not found!')
                            break
        return new_wf

    def calc_norm(self):
        if self.WF_type == 'CISD':
            self.norm = 1.0
            for S in self.singles:
                self.norm += 2 * S.t**2
            for D in self.doubles:
                if D.a == D.b:
                    self.norm += (1 if D.i == D.j else 2) * D.t**2
                elif D.a > D.b:
                    t_compl = 0.0
                    for Dba in self.doubles:
                        if (D.i == Dba.i
                            and D.j == Dba.j
                            and D.a == Dba.b
                            and D.b == Dba.a):
                            t_compl = Dba.t
                            break
                    if D.i == D.j:
                        self.norm += 0.5 * (D.t + t_compl)**2
                    else:
                        self.norm += 4 * (D.t**2 + t_compl**2 - D.t*t_compl)
                else:
                    continue
            self.norm = math.sqrt(self.norm)
        elif self.WF_type == 'CCSD':
            raise NotImplementedError('We can not calculate norm for CCSD yet!')
        else:
            raise ValueError('We do not know how to calculate the norm for '
                             +  self.WF_type + '!')
    
    def __repr__(self):
        """Return representation of the wave function."""
        x = ['']
        if self.singles is not None:
            x.append('Amplitudes of single excitations:')
            for S in self.singles:
                x.append("{1:d} -> {2:d}    {0:12.8f}".format(*S))
            x.append('-'*50)
        if self.doubles is not None:
            x.append('Amplitudes of double excitations:')
            for D in self.doubles:
                x.append("{1:d} {2:d} -> {3:d} {4:d}    {0:12.8f}".format(*D))
            x.append('-'*50)
        return ('<Wave Function in Intermediate normalisation>\n'
                + super().__repr__()
                + '\n'.join(x))
    
    def __str__(self):
        """Return string version the wave function."""
        S_main = None
        D_main = None
        if self.singles is not None:
            for S in self.singles:
                if S_main is None or abs(S_main.t) < abs(S.t):
                    S_main = S
        if self.doubles is not None:
            for D in self.doubles:
                if D_main is None or abs(D_main.t) < abs(D.t):
                    D_main = D
        return ('|0> + {0:5f} |{1:d} -> {2:d}> + ... + {3:5f} |{4:d},{5:d} -> {6:d},{7:d}> + ...'.\
                format(
                    S_main.t, S_main.i, S_main.a,
                    D_main.t, D_main.i, D_main.j, D_main.a, D_main.b))

    def distance_to_det(self, U, assume_orth = False):
        """Calculates the distance to the determinant U
        
        See dGr_FCI_Molpro.Molpro_FCI_Wave_Function.distance_to_det
        """
        if isinstance(U, tuple):
            Ua, Ub = U
        else:
            Ua = Ub = U
        f0_a = Absil.calc_fI(Ua, range(self.n_alpha))
        f0_b = Absil.calc_fI(Ub, range(self.n_beta))
        f = f0_a * f0_b
        for S in self.singles:
            fI_a = Absil.calc_fI(Ua, get_I(self.n_alpha, S.i, S.a)) * f0_b
            fI_b = Absil.calc_fI(Ub, get_I(self.n_beta, S.i, S.a)) * f0_a
            if S.i + self.n_alpha % 2 == 1:
                fI_a *= -1
            if S.i + self.n_beta % 2 == 1:
                fI_b *= -1
            f += S.t * (fI_a + fI_b)
        for D in self.doubles:
            if D.a < D.b:
                continue
            fI = (Absil.calc_fI(Ua, get_I(self.n_alpha, D.i, D.a))
                  * Absil.calc_fI(Ub, get_I(self.n_beta, D.j, D.b)))
            if D.a != D.b:
                fI += (Absil.calc_fI(Ua, get_I(self.n_alpha, D.i, D.b))
                       * Absil.calc_fI(Ub, get_I(self.n_beta, D.j, D.a)))
            if D.i != D.j:
                fI += (Absil.calc_fI(Ua, get_I(self.n_alpha, D.j, D.a))
                       * Absil.calc_fI(Ub, get_I(self.n_beta, D.i, D.b)))
                if D.a != D.b:
                    fI += (Absil.calc_fI(Ua, get_I(self.n_alpha, D.j, D.b))
                           * Absil.calc_fI(Ub, get_I(self.n_beta, D.i, D.a)))
            t_compl = 0.0
            if D.a != D.b:
                fI /= 2    
                for Dba in self.doubles:
                    if (D.i == Dba.i
                        and D.j == Dba.j
                        and D.a == Dba.b
                        and D.b == Dba.a):
                        t_compl = Dba.t
                        break
            fI *= D.t + t_compl
            if D.a != D.b and D.i != D.j:
                fI2 = Absil.calc_fI(Ua, get_I(self.n_alpha, [D.i, D.j], [D.b, D.a])) * f0_b
                fI2 += Absil.calc_fI(Ua, get_I(self.n_beta, [D.i, D.j], [D.b, D.a])) * f0_a
                fI += fI2 * (t_compl - D.t)
            if D.i + D.j % 2 == 1:
                fI = -fI
            f += fI
        f /= self.norm
        if not assume_orth:
            Da = linalg.det(np.matmul(Ua.T, Ua))
            Db = linalg.det(np.matmul(Ub.T, Ub))
            f /= math.sqrt(Da * Db)
        return f

    def get_ABC_matrices(self, U, thresh_cI=1E-10):
        """Calculates the arrays A,B,C needed for Absil's algorithm"""
        pass
