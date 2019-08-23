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
    
    def all_dets(self):
        """Generate all determinants, yielding normalised CI-like wave function"""
        if self.norm is None:
            raise ValueError('Norm has not been calculated yet!')
        if self.WF_type != 'CISD':
            raise NotImplementedError('Curently works only for CISD')
        yield genWF.Ref_Det(c = 1.0/self.norm)
        for S in self.singles:
            yield genWF.Singly_Exc_Det(c = (S.t/self.norm
                                            if (S.i + self.n_alpha-1) % 2 == 0 else
                                            -S.t/self.norm),
                                       i = S.i,
                                       a = S.a,
                                       spin = 1)
            yield genWF.Singly_Exc_Det(c = (S.t/self.norm
                                            if (S.i + self.n_beta-1) % 2 == 0 else
                                            -S.t/self.norm),
                                       i = S.i,
                                       a = S.a,
                                       spin = -1)
        for D in self.doubles:
            if D.a < D.b:
                continue
            if D.a == D.b:
                coeff = D.t / self.norm
                if (D.i + D.j) % 2 == 1:
                    coeff = -coeff
                yield genWF.Doubly_Exc_Det(c = coeff,
                                           i = D.i, a = D.a, spin_ia = +1,
                                           j = D.j, b = D.b, spin_jb = -1)
                if D.i != D.j:
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.j, a = D.a, spin_ia = +1,
                                               j = D.i, b = D.b, spin_jb = -1)
            else:
                t_compl = 0.0
                for Dba in self.doubles:
                    if (D.i == Dba.i
                        and D.j == Dba.j
                        and D.a == Dba.b
                        and D.b == Dba.a):
                        t_compl = Dba.t
                        break
                if D.i == D.j:
                    coeff = (D.t + t_compl) / (2 * self.norm)
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.i, a = D.a, spin_ia = +1,
                                               j = D.j, b = D.b, spin_jb = -1)
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.i, a = D.b, spin_ia = +1,
                                               j = D.j, b = D.a, spin_jb = -1)
                else:
                    coeff = (t_compl - D.t)/self.norm
                    if (D.i + D.j) % 2 == 1:
                        coeff = -coeff
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.i, a = D.a, spin_ia = +1,
                                               j = D.j, b = D.b, spin_jb = +1)
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.i, a = D.a, spin_ia = -1,
                                               j = D.j, b = D.b, spin_jb = -1)
                    coeff = D.t / self.norm
                    if (D.i + D.j) % 2 == 1:
                        coeff = -coeff
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.i, a = D.a, spin_ia = +1,
                                               j = D.j, b = D.b, spin_jb = -1)
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.j, a = D.b, spin_ia = +1,
                                               j = D.i, b = D.a, spin_jb = -1)
                    coeff = t_compl / self.norm
                    if (D.i + D.j) % 2 == 1:
                        coeff = -coeff
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.i, a = D.b, spin_ia = +1,
                                               j = D.j, b = D.a, spin_jb = -1)
                    yield genWF.Doubly_Exc_Det(c = coeff,
                                               i = D.j, a = D.a, spin_ia = +1,
                                               j = D.i, b = D.b, spin_jb = -1)

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
