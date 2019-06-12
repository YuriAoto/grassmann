"""Probably not correct: the minimisation has to explore
symmetry broken wave functions.

"""
import numpy as np
from numpy import linalg
from scipy.linalg import expm, lu
import copy
import math

class Molpro_FCI_Wave_Function():
    
    def __init__(self, file_name = None, state='1.1'):
        self.determinants = []
        self.orb_dim = None
        self.n_frozen = None
        self.n_elec = None
        self.n_alpha = None
        self.n_beta = None
        self.beta_shift = None
        self.hf_occ = None
        self.hf_det = None
        sgn_invert = False
        if file_name is not None:
            FCI_found = False
            FCI_prog_found = False
            with open (file_name, 'r') as f:
                for l in f:
                    # if 'NUMBER OF CONTRACTIONS' in l:
                    #     self.orb_dim = map(lambda x: int(re.search('(\d+)',x).group()), 
                    #                        l.split('(')[1].split('+'))
                    if 'FCI STATE  '+state+' Energy' in l and 'Energy' in l:
                        FCI_found = True
                        continue
                    if 'PROGRAM * FCI (Full CI)' in l:
                        FCI_prog_found = True
                        continue
                    if 'Final occupancy:' in l:
                        self.hf_occ = map(int, l.split()[2:])
                    if FCI_found:
                        if 'CI Vector' in l:
                            continue
                        if 'EOF' in l:
                            break
                        lspl = l.split()
                        coef = float(lspl[0])
                        if len(self.determinants) == 0:
                            sgn_invert = coef < 0.0
                        if sgn_invert:
                            coef = -coef
                        det_descr = lspl[pos_a_ini:pos_a_fin] + lspl[pos_b_ini:pos_b_fin]
                        det_descr = map(lambda x: int(x) - self.n_frozen, det_descr)
                        self.determinants.append( [coef] + det_descr)
                        if det_descr == self.hf_det[1:]:
                            self.hf_det[0] = self.determinants[-1]
                    elif FCI_prog_found:
                        if 'Frozen orbitals:' in l:
                            self.n_frozen = map(int, re.findall('\d+', l.split('(')[1]))
                        if 'Active orbitals:' in l:
                            self.orb_dim = map(int, re.findall('\d+', l.split('(')[1]))
                            self.hf_det = [0.0]
                            cur_orb = 0
                            for occ in self.hf_occ:
                                for i_orb in range(wf.n_frozen[i_hf_occ] + wf.orb_dim[i_hf_occ]):
                                    cur_orb += 1
                                    if i_orb <= occ:
                                        self.hf_det.append(cur_orb)
                            l = len(self.hf_det)-1
                            for i in range(l):
                                self.hf_det.append(hf_det[i+1])

                        if 'Active electrons:' in l:
                            self.n_elec = int(l.split()[2])
                            if self.n_elec%2 != 0:
                                raise Exception('Only even number of electrons!')
                            self.n_alpha = self.n_elec/2
                            self.n_beta = self.n_alpha
                            self.beta_shift = self.n_alpha*(sum(self.orb_dim) - self.n_alpha)
                            pos_a_ini = 1 + self.n_frozen
                            pos_a_fin = pos_a_ini + self.n_alpha
                            pos_b_ini = 1 + 2*self.n_frozen + self.n_alpha
                            pos_b_fin = pos_b_ini + self.n_beta

            if not FCI_prog_found or not FCI_found:
                raise Exception('FCI wave function not found!')

    def __str__(self):
        x = '='*50 + '\n'
        x += 'n frozen: ' + str(self.n_frozen) + '\n'
        x += 'n electrons: ' + str(self.n_elec) + '\n'
        x += 'n alpha: ' + str(self.n_alpha) + '\n'
        x += 'n beta: ' + str(self.n_beta) + '\n'
        x += 'orb dim: ' + str(self.orb_dim) + '\n'
        x += 'beta shift: ' + str(self.beta_shift) + '\n'
        for det in self.determinants:
            x += '{0:10.6f} '.format(det[0])
            for i_occ in det[1:]:
                x += '{0:4d} '.format(i_occ)
            x += ' >> RANK: {0:d}\n'.format(rank_of_exc(det))
        x += '='*50 + '\n'
        return x

def rank_of_exc(determinant, reference_det):
    """ Works well for closed shell case, with n_alpha = n_beta
    """
    raise Exception('YAA not complete!')
    rank = 0
    N = (len(determinant)-1)/2 # dim of orb basis
    for i in determinant[1:]:
        if i > N:
            rank += 1
    return rank

def print_matrix(X, f_str):
    for i in X:
        for j in i:
            f_str.write(' {0:10.6f} '.format(j)\
                        if abs(j) > 1.0E-7 else
                        (' ' + '-'*10 + ' '))
        f_str.write('\n')
    f_str.write('\n')


def get_trans_max_coef(wf):
    """
    Returns Ua and Ub that transforms the orbitals
    such that the largest coefficient of wf
    is in the first determinant: 1,2,3,..., 1,2,3,...
    """
    det_max_coef = None
    for det in wf.determinants:
        if det_max_coef is None or abs(det[0]) > abs(det_max_coef[0]):
            det_max_coef = det

    Ua = np.identity(sum(wf.orb_dim))
    Ub = np.identity(sum(wf.orb_dim))

    # alpha orbitals
    extra_in_det = []
    miss_in_det = []
    for i in det_max_coef[1:wf.n_alpha+1]:
        if not (i-1 in range(wf.n_alpha)):
            extra_in_det.append(i)
    for i in range(wf.n_alpha):
        if not (i+1 in det_max_coef[1:wf.n_alpha+1]):
            miss_in_det.append(i+1)
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
            extra_in_det.append(i)
    for i in range(wf.n_beta):
        if not (i+1 in det_max_coef[wf.n_alpha+1:]):
            miss_in_det.append(i+1)
    for i,j in zip(extra_in_det, miss_in_det):
        Ub[i][i] = 0.0
        Ub[j][j] = 0.0
        Ub[j][i] = 1.0
        Ub[i][j] = 1.0

    print Ua
    print
    print Ub
    print

    return Ua, Ub




def transform_wf(wf, Ua, Ub, just_C0 = False):
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
                #print '='*20
                #print 'det_J: ', det_J
                #print 'det_I: ', det_I
                #print U_minor
                #print '='*20
        new_wf.determinants.append(new_det)
        if just_C0:
            break
    #print 'Number of det calculations:',  n_calcs
    return new_wf







#
# Parametrization of orb rotations:
#
# |0'> = exp(-K) |0>
#
# where K = K(alpha) + K(beta)
#    K(sigma) = sum_{i,a} K_i^a (a_i^a - a_a^i)
#
#  K_1^{n+1}    K_1^{n+2}   ...  K_1^orb_dim
#  K_2^{n+1}    K_2^{n+2}   ...  K_2^orb_dim
#           ...
#                      K_i^a
#           ...
#  K_n^{n+1}    K_n^{n+2}   ...  K_n^orb_dim
#
#   --> packed as:
#
#    Jac[0]   = K_1^{n+1}
#    Jac[1]   = K_2^{n+1}
#       ...
#    Jac[n-1] = K_n^{n+1}
#    Jac[0   + n] = K_1^{n+2}
#    Jac[1   + n] = K_2^{n+2}
#       ...
#    Jac[n-1 + n] = K_n^{n+2}
#    Jac[0   + 2n] = K_1^{n+3}
#       ...
#    Jac[i-1 + (a - n - 1)*n] = K_i^a
#      ...
#    Jac[n-1 + (orb_dim - n - 1)*n = n*(orb_dim - n) - 1] = K_n^orb_dim    <<< last alpha
#    Jac[n*(orb_dim - n) + 0] = K_1^2     <<< first beta
#      ...
#
#    -> beta are shifted by  b_shift = n*(orb_dim - n)
#

def get_position_in_jac(i, a, n, spin_shift):
    """Packs Jacobian: returns the position
    """
    return i-1 + (a - n-1)*n + spin_shift

def get_i_a_from_pos_in_jac(pos, n, b_shift):
    """ Unpacks Jacobian: returns (i,a)
    """
    s = 0 if pos < b_shift else b_shift
    return (1 + (pos - s)%n, # i
            1 + (pos - s)/n + n) # a


def construct_Jac_Hess(wf, f_str, analytic = True, verb = 0):
    if verb > 50:
        f_str.write('='*50 + '\n')
        f_str.write('Building Jacobian and Hessian')
        f_str.write(('Analytic' if analytic else 'Numerical') + ' procudure')
    n_param = 0
    for i, n_el in enumerate(wf.hf_occ):
        n_param += n_el*(wf.orb_dim[i] - n_el)
    n_param *= 2
    Jac = np.zeros(n_param)
    Hess = np.zeros((n_param, n_param))
    if analytic:
        for det in wf.determinants:
            exc_rank = rank_of_exc(det, wf.hf_det)

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

    if verb > 30:
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
    if verb > 30:
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
    if verb > 30:
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
    if verb > 5:
        f_str.write('# Matrices Ua and Ub are done. Going to calculate transformed wf now...\n')
    return (transform_wf(cur_wf, Ua, Ub, just_C0 = just_C0), Ua, Ub)

