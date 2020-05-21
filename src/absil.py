"""Functions related to the Newton-Grassmann method of Absil

Here, we consider the function f given by

f(x) = <0|ext>

where |0> is a Slater determinant and |ext> a correlated wave function.
The functions here are used to search critical points of this function
by the Newton method on the Grassmannian, as described in:

P-A Absil, R Mahony, and R Sepulchre, "Riemannian Geometry of Grassmann
Manifolds with a View on Algorithmic Computation", Acta App. Math. 80,
1999-220, 2004

Functions:
----------

calc_all_F
overlap_to_det
generate_lin_system
check_Newton_Absil_eq

"""
import math
import logging
import copy

import numpy as np
from scipy import linalg

from wave_functions import general as gen_wf
from wave_functions.cisd import Wave_Function_CISD
from util import logtime, get_n_from_triang
from exceptions import dGrValueError

logger = logging.getLogger(__name__)


def _calc_fI(U, det_indices):
    """Calculate the contribution of U[det_indices,:] to f
    
    Parameters:
    -----------
    
    U (numpy.ndarray)
        The coefficients matrix
    
    det_indices (ndarray if int8)
        A list of subindices for the columns of U
    
    Return:
    -------
    
    det(U[det_indices, :])
    """
    if len(det_indices) == 0:
        return 1.0
    return linalg.det(U[det_indices, :])


def _calc_G(U, det_indices, i, j):
    """Calculate the element ij of matrix G
    
    Behaviour:
    ----------
    
    Calculates the following determinant:
    det(U[det_indices, :] <-j- e_i )
    See _calc_H for details.
    
    Parameters:
    -----------
    
    U (numpy.ndarray)
        The coefficients matrix
    
    det_indices (ndarray if int8)
        A list of subindices
    
    i,j (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i )
    """
    if i not in det_indices:
        return 0.0
    if U.shape[1] == 1:
        return 1.0
    sign = 1 if (j + np.where(det_indices == i)[0][0]) % 2 == 0 else -1
    row_ind = np.array([x for x in det_indices if x != i], dtype=int)
    col_ind = np.array([x for x in range(U.shape[1]) if x != j], dtype=int)
    return sign * linalg.det(U[row_ind[:, None], col_ind])


def _calc_H(U, det_indices, i, j, k, l):
    """Calculate the element ijkl of matrix H
    
    Behaviour:
    ----------
    
    Calculates the following determinant:
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    where <-j- e_i means that the j-th column
    of the matrix is replaced by the vector e_i
    (with all zeros except at i, that has a 1).
    i is between 0 and U.shape[0], and by expansion
    on the replaced jth column, i+1 must be in det_indices
    for non vanishing determinant.
    Analogous for <-l- e_k.
    if j == l (trying to replace the same column)
    gives 0 (irrespective of i and k!)
    
    Parameters:
    -----------
    
    U (numpy.ndarray)
        The coefficients matrix
    
    det_indices (ndarray if int8)
        A list of subindices
    
    i,j,k,l (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    if j == l or i == k:
        return 0.0
    if i not in det_indices or k not in det_indices:
        return 0.0
    sign = 1.0 if ((i < k) == (j < l)) else -1.0
    if U.shape[1] == 2:
        return sign
    if (j + np.where(det_indices == i)[0][0]
            + l + np.where(det_indices == k)[0][0]) % 2 == 1:
        sign = -sign
    row_ind = np.array([x for x in det_indices if (x != i and x != k)],
                       dtype=int)
    col_ind = np.array([x for x in range(U.shape[1]) if (x != j and x != l)],
                       dtype=int)
    return sign * linalg.det(U[row_ind[:, None], col_ind])


def calc_all_F(wf, U):
    """Calculate all F needed in an iteration
    
    Behaviour:
    ----------
    
    Calculate all possible _calc_fI for the spirreps
    and string indices.
    
    Parameters:
    -----------
    wf (dGr_general_WF.Wave_Function)
    
    U (list of np.ndarray)
        See overlap_to_det for the details
    
    Return:
    -------
    
    A wf.n_spirrep list of 1D np.ndarray, in the order of
    spirrep for the list and string_indices for the array.
    """
    F = []
    for spirrep in wf.spirrep_blocks(restricted=False):
        F.append(np.zeros(wf.n_strings(spirrep, U[spirrep].shape[1])))
        for I in wf.string_indices(spirrep=spirrep,
                                   only_this_occ=U[spirrep].shape[1]):
            F[-1][int(I)] = _calc_fI(U[spirrep], I.occ_orb)
    return F


def overlap_to_det(wf, U, F=None, assume_orth=True):
    """Calculate the overlap between wf and the determinant U
    
    Behaviour:
    ----------
    
    Calculates f(x) = <wf|U>, where wf is a normalised wave function
    and U is a Slater determinant, not necessarily on the same orbital
    basis of wf.
    
    Limitations:
    ------------
    
    Only for unrestricted cases (well, restricted cases work, if
    the parameters are given in a redundant unrestricted format)
    
    Parameters:
    -----------
    
    wf (dGr_general_WF.Wave_Function)
        The external wave function
    
    U (list of np.ndarray)
        An element of the Stiefel manifold that represents a
        Slater determinant. It should be a list such as
        [U_a^1, ..., U_a^g, U_b^1, ..., U_b^g]
        where U_sigma^i is the U for spin sigma
        (alpha=a or beta=b) and irrep i.
    
    F (list of np.ndarray, default = None)
        The result of calc_all_F. Calculate if None.
    
    assume_orth (bool, default = True)
        If not True, the result is divided by the normalisation of U.
        Remember that wf is assumed to be normalised already!
        
    Return:
    -------
    
    The float <wf|U>
    """
    if isinstance(wf, Wave_Function_CISD):
        return _overlap_to_det_from_restricted_CISD(
            wf, U, assume_orth=assume_orth)
    elif isinstance(wf, gen_wf.Wave_Function):
        return _overlap_to_det_from_genWF(
            wf, U, F=F, assume_orth=assume_orth)
    else:
        raise dGrValueError('Unknown type of wave function')


def _overlap_to_det_from_restricted_CISD(wf, U, assume_orth=True):
    F0 = []
    Fs = []
    for irrep in wf.spirrep_blocks(restricted=True):
        Fs.append(np.zeros((wf.n_corr_orb[irrep], wf.n_ext[irrep])))
        Index = np.arange(U[irrep].shape[1])
        F0.append(_calc_fI(U[irrep], Index))
        for i, a, Index in _all_singles(U[irrep].shape[1],
                                        wf.n_corr_orb[irrep],
                                        wf.n_ext[irrep]):
            Fs[irrep][i, a] = _calc_fI(U[irrep], Index)
    f = wf.C0 * _calc_Fprod(F0, (), wf.n_irrep)**2
    for irrep in wf.spirrep_blocks(restricted=True):
        contr_irrep = np.einsum('ia,ia',
                                Fs[irrep], wf.Cs[irrep])
        for i, j, a, b, Index in _all_doubles(U[irrep].shape[1],
                                              wf.n_corr_orb[irrep],
                                              wf.n_ext[irrep]):
            contr_irrep += (_calc_fI(U[irrep], Index)
                            * wf.Cd[irrep][get_n_from_triang(
                                i, j, with_diag=False),
                                           get_n_from_triang(
                                               a, b, with_diag=False)])
        contr_irrep *= 2 * F0[irrep]
        contr_irrep += np.einsum('ijkl,ij,kl',
                                 wf.Csd[irrep][irrep], Fs[irrep], Fs[irrep])
        f += (_calc_Fprod(F0,
                          (irrep,),
                          wf.n_irrep)**2
              * contr_irrep)
        for irrep2 in range(irrep):
            f += (_calc_Fprod(F0,
                              (irrep, irrep2),
                              wf.n_irrep)**2
                  * 2 * F0[irrep] * F0[irrep2]
                  * np.einsum('iajb,ia,jb',
                              wf.Csd[irrep][irrep2],
                              Fs[irrep], Fs[irrep2]))
    if not assume_orth:
        for U_spirrep in U:
            if U_spirrep.shape[0] * U_spirrep.shape[1] != 0:
                f /= linalg.det(np.matmul(U_spirrep.T, U_spirrep))
    return f


def _overlap_to_det_from_genWF(wf, U, F=None, assume_orth=True):
    if F is None:
        F = calc_all_F(wf, U)
    f = 0.0
    for I in wf.string_indices(no_occ_orb=True,
                               only_this_occ=gen_wf.Orbitals_Sets(
                                   list(map(lambda U_i: U_i.shape[1], U)))):
        f_contr = 1.0
        for spirrep, I_spirrep in enumerate(I):
            if len(I_spirrep) == 0:
                continue
            I_spirrep.wf = wf
            I_spirrep.spirrep = spirrep
            f_contr *= F[spirrep][int(I_spirrep)]
        f += wf[I] * f_contr
    if not assume_orth:
        for U_spirrep in U:
            if U_spirrep.shape[0] * U_spirrep.shape[1] != 0:
                f /= math.sqrt(linalg.det(np.matmul(U_spirrep.T, U_spirrep)))
    return f


def make_slice_XC(U):
    """Generate the slices for the X,C matrix from generate_lin_system
    
    Parameters:
    -----------
    U (lsit of np.ndarrays)
        The orbital coefficients, as in generate_lin_system
    
    Return:
    -------
    A list of slices, for each spirrep block
    """
    new_slice = []
    for i, Ui in enumerate(U):
        if Ui.shape[1] > 0:
            logger.debug('U[%d]; shape = %s:\n%s',
                         i, Ui.shape, Ui)
        else:
            logger.debug('No electrons in irrep = %d', i)
        ini = 0 if i == 0 else new_slice[-1].stop
        new_slice.append(
            slice(ini, ini + Ui.shape[0] * Ui.shape[1]))
    logger.debug('new slice for XC matrices:\n%r',
                 new_slice)
    return new_slice


def generate_lin_system(
        wf, U, slice_XC, F=None, with_full_H=True):
    """Generate the linear system for Absil's method
    
    This is a wrapper to the actual functions.
    
    Behaviour:
    ----------
    
    Calculate the matrices that define the main linear system of
    equations in Absil's method: X @ eta = C
    
    Limitations:
    ------------
    
    Only for restricted calculations if wf is instance of
    Wave_Function_CISD, and only for unrestricted otherwise.
    
    Parameters:
    -----------
    
    U (list of np.ndarray)
    
    wf (dGr_general_WF.Wave_Function)
    
    F (list of np.ndarray, default=None)
    
        See overlap_to_det for the details of the above parameters
    
    slice_XC (list of slices)
        Slices of each spirrep in the blocks of matrices X and C
    
    with_full_H (bool, default=True)
        If True, calculates and store the full matrix H
        If False, store only row-wise.
        Although storing the full matrix uses more memory,
        numpy's broadicasting should speed up the calculation
    
    Return:
    -------
    
    The tuple (X, C), such that eta satisfies X @ eta = C
    """
    if isinstance(wf, Wave_Function_CISD):
        return _generate_lin_system_from_restricted_CISD(
            wf, U, slice_XC)
    elif isinstance(wf, gen_wf.Wave_Function):
        return _generate_lin_system_from_genWF(
            wf, U, slice_XC, F=F, with_full_H=with_full_H)
    else:
        raise dGrValueError('Unknown type of wave function')


def _calc_Fprod(F0, indices, max_ind):
    """Calculate the product of F0[i] with i not in indices"""
    F = 1.0
    for irrep in range(max_ind):
        if irrep in indices:
            continue
        F *= F0[irrep]
    return F


def _all_singles(n_el, n_corr, n_ext):
    """Generator that yield all single excitations, as (i,a,I)"""
    n_core = n_el - n_corr
    Index = np.zeros(n_el, dtype=int)
    Index[:n_core] = np.arange(n_core)
    Index[n_core:-1] = np.arange(n_core + 1, n_el)
    for i in range(n_corr):
        for a in range(n_ext):
            Index[-1] = n_el + a
            yield i, a, Index
        Index[n_core + i] = n_core + i


def _all_doubles(n_el, n_corr, n_ext):
    """Generator that yield all double excitations, as (i,j,a,b,I)"""
    n_core = n_el - n_corr
#    print('--------')
#    print('n_el = {}, n_corr = {}, n_ext = {}'.format(
#        n_el, n_corr, n_ext))
    Index = np.zeros(n_el, dtype=int)
    Index[:n_core] = np.arange(n_core)
    Index[n_core:-2] = np.arange(n_core + 2, n_el)
    for j in range(n_corr):
        for i in range(j + 1, n_corr):
            for a in range(n_ext):
                Index[-1] = n_el + a
                for b in range(a):
                    Index[-2] = (n_el
                                 if b == 0 else
                                 (Index[-2] + 1))
#                    print('i={},j={},a={},b={}\nI={}'.format(i,j,a,b,I))
                    yield i, j, a, b, Index
            Index[i - 1] = i
        if j < n_corr:
            Index[j] = j
            Index[j + 1:-2] = np.arange(j + 3, n_el, dtype=int)


def _generate_lin_system_from_restricted_CISD(
        wf, U, slice_XC):
    """Generate the linear system for Absil's method
    
    This is the specific implementation, for a restricted CISD
    wave function (dGr_CISD_WF.Wave_Function_CISD), using
    restricted orbitals, and for the component of the Grassmannian
    with same occupation as reference wave function
    
    Details:
    -------
    The indices of lists and arrays are in the following order:
    
    F0[irrep]
    G0[irrep][p,q]
    Fs[irrep][i,a]
    Gs[irrep][i,a,p,q]
    H[p,q,r,s]
    
    where:
    irrep      runs over all irreps
    p,r        run over all orbitals (of that irrep):
               0 <= p,r < K[irrep]
    q,s        run over all occupied orbitals (of that irrep):
               0 <= q,s < n[irrep]
    i,j        run over correlated occupied orbitals (of that irrep):
               0 <= i,j < wf.n_corr_orb[irrep]
               (add wf.n_core[irrep] to get corresponding position in U)
    a,b        run over virtual orbitals (of that irrep):
               0 <= a,b < wf.n_ext[irrep]
               (add wf.ref_occ[irrep] to get corresponding position in U)
   
    The relation between indices and the notation for X:
    
    X[irrep,irrep2] = X_irrep^irrep2
    
    """
    K = [U_irrep.shape[0] for U_irrep in U]
    n = [U_irrep.shape[1] for U_irrep in U]
    nK = [K[irrep] * n[irrep] for irrep in range(len(K))]
    sum_Kn = sum([K[irrep] * n[irrep] for irrep in range(len(K))])
    sum_nn = sum([n[irrep]**2 for irrep in range(len(K))])
    X = np.zeros((sum_Kn + sum_nn, sum_Kn))
    C = np.zeros(sum_Kn + sum_nn)
    f = 1.0
    Pi = []
    F0 = []
    G0 = []
    Fs = []
    Gs = []
    L = []
    Kmix = np.zeros((wf.n_irrep, wf.n_irrep))
    Fp = np.zeros((wf.n_irrep, wf.n_irrep))
    for irrep in wf.spirrep_blocks(restricted=True):
        Index = np.arange(n[irrep])
        F0.append(_calc_fI(U[irrep], Index))
        f *= F0[irrep]
        G0.append(np.zeros((K[irrep], n[irrep])))
        Fs.append(np.zeros((wf.n_corr_orb[irrep], wf.n_ext[irrep])))
        # -> not needed, right        Index = np.arange(n[irrep])
        for p in range(K[irrep]):
            for q in range(n[irrep]):
                G0[irrep][p, q] = _calc_G(U[irrep], Index,
                                          p, q)
        for i in range(wf.n_corr_orb[irrep]):
            for a in range(wf.n_ext[irrep]):
                Fs[irrep][i, a] = np.dot(U[irrep][wf.ref_occ[irrep] + a, :],
                                         G0[irrep][wf.n_core[irrep] + i, :])
            if (wf.n_core[irrep] + i + n[irrep] - 1) % 2 == 1:
                Fs[irrep][i, :] *= -1
        Pi.append(np.identity(K[irrep]) - U[irrep] @ U[irrep].T)
        if n[irrep] > 0:
            logger.debug('For irrep = %d:\nF0 = %f\nFs:\n%r\nG0:\n%r\nPi:\n%r',
                         irrep, F0[irrep], Fs[irrep], G0[irrep], Pi[irrep])
    f = wf.C0 * f**2
    logger.debug('C0 * Fprod (first contrib. to f(Y)) = %f', f)
    for irrep in wf.spirrep_blocks(restricted=True):
        Gs.append(np.zeros((wf.n_corr_orb[irrep], wf.n_ext[irrep],
                            K[irrep], n[irrep])))
        H = np.zeros((K[irrep], n[irrep],
                      K[irrep], n[irrep]))
        Index = np.arange(n[irrep])
        for p in range(K[irrep]):
            for q in range(n[irrep]):
                H[p, q, p, q] = -F0[irrep]
                for r in range(p):
                    for s in range(q):
                        H[r, s, p, q] = _calc_H(U[irrep],
                                                Index,
                                                r, s, p, q)
                        H[r, q, p, s] = H[p, s, r, q] = -H[r, s, p, q]
                        H[p, q, r, s] = H[r, s, p, q]
        for i in range(wf.n_corr_orb[irrep]):
            for a in range(wf.n_ext[irrep]):
                for p in range(n[irrep]):
                    if p == wf.n_core[irrep] + i:
                        continue
                    for q in range(n[irrep]):
                        Gs[irrep][i, a, p, q] = (
                            np.dot(U[irrep][wf.ref_occ[irrep] + a, :],
                                   H[p, q, wf.n_core[irrep] + i, :])
                            - (U[irrep][wf.ref_occ[irrep] + a, q]
                               * H[p, q, wf.n_core[irrep] + i, q]))
                Gs[irrep][i, a, wf.ref_occ[irrep] + a, :] = (
                    G0[irrep][wf.n_core[irrep] + i, :])
            if (wf.n_core[irrep] + i + n[irrep] - 1) % 2 == 1:
                Gs[irrep][i, :, :, :] *= -1
        if n[irrep] > 0:
            logger.debug('For irrep = %d:\nH_I0:\n%r\nGs:\n%r',
                         irrep, H, Gs[irrep])
        D = 0.0
        D2 = np.array(wf.Cs[irrep])
        for irrep2 in wf.spirrep_blocks(restricted=True):
            if irrep == irrep2:
                continue
            if irrep > irrep2:
                D2 += np.einsum('iajb,jb->ia',
                                wf.Csd[irrep][irrep2], Fs[irrep2]) / F0[irrep2]
            else:
                D2 += np.einsum('iajb,ia->jb',
                                wf.Csd[irrep2][irrep], Fs[irrep2]) / F0[irrep2]
        for i in range(wf.n_corr_orb[irrep]):
            # Here, b<a always
            sign = 1 if (wf.n_core[irrep] + i + n[irrep] + 1) % 2 == 1 else -1
            for a in range(wf.n_ext[irrep]):
                for j in range(i):
                    ij = get_n_from_triang(i, j, with_diag=False)
                    for b in range(a):
                        ab = get_n_from_triang(a, b, with_diag=False)
                        tmp = sign * np.dot(U[irrep][wf.ref_occ[irrep] + a, :],
                                            Gs[irrep][j, b,
                                                      wf.n_core[irrep] + i, :])
                        logger.debug('Current F_{i,j=%d,%d}^{a,b=%d,%d} = %f',
                                     i, j, a, b, tmp)
                        D += wf.Cd[irrep][ij, ab] * tmp
        logger.debug('Cd[ijab]*Fd[ijab] (contr. of doubles) for irrep %d = %f',
                     irrep, D)
        tmp = np.einsum('ijkl,ij,kl',
                        wf.Csd[irrep][irrep], Fs[irrep], Fs[irrep])
        L.append(tmp)
        logger.debug('L[%d] <- %f (contrib of doubles=singles*singles)',
                     irrep, tmp)
        tmp = np.einsum('ia,ia',
                        Fs[irrep], wf.Cs[irrep])
        logger.debug('Cs*Fs for irrep %d = %f (contrib of singles) ',
                     irrep, tmp)
        L[-1] += 2 * F0[irrep] * (D + tmp)
        tmp = F0[irrep] * wf.C0 + D
        C[slice_XC[irrep]] = np.ravel(G0[irrep]) * tmp
        logger.debug('C at 1 [%d]:\n%r', irrep, C[slice_XC[irrep]])
        C[slice_XC[irrep]] += np.ravel(
            np.einsum('ijkl,ij,klmn->mn',
                      wf.Csd[irrep][irrep], Fs[irrep], Gs[irrep]))
        logger.debug('C at 2 [%d]:\n%r', irrep, C[slice_XC[irrep]])
        tmp += np.tensordot(D2, Fs[irrep], axes=2)
        X[slice_XC[irrep],
          slice_XC[irrep]] = np.reshape(H, (nK[irrep],
                                            nK[irrep])) * tmp
        # Here, H_I0 in H is not needed anymore. We will use H for the "bigG"
        H = wf.C0 * (np.einsum('ij,kl->ijkl',
                               G0[irrep], G0[irrep]))
        H += np.einsum('iajb,iapq,jbrs->pqrs',
                       wf.Csd[irrep][irrep], Gs[irrep], Gs[irrep])
        H += np.einsum('ia,iapq,rs->pqrs',
                       D2, Gs[irrep], G0[irrep])
        H += np.einsum('ia,pq,iars->pqrs',
                       D2, G0[irrep], Gs[irrep])
        Gs[irrep] = F0[irrep] * Gs[irrep] + np.einsum('ia,pq->iapq',
                                                      Fs[irrep], G0[irrep])
        C[slice_XC[irrep]] += np.ravel(np.einsum('ia,iapq->pq',
                                                 D2, Gs[irrep]))
        logger.debug(r'\hat{Gs}[irrep = %d]:\n%r',
                     irrep, Gs[irrep])
        logger.debug('C at 3:\n%r', C[slice_XC[irrep]])
        Gd = np.zeros((K[irrep], n[irrep]))
        for i, j, a, b, Index in _all_doubles(n[irrep],
                                              wf.n_corr_orb[irrep],
                                              wf.n_ext[irrep]):
            ij = get_n_from_triang(i, j, with_diag=False)
            ab = get_n_from_triang(a, b, with_diag=False)
            for p in range(K[irrep]):
                for q in range(n[irrep]):
                    Gd[p, q] = _calc_G(U[irrep], Index,
                                       p, q)
            C[slice_XC[irrep]] += np.ravel(F0[irrep]
                                           * wf.Cd[irrep][ij, ab] * Gd)
            H += wf.Cd[irrep][ij, ab] * np.einsum('pq,rs->pqrs',
                                                  Gd, G0[irrep])
            H += wf.Cd[irrep][ij, ab] * np.einsum('pq,rs->pqrs',
                                                  G0[irrep], Gd)
        logger.debug("bigG[%d]:\n%r", irrep, H)
        # Here: bigG is complete and is used only in the line below
        X[slice_XC[irrep],
          slice_XC[irrep]] += np.reshape(np.einsum('pqts,rt->pqrs',
                                                   H, Pi[irrep]),
                                         (nK[irrep], nK[irrep]))
        logger.debug("bigG[%d] @ Pi[%d]:\n%r",
                     irrep, irrep,
                     np.einsum('pqts,rt->pqrs',
                               H, Pi[irrep]))
        for i, a, Index in _all_singles(n[irrep],
                                        wf.n_corr_orb[irrep],
                                        wf.n_ext[irrep]):
            for p in range(K[irrep]):
                for q in range(n[irrep]):
                    H[p, q, p, q] = -Fs[irrep][i, a]
                    for r in range(p + 1):
                        for s in range(q + 1):
                            if r == p and s == q:
                                continue
                            H[r, s, p, q] = _calc_H(U[irrep],
                                                    Index,
                                                    r, s, p, q)
                            H[r, q, p, s] = H[p, s, r, q] = -H[r, s, p, q]
                            H[p, q, r, s] = H[r, s, p, q]
            logger.debug('H_I_{i=%d}^{a=%d}[%d]:\n%r', i, a, irrep, H)
            tmp = (D2[i, a] * F0[irrep]
                   + np.tensordot(wf.Csd[irrep][irrep][i, a, :, :],
                                  Fs[irrep],
                                  axes=2))
            X[slice_XC[irrep],
              slice_XC[irrep]] += np.reshape(H,
                                             (nK[irrep],
                                              nK[irrep])) * tmp
        for i, j, a, b, Index in _all_doubles(n[irrep],
                                              wf.n_corr_orb[irrep],
                                              wf.n_ext[irrep]):
            ij = get_n_from_triang(i, j, with_diag=False)
            ab = get_n_from_triang(a, b, with_diag=False)
            for p in range(K[irrep]):
                for q in range(n[irrep]):
                    H[p, q, p, q] = -_calc_fI(U[irrep], Index)
                    for r in range(p):
                        for s in range(q):
                            H[r, s, p, q] = _calc_H(U[irrep],
                                                    Index,
                                                    r, s, p, q)
                            H[r, q, p, s] = H[p, s, r, q] = -H[r, s, p, q]
                            H[p, q, r, s] = H[r, s, p, q]
            logger.debug('i=%d,j=%d -> a=%d,b=%d (I=%r): H:\n%r',
                         i, j, a, b, Index, H)
            tmp = F0[irrep] * wf.Cd[irrep][ij, ab]
            logger.debug('F0*Cd = %e', tmp)
            logger.debug(X[slice_XC[irrep],
                           slice_XC[irrep]])
            X[slice_XC[irrep],
              slice_XC[irrep]] += np.reshape(H,
                                             (nK[irrep],
                                              nK[irrep])) * tmp
            logger.debug(X[slice_XC[irrep],
                           slice_XC[irrep]])
        D = 1.0
        logger.debug('bigH[%d] + bigG@Pi:\n%r', irrep, X[slice_XC[irrep],
                                                         slice_XC[irrep]])
        # Fill only X[irrep][irrep2] with irrep >= irrep2
        for irrep2 in wf.spirrep_blocks(restricted=True):
            if irrep2 == irrep:
                continue
            D *= F0[irrep2]
            if irrep2 < irrep:
                X[slice_XC[irrep],
                  slice_XC[irrep2]] += F0[irrep2] * np.outer(
                      C[slice_XC[irrep]],
                      G0[irrep2])
            elif irrep2 > irrep:
                X[slice_XC[irrep2],
                  slice_XC[irrep]] += F0[irrep2] * np.outer(G0[irrep2],
                                                            C[slice_XC[irrep]])
            logger.debug('X[irrep2=%d,irrep=%d] (X_%d^%d):\n%r',
                         irrep2, irrep, irrep2, irrep,
                         X[slice_XC[irrep2],
                           slice_XC[irrep]])
            logger.debug('X[irrep=%d,irrep2=%d] (X_%d^%d):\n%r',
                         irrep, irrep2, irrep, irrep2,
                         X[slice_XC[irrep],
                           slice_XC[irrep2]])
        D *= D
        f += D * L[irrep]
        logger.debug(r'\Prod(irrep != %d) F0[irrep]**2 = %f', irrep, D)
        X[slice_XC[irrep],
          slice_XC[irrep]] -= np.outer(C[slice_XC[irrep]], U[irrep])
        if n[irrep] > 0:
            logger.debug('For irrep = %d:\nprod F0[i]**2 (for i != irrep) = %f'
                         + '\nL = %f\nM:\n%r\n'
                         + '-(MxU - bigG@Pi - bigH):\n%r',
                         irrep, D,
                         L[irrep], C[slice_XC[irrep]],
                         X[slice_XC[irrep],
                           slice_XC[irrep]])
        X[slice_XC[irrep],
          slice_XC[irrep]] *= -D
        C[slice_XC[irrep]] *= D
    for irrep in wf.spirrep_blocks(restricted=True):
        for irrep2 in range(irrep):
            X[slice_XC[irrep],
              slice_XC[irrep2]] += np.reshape(np.einsum('iajb,iapq,jbrs->pqrs',
                                                        wf.Csd[irrep][irrep2],
                                                        Gs[irrep],
                                                        Gs[irrep2]),
                                              (nK[irrep],
                                               nK[irrep2])) / 2
            logger.debug('X[irrep=%d,irrep2=%d] (X_%d^%d) after Csd:\n%r',
                         irrep, irrep2, irrep, irrep2,
                         X[slice_XC[irrep],
                           slice_XC[irrep2]])
            Kmix[irrep, irrep2] = np.einsum('iajb,ia,jb',
                                            wf.Csd[irrep][irrep2],
                                            Fs[irrep], Fs[irrep2])
            Kmix[irrep, irrep2] *= 2 * F0[irrep] * F0[irrep2]
            Kmix[irrep2, irrep] = Kmix[irrep, irrep2]
            Fp[irrep, irrep2] = _calc_Fprod(F0, (irrep, irrep2), wf.n_irrep)
            Fp[irrep, irrep2] *= Fp[irrep, irrep2]
            Fp[irrep2, irrep] = Fp[irrep, irrep2]
            f += Fp[irrep, irrep2] * Kmix[irrep, irrep2]
            Gd = np.einsum('ia,jbpq,jbia->pq',
                           Fs[irrep2], Gs[irrep], wf.Csd[irrep][irrep2])
            if n[irrep] > 0 and n[irrep2] > 0:
                logger.debug('For irrep=%d, irrep2=%d:\n'
                             + r'\Prod(i != irrep, irrep2) F0[i]**2 = %f'
                             + '\n'
                             + 'Kmix[irrep,irrep2] = %f\n'
                             + 'Fs[irrep2]Gs[irrep]Csd[irrep,irrep2]:\n%r',
                             irrep, irrep2,
                             Fp[irrep][irrep2],
                             Kmix[irrep, irrep2],
                             Gd)
            X[slice_XC[irrep],
              slice_XC[irrep2]] -= np.outer(Gd, G0[irrep2])
            Gd = np.einsum('ia,jbpq,iajb->pq',
                           Fs[irrep], Gs[irrep2], wf.Csd[irrep][irrep2])
            if n[irrep] > 0 and n[irrep2] > 0:
                logger.debug('For irrep=%d, irrep2=%d:\n'
                             + 'Fs[irrep]Gs[irrep2]Csd[irrep,irrep2]:\n%r',
                             irrep, irrep2,
                             Gd)
            X[slice_XC[irrep],
              slice_XC[irrep2]] -= np.outer(G0[irrep], Gd)
            X[slice_XC[irrep],
              slice_XC[irrep2]] *= Fp[irrep, irrep2]
    for irrep in wf.spirrep_blocks(restricted=True):
        H = np.zeros((K[irrep], n[irrep],
                      K[irrep], n[irrep]))
        Index = np.arange(n[irrep])  # Index for reference
        for p in range(K[irrep]):
            for q in range(n[irrep]):
                H[p, q, p, q] = -F0[irrep]
                for r in range(p):
                    for s in range(q):
                        H[r, s, p, q] = _calc_H(U[irrep],
                                                Index,
                                                r, s, p, q)
                        H[r, q, p, s] = H[p, s, r, q] = -H[r, s, p, q]
                        H[p, q, r, s] = H[r, s, p, q]
        H *= F0[irrep]
        H += np.einsum('pq,ts,rt ->pqrs',
                       G0[irrep], G0[irrep], Pi[irrep])
        Gd = F0[irrep] * G0[irrep]
        H -= np.einsum('pq,rs->pqrs',
                       Gd, U[irrep])
        H *= -1
        D = 0.0
        for irrep2 in wf.spirrep_blocks(restricted=True):
            if irrep2 == irrep:
                continue
            D += L[irrep2] * Fp[irrep, irrep2]
            for irrep3 in range(irrep2):
                if irrep3 == irrep:
                    continue
                Fg1g2g3 = _calc_Fprod(F0,
                                      (irrep, irrep2, irrep3),
                                      wf.n_irrep)
                D += Kmix[irrep2, irrep3] * Fg1g2g3**2
        C[slice_XC[irrep]] += np.ravel(D * Gd)
        X[slice_XC[irrep],
          slice_XC[irrep]] += np.reshape(H,
                                         (nK[irrep],
                                          nK[irrep])) * D
    for irrep in wf.spirrep_blocks(restricted=True):
        logger.debug('C before proj (irrep = %d):\n%r',
                     irrep, C)
        C[slice_XC[irrep]] = np.ravel(
            Pi[irrep] @ np.reshape(C[slice_XC[irrep]],
                                   (K[irrep],
                                    n[irrep])))
        logger.debug('X after reshape:\n%r',
                     np.reshape(X[slice_XC[irrep],
                                  slice_XC[irrep]],
                                (K[irrep], n[irrep],
                                 K[irrep], n[irrep])))
        logger.debug('projX before re-reshape:\n%r',
                     np.einsum('pt,tqrs->pqrs',
                               Pi[irrep], np.reshape(X[slice_XC[irrep],
                                                       slice_XC[irrep]],
                                                     (K[irrep], n[irrep],
                                                      K[irrep], n[irrep]))))
        X[slice_XC[irrep],
          slice_XC[irrep]] = np.reshape(
              np.einsum('pt,tqrs->pqrs',
                        Pi[irrep],
                        np.reshape(X[slice_XC[irrep],
                                     slice_XC[irrep]],
                                   (K[irrep], n[irrep],
                                    K[irrep], n[irrep]))),
              (nK[irrep],
               nK[irrep]))
        for irrep2 in range(irrep):
            D = -wf.C0 * Fp[irrep, irrep2]
            for irrep3 in wf.spirrep_blocks(restricted=True):
                if irrep3 == irrep or irrep3 == irrep2:
                    continue
                Fg1g2g3 = _calc_Fprod(F0,
                                      (irrep, irrep2, irrep3),
                                      wf.n_irrep)
                D += L[irrep3] * Fg1g2g3**2
                for irrep4 in range(irrep3):
                    if irrep4 == irrep or irrep4 == irrep2:
                        continue
                    Fg1g2g3g4 = _calc_Fprod(F0,
                                            (irrep, irrep2, irrep3, irrep4),
                                            wf.n_irrep)
                    D += Kmix[irrep3][irrep4] * Fg1g2g3g4**2
            X[slice_XC[irrep],
              slice_XC[irrep2]] += (np.outer(G0[irrep],
                                             G0[irrep2])
                                    * D * F0[irrep] * F0[irrep2])
            logger.debug('X[irrep=%d,irrep2=%d] (before projection):\n%r',
                         irrep, irrep2,
                         X[slice_XC[irrep],
                           slice_XC[irrep2]])
            X[slice_XC[irrep],
              slice_XC[irrep2]] = -2 * np.reshape(
                  np.einsum('pt,tqus,ru->pqrs',
                            Pi[irrep],
                            np.reshape(X[slice_XC[irrep],
                                         slice_XC[irrep2]],
                                       (K[irrep], n[irrep],
                                        K[irrep2], n[irrep2])),
                            Pi[irrep2]),
                  (nK[irrep],
                   nK[irrep2]))
            X[slice_XC[irrep2],
              slice_XC[irrep]] = X[slice_XC[irrep],
                                   slice_XC[irrep2]].T
    # Terms to guarantee orthogonality to U:
    shift_irrep_pq = 0
    pos_rs = 0
    for irrep, U_irrep in enumerate(U):
        for p in range(K[irrep]):
            pos_pq = sum_Kn + shift_irrep_pq
            for q in range(n[irrep]):
                logger.debug('To guarantee orthogonality:\n'
                             + ' Adding at [%d: %d + %d, %d] U[%d,:]:\n%r',
                             pos_pq, pos_pq, n[irrep], pos_rs,
                             p, U_irrep[p, :])
                X[pos_pq: pos_pq + n[irrep],
                  pos_rs] = U_irrep[p, :]
                pos_rs += 1
                pos_pq += n[irrep]
        shift_irrep_pq += n[irrep]**2
    return f, X, C


def _generate_lin_system_from_genWF(
        wf, U, slice_XC, F=None, with_full_H=True):
    """Generate the linear system for Absil's method
    
    This is the general implementation, for a general wave function
    (dGr_general_WF.Wave_Function)
    
    It is actually slow.
    
    """
    if not with_full_H:
        raise NotImplementedError('with_full_H = False is not Implemented')
    if F is None:
        F = calc_all_F(wf, U)
    f = overlap_to_det(wf, U, F)
    K = [U_spirrep.shape[0] for U_spirrep in U]
    n = [U_spirrep.shape[1] for U_spirrep in U]
    sum_Kn = sum([K[i] * n[i] for i in range(len(K))])
    sum_nn = sum([n[i]**2 for i in range(len(K))])
    X = np.zeros((sum_Kn + sum_nn, sum_Kn))
    C = np.zeros(sum_Kn + sum_nn)
    for spirrep_1 in wf.spirrep_blocks(restricted=False):
        logger.info('Starting spirrep_1 = %d', spirrep_1)
        logger.info('n_irrep = %d', wf.n_irrep)
        Pi = np.identity(K[spirrep_1]) - U[spirrep_1] @ U[spirrep_1].T
        logger.debug('Pi = %s', Pi)
        for I_1 in wf.string_indices(spirrep=spirrep_1):
            with logtime('Calc H, G'):
                logger.debug('At I_1 = %s', I_1)
                if len(I_1) != n[spirrep_1]:
                    continue
                H = np.zeros((K[spirrep_1], n[spirrep_1],
                              K[spirrep_1], n[spirrep_1]))
                G_1 = np.zeros((K[spirrep_1], n[spirrep_1]))
                for i in range(K[spirrep_1]):
                    for j in range(n[spirrep_1]):
                        H[i, j, i, j] = -F[spirrep_1][int(I_1)]
                        for k in range(i):
                            for l in range(j):
                                H[k, l, i, j] = _calc_H(U[spirrep_1],
                                                        I_1.occ_orb,
                                                        k, l, i, j)
                                H[k, j, i, l] = H[i, l, k, j] = -H[k, l, i, j]
                                H[i, j, k, l] = H[k, l, i, j]
                        #  G_1[i,j] = np.dot(H[:,0,i,j], U[spirrep_1][:,0])
                        G_1[i, j] = _calc_G(U[spirrep_1],
                                            I_1.occ_orb,
                                            i, j)
                logger.debug('current H:\n%s', H)
                logger.debug('current G:\n%s', G_1)
                H = Pi @ (np.multiply.outer(U[spirrep_1], G_1) - H)
                logger.debug('Pi (U G - H):\n%s', H)
            with logtime('Calc S'):
                S = 0.0
                logger.info('spirrep_1 = %d; I_1 = %s', spirrep_1, I_1)
                for I_full in wf.string_indices(
                        coupled_to=(gen_wf.Spirrep_Index(spirrep=spirrep_1,
                                                         Index=I_1),)):
                    if list(map(len, I_full)) != list(
                            map(lambda x: x.shape[1], U)):
                        continue
                    F_contr = 1.0
                    for spirrep_other, I_other in enumerate(I_full):
                        if spirrep_other != spirrep_1 and len(I_other) > 0:
                            F_contr *= F[spirrep_other][int(I_other)]
                    S += wf[I_full] * F_contr
                logger.debug('S = %s; H:\n%s', S, H)
            with logtime('Calc Xdiag, C'):
                X[slice_XC[spirrep_1],
                  slice_XC[spirrep_1]] += S * np.reshape(
                      H,
                      (K[spirrep_1] * n[spirrep_1],
                       K[spirrep_1] * n[spirrep_1]),
                      order='F').T
                G_1 -= F[spirrep_1][int(I_1)] * U[spirrep_1]
                logger.debug('spirrep=%d: S = %s; G_1:\n %s',
                             spirrep_1, S, G_1)
                C[slice_XC[spirrep_1]] += S * np.reshape(
                    G_1,
                    (K[spirrep_1] * n[spirrep_1],),
                    order='F')
            for spirrep_2 in wf.spirrep_blocks(restricted=False):
                if spirrep_2 <= spirrep_1:
                    continue
                logger.info('At spirrep_2 = %d', spirrep_2)
                with logtime('spirrep_2'):
                    G_2 = np.zeros((K[spirrep_2], n[spirrep_2]))
                    for I_2 in wf.string_indices(
                            spirrep=spirrep_2,
                            coupled_to=(gen_wf.Spirrep_Index(spirrep=spirrep_1,
                                                             Index=I_1),)):
                        if len(I_2) != n[spirrep_2]:
                            continue
                        logger.debug('I_2 = %s', I_2)
                        for k in range(K[spirrep_2]):
                            for l in range(n[spirrep_2]):
                                G_2[k, l] = _calc_G(U[spirrep_2],
                                                    I_2.occ_orb,
                                                    k, l)
                        G_2 -= F[spirrep_2][int(I_2)] * U[spirrep_2]
                        S = 0.0
                        for I_full in wf.string_indices(
                                coupled_to=(
                                    gen_wf.Spirrep_Index(
                                        spirrep=spirrep_1,
                                        Index=I_1),
                                    gen_wf.Spirrep_Index(
                                        spirrep=spirrep_2,
                                        Index=I_2))):
                            if list(map(len, I_full)) != list(
                                    map(lambda x: x.shape[1], U)):
                                continue
                            F_contr = 1.0
                            for spirrep_other, I_other in enumerate(I_full):
                                if (wf.ref_occ[spirrep_other] > 0
                                    and spirrep_other != spirrep_1
                                        and spirrep_other != spirrep_2):
                                    F_contr *= F[spirrep_other][int(I_other)]
                            S += wf[I_full] * F_contr
                        logger.debug('G_1:\n%s', G_1)
                        logger.debug('G_2:\n%s', G_2)
                        logger.debug('S = %s', S)
                        X[slice_XC[spirrep_1],
                          slice_XC[spirrep_2]] -= S * np.reshape(
                              np.multiply.outer(G_1, G_2),
                              (K[spirrep_1] * n[spirrep_1],
                               K[spirrep_2] * n[spirrep_2]),
                              order='F')
            for spirrep_2 in wf.spirrep_blocks(restricted=False):
                if spirrep_1 > spirrep_2:
                    X[slice_XC[spirrep_1],
                      slice_XC[spirrep_2]] = X[slice_XC[spirrep_2],
                                               slice_XC[spirrep_1]].T
    # Terms to guarantee orthogonality to U:
    prev_ij = sum_Kn
    prev_kl = 0
    for spirrep, U_spirrep in enumerate(U):
        for i in range(n[spirrep]):
            logger.debug('Adding U^T:\n%s', U_spirrep.T)
            logger.debug('Position of U^T = [%s: %s + %s, %s: %s + %s',
                         prev_ij, prev_ij, n[spirrep],
                         prev_kl, prev_kl, K[spirrep])
            X[prev_ij: prev_ij + n[spirrep],
              prev_kl: prev_kl + K[spirrep]] = U_spirrep.T
            prev_ij += n[spirrep]
            prev_kl += K[spirrep]
    return f, X, C


def calc_storage_CISD(occ, corr, virt):
    """
    Calculates the storage needed for the algorithm for CISD wave functions
    """
    n_inp = 1
    n_out = Fs = Gd = G0 = Gs = bigG = H = others = 0
    for irrep in range(len(occ)):
        nv = corr[irrep] * virt[irrep]
        nK = occ[irrep] * (virt[irrep] + occ[irrep])
        n_inp += nK  # Y
        n_inp += nv  # singles
        n_inp += (virt[irrep]**2) * corr[irrep] * (corr[irrep] + 1) // 2
        n_inp += (corr[irrep] * (corr[irrep] - 1)
                  * virt[irrep] * (virt[irrep] - 1)) // 4
        for irrep2 in range(irrep):
            n_inp += nv * corr[irrep2] * virt[irrep2]
        n_out += nK
        Fs += nv
        Gs += nv * nK
        Gd = max(Gd, nK)
        H = max(H, nK**2)
        others = max(others, nv)
    G0 = n_out
    bigG = H
    n_out = n_out * (n_out + 1)
    others += 2 * (1 + len(occ) + len(occ)**2)
    total = n_inp + n_out + Fs + Gd + G0 + Gs + bigG + H + others
    return (n_inp, n_out, Fs, Gd, G0, Gs, bigG, H, others, total)


def check_Newton_eq(wf, U, eta, restricted, eps=0.001):
    """Check, numerically, if eta satisfies Absil equation
    
    Parameters:
    -----------
    
    wf (dGr_general_WF.Wave_Function)
    
    U (list of np.ndarray)
    
        See overlap_to_det for the details of the above parameters
    
    eta (list of np.ndarray)
        An element of the Stiefel manifold that contains the
        (supposed) solution of Absil equations.
        It has same number of elements as U
    
    restricted (bool)
        If True, assumes that U contains only one set of spatial orbitals
        If False, U should contain alpha and beta elements
    
    eps (float, optional, default = 0.001)
        Step size to calculate numerical derivatives
    
    Behaviour:
    ----------
    
    Print in the log (INFO) the main elements of
    Absil equation for the Newton step on the Grassmannian.
    
    Return:
    -------
    True if, for all irreps, eta satisfies the equation
    and is orthogonal to U.
    False if the test fails in some case
    """
    def get_shifted_U(shift):
        """return U plus shift"""
        Umod = copy.deepcopy(U)
        for spirrep in range(len(U)):
            Umod[spirrep] += shift[spirrep]
        return Umod
    
    def calc_grad(this_U, spirrep):
        grad = np.zeros(this_U[spirrep].shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                this_U[spirrep][i, j] += eps
                grad[i, j] = overlap_to_det(wf, this_U, assume_orth=False)
                this_U[spirrep][i, j] -= 2 * eps
                grad[i, j] -= overlap_to_det(wf, this_U, assume_orth=False)
                grad[i, j] /= 2 * eps
                this_U[spirrep][i, j] += eps
        return grad
    all_true = True
    Uplus = get_shifted_U([eps * eta_irp for eta_irp in eta])
    Uminus = get_shifted_U([-eps * eta_irp for eta_irp in eta])
    for spirp, Ueta in enumerate(zip(U, eta)):
        U_irp, eta_irp = Ueta
        if U_irp.shape[0] * U_irp.shape[1] == 0:
            logger.info('Skipping irrep %d for having empty U', spirp)
            continue
        U_T_eta = np.matmul(U_irp.T, eta_irp)
        Proj = np.identity(U_irp.shape[0]) - np.matmul(U_irp,
                                                       U_irp.T)
        grad = calc_grad(U, spirp)
        RHS = -np.matmul(Proj, grad)
        # ---
        Dgrad_plus = calc_grad(Uplus, spirp)
        UT_U_inv = linalg.inv(np.matmul(Uplus[spirp].T, Uplus[spirp]))
        Proj_plus = (np.identity(Uplus[spirp].shape[0])
                     - np.matmul(Uplus[spirp],
                                 np.matmul(UT_U_inv, Uplus[spirp].T)))
        Dgrad_minus = calc_grad(Uminus, spirp)
        UT_U_inv = linalg.inv(np.matmul(Uminus[spirp].T, Uminus[spirp]))
        Proj_minus = (np.identity(Uminus[spirp].shape[0])
                      - np.matmul(Uminus[spirp],
                                  np.matmul(UT_U_inv, Uminus[spirp].T)))
        Dgrad = (np.matmul(Proj_plus, Dgrad_plus)
                 - np.matmul(Proj_minus, Dgrad_minus)) / (2 * eps)
        # ---
        LHS = np.matmul(Proj, Dgrad)
        # ---
        hess = (Dgrad_plus - Dgrad_minus) / (2 * eps)
        LHS_mod = hess - np.matmul(np.matmul(eta_irp,
                                             U_irp.T),
                                   grad)
        # ---
        logger.info('U.T @ U; spirrep = %d:\n%s',
                     spirp, np.matmul(U_irp.T, U_irp))
        logger.info('U[%d]:\n%s', spirp, U_irp)
        logger.info('eta[%d]:\n%s', spirp, eta_irp)
        logger.info('U.T @ eta [%d] (eta should be orth to U):\n%s',
                    spirp, U_T_eta)
        logger.info('Proj[%d]:\n%s', spirp, Proj)
        logger.info('grad[%d]:\n%s', spirp, grad)
        logger.info('Dgrad_plus[%d]:\n%s', spirp, Dgrad_plus)
        logger.info('Proj_plus @ Dgrad_plus[%d]:\n%s',
                    spirp, np.matmul(Proj_plus, Dgrad_plus))
        logger.info('Dgrad_minus[%d]:\n%s', spirp, Dgrad_minus)
        logger.info('Proj_minus @ Dgrad_minus[%d]:\n%s',
                    spirp, np.matmul(Proj_minus, Dgrad_minus))
        logger.info('RHS[%d]:\n%s', spirp, RHS)
        logger.info('LHS[%d]:\n%s', spirp, LHS)
#        logger.info('LHS/RHS[%d]:\n%s', spirp, LHS/RHS)
        logger.info('D(Pi grad f)(U)[eta]; spirp=%d:\n%s',
                    spirp, Dgrad)
        logger.info('hess = d/dt grad f(U + t*eta); spirp=%d:\n%s',
                    spirp, hess)
        logger.info('LHS[%d]; usind mod. eq., before projection:\n%s',
                    spirp, LHS_mod)
        LHS_mod = np.matmul(Proj, LHS_mod)
        logger.info('LHS[%d]; usind mod. eq.:\n%s',
                    spirp, LHS_mod)
        all_true = all_true and np.allclose(LHS, LHS_mod,
                                            rtol=1e-05, atol=1e-08)
        all_true = all_true and np.allclose(LHS, RHS,
                                            rtol=1e-03, atol=1e-03)
        all_true = all_true and np.allclose(U_T_eta, np.zeros(U_T_eta.shape),
                                            rtol=1e-05, atol=1e-08)
    return all_true
