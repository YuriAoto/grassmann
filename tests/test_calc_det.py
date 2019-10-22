import sys
import datetime
import time

from bitarray import bitarray
import numpy  as np
from scipy import linalg

sys.path.append('/home/yuriaoto/Documents/Codes/min_dist_Gr/')

from dGr_util import logtime


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
    with logtime('init numpy') as T_init:
        if j == l or i == k:
            return 0.0
        if i not in det_indices or k not in det_indices:
            return 0.0
        sign = 1.0 if ((i<k) == (j<l)) else -1.0
        if U.shape == 2:
            return sign
    with logtime('sign numpy') as T_sign:
        if int(j + np.where(det_indices == i)[0] + l + np.where(det_indices == k)[0]) % 2 == 1:
            sign = -sign
    with logtime('indices numpy') as T_indices:
        row_ind = np.array([x for x in det_indices       if (x!=i and x!=k)], dtype=int)
        col_ind = np.array([x for x in range(U.shape[1]) if (x!=j and x!=l)], dtype=int)
    with logtime('det numpy') as T_det:
        det = linalg.det(U[row_ind[:,None],col_ind])
    return sign * det, T_init, T_sign, T_indices, T_det


def _calc_H_bitarr(U, det_indices, i, j, k, l):
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
    
    det_indices (bitarray)
        A bitarray indicating the indices of the submatrix
    
    i,j,k,l (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    with logtime('init bitarray') as T_init:
        if j == l or i == k:
            return 0.0
        if not det_indices[i] or not det_indices[k]:
            return 0.0
        sign = 1.0 if ((i<k) == (j<l)) else -1.0
        if U.shape[1] == 2:
            return sign
    with logtime('sign bitarray') as T_sign:
        if (j + det_indices[:i].count() + l + det_indices[:k].count()) % 2 == 1:
            sign = -sign
    with logtime('indices bitarray') as T_indices:
        row_ind = np.array([ii for ii,x in enumerate(det_indices) if (x and ii!=i and ii!=k)],
                           dtype=int)
        col_ind = np.array([x for x in range(U.shape[1]) if (x!=j and x!=l)], dtype=int)
    with logtime('det bitarray') as T_det:
        det = linalg.det(U[row_ind[:,None],col_ind])
    return sign * det, T_init, T_sign, T_indices, T_det



def _calc_H_bitarr_2(U, det_indices, i, j, k, l):
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
    
    det_indices (bitarray)
        A bitarray indicating the indices of the submatrix
    
    i,j,k,l (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    with logtime('init bitarray') as T_init:
        if j == l or i == k:
            return 0.0
        if not det_indices[i] or not det_indices[k]:
            return 0.0
        sign = 1.0 if ((i<k) == (j<l)) else -1.0
        if U.shape[1] == 2:
            return sign
    with logtime('sign bitarray') as T_sign:
        if (j + det_indices[:i].count() + l + det_indices[:k].count()) % 2 == 1:
            sign = -sign
    with logtime('indices bitarray') as T_indices:
        row_ind = det_indices
        row_ind[i] = False
        row_ind[k] = False
        col_ind = bitarray([True for i in range(U.shape[1])])
        col_ind[j] = False
        col_ind[l] = False
        row_ind = np.asarray(list(row_ind))
        col_ind = np.asarray(list(col_ind))
    with logtime('det bitarray') as T_det:
#        det = linalg.det(U[row_ind][:,col_ind])
        det = linalg.det(U[np.outer(row_ind[:,None],
                                    col_ind)].reshape(U.shape[1]-2,
                                                     U.shape[1]-2))
    return sign * det, T_init, T_sign, T_indices, T_det


def _calc_H_bitarr_3(U, det_indices, i, j, k, l):
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
    
    det_indices (bitarray)
        A bitarray indicating the indices of the submatrix
    
    i,j,k,l (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    with logtime('init bitarray') as T_init:
        if j == l or i == k:
            return 0.0
        if not det_indices[i] or not det_indices[k]:
            return 0.0
        sign = 1.0 if ((i<k) == (j<l)) else -1.0
        if U.shape[1] == 2:
            return sign
    with logtime('sign bitarray') as T_sign:
        if (j + det_indices[:i].count() + l + det_indices[:k].count()) % 2 == 1:
            sign = -sign
    with logtime('indices bitarray') as T_indices:
        row_ind = det_indices
        row_ind[i] = False
        row_ind[k] = False
        row_ind = np.asarray(list(row_ind))
    with logtime('det bitarray') as T_det:
        det = linalg.det(np.delete(U[row_ind],(j,l),1))
    return sign * det, T_init, T_sign, T_indices, T_det



def _calc_H_bitarr_4(U, det_indices, i, j, k, l):
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
    
    det_indices (bitarray)
        A bitarray indicating the indices of the submatrix
    
    i,j,k,l (int)
        The indices
    
    Return:
    -------
    
    det(U[det_indices, :] <-j- e_i <-l- e_k )
    """
    with logtime('init bitarray') as T_init:
        if j == l or i == k:
            return 0.0
        if not det_indices[i] or not det_indices[k]:
            return 0.0
        sign = 1.0 if ((i<k) == (j<l)) else -1.0
        if U.shape[1] == 2:
            return sign
    with logtime('sign bitarray') as T_sign:
        if (j + det_indices[:i].count() + l + det_indices[:k].count()) % 2 == 1:
            sign = -sign
    with logtime('indices bitarray') as T_indices:
        row_ind = det_indices
        row_ind[i] = False
        row_ind[k] = False
        row_ind = np.asarray(list(row_ind))
    with logtime('det bitarray') as T_det:
        mask = np.ones(U.shape[1], dtype=bool)
        mask[[j,l]] = False
        det = linalg.det(U[row_ind,:][:,mask])
    return sign * det, T_init, T_sign, T_indices, T_det


N = 1000

n=50
K=700
i,j,k,l = 0,1,2,3

U = np.reshape(np.random.random(n*K), (K,n))
det_dumb = linalg.det(U[np.arange(n)[:]])
print (det_dumb)

ini_time = time.time()

elapsed_time_load_bt = ini_time - ini_time
elapsed_time_init_bt = ini_time - ini_time
elapsed_time_sign_bt = ini_time - ini_time
elapsed_time_indices_bt = ini_time - ini_time
elapsed_time_det_bt = ini_time - ini_time
elapsed_time_total_bt = ini_time - ini_time

elapsed_time_load_np = ini_time - ini_time
elapsed_time_init_np = ini_time - ini_time
elapsed_time_sign_np = ini_time - ini_time
elapsed_time_indices_np = ini_time - ini_time
elapsed_time_det_np = ini_time - ini_time
elapsed_time_total_np = ini_time - ini_time

for test_i in range(N):
    U = np.reshape(np.random.random(n*K), (K,n))

    with logtime('Load bitarray') as T_load_bt:
        a = bitarray(K)
        a.setall(False)
        a[:n] = True

    with logtime('Calc bitarray') as T_calc_bt:
        res_bt, T_init_bt, T_sign_bt, T_indices_bt, T_det_bt = _calc_H_bitarr_4(U, a, i,j,k,l)

    elapsed_time_load_bt += T_load_bt.end_time - T_load_bt.ini_time
    elapsed_time_total_bt += T_calc_bt.end_time - T_calc_bt.ini_time
    elapsed_time_init_bt += T_init_bt.end_time - T_init_bt.ini_time
    elapsed_time_sign_bt += T_sign_bt.end_time - T_sign_bt.ini_time
    elapsed_time_indices_bt += T_indices_bt.end_time - T_indices_bt.ini_time
    elapsed_time_det_bt += T_det_bt.end_time - T_det_bt.ini_time

    U = np.reshape(np.random.random(n*K), (K,n))

    with logtime('Load numpy') as T_load_np:
        a = np.arange(n)

    with logtime('Calc numpy') as T_calc_np:
        res_np, T_init_np, T_sign_np, T_indices_np, T_det_np = _calc_H(U, a, i,j,k,l)
    
    elapsed_time_load_np += T_load_np.end_time - T_load_np.ini_time
    elapsed_time_total_np += T_calc_np.end_time - T_calc_np.ini_time
    elapsed_time_init_np += T_init_np.end_time - T_init_np.ini_time
    elapsed_time_sign_np += T_sign_np.end_time - T_sign_np.ini_time
    elapsed_time_indices_np += T_indices_np.end_time - T_indices_np.ini_time
    elapsed_time_det_np += T_det_np.end_time - T_det_np.ini_time


    
print('With numpy: [0 1 2 3 4 ... ]')
print('Elapsed time (load):', str(datetime.timedelta(seconds=elapsed_time_load_np)))
print('Elapsed time (calc):', str(datetime.timedelta(seconds=elapsed_time_total_np)))
print('Elapsed time (init):', str(datetime.timedelta(seconds=elapsed_time_init_np)))
print('Elapsed time (sign):', str(datetime.timedelta(seconds=elapsed_time_sign_np)))
print('Elapsed time (ind) :', str(datetime.timedelta(seconds=elapsed_time_indices_np)))
print('Elapsed time (det) :', str(datetime.timedelta(seconds=elapsed_time_det_np)))
print()

print('With bitarray: (111110000)')
print('Elapsed time (load):', str(datetime.timedelta(seconds=elapsed_time_load_bt)))
print('Elapsed time (calc):', str(datetime.timedelta(seconds=elapsed_time_total_bt)))
print('Elapsed time (init):', str(datetime.timedelta(seconds=elapsed_time_init_bt)))
print('Elapsed time (sign):', str(datetime.timedelta(seconds=elapsed_time_sign_bt)))
print('Elapsed time (ind) :', str(datetime.timedelta(seconds=elapsed_time_indices_bt)))
print('Elapsed time (det) :', str(datetime.timedelta(seconds=elapsed_time_det_bt)))
print()


print('bitarray/numpy:')
print('load:{:.3f}'.format(elapsed_time_load_bt   /elapsed_time_load_np   ))
print('calc:{:.3f}'.format(elapsed_time_total_bt  /elapsed_time_total_np  ))
print('init:{:.3f}'.format(elapsed_time_init_bt   /elapsed_time_init_np   ))
print('sign:{:.3f}'.format(elapsed_time_sign_bt   /elapsed_time_sign_np   ))
print('ind :{:.3f}'.format(elapsed_time_indices_bt/elapsed_time_indices_np))
print('det :{:.3f}'.format(elapsed_time_det_bt    /elapsed_time_det_np    ))
print()


exit()
