"""Some functions to work with indices of arrays



"""
from libc.math cimport sqrt, floor
import cython

# def get_I(n, i=None, a=None):
#     """Return range(n).remove(i) + [a]"""
#     if type(i) != type(a):
#         raise ValueError('Both i and a must be of same type!')
#     if i is None:
#         return list(range(n))
#     if isinstance(i, int):
#         return [x for x in range(n) if x != i] + [a]
#     else:
#         return [x for x in range(n) if x not in i] + a


cpdef inline int triangular(int n):
    r"""The n-th trianglar number = \sum_i^n i"""
    return ((n + 1) * n) // 2


cpdef inline int n_from_triang(int i, int j):
    """Return the position in a triangular arrangement (i < j), i runs faster
    
    0,1                    0
    0,2  1,2               1   2
    0,3  1,3  2,3          3   4   5
    0,4  1,4  2,4  3,4     6   7   8   9
    ...  i,j
    
    """
    return i + triangular(j - 1)


cpdef inline (int, int) ij_from_triang(int n):
    """Returns (i,j). Inverse of n_from_triang"""
    cdef int j = <int>(floor((sqrt(1 + 8 * n) - 1) / 2))
    return n - j * (j + 1) // 2, j + 1


cpdef inline int n_from_triang_with_diag(int i, int j):
    """Return the position in a triangular arrangement (i <= j), i runs faster
    
    0,0                          0
    0,1  1,1                     1   2
    0,2  1,2  2,2                3   4   5
    0,3  1,3  2,3  3,3           6   7   8   9
    0,4  1,4  2,4  3,4  4,4     10  11  12  13  14
    ...  i,j
    """
    return i + triangular(j)


cpdef inline (int, int) ij_from_triang_with_diag(int n):
    """Return (i, j). Inverse of n_from_triang_with_diag"""
    cdef int j = <int>(floor((sqrt(1 + 8 * n) - 1) / 2))
    return n - j * (j + 1) // 2, j


cpdef inline int n_from_rect(int i, int j, int m):
    """Return i*m + j (position in row-major, C order), j runs faster
    
    i, j                          n
    
    0,0   0,1   ...   0,m-1       0    1  ...    m-1
    1,0   1,1   ...   1,m-1       m  m+1  ...   2m-1
    2,0   2,1   ...   2,m-1      2m 2m+1  ...   3m-1
    ....        i,j                     i*m + j
    """
    return i * m + j


@cython.cdivision(True)
cpdef inline (int, int) ij_from_rect(int n, int m):
    """Returns (i, j). Inverse of n_from_rect"""
    return n // m, n % m
