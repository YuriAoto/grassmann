"""Some functions to work with indices of arrays



"""
from libc.math cimport sqrt, floor


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


cpdef (int, int) get_ij_from_triang(int n, with_diag=True):
    """Returns (i,j). Inverse of get_n_from_triang"""
    i = int(floor((sqrt(1 + 8 * n) - 1) / 2))
    j = n - i * (i + 1) // 2
    if not with_diag:
        i += 1
    return i, j


cpdef inline int get_n_from_triang(int i, int j, with_diag=True):
    """Return the position in a triangular arrangement (i>=j):
    
    with_diag=True:
    
    0,0                      0
    1,0  1,1                 1  2
    2,0  2,1  2,2            3  4  5
    3,0  3,1  3,2   3,3      6  7  8  9
    ...  i,j
    
    with_diag=False:
    
    1,0               0
    2,0  2,1          1  2
    3,0  3,1  3,2     3  4  5
    ...  i,j
    
    """
    if with_diag:
        return j + triangular(i)
    else:
        return j + triangular(i - 1)


cpdef inline int get_pos_from_rectangular(int i, int a, int n):
    """Returns i*n + a (position in row-major, C order)
    
    i,a                           pos
    
    0,0   0,1   ...   0,n-1       0    1  ...   n-1
    1,0   1,1   ...   1,n-1       n  n+1  ...   2n-1
    2,0   2,1   ...   2,n-1      2n 2n+1  ...   3n-1
    ....        i,a                     i*n + a
    """
    return i * n + a


cpdef inline int get_ia_from_rectangular(int pos, int n):
    """Returns (i,a). Inverse of get_pos_from_rectangular"""
    return pos // n, pos % n
