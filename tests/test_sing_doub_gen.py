import os
import sys

import numpy as np

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + '/../')

from dGr_Absil import _all_doubles, _all_singles

n_el = 3
n_corr = 3
n_ext = 7

print('For n_el={}, n_corr={}, n_ext={}:'.format(
    n_el, n_corr, n_ext))
I=np.arange(n_el)
print('Reference: {}'.format(I))
print('Singles:')
n = 0
for i,a,I in _all_singles(n_el, n_corr, n_ext):
    print('i={}, a={}, I={}'.format(
    i, a, I))
    n += 1
print('Total of singles: {}'.format(n))
print('Doubles:')
n = 0
for i,j,a,b,I in _all_doubles(n_el, n_corr, n_ext):
    print('i={}, j={}, a={}, b={}, I={}'.format(
        i, j, a, b, I))
    n += 1
print('Total of doubles: {}'.format(n))
