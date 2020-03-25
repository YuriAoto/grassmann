import os
import sys

import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((10,3)))


U[0][0,0] = 0.332
U[0][1,0] = -0.441232
U[0][2,0] = 0.22
U[0][3,0] = 0.54
U[0][4,0] = 0.233
U[0][5,0] = -0.11
U[0][6,0] = 0.14
U[0][7,0] = 0.1532
U[0][8,0] = -0.32
U[0][9,0] = 0.1312

U[0][0,1] = -0.44312
U[0][1,1] = -0.33
U[0][2,1] = 0.423
U[0][3,1] = 0.4
U[0][4,1] = 0.08645
U[0][5,1] = 0.14400
U[0][6,1] = -0.233
U[0][7,1] = 0.24
U[0][8,1] = 0.67
U[0][9,1] = -0.42

U[0][0,2] = 0.00864
U[0][1,2] = 0.52342
U[0][2,2] = 0.789
U[0][3,2] = -0.1112
U[0][4,2] = 0.1782
U[0][5,2] = -0.12673
U[0][6,2] = 0.34
U[0][7,2] = 0.233
U[0][8,2] = 0.555
U[0][9,2] = -0.111


zero_skip_linalg = 1.0E-8

for i, Ui in enumerate(U):
    norm_Ui = linalg.norm(Ui)
    if norm_Ui > zero_skip_linalg:
        U[i] = linalg.orth(Ui)
    print(U[i].T @ U[i])


np.savez(os.path.dirname(os.path.abspath('__file__'))
         + '/restricted_ini_U', *U)

#for i in range(8):
#    U.append(np.array(U[i]))

#np.savez('unrestricted_ini_U', *U)

exit()
