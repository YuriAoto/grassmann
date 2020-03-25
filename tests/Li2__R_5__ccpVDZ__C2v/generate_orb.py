import os
import sys

import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((14,3)))
U.append(np.zeros((6,0)))
U.append(np.zeros((6,0)))
U.append(np.zeros((2,0)))


U[0][0,0] = 0.332
U[0][1,0] = -0.441232
U[0][2,0] = 0.22
U[0][3,0] = 0.54
U[0][4,0] = -0.233
U[0][5,0] = 0.11
U[0][6,0] = 0.5523
U[0][7,0] = -0.21212
U[0][8,0] = 0.909956
U[0][9,0] = 0.09987
U[0][10,0] = -0.1232
U[0][11,0] = 0.187576
U[0][12,0] = 0.2131
U[0][13,0] = -0.123

U[0][0,1] = 0.12
U[0][1,1] = 0.3
U[0][2,1] = -0.21
U[0][3,1] = -0.43
U[0][4,1] = 0.1
U[0][5,1] = 0.13
U[0][6,1] = 0.4132
U[0][7,1] = 0.1
U[0][8,1] = -0.35
U[0][9,1] = 0.1
U[0][10,1] = 0.4
U[0][11,1] = -0.43531
U[0][12,1] = 0.43
U[0][13,1] = -0.14

U[0][0,2] = -0.55
U[0][1,2] = 0.62
U[0][2,2] = -0.1
U[0][3,2] = 0.13
U[0][4,2] = 0.8
U[0][5,2] = 0.1
U[0][6,2] = -0.51
U[0][7,2] = 0.8
U[0][8,2] = 0.12
U[0][9,2] = -0.22
U[0][10,2] = 0.2
U[0][11,2] = 0.11
U[0][12,2] = -0.11
U[0][13,2] = 0.887


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
