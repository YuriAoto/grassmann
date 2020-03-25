import os
import sys

import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((14,2)))
U.append(np.zeros((14,1)))


U[0][0,0] = 0.332
U[0][1,0] = .73
U[0][2,0] = 0.723
U[0][3,0] = 0.21
U[0][4,0] = -0.123
U[0][5,0] = 0.86
U[0][6,0] = -0.21
U[0][7,0] = -0.64
U[0][8,0] = 0.12
U[0][9,0] = 0.86
U[0][10,0] = -0.12
U[0][11,0] = 0.10
U[0][12,0] = 0.01
U[0][13,0] = -0.123

U[0][0,1] = 0.12
U[0][1,1] = 0.3
U[0][2,1] = -0.64
U[0][3,1] = -0.21
U[0][4,1] = 0.15
U[0][5,1] = -0.13
U[0][6,1] = 0.4132
U[0][7,1] = 0.212
U[0][8,1] = -0.32
U[0][9,1] = 0.1
U[0][10,1] = 0.32
U[0][11,1] = -0.12
U[0][12,1] = -0.43
U[0][13,1] = -0.43

U[1][0,0] = -0.55
U[1][1,0] = 0.32
U[1][2,0] = -0.1
U[1][3,0] = 0.143
U[1][4,0] = 0.8
U[1][5,0] = 0.55
U[1][6,0] = -0.33
U[1][7,0] = 0.8
U[1][8,0] = 0.43
U[1][9,0] = -0.22
U[1][10,0] = 0.432
U[1][11,0] = 0.12
U[1][12,0] = -0.13
U[1][13,0] = 0.32


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
