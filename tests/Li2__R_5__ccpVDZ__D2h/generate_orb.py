import os
import sys

import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((7,2)))
U.append(np.zeros((3,0)))
U.append(np.zeros((3,0)))
U.append(np.zeros((1,0)))
U.append(np.zeros((7,1)))
U.append(np.zeros((3,0)))
U.append(np.zeros((3,0)))
U.append(np.zeros((1,0)))


U[0][0,0] = 0.332
U[0][1,0] = -0.441232
U[0][2,0] = 0.22
U[0][3,0] = 0.54
U[0][4,0] = 0.233
U[0][5,0] = 0.11
U[0][6,0] = -0.11

U[0][0,1] = 0.12
U[0][1,1] = -0.23
U[0][2,1] = 0.41
U[0][3,1] = -0.25
U[0][4,1] = 0.3
U[0][5,1] = 0.6
U[0][6,1] = -0.7

U[4][0,0] = -0.09
U[4][1,0] = 0.3
U[4][2,0] = -0.1
U[4][3,0] = -0.11
U[4][4,0] = 0.4
U[4][5,0] = -0.2
U[4][6,0] = 0.4


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
