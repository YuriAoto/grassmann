import os
import sys

import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((6,3)))
U.append(np.zeros((2,0)))
U.append(np.zeros((2,0)))
U.append(np.zeros((0,0)))


U[0][0,0] = 0.332
U[0][1,0] = 0.441232
U[0][2,0] = 0.22
U[0][3,0] = 0.54
U[0][4,0] = 0.233
U[0][5,0] = 0.11

U[0][0,1] = 0.11
U[0][1,1] = -0.441232
U[0][2,1] = 0.332
U[0][3,1] = 0.11312
U[0][4,1] = 0.0931321
U[0][5,1] = -0.5421

U[0][0,2] = -0.332
U[0][1,2] = 0.73128
U[0][2,2] = -0.4431
U[0][3,2] = 0.5434
U[0][4,2] = -0.5432
U[0][5,2] = 0.2122

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
