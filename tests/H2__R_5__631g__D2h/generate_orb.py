import os
import sys

import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((2,1)))
U.append(np.zeros((0,0)))
U.append(np.zeros((0,0)))
U.append(np.zeros((0,0)))
U.append(np.zeros((2,1)))
U.append(np.zeros((0,0)))
U.append(np.zeros((0,0)))
U.append(np.zeros((0,0)))


U[0][0,0] = 0.345
U[0][1,0] = 0.441232


U[4][0,0] = 0.345
U[4][1,0] = -0.74

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