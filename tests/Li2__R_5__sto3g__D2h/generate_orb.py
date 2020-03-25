import numpy as np
from scipy import linalg

U = []

U.append(np.zeros((3,2)))
U.append(np.zeros((1,0)))
U.append(np.zeros((1,0)))
U.append(np.zeros((0,0)))
U.append(np.zeros((3,1)))
U.append(np.zeros((1,0)))
U.append(np.zeros((1,0)))
U.append(np.zeros((0,0)))


U[0][0,0] = 0.332
U[0][1,0] = 0.441232
U[0][2,0] = 0.22

U[0][0,1] = -0.33213
U[0][1,1] = 0.332
U[0][2,1] = 0.45663

U[4][0,0] = 0.2322
U[4][1,0] = 0.654343
U[4][2,0] = 0.321121


zero_skip_linalg = 1.0E-8

for i, Ui in enumerate(U):
    norm_Ui = linalg.norm(Ui)
    if norm_Ui > zero_skip_linalg:
        U[i] = linalg.orth(Ui)
    print(U[i].T @ U[i])


np.savez('restricted_ini_U', *U)

#for i in range(8):
#    U.append(np.array(U[i]))

#np.savez('unrestricted_ini_U', *U)

exit()
